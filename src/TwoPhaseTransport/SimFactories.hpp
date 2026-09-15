// SimFactories.hpp
//  - Mesh (uniform/nonuniform × homo/hetero K)
//  - Pressure (steady): Regular or CRIS
//  - Transport (transient): Regular or CRIS
//  - Problem wrappers (Steady / DiscreteProblemFIM)
//  - InexactNewtonSolver with Standard vs LineSearch updater

#ifndef __SIMFACTORIES_HPP_
#define __SIMFACTORIES_HPP_

#include <variant>
#include <vector>
#include <array>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
//#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>


#include "SimConfig.hpp"
#include "MeshRepresentations/StructuredConnectivity.hpp"
#include "MeshRepresentations/CartesianMesh.hpp"
#include "MeshRepresentations/UnstructuredMesh.hpp"
#include "MeshRepresentations/UniformField.hpp"
#include "WellsManager.hpp"
#include <cstdlib>
#include "WellsUIParser.hpp"

#include "DiscreteProblem/Steady.hpp"
#include "DiscreteProblem/TransientBackwardEuler.hpp"
#include "Solvers/Nonlinear/InexactNewtonSolver.hpp"
#include "Solvers/Nonlinear/NewtonUpdaters.hpp"
#include "TwoPhaseTransport.hpp"
#include "LinearDiffusivitySteady.hpp" 
#include "CRIS.hpp"
#include "TPTAdapterCRIS.hpp"


namespace simfact {

    template <typename MatrixT>
    void write_matrix_market(const MatrixT& matrix, const std::string& filename)
    {
        std::ofstream file(filename);
        ASSERT_WITH_MSG(file.is_open(), "Cannot write Jacobian snapshot: " + filename);
        file << "%%MatrixMarket matrix coordinate real general\n";
        file << matrix.N() << " " << matrix.N() << " " << matrix.NNZ() << "\n";
        file << std::setprecision(17);
        for (std::size_t row = 0; row < matrix.N(); ++row)
            for (std::size_t k = matrix.rowptr()[row]; k < matrix.rowptr()[row + 1]; ++k)
                file << row + 1 << " " << matrix.colind()[k] + 1 << " " << matrix.value()[k] << "\n";
    }

    enum class Problem { Pressure, Transport };

    // Generic binary writer for StateT (vector-like with operator[])
    template <typename StateT>
    void write_state_binary(const StateT& state, const std::string& filename)
    {
        std::ofstream file(filename, std::ios::binary);
        ASSERT_WITH_MSG(file.is_open(), "Cannot open file: " + filename);

        const uint64_t n = static_cast<uint64_t>(state.size());
        file.write(reinterpret_cast<const char*>(&n), sizeof(n));
        ASSERT_WITH_MSG(!file.fail(), "Failed writing vector size to: " + filename);

        for (uint64_t i = 0; i < n; ++i) {
            double val;
            if constexpr (std::is_same_v<typename StateT::value_type, double>) {
                val = static_cast<double>(state[i]);
            }
            else {
                val = static_cast<double>(state[i].value()); // AD scalar type
            }
            file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            ASSERT_WITH_MSG(!file.fail(), "Failed writing data to: " + filename);
        }
    }

    template<class T, class... Alts>
    T& get_or_throw(std::variant<Alts...>& v, const char* msg) {
        if (!std::holds_alternative<T>(v))
            throw std::runtime_error(msg);
        return std::get<T>(v);
    }
    template<class T, class... Alts>
    const T& get_or_throw(const std::variant<Alts...>& v, const char* msg) {
        if (!std::holds_alternative<T>(v))
            throw std::runtime_error(msg);
        return std::get<T>(v);
    }

    // ============================================================================
    // 1) Mesh factory: For now implementation only exists for Structured grids with Sequential ikj policy
    // ============================================================================

    using UCartesianBFacesB = UCartesianBFaces<bool, IJKSequentialFaces>;
    using UCartesianBFacesD = UCartesianBFaces<double, IJKSequentialFaces>;
    using UCartesianIFacesD = UCartesianIFaces<double, IJKSequentialFaces>;

    using Conn = StructuredConnectivity<IJKSequentialFaces, std::size_t>;

    using GridU = CartesianGrid<Conn, UniformCartesianGeometry>;
    using GridN = CartesianGrid<Conn, NonuniformCartesianGeometry<std::size_t>>;
    using GridVariant = std::variant<GridU, GridN, UnstructuredGrid>;
	
    using CoeffVariant = std::variant<HomogeneousAnisotropicK<double>, HeterogeneousAnisotropicK<double> >;
    using UMeshU = UnstructuredMeshView< uniform_field<double>>;
    using UMeshN = UnstructuredMeshView<std::vector<double>>;


    using MeshView_UU = TPFAMeshView<GridU, HomogeneousAnisotropicK<double>, TPFAOperator, HarmonicPolicy>;
    using MeshView_UN = TPFAMeshView<GridU, HeterogeneousAnisotropicK<double>, TPFAOperator, HarmonicPolicy>;
    using MeshView_NU = TPFAMeshView<GridN, HomogeneousAnisotropicK<double>, TPFAOperator, HarmonicPolicy>;
    using MeshView_NN = TPFAMeshView<GridN, HeterogeneousAnisotropicK<double>, TPFAOperator, HarmonicPolicy>;
    using MeshViewVariant = std::variant<MeshView_UU, MeshView_UN, MeshView_NU, MeshView_NN, UMeshU, UMeshN>;

    using IFaceVariant = std::variant<std::vector<double>, UCartesianIFacesD>;
    using BFaceVariant = std::variant<std::vector<double>, UCartesianBFacesD>;
    using InitFieldVariant = std::variant < std::vector<double>, uniform_field<double> >;

    


    inline GridVariant build_grid_from_config(const SimConfig& cfg)
    {

        if (cfg.mesh_type == SimConfig::MeshType::Unstructured) {
            auto clist = read_vector_file<std::size_t>(cfg.cl_file);
            
            auto vol = read_vector_file(cfg.vol_file);
            auto depth = read_vector_file(cfg.depth_file);
            if (cfg.global_idx_file.empty())
                return UnstructuredGrid(clist, vol, depth, cfg.nx, cfg.ny, cfg.nz);
            auto global_idx_map = read_vector_file<std::size_t>(cfg.global_idx_file);
            return UnstructuredGrid(clist, vol, depth, cfg.nx, cfg.ny, cfg.nz, global_idx_map);
        }

        Conn conn(cfg.nx, cfg.ny, cfg.nz);
        const std::size_t nc = conn.num_cells();
        if (cfg.geom_kind == SimConfig::GeometryKind::Uniform) {

            UniformCartesianGeometry geom(cfg.dx, cfg.dy, cfg.dz);
            return GridU(std::move(conn), std::move(geom));
        }
        auto dx = read_vector_file(cfg.dx_file);
        auto dy = read_vector_file(cfg.dy_file);
        auto dz = read_vector_file(cfg.dz_file);
        if (dx.size() != nc || dy.size() != nc || dz.size() != nc)
            throw std::runtime_error("Nonuniform geometry size mismatch");
        NonuniformCartesianGeometry<std::size_t> geom(std::move(dx), std::move(dy), std::move(dz));
        return GridN(std::move(conn), std::move(geom));

    }
	
    inline CoeffVariant build_coeff(const SimConfig::MeshType mesh_type, const SimConfig::FieldType coeff_field_type, double kx, double ky, double kz, const std::string& kx_file,
        const std::string& ky_file, const std::string& kz_file)
    {
        if (coeff_field_type == SimConfig::FieldType::Homogeneous || mesh_type == SimConfig::MeshType::Unstructured)
            return HomogeneousAnisotropicK<double>({ kx, ky, kz });
        auto Kx = read_vector_file(kx_file);
        auto Ky = read_vector_file(ky_file);
        auto Kz = read_vector_file(kz_file);
        return HeterogeneousAnisotropicK<double>(std::move(Kx), std::move(Ky), std::move(Kz));
    }
  


	template<typename grid_t , typename coeff_t>
	inline MeshViewVariant build_mesh_view (const SimConfig& cfg, const grid_t& grid,  coeff_t& coeff, Problem problem_t)
	{
		if constexpr (std::is_same_v<grid_t, GridU>) {
			if constexpr(std::is_same_v<coeff_t, HomogeneousAnisotropicK<double>>)
				return MeshView_UU(grid, std::move(coeff));
            if constexpr(std::is_same_v<coeff_t, HeterogeneousAnisotropicK<double>>)
			    return MeshView_UN(grid, std::move(coeff));
		}
        if constexpr (std::is_same_v<grid_t, GridN >) {
            if constexpr (std::is_same_v<coeff_t, HomogeneousAnisotropicK<double>>)
                return MeshView_NU(grid, std::move(coeff));
            if constexpr (std::is_same_v<coeff_t, HeterogeneousAnisotropicK<double>>)
                return MeshView_NN(grid, std::move(coeff));
        }
        if constexpr (std::is_same_v<grid_t, UnstructuredGrid>)
        {
            //auto Tintr = read_vector_file(cfg.T_file);
            //return UMeshN(grid, Tintr);
            if (problem_t == Problem::Pressure) {
                auto Tintr = read_vector_file(cfg.T_file);
                return UMeshN(grid, Tintr);
            }
            else {
                uniform_field<double> disp{ grid.num_intr_faces(), cfg.dispx };
                return UMeshU(grid, disp);
            }
                
        }

        //return MeshView_UU(grid, std::move(coeff));
	}
	
	inline MeshViewVariant build_PMesh(const SimConfig& cfg, const GridVariant& gridv)
	{
	CoeffVariant coeffv{build_coeff(cfg.mesh_type, cfg.k_kind, cfg.kx, cfg.ky, cfg.kz, cfg.kx_file, cfg.ky_file, cfg.kz_file)};
    return std::visit([&](auto& grid, auto coeff)->MeshViewVariant {return build_mesh_view(cfg, grid, coeff, Problem::Pressure);}, gridv, coeffv);
	}		
	
	inline MeshViewVariant build_SMesh(const SimConfig& cfg, const GridVariant& gridv)
	{
	CoeffVariant coeffv{build_coeff(cfg.mesh_type, cfg.disp_kind, cfg.dispx, cfg.dispy, cfg.dispz, cfg.dispx_file, cfg.dispy_file, cfg.dispz_file)};
    return std::visit([&](auto& grid, auto coeff)->MeshViewVariant {return build_mesh_view(cfg, grid, coeff, Problem::Transport);}, gridv, coeffv);
	}	


    struct MeshBundle {
		MeshBundle(const SimConfig& cfg): grid(build_grid_from_config(cfg)), 
		pmesh(build_PMesh(cfg, grid)), smesh(build_SMesh(cfg, grid)){ }
        const GridVariant grid;
        const MeshViewVariant pmesh;
        const MeshViewVariant smesh;
    };	


    // ============================================================================
    // 2) Utility: choose Newton updater (enum -> concrete updater object)
    // ============================================================================

    using UpdaterVariant = std::variant<StandardNewtonUpdater, LineSearchUpdater>;

    inline UpdaterVariant make_updater(const SimConfig& cfg)
    {
        if (cfg.newton_updater == SimConfig::NewtonUpdater::Standard)
            return StandardNewtonUpdater{};
        return LineSearchUpdater{};
    }


    // ============================================================================
    // 3) Transport bundle (ModelVariant, TSUpdater-owned, ProblemVariant, NewtonVariant)
    // ============================================================================

    template <typename Mesh>
    IFaceVariant make_intr_vel_field(const SimConfig& cfg, const Mesh& mesh, const WellsManager<Mesh>& wm)
    {
        if (cfg.IVel_type == SimConfig::FieldType::Homogeneous && wm.wells().empty())
        {
            return UCartesianIFacesD(mesh.num_intr_faces(), cfg.intr_vel_val, mesh.layout());

        }
        else if (wm.wells().empty()) {
            return read_vector_file(cfg.intr_vel_file);
        }
        else return std::vector<double>(mesh.num_intr_faces(), 0.0);
    }
    template <typename Mesh>
    BFaceVariant make_bndr_vel_field(const SimConfig& cfg, const Mesh& mesh, const WellsManager<Mesh>& wm)
    {
        if (cfg.BVel_type == SimConfig::FieldType::Homogeneous && wm.wells().empty())
        {
            return UCartesianBFacesD(mesh.num_bndr_faces(), cfg.bndr_vel_val, mesh.layout());

        }
        else if (wm.wells().empty()) {
            return read_vector_file(cfg.bndr_vel_file);
        }
        else return UCartesianBFacesD(mesh.num_bndr_faces(), { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, mesh.layout());
    }
    template <typename Mesh>
    BFaceVariant make_bc_field(const SimConfig& cfg, const Mesh& mesh)
    {
        if (cfg.bc_type == SimConfig::FieldType::Homogeneous)
        {
            return UCartesianBFacesD(mesh.num_bndr_faces(), cfg.bndry_val, mesh.layout());

        }
        else {
            return read_vector_file(cfg.bc_file);
        }
    }
    template <typename Mesh>
    InitFieldVariant make_init_field(const SimConfig& cfg, const Mesh& mesh)
    {
        if (cfg.ic_type == SimConfig::FieldType::Homogeneous)
        {
            return uniform_field(mesh.num_cells(), cfg.init_val);

        }
        else {
            return read_vector_file(cfg.init_file);
        }
    }
    
    template<typename Mesh, typename wells_t, typename LNsolver,
        typename IFVel_t, typename BFVel_t, typename BCField_t, typename init_t>
    struct TransportBundle
    {
        using SRDMModel = TwoPhaseTransport<Mesh, init_t, BCField_t, IFVel_t, BFVel_t>;
        using NetType = FCN;
        using AdapterT = TPTAdapter<init_t, BCField_t, IFVel_t, BFVel_t>;
        using CRISModel = CRIS<NetType, Mesh, AdapterT>;

        using ModelVariant = std::variant<SRDMModel, CRISModel>;

        using SRDMProb = DiscreteProblemFIM<SRDMModel>;
        using CRISProb = DiscreteProblemFIM<CRISModel>;
        using ProblemVariant = std::variant<SRDMProb, CRISProb>;

        using NewtonStd_SRDM = InexactNewtonSolver<SRDMProb, LNsolver, StandardNewtonUpdater>;
        using NewtonLS_SRDM = InexactNewtonSolver<SRDMProb, LNsolver, LineSearchUpdater>;
        using NewtonStd_CRIS = InexactNewtonSolver<CRISProb, LNsolver, StandardNewtonUpdater>;
        using NewtonLS_CRIS = InexactNewtonSolver<CRISProb, LNsolver, LineSearchUpdater>;
        using NewtonVariant = std::variant<NewtonStd_SRDM, NewtonLS_SRDM, NewtonStd_CRIS, NewtonLS_CRIS>;

        ModelVariant   model;
        ProblemVariant prob;
        NewtonVariant  newton;

        TransportBundle(const SimConfig& cfg,
            const Mesh& mesh, const wells_t& wm,
            IFVel_t& intr_face_vel, BFVel_t& bndr_face_vel,
            BCField_t& bndr_val, UCartesianBFaces<bool, IJKSequentialFaces>& drchlt_bndr,
            init_t& uinit)
            : model(make_model(cfg, mesh, wm, intr_face_vel, bndr_face_vel, bndr_val, drchlt_bndr, uinit))
            , prob(make_problem(cfg))
            , newton(make_newton(cfg))
        {
            std::visit([&](auto& solver) {
                solver.set_refresh_transport_preconditioner(cfg.refresh_transport_preconditioner);
                solver.set_linear_solve_policy(cfg.linear_solve_policy, cfg.linear_tolerance_floor);
            }, newton);
        }

        template<class F>
        void visit_solver_problem(F&& f) {
            std::visit([&](auto& p) {
                using P = std::decay_t<decltype(p)>;

                std::visit([&](auto& s) {
                    using S = std::decay_t<decltype(s)>;

                    if constexpr (
                        (std::is_same_v<P, SRDMProb> &&
                            (std::is_same_v<S, NewtonStd_SRDM> || std::is_same_v<S, NewtonLS_SRDM>))
                        ||
                        (std::is_same_v<P, CRISProb> &&
                            (std::is_same_v<S, NewtonStd_CRIS> || std::is_same_v<S, NewtonLS_CRIS>))
                        ) {
                        f(s, p);
                    }
                    }, newton);

                }, prob);
        }

    private:
        ModelVariant make_model(const SimConfig& cfg, const Mesh& mesh, const wells_t& wm,
            IFVel_t& intr_face_vel, BFVel_t& bndr_face_vel,
            BCField_t& bndr_val, UCartesianBFacesB& drchlt_bndr, init_t& uinit)
        {
            const bool is_SRDM = (cfg.mode == SimConfig::RunMode::SRDM);
            const bool is_exact_cris = (cfg.mode == SimConfig::RunMode::CRIS_EXACT);
            auto adv_flux_func = CocurrentFracFlux{ cfg.mu_nw / cfg.mu_w, cfg.n_w, cfg.n_nw, cfg.Ng };
            auto diff_flux_func = ExponentialFlux{ cfg.lambda };

            if (is_SRDM) {
                return SRDMModel(mesh,
                    std::move(adv_flux_func), std::move(diff_flux_func),
                    std::move(intr_face_vel), std::move(bndr_face_vel),
                    wm.wells(),
                    std::move(bndr_val), std::move(drchlt_bndr),
                    std::move(uinit),
                    cfg.RTOL, cfg.DUTOL);
            }
            else if (is_exact_cris) {
                ASSERT_WITH_MSG(cfg.n_w == 2.0 && (cfg.n_nw == 2.0 || cfg.n_nw == 0.2) && cfg.Ng == 0.0,
                    "MODE CRIS_EXACT supports regular or IMP advection with n_w=2, Ng=0");
                auto net = std::make_shared<NetType>(cfg.mu_nw / cfg.mu_w, cfg.n_nw == 0.2);
                AdapterT adapt(std::move(adv_flux_func), std::move(diff_flux_func),
                    std::move(intr_face_vel), std::move(bndr_face_vel),
                    wm.wells(),
                    std::move(bndr_val), std::move(drchlt_bndr),
                    std::move(uinit), cfg.embedding_trace_file);
                return CRISModel(std::move(net), mesh, std::move(adapt), cfg.RTOL, cfg.DUTOL);
            }
            else {
                std::vector<LayerDesc> layers;
                std::vector<double> IR, OR, params;
                std::vector<int> input_transform_codes;
                parse_network(cfg.cris_model_path, layers, IR, OR, params, &input_transform_codes);
                auto net = std::make_shared<NetType>(layers, IR, OR, params.data(), input_transform_codes);

                AdapterT adapt(std::move(adv_flux_func), std::move(diff_flux_func),
                    std::move(intr_face_vel), std::move(bndr_face_vel),
                    wm.wells(),
                    std::move(bndr_val), std::move(drchlt_bndr),
                    std::move(uinit), cfg.embedding_trace_file);

                return CRISModel(std::move(net), mesh, std::move(adapt), cfg.RTOL, cfg.DUTOL);
            }
        }

        ProblemVariant make_problem(const SimConfig& cfg)
        {
            const double dt_init = cfg.dt_init;
            return std::visit([&](auto& m) -> ProblemVariant {
                using M = std::decay_t<decltype(m)>;
                // IMPORTANT: DiscreteProblemFIM should take TSUpdater by VALUE (recommended).
                return DiscreteProblemFIM<M>(m, TSUpdater{ cfg.dt_max, cfg.dt_min, cfg.dt_incr, cfg.dt_decr }, dt_init);
                }, model);
        }

        NewtonVariant make_newton(const SimConfig& cfg)
        {
            const bool use_ls = (cfg.newton_updater == SimConfig::NewtonUpdater::Linesearch);
            //const int verbosity = 3;
            //const double eta = 1e-8;
            return std::visit([&](auto& p) -> NewtonVariant {
                using P = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<P, SRDMProb>) {
                    if (!use_ls)
                        return NewtonStd_SRDM(p, cfg.RTOL, cfg.DUTOL, cfg.max_steps_newton,cfg.LINMAX, cfg.verbosity, cfg.eta, 0.9, 2.0, StandardNewtonUpdater{});
                    else
                        return NewtonLS_SRDM(p, cfg.RTOL, cfg.DUTOL, cfg.max_steps_newton, cfg.LINMAX, cfg.verbosity, cfg.eta, 0.9, 2.0, LineSearchUpdater{});
                }
                else {
                    if (!use_ls)
                        return NewtonStd_CRIS(p, cfg.RTOL, cfg.DUTOL, cfg.max_steps_newton, cfg.LINMAX, cfg.verbosity, cfg.eta, 0.9, 2.0, StandardNewtonUpdater{});
                    else
                        return NewtonLS_CRIS(p, cfg.RTOL, cfg.DUTOL, cfg.max_steps_newton, cfg.LINMAX, cfg.verbosity, cfg.eta, 0.9, 2.0, LineSearchUpdater{});
                }
                }, prob);
        }
    };

   
    // ============================================================================
    // 4) Pressure bundle (Steady wrapper, Newton solver)
    // ============================================================================

    template<class Mesh, class LNsolver>
    struct PressureBundle
    {
        using Model = LinearDiffusivitySteady<Mesh>;
        using Problem = discrete_systems::Steady<Model>;
        using Newton = InexactNewtonSolver<Problem, LNsolver, StandardNewtonUpdater>;

        Model   model;
        Problem prob;
        Newton  newton;

        PressureBundle(const SimConfig& cfg, const Mesh& mesh, WellsManager<Mesh>& wm)
            : model(mesh, wm, cfg.rho, cfg.g, 1e-3, cfg.DUTOL)
            , prob(model)
            , newton(prob,
                /*residTol*/ 1e-3,
                /*updateTol*/ cfg.DUTOL,
                /*maxiter*/ 100000, cfg.pressure_linear_budget(),
                /*verbosity*/ cfg.verbosity,
                /*eta0*/ 1e-6, /*gamm*/ 0.9, /*omeg*/ 2.0,
                StandardNewtonUpdater{})
        {
        }
    };

    // ============================================================================
    // 5) Sequential PT composition
    // ============================================================================

    template <typename Solver, typename Problem, typename State>
    inline void solve_pressure(Solver& solver,  Problem& prob, State& p,
        std::vector<double>& intr_face_vel)
    {
        std::cout << "Starting Pressure solve" << std::endl;
        prob.initial_guess(p);
        auto rep = solver.solve(prob, p);
        std::cout << "Pressure solve Done" << std::endl;
        if (rep.is_failed)
            throw std::runtime_error("Pressure solve failed");
        prob.model().update_flow(p, intr_face_vel);
        //for (auto vel : intr_face_vel)
        //    std::cout << vel << ", ";
        std::cout << std::endl;
    }


    template <typename Solver, typename Problem, typename State>
    inline void run_transport_to_event(const SimConfig& cfg, Solver& solver, Problem& prob, State& u, double t_event, std::ofstream& perf_rprt_csv, std::ofstream& wells_rprt_csv, const std::vector<Well>& wells, bool& snapshot_written)
    {
        prob.next_event(t_event);
        while (prob.accepted_time() < t_event) {
            std::cout << "Time till next event: " << t_event - prob.accepted_time() << std::endl;
            prob.initialize_timestep(u);

            auto rep = solver.solve(prob, u);
            if (cfg.write_jacobian_snapshot && !snapshot_written) {
                write_matrix_market(solver.initial_jacobian(), cfg.output_folder + "/jacobian_initial.mtx");
                snapshot_written = true;
            }
            if (perf_rprt_csv.is_open())
                perf_rprt_csv << prob.DT() << ","
                << (rep.is_converged && !rep.is_failed ? 1 : 0) << ","
                << rep.niter_nln << ","
                << rep.niter_ln << ","
                << rep.solve_s << ","
                << rep.nfeval << ","
                << rep.n_linear_tolerance_misses << ","
                << rep.n_budget_continuations << ","
                << rep.n_linear_breakdowns << ","
                << rep.last_linear_tolerance << ","
                << rep.ln_report.relative_residual << std::endl;
            const bool converged = rep.is_converged && !rep.is_failed;
            // Opt-in diagnostic: preserve the attempted state before timestep
            // rejection restores the previous state. Never label it accepted.
            const char* snapshot_probe = std::getenv("TPT_SAVE_FIRST_ATTEMPT");
            if (snapshot_probe && std::string(snapshot_probe) == "1") {
                write_state_binary(u, cfg.output_folder + "/attempted_saturation.bin");
                std::cout << "Diagnostic attempted-state snapshot saved; stopping before timestep acceptance/rejection." << std::endl;
                throw std::runtime_error("Diagnostic stop after first transport attempt");
            }
            if (!prob.advance(u, converged))
                throw std::runtime_error("Transport dt too small; Aborting Simulation.");

            if (converged && wells_rprt_csv.is_open()) {
                wells_rprt_csv << prob.accepted_time();
                for (auto& well : wells) {
                    if (well.type == WellType::Producer)
                    {
                        double SwA = 0.0;
                        for (std::size_t ci = 0; ci < well.comps.size(); ++ci)
                        {

                            const auto& comp = well.comps[ci];
                            const std::size_t l = comp.cell;
                            const double q = well.rates.q_comp[ci];
                            double u_l;
                            if constexpr (std::is_same_v<double, std::decay_t<decltype(u[l])>>)
                                u_l = u[l];
                            else
                                u_l = u[l].value();
                            SwA += q * std::clamp(u_l, 0.0, 1.0);
                        }
                        SwA /= well.rates.q_total;
                        wells_rprt_csv << "," << SwA;
                    }

                }
                wells_rprt_csv << "\n";
            }
        }
        std::cout << "Nonlinear steps: " << solver.full_stats.niter_nln <<", Fevals: "<<solver.full_stats.nfeval<< ", Linear steps: " << solver.full_stats.niter_ln <<", LinSolve time (s) : "<<solver.full_stats.solve_s <<", Wasted nonlinear steps: " << solver.full_stats.n_failed << std::endl;
    }

    // Transport-only time loop until t_end
    template <typename Mesh, typename NLNSolver, typename ProbT>
    inline void run_transport_to_time(const SimConfig& cfg, Mesh& mesh, NLNSolver& solver, ProbT& prob, std::ofstream& perf_rprt_csv)
    {
        std::ofstream dummy_stream;
        using SolverT = std::decay_t<decltype(solver)>;
        //using ProbT = std::decay_t<decltype(prob)>;
        static_assert(std::is_same_v<typename SolverT::Problem, ProbT>,
            "Solver/Problem type mismatch in variant visitation");
        using StateT = typename ProbT::State;
        StateT u{ prob.initialize_state() };
        bool snapshot_written = false;
        double t_next_report = cfg.report_freq;
        write_state_binary(u, cfg.output_folder + "/saturation_0.bin");
        while (t_next_report < cfg.t_end)
        {
            run_transport_to_event(cfg, solver, prob, u, t_next_report, perf_rprt_csv, dummy_stream, std::vector<Well>{}, snapshot_written);
            std::cout << "Reporting " << std::endl;
            write_state_binary(u, cfg.output_folder + "/saturation_" + std::to_string(static_cast<int>(t_next_report)) + ".bin");
            t_next_report += cfg.report_freq;

        }
        run_transport_to_event(cfg, solver, prob, u, cfg.t_end, perf_rprt_csv, dummy_stream, std::vector<Well>{}, snapshot_written);
        write_state_binary(u, cfg.output_folder + "/saturation_" + std::to_string(static_cast<int>(cfg.t_end)) + ".bin");
    }

    // Conservative maximum-cell Courant coefficient for the transport update.
    // Multiplying this by a trial timestep gives the dimensionless CFL used in
    // result plots.  Face throughput is accumulated on both adjacent cells;
    // well throughput is included cell-wise.
    template <typename MeshT, typename WellsT>
    inline std::pair<double, double> cell_courant_rates(const MeshT& mesh, const std::vector<double>& intr_face_vel,
        const WellsT& wells)
    {
        std::vector<double> throughput(mesh.num_cells(), 0.0);
        for (std::size_t f = 0; f < mesh.num_intr_faces(); ++f) {
            const auto [l1, l2] = mesh.intr_face_to_cells(f);
            const double q = std::abs(intr_face_vel[f]);
            throughput[l1] += q;
            throughput[l2] += q;
        }
        for (const auto& well : wells)
            for (std::size_t c = 0; c < well.comps.size(); ++c)
                throughput[well.comps[c].cell] += std::abs(well.rates.q_comp[c]);

        double maximum = 0.0, sum = 0.0;
        for (std::size_t cell = 0; cell < mesh.num_cells(); ++cell) {
            const double rate = throughput[cell] / mesh.cell_volume(cell);
            maximum = std::max(maximum, rate);
            sum += rate;
        }
        return { maximum, sum / mesh.num_cells() };
    }





    template <typename PMeshT, typename SMeshT, typename PNLNSolver, typename SNLNSolver, typename PProbT, typename SProbT>
    inline void run_sequential_PT(const SimConfig& cfg,  WellsManager<PMeshT>& wm, const PMeshT& pmesh, const SMeshT& smesh,
        PNLNSolver& psolver, SNLNSolver& ssolver, PProbT& pprob, SProbT& sprob, std::ofstream& prf_rprt_csv, std::ofstream& wells_rprt_csv )
    {
        using SolverT = std::decay_t<decltype(ssolver)>;
        using ProbT = std::decay_t<decltype(sprob)>;
        static_assert(std::is_same_v<typename SolverT::Problem, ProbT>,
            "Solver/Problem type mismatch in variant visitation");
        using PStateT = typename PProbT::State;
        using SStateT = typename SProbT::State;
        using iFVEl_t = std::decay_t<decltype(sprob.model().get_intr_vel())>;
        static_assert(std::is_same_v<iFVEl_t, std::vector<double>>,
            "Solver/Problem type mismatch in variant visitation");
        std::vector<double>& intr_face_vel = sprob.model().get_intr_vel();
        SStateT s{ sprob.initialize_state() };
        
        PStateT p{ pprob.initialize_state() };
        
        double t_next_report = cfg.report_freq;
        double t_next_well_event = wm.update_event_time(); //Modify to update well state with time event
        double t_event{ 0 };
        bool snapshot_written = false;
        std::ofstream cfl_rprt_csv(cfg.output_folder + "/cfl_reference.csv");
        cfl_rprt_csv << "TIME,MAX_CELL_COURANT_RATE,MEAN_CELL_COURANT_RATE\n";
        solve_pressure(psolver, pprob, p, intr_face_vel);
        const auto [max_cfl_rate, mean_cfl_rate] = cell_courant_rates(smesh, intr_face_vel, wm.wells());
        cfl_rprt_csv << std::setprecision(17) << t_event << "," << max_cfl_rate << "," << mean_cfl_rate << "\n" << std::flush;
        std::cout << "Reporting " << std::endl;
        write_state_binary(p, cfg.output_folder + "/pressure_" + std::to_string(static_cast<int>(t_event)) + ".bin");


        while (true) {
            t_event = std::min({ t_next_report, t_next_well_event, cfg.t_end });
            run_transport_to_event(cfg, ssolver, sprob, s, t_event, prf_rprt_csv, wells_rprt_csv, wm.wells(), snapshot_written );
            //t_current_well_event = t_next_well_event;

            //t = sprob.accepted_time();
            if (t_event == cfg.t_end)
            {
                std::cout << "Simulation Successfully Completed" << std::endl;
                std::cout << "Reporting " << std::endl;
                write_state_binary(s, cfg.output_folder + "/saturation_" + std::to_string(static_cast<int>(t_event)) + ".bin");
                break;
            }
            if (t_event == t_next_report)
            {
                //save result
                std::cout << "Reporting " << std::endl;
                write_state_binary(s, cfg.output_folder + "/saturation_" + std::to_string(static_cast<int>(t_next_report)) + ".bin");
                t_next_report += cfg.report_freq;
            }
            if (t_event == t_next_well_event)
            {
                //update well time to new_well_time_event
                t_next_well_event = wm.update_event_time(); //modify to update well state with time event
                solve_pressure(psolver, pprob, p, intr_face_vel);
                const auto [max_cfl_rate, mean_cfl_rate] = cell_courant_rates(smesh, intr_face_vel, wm.wells());
                cfl_rprt_csv << std::setprecision(17) << t_event << "," << max_cfl_rate << "," << mean_cfl_rate << "\n" << std::flush;
                std::cout << "Reporting " << std::endl;
                write_state_binary(p, cfg.output_folder + "/pressure_" + std::to_string(static_cast<int>(t_event)) + ".bin");
                 
            }

            

        }


    }

}

#endif // __SIMFACTORIES_HPP_
