#include "WellsManager.hpp"
#include "SimConfig.hpp"
#include "SimFactories.hpp"
#include <type_traits>
#include "Solvers/Linear/AMGCL_Solver.hpp"

#include <string>

int main(int argc, char** argv)
{
    const std::string case_folder = (argc > 1) ? std::string(argv[1]) : std::string("case");
    std::cout << "Starting Simulation" << std::endl;

    const SimConfig cfg{ parse_sim_config(case_folder) };
    std::cout << "Config file Parsed" << std::endl;
    std::cout << "Transport linear policy: "
        << (cfg.linear_solve_policy == LinearSolvePolicy::Strict ? "STRICT" : "CONTINUE_ON_BUDGET")
        << ", LINTOL_MIN=" << cfg.linear_tolerance_floor << std::endl;
    std::cout << "Linear budgets: transport=" << cfg.LINMAX
        << ", pressure=" << cfg.pressure_linear_budget() << std::endl;

    using LNsolver = AMGCL<>;
    simfact::MeshBundle gridBundle{ cfg };
    std::cout << "Grid generated" << std::endl;
    std::ofstream prf_rprt_csv(cfg.output_folder + "/solver_report.csv");
    prf_rprt_csv << std::setprecision(17);
    std::ofstream well_rprt_csv;
    if (prf_rprt_csv.is_open())
        prf_rprt_csv << "DT,converged,NLNSTEPS,LINSTEPS,LNSOLVE TIME (S),NFEVAL,LINEAR_TOL_MISSES,BUDGET_CONTINUATIONS,LINEAR_BREAKDOWNS,LAST_LINTOL,LAST_LINEAR_RELRES\n";

    std::visit([&](auto& /*grid*/, auto& pmesh, auto& smesh)
        {
            using PMeshT = std::decay_t<decltype(pmesh)>;
            using SMeshT = std::decay_t<decltype(smesh)>;

            WellsManager<PMeshT> wm(pmesh);
            std::cout << "Wells Created" << std::endl;

            if (!cfg.wells_file.empty()) {
                well_rprt_csv = std::ofstream(cfg.output_folder + "/well_report.csv");
                if (well_rprt_csv.is_open())
                    well_rprt_csv << "Time";

                std::cout << "Populating Wells" << std::endl;
                auto specs = wellparser::parse_wells_file(cfg.wells_file); // std::vector<std::variant<...>>
                std::cout << "Wells file parsed" << std::endl;

                for (auto& v : specs) {
                    std::visit([&](auto&& s) {
                        using SpecT = std::remove_reference_t<decltype(s)>;

                        // Report header: only producers
                        if (well_rprt_csv.is_open()) {
                            if (s.type == WellType::Producer)
                                well_rprt_csv << "," << s.name;
                        }

                        // Dispatch to correct add_* based on spec type
                        if constexpr (std::is_same_v<SpecT, VerticalWellSpec>) {
                            wm.add_vertical_well(std::move(s));
                        }
                        else if constexpr (std::is_same_v<SpecT, GenericWellSpec>) {
                            wm.add_generic_well(std::move(s));
                        }
                        else {
                            static_assert(!sizeof(SpecT), "Unhandled well spec type in variant");
                        }
                        }, v);
                }

                if (well_rprt_csv.is_open())
                    well_rprt_csv << "\n";

                std::cout << "Wells Populated" << std::endl;
            }


            auto intr_face_vel = simfact::make_intr_vel_field(cfg, pmesh, wm);
            auto bndr_face_vel = simfact::make_bndr_vel_field(cfg, pmesh, wm);
            auto bndr_val = simfact::make_bc_field(cfg, pmesh);
            auto db = simfact::UCartesianBFacesB(pmesh.num_bndr_faces(), cfg.drchlet_bndry, pmesh.layout());
            auto uinit = simfact::make_init_field(cfg, pmesh);
            std::cout << "Fields generated" << std::endl;
            simfact::PressureBundle<PMeshT, PLNSolver> pb(cfg, pmesh, wm);

            std::visit([&](auto& iv, auto& bv, auto& bc, auto& ic)
                {
                    using IFVel_t = std::decay_t<decltype(iv)>;
                    using BFVel_t = std::decay_t<decltype(bv)>;
                    using BCField_t = std::decay_t<decltype(bc)>;
                    using init_t = std::decay_t<decltype(ic)>;

                    simfact::TransportBundle<SMeshT, WellsManager<PMeshT>, TLNSolver, IFVel_t, BFVel_t, BCField_t, init_t>
                        tb(cfg, smesh, wm, iv, bv, bc, db, ic);
                    std::cout << "Created Transport Problem" << std::endl;
                    tb.visit_solver_problem([&](auto& ssolver, auto& sprob)
                        {
                            if (cfg.wells_file.empty()) {
                                std::cout << "Starting Transport Simulation" << std::endl;
                                simfact::run_transport_to_time(cfg, smesh, ssolver, sprob, prf_rprt_csv);
                            }
                            else {
                                
                                // sequential PT requires intr_face_vel to be std::vector<double>
                                (void)simfact::get_or_throw<std::vector<double>>(intr_face_vel,
                                    "Sequential PT requires intr_face_vel as std::vector<double>");

                                if constexpr (std::is_same_v<IFVel_t, std::vector<double>>) {
                                    std::cout << "Starting Sequential Pressure/Transport Simulation" << std::endl;
                                    simfact::run_sequential_PT(cfg, wm, pmesh, smesh,
                                        pb.newton, ssolver,
                                        pb.prob, sprob, prf_rprt_csv, well_rprt_csv);
                                }
                            }
                        });

                }, intr_face_vel, bndr_face_vel, bndr_val, uinit);

        }, gridBundle.grid, gridBundle.pmesh, gridBundle.smesh);
    std::cout << "Simulation Completed Successfully" << std::endl;
    return 0;
}
