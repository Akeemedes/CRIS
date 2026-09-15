#ifndef __TWOPHASETRANSPORT_HPP_
#define __TWOPHASETRANSPORT_HPP_

#include <cstddef>
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>

#include "DiscreteProblem/MultivariableNorm.hpp"
#include "FluxFunctors.hpp"
#include "MeshRepresentations/UniformField.hpp"
#include "WellsManager.hpp"
#include "CRIS/CSR_Pattern.hpp"

template < 
    typename TPFA_Mesh, 
    typename init_t    = std::vector<double>, 
    typename BCField_t = std::vector<double>,
    typename IFVel_t = std::vector<double>, 
    typename BFVel_t = std::vector<double>,  
    typename FFunctor  = CocurrentFracFlux, 
    typename GFunctor  = ExponentialFlux
>
class TwoPhaseTransport {
public:
    using State = std::vector<double>;
    //using ResidualNorm = MultivariableNorm<1, TwoNormFunct>;
    using ResidualNorm = MultivariableNorm<1, InfNormFunct>;
    using UpdateNorm = MultivariableNorm<1, InfNormFunct>;
    using DBField_t = UCartesianBFaces<bool, IJKSequentialFaces>;
    
    TwoPhaseTransport(const TPFA_Mesh& _mesh,
        const FFunctor& _adv_flux_func,
        const GFunctor& _diff_flux_func,
        const IFVel_t& _intr_face_vel,
        const BFVel_t& _bndr_face_vel,
        const std::vector<Well>& _wells,
        const BCField_t& _bndr_val,
        const DBField_t& _drchlt_bndr,
        const init_t& _uinit,
        double _RTOL = 1.0e-8,
        double _DUTOL = 1.0e-7)
        : diff_flux_func(_diff_flux_func)
        , adv_flux_func(_adv_flux_func)
        , intr_face_vel(_intr_face_vel)
        , bndr_face_vel(_bndr_face_vel)
        , bndr_val(_bndr_val)
        , wells(_wells)
        , drchlt_bndr(_drchlt_bndr)
        , RTOL(_RTOL)
        , DUTOL(_DUTOL)
        , mesh(_mesh)
        , uinit(_uinit)
        , diag(_mesh.num_cells())
        , k_12(_mesh.num_intr_faces())
        , k_21(_mesh.num_intr_faces())
    {
        //std::cout << "unit size: " << uinit.size() << " mesh size: " << mesh.num_cells() << std::endl;
        assert(intr_face_vel.size() == mesh.num_intr_faces());
        assert(bndr_val.size() == mesh.num_bndr_faces());
        assert(drchlt_bndr.size() == mesh.num_bndr_faces());
        assert(uinit.size() == mesh.num_cells());
    }

    std::size_t neqs() const { return mesh.num_cells(); }
    std::size_t neq_per_cell() const { return 1; }
    std::size_t max_nnz() const { return mesh.max_num_neighbors(); }

    //Initialize uold with initial condition
    void initial_condition(std::vector<double>& uold) const {
        uold.resize(neqs());
        for (std::size_t l = 0; l < neqs(); ++l)
            uold[l] = uinit[l];
    }


    template<typename V1, typename V2, typename M>
    void allocate_containers(V1& du, V2& r, M& J) const {

        J.block_size = neq_per_cell();
        J.reserve(neqs(), max_nnz());
        csr_pattern::build_pattern(J, mesh, diag, k_12, k_21);
        J.reset_size();
        r.resize(neqs(), 0.0);
        du.resize(neqs(), 0.0);
    }

    template<typename V1, typename V2, typename M>
    void initialize_containers(V1& du, V2&, M&) const {
        std::fill(du.begin(), du.end(), 0.0);
    }

    //Updates the old state after solving a timestep
    void bind_to_old_state(const State& unew,
        std::vector<double>& uold,
        double /*tnew*/,
        double /*dt*/)
    {
        assert(unew.size() == neqs());
        for (std::size_t l = 0; l < mesh.num_cells(); ++l)

            uold[l] = unew[l];
    }

    template<typename Vresid>
    void evaluate(const State& u, const std::vector<double>& uold, double dt, Vresid& resid)
    {
        const std::size_t nc = mesh.num_cells();
        const std::size_t nif = mesh.num_intr_faces();
        const std::size_t nbf = mesh.num_bndr_faces();
        assert(u.size() == nc);
        assert(uold.size() == nc);
        assert(dt > 0.0);

        for (std::size_t l = 0; l < nc; ++l) {
            resid[l] = u[l] - uold[l];
        }

        for (std::size_t f = 0; f < nif; ++f) {

            const auto [l1, l2] = mesh.intr_face_to_cells(f);
            const double u_l1 = u[l1];
            const double u_l2 = u[l2];
            const double v_12 = intr_face_vel[f];
            const double T_12 = mesh.intr_face_coef(f);
            const double v1 = mesh.cell_volume(l1);
            const double v2 = mesh.cell_volume(l2);
            //pow(u_l1, 2.0)
            //auto flux_12 = dt * ((v_12 > 0.0) * v_12 * adv_flux_func(u_l1) + (v_12 < 0.0) * v_12 * adv_flux_func(u_l2) + T_12 * (diff_flux_func(u_l1) - diff_flux_func(u_l2)));
            //std::cout << "Flux: " << flux_12 << std::endl;
            resid[l1] += dt* ((v_12 > 0.0) * v_12 * adv_flux_func(u_l1) + (v_12 < 0.0) * v_12 * adv_flux_func(u_l2) + T_12 * (diff_flux_func(u_l1) - diff_flux_func(u_l2)))/v1;
            resid[l2] -= dt * ((v_12 > 0.0) * v_12 * adv_flux_func(u_l1) + (v_12 < 0.0) * v_12 * adv_flux_func(u_l2) + T_12 * (diff_flux_func(u_l1) - diff_flux_func(u_l2)))/v2;
            //std::cout << "Intr face contrib: " << resid[l1]<<" "<<resid[l2] << std::endl;
        }

        for (std::size_t bf = 0; bf < nbf; ++bf) {
            const auto [l, sdir] = mesh.bndr_face_to_intr_cell(bf);
            const double u_l = u[l];
            (void)sdir;
            const bool DB = drchlt_bndr[bf];
            const double bc = bndr_val[bf];
            const double Tb = mesh.bndr_face_coef(bf);
            const double vb = bndr_face_vel[bf];
            const double v = mesh.cell_volume(l);
            //scalar f = pow(u_l, 2.0);
            //std::cout << f << std::endl;
            //std::cout << "Boundary vel: " << vb << std::endl;
            //scalar b_resid = dt * (DB * (Tb * (diff_flux_func(u_l) - diff_flux_func(bc)) + (vb > 0) * vb * adv_flux_func(u_l) + (vb < 0) * vb * adv_flux_func(bc)) - (!DB) * bc);
            resid[l] += dt * (DB * (Tb * (diff_flux_func(u_l) - diff_flux_func(bc)) + (vb > 0) * vb * adv_flux_func(u_l) + (vb < 0) * vb * adv_flux_func(bc)) - (!DB) * bc)/v;
            //std::cout << "Tb: " << Tb<<" bc: "<<bc << " flux: " << diff_flux_func(bc) << std::endl;
        }

        for (auto const& w : wells) {

            for (std::size_t ci = 0; ci < w.comps.size(); ++ci) {

                const auto& comp = w.comps[ci];
                const std::size_t l = comp.cell;
                const double u_l = u[l];
                const double q = w.rates.q_comp[ci]; // + inject, - produce
                const double v = mesh.cell_volume(l);
                // injection: q
                // production: q * fw(S)
               
                resid[l] -= dt *((q > 0.0) * q + (q < 0.0) * q * adv_flux_func(u_l))/v;
            }
        }

    }

    template<typename Vresid, typename M>
    bool discretize(const State& u,
        const std::vector<double>& uold,
        double dt,
        Vresid& resid,
        M& J)
    {

        evaluate(u, uold, dt, resid);
        std::fill(J.value().begin(), J.value().end(), 0.0);

        const std::size_t nif = mesh.num_intr_faces();
        const std::size_t nbf = mesh.num_bndr_faces();

        // Storage contribution.
        for (std::size_t l = 0; l < neqs(); ++l)
            J.value()[diag[l]] = 1.0;

        // Interior-face flux contributions.  The residual convention is
        // R_l1 += dt * F_12 / V_l1 and R_l2 -= dt * F_12 / V_l2.
        for (std::size_t f = 0; f < nif; ++f) {
            const auto [l1, l2] = mesh.intr_face_to_cells(f);
            const double v = intr_face_vel[f];
            const double T = mesh.intr_face_coef(f);
            const double c1 = dt / mesh.cell_volume(l1);
            const double c2 = dt / mesh.cell_volume(l2);

            if (v > 0.0) {
                const double df = adv_flux_func.grad(u[l1]);
                J.value()[diag[l1]] += c1 * v * df;
                // R_l2 receives -c2*v*f(u_l1), hence this derivative
                // belongs in row l2, column l1 (not on l2's diagonal).
                J.value()[k_21[f]] -= c2 * v * df;
            }
            else if (v < 0.0) {
                const double df = adv_flux_func.grad(u[l2]);
                J.value()[k_12[f]] += c1 * v * df;
                J.value()[diag[l2]] -= c2 * v * df;
            }

            const double dg1 = diff_flux_func.grad(u[l1]);
            const double dg2 = diff_flux_func.grad(u[l2]);
            J.value()[diag[l1]] += c1 * T * dg1;
            J.value()[k_12[f]] -= c1 * T * dg2;
            J.value()[k_21[f]] -= c2 * T * dg1;
            J.value()[diag[l2]] += c2 * T * dg2;
        }

        // Boundary and well terms are cell-local in this formulation.
        for (std::size_t bf = 0; bf < nbf; ++bf) {
            const auto [l, sdir] = mesh.bndr_face_to_intr_cell(bf);
            (void)sdir;
            if (!drchlt_bndr[bf]) continue;

            const double c = dt / mesh.cell_volume(l);
            const double Tb = mesh.bndr_face_coef(bf);
            const double vb = bndr_face_vel[bf];
            J.value()[diag[l]] += c * (
                Tb * diff_flux_func.grad(u[l])
                + (vb > 0.0 ? vb * adv_flux_func.grad(u[l]) : 0.0));
        }

        for (const auto& w : wells) {
            for (std::size_t ci = 0; ci < w.comps.size(); ++ci) {
                const auto& comp = w.comps[ci];
                const double q = w.rates.q_comp[ci];
                if (q < 0.0) {
                    const double c = dt / mesh.cell_volume(comp.cell);
                    J.value()[diag[comp.cell]] -= c * q * adv_flux_func.grad(u[comp.cell]);
                }
            }
        }

        J.reset_size();
        return true;
    }


    template<typename V>
    std::pair<bool, ResidualNorm> is_residual_norm_converged(const V& resid) const {
        ResidualNorm rnrm;
        rnrm.evaluate(resid);
        return { rnrm.isLessEqual(RTOL), rnrm };
    }

    template<typename V>
    std::pair<bool, UpdateNorm> is_update_norm_converged(const V& upd) const {
        UpdateNorm dunrm;
        dunrm.evaluate(upd);
        return { dunrm.isLessEqual(DUTOL), dunrm };
    }
    IFVel_t& get_intr_vel() { return intr_face_vel; }

    State initialize_state() const
    {
        std::size_t size = neqs();
        State state(size);
        for (std::size_t i = 0; i < size; ++i)
            state[i] = uinit[i];
        return state;

    }
    void initialize_timestep(State& /*unew*/, const std::vector<double>& /*uold*/, double /*DT*/) const {

    }

    //Todo Rethink resource ownership

private:

    const GFunctor diff_flux_func;
    const FFunctor adv_flux_func;

    IFVel_t intr_face_vel;
    const BFVel_t bndr_face_vel;
    const BCField_t bndr_val;
    const DBField_t drchlt_bndr;
    const std::vector<Well>& wells;

    const double RTOL;
    const double DUTOL;

    const TPFA_Mesh& mesh;

    const init_t uinit;
    mutable std::vector<std::size_t> diag, k_12, k_21;
    //double s_inj = 1.0;
};

#endif // __TWOPHASETRANSPORT_HPP_
