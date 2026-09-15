#ifndef __DISCRETEPROBLEM_HPP_
#define __DISCRETEPROBLEM_HPP_
#include <vector>
#include <utility>
#include <tuple>
#include <cmath>
#include <cassert>
#include <cstddef>
#include "SimUtils.hpp"



struct TSUpdater {
    TSUpdater(double _max_step, double _min_step, double _ts_incr, double _ts_decr)
        : maxStep(_max_step), minStep(_min_step), stepIncrMult(_ts_incr), stepDecrMult(_ts_decr)
    {
        assert(_ts_incr >= 1.0);
        assert(_ts_decr < 1.0);
        assert(_max_step >= _min_step);
    }

    bool operator()(double& dt, bool converged) const {
        if (converged) {
            dt *= stepIncrMult;
            dt = std::min(dt, maxStep);
            return true;
        }
        else {
            dt *= stepDecrMult;
            return dt >= minStep;
        }
    }

    double minstep() const { return minStep; }
    double maxstep() const { return maxStep; }

private:
    double maxStep, minStep, stepIncrMult, stepDecrMult;
};

template<typename Model>
class DiscreteProblemFIM {
public:
    using State = typename Model::State;
    using ResidualNorm = typename Model::ResidualNorm;
    using UpdateNorm = typename Model::UpdateNorm;

    DiscreteProblemFIM(Model& _model, TSUpdater _updater, double init_dt)
        : mModel(_model)
        , updater(std::move(_updater))
        , uold(mModel.neqs())
        , t_accept(0.0)
        , dt_trial(init_dt)
    {
        assert("Invalid Initial Timestep. Initial timestep size must be between the set min and max step sizes" && dt_trial >= updater.minstep() && dt_trial <= updater.maxstep()
            );
        mModel.initial_condition(uold);
    }

    std::size_t max_num_eqns() const { return mModel.neqs(); }
    std::size_t max_nnz() const { return mModel.max_nnz(); }
    double accepted_time() const { return t_accept; }
    

    template<typename V1, typename V2, typename M>
    void allocate_for_solution(V1& update, V2& resid, M& J) const {
        mModel.allocate_containers(update, resid, J);
    }

    template<typename V1, typename V2, typename M>
    void initialize_for_solution(V1& update, V2& resid, M& J) const {
        std::cout << "Starting solution with timestep size: " << std::min(dt_trial, t_next_event - t_accept) << std::endl;
        mModel.initialize_containers(update, resid, J);
    }

    State initialize_state () const {
        return mModel.initialize_state();
    }

    void initialize_timestep(State& unew)
    {
        mModel.initialize_timestep(unew, uold, std::min(dt_trial, t_next_event - t_accept));
    }

    template <typename V>
    void evaluate(const State& u, V& resid) {
        mModel.evaluate(u, uold, std::min(dt_trial, t_next_event - t_accept), resid);
    }

    template<typename V, typename M>
    std::tuple<bool, bool, ResidualNorm> evaluate(const State& u, V& resid, M& J) {
        mModel.discretize(u, uold, std::min(dt_trial, t_next_event - t_accept), resid, J);

        bool is_bad_residual = false;
        for (std::size_t i = 0; i < resid.size(); ++i) {
            double r = static_cast<double>(resid[i]);
            if (!std::isfinite(r)) {
                is_bad_residual = true;
                break;
            }
        }

        ResidualNorm rnrm;
        bool is_rnrm_converged = false;
        std::tie(is_rnrm_converged, rnrm) = mModel.is_residual_norm_converged(resid);

        return { is_bad_residual, is_rnrm_converged, rnrm };
    }

    void reset_state(State& unew) {
        for (std::size_t l = 0; l < max_num_eqns(); ++l)
        {
            unew[l] = uold[l];
            make_independent(unew[l], l);
        }

    }

    void next_event(double t_event) {
        t_next_event = t_event;
        //dt_trial = std::min(dt_trial, t_event - t_accept);
    }

    bool advance(State& unew, bool converged) {
        double dt_trial_old = std::min(dt_trial, t_next_event - t_accept);
        if (!converged) {
            if (!updater(dt_trial_old, converged))
                return false;
            dt_trial = dt_trial_old;
            std::cout << "Timestep failed. Restarting time " << t_accept << " with step size " << dt_trial << std::endl;
            reset_state(unew);
        }
        else {
            t_accept += dt_trial_old;
            if (dt_trial_old == dt_trial)
                updater(dt_trial, true);
            double dt_trial_clipped = std::min(dt_trial, t_next_event - t_accept);
            mModel.bind_to_old_state(unew, uold, t_accept, dt_trial_clipped);
            std::cout << "Timestep converged. Starting time " << t_accept << " with step size " << dt_trial_clipped << std::endl;
            

        }

        return true;
    }

    template<typename V>
    auto is_update_norm_converged(const V& du) const {
        return mModel.is_update_norm_converged(du);
    }

    //const Model& model() const { return mModel; }
    Model& model() const { return mModel; }
    double DT() const { return std::min(dt_trial, t_next_event - t_accept); }

private:
    double trial_dt() const { return dt_trial; }
    double trial_time() const { return t_accept + DT(); }

    Model& mModel;
    TSUpdater updater;
    std::vector<double> uold;
    double t_accept;
    double dt_trial;
    double t_next_event{ 0 };
};

#endif // __DISCRETEPROBLEM_HPP_