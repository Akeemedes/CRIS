#ifndef __STEADY_HPP_
#define __STEADY_HPP_

#include <vector>
#include <utility>

namespace discrete_systems {

    template< typename Model >
    class Steady {
    public:
        using State = typename Model::State;
        using ResidualNorm = typename Model::ResidualNorm;
        using UpdateNorm = typename Model::UpdateNorm;

        Steady(Model& _model) : mModel(_model) {}

        std::size_t max_num_eqns() const { return mModel.neqs(); }
        std::size_t max_nnz() const { return mModel.max_nnz(); }


        template< typename V1, typename V2, typename M >
        void allocate_for_solution(V1& update, V2& resid, M& J) const
        {
            mModel.allocate_containers(update, resid, J);
        }

        template< typename V1, typename V2, typename M >
        void initialize_for_solution(V1& update, V2& resid, M& J) const
        {
            mModel.initialize_containers(update, resid, J);
        }

        State  initialize_state() const {
           return mModel.initialize_state();
        }

        void initial_guess(State& unew) const
        {
            mModel.initial_guess(unew);
        }

        template <typename V>
        void evaluate(const State& u, V& resid)
        {
            mModel.evaluate(u, resid);
        }

        template< typename V, typename M >
        std::tuple<bool, bool, ResidualNorm > evaluate(const State& u, V& resid, M& J) {
            //std::cout << "Begining discretization with "<<resid.size() << std::endl;
            mModel.discretize(u, resid, J);
            //std::cout << "Discretization complete with " << resid.size() << std::endl;
            ResidualNorm rnrm;
            bool is_rnrm_converged = false;
            bool is_bad_residual = false;
            for (std::size_t i = 0; i < resid.size(); ++i)
                if (!std::isfinite(resid[i])) is_bad_residual = true;
            std::tie(is_rnrm_converged, rnrm) = mModel.is_residual_norm_converged(resid);
            return { is_bad_residual, is_rnrm_converged, rnrm };
        }

        template<typename V>
        auto is_update_norm_converged(const V& du) const {
            return mModel.is_update_norm_converged(du);
        }

        //const Model& model() const { return mModel; }
        Model& model() { return mModel; }


    private:
        Model& mModel;
    };
}
#endif // STEADY
