#ifndef __TPTADAPTERCRISHPP_
#define __TPTADAPTERCRISHPP_

//#include <cstddef>
#include <vector>
#include <torch/torch.h>
#include <cassert>
#include <cstdint>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "FluxFunctors.hpp"
#include "MeshRepresentations/UniformField.hpp"
#include "WellsManager.hpp"




template<
    typename init_t = std::vector<double>,
    typename BCField_t = std::vector<double>,
    typename IFVel_t = std::vector<double>,
    typename BFVel_t = std::vector<double>,
    typename FFunctor = CocurrentFracFlux,
    typename GFunctor = ExponentialFlux >
class TPTAdapter {
public:
    using State = std::vector<double>;
    using DBField_t = UCartesianBFaces<bool, IJKSequentialFaces>;//Modify to make generic here and TwoPhaseTransport
    TPTAdapter(
        const FFunctor& _adv_flux_func,
        const GFunctor& _diff_flux_func,
        const IFVel_t& _intr_face_vel,
        const BFVel_t& _bndr_face_vel,
        const std::vector<Well>& _wells,
        const BCField_t& _bndr_val,
        const DBField_t& _drchlt_bndr,
        const init_t& _uinit,
        const std::string& _embedding_trace_file = {}
    )
        : diff_flux_func(_diff_flux_func)
        , adv_flux_func(_adv_flux_func)
        , intr_face_vel(_intr_face_vel)
        , bndr_face_vel(_bndr_face_vel)
        , bndr_val(_bndr_val)
        , wells(_wells)
        , drchlt_bndr(_drchlt_bndr)
        , uinit(_uinit)
    {
        if (!_embedding_trace_file.empty()) {
            embedding_trace_ = std::make_shared<std::ofstream>(_embedding_trace_file);
            if (!*embedding_trace_) {
                throw std::runtime_error("Unable to open embedding trace: " + _embedding_trace_file);
            }
            *embedding_trace_ << "evaluation,cell,dt,gradient_requested,a,beta,u_old\n";
        }
    }

protected:
    // ---- storage owned by this adapter object ----
    const GFunctor diff_flux_func;
    const FFunctor adv_flux_func;

    IFVel_t intr_face_vel;
    const BFVel_t bndr_face_vel;
    const BCField_t bndr_val;
    const DBField_t drchlt_bndr;
    const std::vector<Well>& wells;
    const init_t uinit;


    std::vector<double> mA, mB, mC;
    torch::Tensor mAT, mBT, mCT;
    std::shared_ptr<std::ofstream> embedding_trace_;
    std::uint64_t embedding_evaluation_{0};

public:
    template<class Derived>
    struct Adapter : public TPTAdapter {
        using Base = TPTAdapter;
        using State = typename Base::State;

        // CRIS passes a fully-constructed TPTAdapter rvalue
        explicit Adapter(Base&& a) : Base(std::move(a)) {}
        explicit Adapter(Base const& a) : Base(a) {}

        Derived& derived() { return static_cast<Derived&>(*this); }
        Derived const& derived() const { return static_cast<Derived const&>(*this); }

        // Must be called once from CRIS ctor body (after mesh exists)
        void init_cris_storage_from_mesh() {
            auto& d = derived();
            this->mA.resize(d.mesh.num_cells(), 0.0);
            this->mB.resize(d.mesh.num_cells(), 0.0);
            this->mC.resize(d.mesh.num_cells(), 0.0);

            assert(this->intr_face_vel.size() == d.mesh.num_intr_faces());
            assert(this->bndr_face_vel.size() == d.mesh.num_bndr_faces());
            assert(this->bndr_val.size() == d.mesh.num_bndr_faces());
            assert(this->drchlt_bndr.size() == d.mesh.num_bndr_faces());

            const long num_cells = static_cast<long>(d.mesh.num_cells());

            this->mAT = torch::from_blob(this->mA.data(),
                { num_cells, 1 },
                torch::TensorOptions().dtype(torch::kFloat64));
            this->mAT.set_requires_grad(false);

            this->mBT = torch::from_blob(this->mB.data(),
                { num_cells, 1 },
                torch::TensorOptions().dtype(torch::kFloat64));
            this->mBT.set_requires_grad(false);

            this->mCT = torch::from_blob(this->mC.data(),
                { num_cells, 1 },
                torch::TensorOptions().dtype(torch::kFloat64));
            this->mCT.set_requires_grad(false);
        }

        void initial_condition(std::vector<double>& uold)
        {
            for (std::size_t l = 0; l < derived().neqs(); ++l)
                uold[l] = this->uinit[l];
        }

        void initialize_embeddings() {
            auto& d = derived();
            std::fill(this->mA.begin(), this->mA.end(), 0.0);
            for (std::size_t b = 0; b < d.mesh.num_bndr_faces(); ++b) {
                auto [l, direc] = d.mesh.bndr_face_to_intr_cell(b);
                (void)direc;

                const double cell_volume = d.mesh.cell_volume(l);
                const double vb = this->bndr_face_vel[b];
                const double Wb = this->bndr_val[b];
                const bool DB = this->drchlt_bndr[b];

                //this->mC[l] -= (!DB) * Wb / cell_volume;

                this->mA[l] -= (vb < 0) * vb * this->adv_flux_func(Wb) / cell_volume;
                //this->mB[l] += (vb > 0) * (vb + DBL_MIN) / cell_volume;
                //std::cout << Wb << " " << this->adv_flux_func(Wb) << " " << this->mA[l] << std::endl;

            }

            for (auto const& w : this->wells)
            {
                for (std::size_t ci = 0; ci < w.comps.size(); ++ci) {
                    const auto& comp = w.comps[ci];
                    const std::size_t l = comp.cell;
                    const double q = w.rates.q_comp[ci]; // + inject, - produce
                    const double cell_volume = d.mesh.cell_volume(l);
                    // injection: q
                    this->mA[l] += (q > 0) * q / cell_volume;
                    //this->mB[l] -= (q < 0) * (q + DBL_MIN) / cell_volume;
                }
            }

           

        }

        void update_embedding() {
            auto& d = derived();

            std::fill(this->mB.begin(), this->mB.end(), 0.0);
            //std::fill(this->mC.begin(), this->mC.end(), 0.0);

            for (std::size_t f = 0; f < d.mesh.num_intr_faces(); ++f) {
                auto [l1, l2] = d.mesh.intr_face_to_cells(f);

                double cell_volume1 = d.mesh.cell_volume(l1);
                double cell_volume2 = d.mesh.cell_volume(l2);
                double vel_12 = this->intr_face_vel[f];

                this->mB[l1] += (vel_12 > 0) * vel_12 / cell_volume1;
                this->mB[l2] -= (vel_12 < 0) * vel_12 / cell_volume2;

                //this->mC[l1] += d.mesh.intr_face_coef(f) / cell_volume1;
                //this->mC[l2] += d.mesh.intr_face_coef(f) / cell_volume2;
            }

            for (std::size_t b = 0; b < d.mesh.num_bndr_faces(); ++b) {
                auto [l1, direc] = d.mesh.bndr_face_to_intr_cell(b);
                (void)direc;

                double cell_volume = d.mesh.cell_volume(l1);
                double vel_b = this->bndr_face_vel[b];

                this->mB[l1] += this->drchlt_bndr[b] * (vel_b > 0) * vel_b / cell_volume;
                //this->mC[l1] += d.mesh.bndr_face_coef(b);
                //std::cout <<"Boundary dispersion coeff: "<< d.mesh.bndr_face_coef(b)<<" mB: "<<this->mB[l1]<<" mC: "<<this->mC[l1] << std::endl;
            }

            for (auto const& w : this->wells)
            {
                for (std::size_t ci = 0; ci < w.comps.size(); ++ci) {
                    const auto& comp = w.comps[ci];
                    const std::size_t l = comp.cell;
                    const double cell_volume = d.mesh.cell_volume(l);
                    const double q = w.rates.q_comp[ci]; // 

                    // production: q
                    this->mB[l] -= (q < 0) * q / cell_volume;
                }
            }

        }


        void bind_to_old_state(
            const State& unew,
            State& uold,
            double tnew,
            double DT)
        {
            auto& d = derived();

            for (std::size_t l = 0; l < d.neqs(); ++l) {
                uold[l] = unew[l];
                this->mC[l] = std::clamp(uold[l], 0.0, 1.0);
            }

            //initialize_embeddings(uold, DT);
            if (velocity_update_flag)
            {
                update_embedding();
                velocity_update_flag = false;
            }

        }

        void velocity_update()
        {
            velocity_update_flag = true;
        }

        void embed_cris_input(const State& u, const std::vector<double>& uold, double DT, bool grad)
        {
            this->mAT.set_requires_grad(grad);

            initialize_embeddings();


            auto& d = derived();

            for (std::size_t f = 0; f < d.mesh.num_intr_faces(); ++f) {
                auto [l1, l2] = d.mesh.intr_face_to_cells(f);

                double cell_volume1 = d.mesh.cell_volume(l1);
                double cell_volume2 = d.mesh.cell_volume(l2);
                double vel_12 = this->intr_face_vel[f];
                double T_12 = d.mesh.intr_face_coef(f);

                this->mA[l1] -= (vel_12 < 0) * vel_12 * this->adv_flux_func(u[l2]) / cell_volume1;


                this->mA[l2] += (vel_12 > 0) * vel_12 * this->adv_flux_func(u[l1]) / cell_volume2;
            }

            d.X = torch::cat({ this->mAT / (this->mBT + DBL_MIN), this->mBT * DT, this->mCT }, 1);
            if (this->embedding_trace_) {
                for (std::size_t l = 0; l < d.mesh.num_cells(); ++l) {
                    *this->embedding_trace_ << this->embedding_evaluation_ << ',' << l << ',' << DT << ','
                        << static_cast<int>(grad) << ',' << this->mA[l] / (this->mB[l] + DBL_MIN) << ','
                        << this->mB[l] * DT << ',' << this->mC[l] << '\n';
                }
                this->embedding_trace_->flush();
            }
            ++this->embedding_evaluation_;
            //auto max = torch::max(this->mBT, 0);
            //auto min = torch::min(this->mBT, 0);
            //std::cout <<"max: " <<std::get<0>(max) << std::endl;
            //std::cout <<"Min: "<< std::get<0>(min) << std::endl;
            //std::cout << d.X << std::endl;
            //std::cout << "uinit: " << uinit[0] << std::endl;
            //d.X = torch::cat({ this->mAT, this->mBT * DT }, 1);
        }
        void zero_grad() {
            if (this->mAT.grad().defined())

                this->mAT.grad().zero_();
        }
        template<class M>
        void propagate_cris_gradients(M& J, double DT, const State& u) {
            auto& d = derived();
            //double max_grad = 0.0;
            if (!this->mAT.grad().defined())
                std::cerr << "CRIS ERROR: gradient not computed for beta\n";

            double* data_ptr = this->mAT.grad().data_ptr<double>();

            for (std::size_t f = 0; f < d.mesh.num_intr_faces(); ++f) {
                auto [l1, l2] = d.mesh.intr_face_to_cells(f);

                double cell_volume1 = d.mesh.cell_volume(l1);
                double cell_volume2 = d.mesh.cell_volume(l2);
                double vel_12 = this->intr_face_vel[f];
                double T_12 = d.mesh.intr_face_coef(f);

                double value_12 = (vel_12 < 0) * (DBL_MIN - vel_12) * this->adv_flux_func.grad(u[l2]) / cell_volume1;

                double value_21 = (vel_12 > 0) * (DBL_MIN + vel_12) * this->adv_flux_func.grad(u[l1]) / cell_volume2;
                //std::cout << "grad 1_2: " << value_12 << " " << data_ptr[l1] << std::endl;
                //std::cout << "grad 2_1: " << value_21 << " " << data_ptr[l2] << std::endl;
                value_12 *= -data_ptr[l1];
                value_21 *= -data_ptr[l2];

                //max_grad = std::abs(value_12) > max_grad ? std::abs(value_12) : max_grad;
                //max_grad = std::abs(value_21) > max_grad ? std::abs(value_21) : max_grad;

                d.update_face_gradients(J, f, value_12, value_21);
            }
            //std::cout << "max grad: " << max_grad << std::endl;
        }
        State initialize_state() 
        {
            std::size_t size = derived().mesh.num_cells();
            State u(size);
            for (std::size_t i = 0; i < size; ++i)
            {
                u[i] = uinit[i];
                this->mC[i] = uinit[i];
            }
            return u;
        }
        void initialize_timestep(State& /*unew*/, const std::vector<double>& /*uold*/, double /*DT*/) {
            //initialize_embeddings(uold, DT);
            if (velocity_update_flag)
            {
                update_embedding();
                velocity_update_flag = false;
            }
        }
        IFVel_t& get_intr_vel() { return this->intr_face_vel; }
    private:
        bool velocity_update_flag = true;
    };
};

#endif // __TPTADAPTERCRISHPP_ included
