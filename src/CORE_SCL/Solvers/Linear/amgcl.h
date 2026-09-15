#ifndef __AMGCL_H__
#define __AMGCL_H__

#include <cstring>
#include <cmath>

#include "CPR_CSR_Matrix.hpp"

#define AMGCL_NO_BOOST // remove other dependencies

#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/crs_tuple.hpp> // needed for std::tie adapter
#include <amgcl/amg.hpp>
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <chrono>


class AMGCL
{
public:
    typedef double                                  double_type;
    typedef int                                     int_type;
    typedef CPR_CSR_Matrix< int_type, double_type > A_type;
    typedef std::vector< double_type >              x_type;
    using Backend = amgcl::backend::builtin<double>;
    using PPrecond = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;
    using SPrecond = amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::ilu0>;
    using Solver = amgcl::make_solver<
        amgcl::preconditioner::cpr< PPrecond, SPrecond>,
        amgcl::solver::bicgstab<Backend>
    >;

    typedef struct
    {
        double   xnrm;
        int      ln_error_code;
        int      niter;
        bool     is_converged;
        bool     is_badsolution;
        double   relative_residual;
        std::chrono::milliseconds timing;
    }                                                  report_t;

public:

    AMGCL(std::size_t, std::size_t) :report(), error(), iters() {
    };
    ~AMGCL() {};

    //SOMTO
    static int_type offset() { return 0; }

    //report_t solve(const A_type& _A, x_type& _x_, const x_type& _b, double_type)
    report_t solve(const A_type& _A, x_type& _x_, const x_type& _b, bool rebuild = false)
    {
        auto start = std::chrono::high_resolution_clock::now();

        int_type rows = _A.N();
        _x_.resize(rows, 0.0);

        // construct solver
        Solver::params prm;
        prm.precond.block_size = _A.block_size;
        auto A = std::tie(rows, _A.rowptr(), _A.colind(), _A.value());
        Solver solve(A, prm);

        // solve
        std::tie(iters, error) = solve(_b, _x_);

        report.relative_residual = error;
        report.is_converged = std::isfinite(error) && error >= 0.0 && error <= prm.solver.tol;
        report.ln_error_code = report.is_converged ? 0 : -1;
        report.niter = iters;
        report.xnrm = 0.0;
        report.is_badsolution = false;
        for (std::size_t i = 0; i < _x_.size(); ++i) {
            report.xnrm += _x_[i] * _x_[i];
            if (!std::isfinite(_x_[i])) report.is_badsolution = true;
        }
        report.xnrm = std::sqrt(report.xnrm);

        auto end = std::chrono::high_resolution_clock::now();

        report.timing = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (report.is_badsolution) {
            report.is_converged = false;
            report.ln_error_code = -1;
        }

        return report;
    }

private:
    report_t   report;
    double_type error;
    int_type   iters;

};


#endif // !AMGCL.H
