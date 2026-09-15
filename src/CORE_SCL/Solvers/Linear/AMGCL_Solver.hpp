#ifndef __AMGCL_SOLVER_HPP__
#define __AMGCL_SOLVER_HPP__

#include <cstring>
#include <cmath>
#include <limits>

#include "CPR_CSR_Matrix.hpp"

#define AMGCL_NO_BOOST // remove other dependencies

#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/zero_copy.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/gmres.hpp>
#include <chrono>


enum class LinSysKind { Pressure, Transport, BlackOil };
enum class TransportKrylov { BiCGStab, GMRES };

template<LinSysKind Kind = LinSysKind::Pressure,
    TransportKrylov TK = TransportKrylov::BiCGStab>

class AMGCL
{
public:
    using double_type = double;
    using int_type = std::size_t;

    using A_type = CPR_CSR_Matrix<int_type, double_type>;
    using x_type = std::vector<double_type>;

    struct report_t {
        double xnrm = 0.0;
        int    ln_error_code = 0;
        int    niter = 0;
        bool   is_converged = false;
        bool   is_badsolution = false;
        double solve_ms = 0.0; // wall time spent in linear solve (ms)
        double relative_residual = std::numeric_limits<double>::infinity();
        bool reached_iteration_limit = false;
        bool is_breakdown = false;
    };

private:
    using Backend = amgcl::backend::builtin<double>;

    using AMGPrec = amgcl::amg<Backend,
        amgcl::coarsening::ruge_stuben,
        amgcl::relaxation::spai0>;

    using ILU0Prec = amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::ilu0>;
    using CPRPrec = amgcl::preconditioner::cpr<AMGPrec, ILU0Prec>;

    using BiCGStab = amgcl::solver::bicgstab<Backend>;
    using GMRES = amgcl::solver::gmres<Backend>;

    using PressureSolver = amgcl::make_solver<AMGPrec, BiCGStab>;
    using TransportSolverBiCG = amgcl::make_solver<ILU0Prec, BiCGStab>;
    using TransportSolverGMRES = amgcl::make_solver<ILU0Prec, GMRES>;
    using BlackOilSolver = amgcl::make_solver<CPRPrec, BiCGStab>;

    using SolverT =
        std::conditional_t<Kind == LinSysKind::Pressure,
        PressureSolver,
        std::conditional_t<Kind == LinSysKind::BlackOil,
        BlackOilSolver,
        std::conditional_t<(Kind == LinSysKind::Transport && TK == TransportKrylov::GMRES),
        TransportSolverGMRES,
        TransportSolverBiCG>>>;

public:
    AMGCL(std::size_t, std::size_t, std::size_t _maxiter = 1000)
        : is_first_call(true), error(0), iters(0), maxiter(_maxiter),
          refresh_transport_preconditioner(Kind == LinSysKind::Transport)
    {
    }

    void set_refresh_transport_preconditioner(bool enabled) {
        refresh_transport_preconditioner = enabled;
    }

    report_t solve(const A_type& A, x_type& x, const x_type& b,
        bool rebuild = false, double_type reltol = 1.0e-6)
    {
        prm.solver.tol = reltol;
        prm.solver.maxiter = maxiter;

        if constexpr (Kind == LinSysKind::BlackOil) {
            prm.precond.block_size = A.block_size;
        }
        if constexpr (Kind == LinSysKind::Transport && TK == TransportKrylov::GMRES) {
            prm.solver.M = 50;
        }

        const int_type rows = A.N();
        x.assign(rows, 0.0);

        auto pA = amgcl::adapter::zero_copy(
            rows,
            A.rowptr().data(),
            A.colind().data(),
            A.value().data()
        );

        if constexpr (Kind == LinSysKind::Transport) {
            rebuild = true;
        }

        if (is_first_call) {
            psolver = std::make_shared<SolverT>(*pA, prm, bprm);
            is_first_call = false;
        }

        if constexpr (Kind == LinSysKind::Transport) {
            if (refresh_transport_preconditioner)
                psolver = std::make_shared<SolverT>(*pA, prm, bprm);
        }

        // Partial update
        if (rebuild || !psolver || psolver->size() != rows) {
            if constexpr (Kind == LinSysKind::BlackOil) {
                psolver->precond().partial_update(*pA, true, bprm);
            }
        }

        // ---- timed solve ----
        const auto t0 = std::chrono::steady_clock::now();
        if (psolver->solver().prm.tol != reltol ||
            psolver->solver().prm.maxiter != static_cast<std::size_t>(maxiter)) {
            // make_solver owns a copy of its construction parameters. Updating
            // our prm does not update a cached Krylov solver. Copy only the
            // iterative solver so the preconditioner is reused as requested;
            // backend work vectors remain shared within this serial wrapper.
            auto iterative = psolver->solver();
            iterative.prm.tol = reltol;
            iterative.prm.maxiter = maxiter;
            std::tie(iters, error) = iterative(*pA, psolver->precond(), b, x);
        } else {
            std::tie(iters, error) = (*psolver)(*pA, b, x);
        }
        const auto t1 = std::chrono::steady_clock::now();

        report_t rep;
        rep.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // AMGCL returns a relative residual, not an integer error/status code.
        rep.relative_residual = error;
        rep.is_converged = std::isfinite(error) && error >= 0.0 && error <= reltol;
        rep.ln_error_code = rep.is_converged ? 0 : -1;
        rep.niter = static_cast<int>(iters);

        rep.xnrm = 0.0;
        rep.is_badsolution = false;
        for (double xi : x) {
            rep.xnrm += xi * xi;
            if (!std::isfinite(xi)) rep.is_badsolution = true;
        }
        rep.xnrm = std::sqrt(rep.xnrm);
        if (rep.is_badsolution) {
            rep.is_converged = false;
            rep.ln_error_code = -1;
        }
        // Budget exhaustion is not tolerance convergence and is not breakdown.
        // Early/non-finite unsuccessful returns are never eligible for continuation.
        rep.is_breakdown = rep.is_badsolution || !std::isfinite(error) || error < 0.0
            || (!rep.is_converged && iters < static_cast<int_type>(maxiter));
        rep.reached_iteration_limit = !rep.is_converged && !rep.is_breakdown
            && maxiter > 0 && iters >= static_cast<int_type>(maxiter);
        if (rep.is_breakdown) rep.ln_error_code = -2;

        return rep;
    }

private:
    bool is_first_call;
    double_type error;
    int_type iters;
    int maxiter;
    bool refresh_transport_preconditioner;
    std::shared_ptr<SolverT> psolver;

    typename SolverT::params prm;
    typename Backend::params bprm;
};


class AMGCL_OLD
{
public:
    typedef double double_type;
    typedef std::size_t int_type;
    typedef CPR_CSR_Matrix< int_type, double_type > A_type;
    typedef std::vector< double_type > x_type;
    using Backend = amgcl::backend::builtin<double>;
    using PPrecond = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::chebyshev>;
    using SPrecond = amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::ilu0>;
    using Solver = amgcl::make_solver <
        amgcl::preconditioner::cpr< PPrecond, SPrecond>,
        amgcl::solver::bicgstab<Backend>

    >;

    typedef struct
    {
        double xnrm;
        int ln_error_code;
        int niter;
        bool is_converged;
        bool is_badsolution;
        double relative_residual;
    } report_t;
public:
    AMGCL_OLD(std::size_t, std::size_t) : is_first_call(true) {
    };
    ~AMGCL_OLD() {};
    report_t solve(const A_type& _A, x_type& x, const x_type& _b, bool rebuild = false, double_type reltol = 1.0e-6)
    {
        // solver parameters
        prm.precond.block_size = _A.block_size;
        prm.solver.tol = reltol;
        prm.solver.maxiter = 400;
        int_type rows = _A.N();
        // setup matrix
        auto pA = amgcl::adapter::zero_copy(rows, _A.rowptr().data(), _A.colind().data(), _A.value().data());
        // build the solver the first time this function is called
        if (is_first_call) {
            psolver = std::make_shared<Solver>(pA, prm, bprm);
            is_first_call = false;
        }
        // Rebuild the solver, if necessary
        if (rebuild ||  !psolver ||  psolver->size() != rows) {
            psolver->precond().partial_update(*pA, true, bprm);
        }
        // solve the problem:
        std::tie(iters, error) = (*psolver)(*pA, _b, x);
        // reporting
        report.relative_residual = error;
        report.is_converged = std::isfinite(error) && error >= 0.0 && error <= reltol;
        report.ln_error_code = report.is_converged ? 0 : -1;
        report.niter = iters;
        report.xnrm = 0.0;
        report.is_badsolution = false;
        for (std::size_t i = 0; i < x.size(); ++i) {
            report.xnrm += x[i] * x[i];
            if (!std::isfinite(x[i])) report.is_badsolution = true;
        }
        report.xnrm = std::sqrt(report.xnrm);
        if (report.is_badsolution) {
            report.is_converged = false;
            report.ln_error_code = -1;
        }
        return report;
    }
private:
    bool is_first_call;
    report_t report;
    double_type error;
    int_type iters;
    std::shared_ptr<Solver> psolver;
    Solver::params prm;
    Backend::params bprm;
};


using PLNSolver = AMGCL<>;
using TLNSolver = AMGCL<LinSysKind::Transport, TransportKrylov::GMRES>;
//using TLNSolver = AMGCL<>;

//using PLNSolver = AMGCL_OLD;
//using TLNSolver = AMGCL_OLD;
#endif // !AMGCL.H
