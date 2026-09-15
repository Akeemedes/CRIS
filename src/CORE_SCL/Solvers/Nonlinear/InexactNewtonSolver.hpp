#ifndef __INEXACTNEWTONSOLVER_HPP__
#define __INEXACTNEWTONSOLVER_HPP__

#include <cstddef>
#include <vector>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include "LinearSolvePolicy.hpp"

// Problem: objects compute the nonlinear residual equation & their Jacobian matrix given an iterate
// LNsolver: objects solve J du = R upto a residual tolerance tol, i.e., ||Jdu - R|| <= \eta ||R||
// Updater: objects that advance the iterate u along an update du with some sort of safguarding unew = u - lambda du
template <typename ProblemType, typename LNsolver, typename Updater >
class InexactNewtonSolver
{
public:
	using Problem = ProblemType;
	struct report_t
	{
		std::size_t niter_nln;
		std::size_t niter_ln;
		std::size_t nfeval;
		typename Problem::ResidualNorm rnrm;
		typename Problem::UpdateNorm ln_xnrm;
		typename LNsolver::report_t ln_report;
		bool is_badresidual;
		bool is_R_converged;
		bool is_DU_converged;
		bool is_converged;
		bool is_tooManyIter;
		bool is_failed;
		double solve_s;
		std::size_t n_linear_tolerance_misses = 0;
		std::size_t n_budget_continuations = 0;
		std::size_t n_linear_breakdowns = 0;
		double last_linear_tolerance = 0.0;
	};

	typedef struct
	{
		std::size_t niter_ln;
		std::size_t niter_nln;
		std::size_t nfeval;
		std::size_t n_failed;
		double solve_s;

	} full_report_t;

public:
	// verbosity == 0 : no printouts
	// verbosity == 1 : printout final result
	// verbosity == 2 : printout first and final iterations
	// verbosity >  2 : printout all iterations
	full_report_t full_stats{ 0, 0, 0, 0, 0.0 };
	const typename LNsolver::A_type& last_jacobian() const { return J; }
	const typename LNsolver::A_type& initial_jacobian() const { return initial_J; }
	void set_refresh_transport_preconditioner(bool enabled) {
		lnsolver.set_refresh_transport_preconditioner(enabled);
	}
	void set_linear_solve_policy(LinearSolvePolicy policy, double tolerance_floor) {
		if (!std::isfinite(tolerance_floor) || tolerance_floor <= 0.0
			|| tolerance_floor > ETA_INIT)
			throw std::invalid_argument("Require 0 < linear tolerance floor <= initial LINTOL");
		linear_policy = policy;
		eta_min = tolerance_floor;
	}
	InexactNewtonSolver(Problem& model, double residTol = 1.0e-6, double updateTol = 1.0e-6,
		std::size_t maxiter = 1, std::size_t linmax=1000, int verbosity = 0,
		double eta0 = 0.1, double gamm = 0.9, double omeg = 2.0, const Updater& _updater = Updater(), bool _lin = false) :
		FTOL(residTol), DUTOL(updateTol), MAXITER(maxiter), verbose(verbosity),
		ETA_INIT(eta0), GAMMA(gamm), OMEGA(omeg), J(), initial_J(), R(), du(), status(),
		lnsolver(model.max_num_eqns(), model.max_nnz(), linmax), fSafeUpdate(_updater), lin(_lin)
	{
		if (!std::isfinite(ETA_INIT) || ETA_INIT <= 0.0 || ETA_INIT >= 1.0)
			throw std::invalid_argument("Initial LINTOL must be finite and in (0,1)");
		eta_min = std::min(eta_min, ETA_INIT);
		model.allocate_for_solution(du, R, J);
		model.allocate_for_solution(du, R, initial_J);
	}

	report_t solve(Problem& model, typename Problem::State& u)
	{
		// initialize data structures
		model.initialize_for_solution(du, R, J);
		//std::cout << "Model Initialized" << std::endl;
		// evaluate first guess
		status.niter_nln = 0;
		status.niter_ln = 0;
		status.solve_s = 0.0;
		status.n_linear_tolerance_misses = 0;
		status.n_budget_continuations = 0;
		status.n_linear_breakdowns = 0;
		status.ln_report = {};
		status.last_linear_tolerance = std::numeric_limits<double>::quiet_NaN();
		std::tie(status.is_badresidual, status.is_R_converged, status.rnrm) = model.evaluate(u, R, J);
		initial_J = J;
		status.nfeval = 1;
		//std::cout << "Resid, Jacobian Computed" << std::endl;
		status.is_DU_converged = false;
		status.is_converged = false;
		status.is_tooManyIter = (status.niter_nln > MAXITER);
		status.is_failed = (status.is_tooManyIter || status.is_badresidual);

		double rnrm_old = std::sqrt(std::inner_product(R.begin(), R.end(), R.begin(), 0.0));
		double rnrm_new = rnrm_old;
		double eta = ETA_INIT;
		double eta_old = eta;

		if (verbose > 1) print_preamble();
		if (verbose > 1) print_first_iter_info( );

		while ((!status.is_failed) && (!status.is_converged))
		{
			// solve J du = R
			//std::cout << "Pre Linear Solve size: " <<R.size()<< std::endl;
			status.ln_report = lnsolver.solve(J, du, R, (status.niter_nln==0), eta);
			status.last_linear_tolerance = eta;
			// Include work spent on rejected corrections and failed timestep attempts.
			status.niter_ln += status.ln_report.niter;
			status.solve_s += status.ln_report.solve_ms / 1000.0;
			if (!status.ln_report.is_converged) ++status.n_linear_tolerance_misses;
			if constexpr (linear_solve_policy_detail::has_outcome<typename LNsolver::report_t>::value) {
				if (status.ln_report.is_breakdown) ++status.n_linear_breakdowns;
			}
			//std::cout << "Post linear Solve size: "<<R.size() << std::endl;

			if (linear_solve_policy_detail::usable(status.ln_report, linear_policy))
			{
				if (!status.ln_report.is_converged) ++status.n_budget_continuations;
				//std::cout << "Updating" << std::endl;
				// perform a version of u = u - du
				//std::cout << "Before updates: " << R.size() << std::endl;
				std::tie( status.is_DU_converged, status.ln_xnrm) = fSafeUpdate.update_state(u, du, R, J, model, status.nfeval);
				//std::cout << "After updates: " << R.size() << std::endl;
				// evaluate iteration status after the update
				++status.niter_nln;
				std::tie(status.is_badresidual, status.is_R_converged, status.rnrm) = model.evaluate(u, R, J);
				++status.nfeval;
				//std::cout << "One more leap: " << R.size() << std::endl;
				status.is_tooManyIter = (status.niter_nln > MAXITER);
				status.is_failed = (status.is_tooManyIter || status.is_badresidual);
				status.is_converged = (status.is_DU_converged && status.is_R_converged);

				if (verbose > 2) print_iter_info(eta);

				// update eta
				rnrm_new = std::sqrt(std::inner_product(R.begin(), R.end(), R.begin(), 0.0));
				if (rnrm_new > 1.0e-7) {
					eta = GAMMA * std::pow(rnrm_new / rnrm_old , OMEGA);
					eta = std::min(eta, ETA_INIT);
					if (eta_old > 0.1 * GAMMA) eta = std::max(eta, GAMMA*eta_old);
					eta_old = eta;
					rnrm_old = rnrm_new;
				}
				else {
					eta = eta_min;
				}
				eta = std::clamp(eta, eta_min, ETA_INIT);
				eta_old = eta;
			}
			else
			{
				std::cout << "LINEAR SOLUTION FAILED WITH ERROR MESSAGE: "
					<< status.ln_report.ln_error_code << std::endl;
				if (verbose > 2) print_iter_info(eta);
				status.is_failed = true;
			}
		}
		if ((verbose == 1) || (verbose == 2)) print_iter_info(status.last_linear_tolerance);
		full_stats.nfeval += status.nfeval;
		full_stats.niter_ln += status.niter_ln;
		full_stats.niter_nln += status.niter_nln;
		full_stats.n_failed += status.is_failed || !status.is_converged;
		full_stats.solve_s += status.solve_s;
		return status;
	}
protected:
	void print_preamble() const {
		std::cout << std::setprecision(6);
		std::cout << "\t"
			<< std::setw(7) << "niter"
			<< std::setw(7) << "conv?"
			<< std::setw(7) << "lnnit"
			<< std::setw(16) << "eta  "
			<< std::setw(16) << "linear_relres" << std::setw(14) << "linear_state"
			<< std::setw(status.ln_xnrm.printout_width()) << "|| DU ||"
			<< std::setw(status.rnrm.printout_width()) << "|| R ||" << std::endl;
		std::cout << "\t"
			<< std::setw(7) << "-----"
			<< std::setw(7) << "-----"
			<< std::setw(7) << "-----"
			<< std::setw(16) << "-----";
		status.ln_xnrm.print_preamble(std::cout);
		status.rnrm.print_preamble(std::cout);
		std::cout << std::endl;
	}
	void print_first_iter_info( ) const {
		std::cout << std::setprecision(6);
		std::cout << "\t"
			<< std::setw(7) << status.niter_nln
			<< std::setw(7) << int(status.is_converged)
			<< std::setw(7) << " - "
			<< std::setw(16) << " - "
			<< std::setw(16) << " - " << std::setw(14) << " - "
			<< std::setw(status.ln_xnrm.printout_width()) << " - "
			<< status.rnrm << std::endl;
	}
	void print_iter_info(double eta) const {
		std::cout << std::setprecision(6);
		std::cout << "\t"
			<< std::setw(7) << status.niter_nln
			<< std::setw(7) << int(status.is_converged)
			<< std::setw(7) << status.ln_report.niter
			<< std::setw(16) << std::scientific  << eta;
		if constexpr (linear_solve_policy_detail::has_outcome<typename LNsolver::report_t>::value) {
			const auto& rep = status.ln_report;
			const char* outcome = rep.is_breakdown ? "breakdown" : rep.is_converged ? "converged"
				: rep.reached_iteration_limit ? "budget" : "failed";
			std::cout << std::setw(16) << rep.relative_residual << std::setw(14) << outcome;
		} else std::cout << std::setw(16) << "n/a" << std::setw(14) << "n/a";
		std::cout << status.ln_xnrm
			<< status.rnrm << std::endl;
	}

private:
	const double FTOL;
	const double DUTOL;
	const std::size_t MAXITER;
	const int verbose;
	const double ETA_INIT = 0.1;
	const double GAMMA = 0.9;
	const double OMEGA = 2.0;
	const bool lin = false;
	LinearSolvePolicy linear_policy = LinearSolvePolicy::Strict;
	double eta_min = 1e-12;
	typename LNsolver::A_type J;
	typename LNsolver::A_type initial_J;
	typename LNsolver::x_type R;
	std::vector<double> du;
	report_t status;
	
	LNsolver lnsolver;
	Updater fSafeUpdate;
};

#endif
