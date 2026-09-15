#ifndef __NEWTONUPDATERS_HPP__
#define __NEWTONUPDATERS_HPP__

#include <cmath>
#include <vector>
#include "SimUtils.hpp"

class StandardNewtonUpdater {
public:
	StandardNewtonUpdater() = default;
	StandardNewtonUpdater(const StandardNewtonUpdater&) = default;

	template< typename T1, typename T2, typename V, typename M, typename Model >
	auto update_state(T1& u, const T2& du, const V&, const M&, const Model& model, std::size_t& /*nfeval*/)
	{
		for (std::size_t i = 0; i < u.size(); ++i) {
			u[i] = u[i] - du[i];
		}
		return model.is_update_norm_converged(du);
	}
};

// Full Newton step projected onto the physical saturation interval [0,1].
// StandardNewtonUpdater applies the unconstrained step.
class ProjectedNewtonUpdater {
public:
	ProjectedNewtonUpdater() = default;
	ProjectedNewtonUpdater(const ProjectedNewtonUpdater&) = default;

	template< typename T1, typename T2, typename V, typename M, typename Model >
	auto update_state(T1& u, const T2& du, const V&, const M&, const Model& model, std::size_t& /*nfeval*/)
	{
		for (std::size_t i = 0; i < u.size(); ++i) {
			u[i] = u[i] - du[i];
			u[i] = std::max(0.0, std::min(1.0, u[i]));
		}
		return model.is_update_norm_converged(du);
	}
};


class MACUpdater {
public:
	MACUpdater() = default;
	MACUpdater(const MACUpdater&) = default;

	template< typename T1, typename T2, typename V, typename M, typename Model >
	auto update_state(T1& u, const T2& du, const V&, const M&, const Model& model, std::size_t& /*nfeval*/)
	{
		for (std::size_t i = 0; i < u.size(); ++i) {
			double dui = du[i];
			dui = dui > 0.2 ? 0.2 : (dui < -0.2 ? -0.2 : dui);
			u[i] -= dui;
			if (u[i] > 1.0) {
				u[i] = 1.0;
				make_independent(u[i], i);
			}
			else if (u[i] < 0.0) {
				u[i] = 0.0;
				make_independent(u[i], i);
			}
			//max(u[i], 1.0);

			//std::cout<<"update "<<i<<": " << du[i] << "\t\t" << u[i] << std::endl;
		}
		return model.is_update_norm_converged(du);
	}
};



class LineSearchUpdater {
public:
	LineSearchUpdater(bool _audit = false) : audit(_audit) {
		set_parameters();
	}

	LineSearchUpdater(const LineSearchUpdater&) = default;

	LineSearchUpdater(double _alpha_min, double _alpha_max,
		double _c1, double _beta) {
		set_parameters(_alpha_min, _alpha_max, _c1, _beta);
	}
	void set_parameters(double _alpha_min = 1e-6, double _alpha_max = 1.0,
		double _c1 = 0.0001, double _beta = 0.5) {
		alpha_min = _alpha_min;
		alpha_max = _alpha_max;
		c1 = _c1;
		beta = _beta;
	}
	template< typename T1, typename T2, typename V, typename M, typename Model >
	auto update_state(T1& x, const T2& dx, const V& residual, const M& J,  Model& model, std::size_t& nfeval)
	{
		// initialize
		//x_old = x;
		state_copy(x, x_old);
		double merit_old = 0.5 * dot_product(residual, residual);
		J.transpose_multiply_v(residual, grad_merit_old); // J^T R
		double gradmeritdot = -c1 * dot_product(grad_merit_old, dx);
		//std::cout << gradmeritdot << std::endl;
		// try first step
		double alpha = std::min(1.0, alpha_max);
		for (std::size_t i = 0; i < x.size(); ++i)
		{
			x[i] = x_old[i] - alpha * dx[i];
			if (x[i] > 1.0) {
				x[i] = 1.0;

			}
			else if (x[i] < 0.0) {
				x[i] = 0.0;

			}
		}
		model.evaluate(x, grad_merit_old);
		++nfeval;
		double merit = 0.5 * dot_product(grad_merit_old, grad_merit_old);

		// perform backtracking
		int n_steps = 0;
		while ((merit - alpha * gradmeritdot > merit_old) && (alpha > alpha_min)) {
			++n_steps;
			alpha *= beta;
			alpha = std::max(alpha, alpha_min);
			double x_max = -10.0;
			for (std::size_t i = 0; i < x.size(); ++i)
			{
				x[i] = x_old[i] - alpha * dx[i];
				if (x[i] > 1.0) {
					x[i] = 1.0;
					
				}
				else if (x[i] < 0.0) {
					x[i] = 0.0;
					
				}
				
				//if (residual[i] > x_max) x_max = residual[i];
				//if (residual[i] < x_min) x_min = residual[i];
			}
			//std::cout << "Max u: " << x_max <<", Min u: "<<x_min<< std::endl;

			model.evaluate(x, grad_merit_old);
			++nfeval;
			merit = 0.5 * dot_product(grad_merit_old, grad_merit_old);
		}
		if (audit) {
			std::cout << "Line search: alpha=" << alpha
				<< ", backtracks=" << n_steps
				<< ", merit_old=" << merit_old
				<< ", merit_new=" << merit << std::endl;
		}
		for (std::size_t i = 0; i < x.size(); ++i) {
			make_independent(x[i], i);
		}
		return model.is_update_norm_converged(dx);
	}

protected:
	template< typename T1, typename T2 >
	double dot_product(const T1& x, const T2& y) {
		assert((x.size() == y.size()));
		double dot = 0.0;
		for (std::size_t i = 0; i < x.size(); ++i)
			dot += x[i] * y[i];
		return dot;
	}

private:
	double c1;
	double alpha_min;
	double alpha_max;
	double beta;
	bool audit;
	std::vector< double> x_old;
	std::vector< double> grad_merit_old;
};

#endif
