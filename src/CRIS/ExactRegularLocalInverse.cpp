#include "ExactRegularLocalInverse.hpp"

#include <cmath>
#include <stdexcept>

namespace {

torch::Tensor fractional_flow(const torch::Tensor& u, double mobility_ratio) {
    const auto nonwetting = 1.0 - u;
    const auto wetting_sq = u * u;
    const auto nonwetting_sq = nonwetting * nonwetting;
    return wetting_sq / (wetting_sq + mobility_ratio * nonwetting_sq);
}

torch::Tensor residual(const torch::Tensor& u, const torch::Tensor& a,
    const torch::Tensor& beta, const torch::Tensor& u_old, double mobility_ratio) {
    return u - u_old + beta * (fractional_flow(u, mobility_ratio) - a);
}

torch::Tensor derivative(const torch::Tensor& u, const torch::Tensor& beta, double mobility_ratio) {
    const auto nonwetting = 1.0 - u;
    const auto denominator = u * u + mobility_ratio * nonwetting * nonwetting;
    const auto flux_derivative = 2.0 * mobility_ratio * u * nonwetting / (denominator * denominator);
    return 1.0 + beta * flux_derivative;
}

} // namespace

namespace cris_exact {

torch::Tensor regular_advection_local_inverse(torch::Tensor input, double mobility_ratio) {
    if (input.dim() != 2 || input.size(1) != 3) {
        throw std::runtime_error("Exact regular inverse expects an [N,3] tensor: a, beta, u_old");
    }
    const auto a = input.select(1, 0);
    const auto beta = input.select(1, 1);
    const auto u_old = input.select(1, 2);
    torch::Tensor root;
    {
        torch::NoGradGuard no_grad;
        auto lo = torch::zeros_like(a);
        auto hi = torch::ones_like(a);
        for (int iteration = 0; iteration < 64; ++iteration) {
            const auto mid = 0.5 * (lo + hi);
            const auto mid_residual = residual(mid, a, beta, u_old, mobility_ratio);
            lo = torch::where(mid_residual < 0.0, mid, lo);
            hi = torch::where(mid_residual > 0.0, mid, hi);
        }
        root = 0.5 * (lo + hi);
    }

    // The bisection result is a detached numerical anchor. Differentiating
    // this one correction at a converged root gives the implicit derivative
    // -R_x/R_u, without retaining a long bisection graph in the global solve.
    const auto corrected = root - residual(root, a, beta, u_old, mobility_ratio)
        / derivative(root, beta, mobility_ratio);
    return corrected.unsqueeze(1);
}

} // namespace cris_exact

namespace cris_exact {
torch::Tensor imp_advection_local_inverse(torch::Tensor input, double mobility_ratio) {
    if (input.dim()!=2 || input.size(1)!=3)
        throw std::runtime_error("Exact IMP inverse expects [N,3]: a,beta,u_old");
    const auto a=input.select(1,0), beta=input.select(1,1), old=input.select(1,2);
    torch::Tensor root, da, db, dc;
    {
        torch::NoGradGuard guard;
        auto lo=torch::zeros_like(a), hi=torch::ones_like(a);
        auto flow=[&](const torch::Tensor& u) {
            return u*u/(u*u+mobility_ratio*torch::pow(1-u,0.2));
        };
        for (int i=0;i<64;++i) {
            auto mid=(lo+hi)*0.5;
            auto r=mid-old+beta*(flow(mid)-a);
            lo=torch::where(r<0,mid,lo);
            hi=torch::where(r>=0,mid,hi);
        }
        root=(lo+hi)*0.5;
        root=torch::where(beta==0,old,root);
        // Exact endpoints, including roots not distinguishable from 1 in float64.
        root=torch::where((old==0)&(a==0),torch::zeros_like(root),root);
        root=torch::where((old==1)&(a==1),torch::ones_like(root),root);
        auto t=1-root;
        auto den=root*root+mobility_ratio*torch::pow(t,0.2);
        auto fp=mobility_ratio*(2*root*torch::pow(t,0.2)+0.2*root*root*torch::pow(t,-0.8))/(den*den);
        // Avoid 0*infinity at beta=0. At u=1 and beta>0, inverse slope tends to zero.
        dc=torch::where(beta==0,torch::ones_like(root),1/(1+beta*fp));
        da=beta*dc;
        db=(a-flow(root))*dc;
    }
    // Detached coefficients provide the first derivative required by Newton,
    // without differentiating bisection or singular endpoint flux expressions.
    return (root+(a-a.detach())*da+(beta-beta.detach())*db+(old-old.detach())*dc).unsqueeze(1);
}
} // namespace cris_exact
