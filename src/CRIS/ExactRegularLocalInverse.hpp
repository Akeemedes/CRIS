#ifndef __EXACT_REGULAR_LOCAL_INVERSE_HPP__
#define __EXACT_REGULAR_LOCAL_INVERSE_HPP__

#include <torch/torch.h>

namespace cris_exact {

// A bracketed, differentiable implicit inverse for
// u - u_old + beta * (f(u) - a) = 0 with the regular n_w=n_nw=2 law.
torch::Tensor regular_advection_local_inverse(torch::Tensor input, double mobility_ratio);
// Degenerate IMP law: n_w=2, n_nw=0.2; first-order implicit derivatives.
torch::Tensor imp_advection_local_inverse(torch::Tensor input, double mobility_ratio);

} // namespace cris_exact

#endif // __EXACT_REGULAR_LOCAL_INVERSE_HPP__
