#pragma once
#include <array>
#include <cmath>
#include <cstddef>

namespace bratu {
// R = a-u+beta*exp(u), lower branch continued from beta=0.
// Metadata fixes a in [-2,6], beta in [0,1], u in [-2,7].
struct Sample { double a, beta, root, root_slope; };
inline std::array<double, 2> inputs(const Sample& s) {
    return {(s.a + 2) / 4 - 1, 2 * std::log1p(s.beta / 1e-6) / std::log1p(1e6) - 1};
}
inline double physical(double y) { return 4.5 * (y + 1) - 2; }
struct Target { double operator()(const Sample& s) const { return (s.root + 2) / 4.5 - 1; } };
struct Recurrence {
    static constexpr std::size_t kResidualsPerSample = 1;
    static constexpr bool kObjectiveEqualsSupervisedMse = false;
    void evaluate(const Sample& s, double y, double* residual, double* derivative) const {
        const double u = physical(y);
        const double reaction = s.beta == 0 ? 0 : s.beta * std::exp(u);
        // Fixed output-scale normalization, without root-Jacobian division.
        // Remains finite at the fold and deliberately retains weak physical
        // sensitivity there. The outer smooth cap only limits cold-start tails.
        residual[0] = (s.a - u + reaction) / 4.5;
        derivative[0] = -1 + reaction;
    }
};
}
