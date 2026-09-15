#pragma once

#include <cmath>
#include <type_traits>
#include <utility>

enum class LinearSolvePolicy { Strict, ContinueOnBudget };

namespace linear_solve_policy_detail {
template<class Report, class = void>
struct has_outcome : std::false_type {};

template<class Report>
struct has_outcome<Report, std::void_t<
    decltype(std::declval<Report>().relative_residual),
    decltype(std::declval<Report>().reached_iteration_limit),
    decltype(std::declval<Report>().is_breakdown)>> : std::true_type {};

template<class Report>
bool usable(const Report& report, LinearSolvePolicy policy) {
    if (report.is_badsolution) return false;
    if constexpr (has_outcome<Report>::value) {
        if (report.is_breakdown || !std::isfinite(report.relative_residual)
            || report.relative_residual < 0.0) return false;
        return report.is_converged || (policy == LinearSolvePolicy::ContinueOnBudget
            && report.reached_iteration_limit);
    } else {
        // An older backend must explicitly identify budget exhaustion before
        // its unconverged corrections can be used.
        return report.is_converged;
    }
}
} // namespace linear_solve_policy_detail
