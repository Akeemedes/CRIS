#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace twophasetransport::training {

// Pointwise physics term for the dispersion-free regular two-phase transport
// recurrence. The network output is normalized to [-1, 1], while the
// recurrence is evaluated in physical saturation u in [0, 1].
//
// R is divided by the exact-root derivative stored with every label. Near the
// selected root this makes 2 R / dR_du(root) equal to the normalized output
// error to first order, so lambda has an interpretable scale relative to MSE.
// Outside [0,1] the flux uses the same clipped physical extension as the
// label oracle; MSE remains responsible for the direct bound-restoring error.
struct RegularAdvectionRecurrenceTerm {
    // Retain the regular default and its arithmetic; IMP sets this to 0.2.
    double nonwetting_exponent{2.0};
    static constexpr std::size_t kResidualsPerSample = 1;
    static constexpr bool kObjectiveEqualsSupervisedMse = false;

    template <class Sample>
    void evaluate(const Sample& sample, double normalized_prediction,
                  double* residuals, double* derivatives_wrt_output) const {
        const double u = 0.5 * (normalized_prediction + 1.0);
        const auto [flux, flux_derivative] = fractional_flow_and_derivative(u);
        if (!(sample.dr_du > 0.0) || !std::isfinite(sample.dr_du)) {
            throw std::runtime_error("Regular recurrence term requires a finite positive root derivative");
        }
        const double recurrence = u - sample.u_old + sample.beta * (flux - sample.a);
        residuals[0] = 2.0 * recurrence / sample.dr_du;
        // d(2R/dR_du(root))/d(normalized_prediction)
        derivatives_wrt_output[0] = (1.0 + sample.beta * flux_derivative) / sample.dr_du;
    }

private:
    std::pair<double, double> fractional_flow_and_derivative(double u) const {
        if (u <= 0.0) return {0.0, 0.0};
        if (u >= 1.0) return {1.0, 0.0};
        constexpr double kMobilityRatio = 0.4;
        const double nonwetting = 1.0 - u;
        if (nonwetting_exponent != 2.0) {
            const double nw = std::pow(nonwetting, nonwetting_exponent);
            const double denominator = u*u + kMobilityRatio*nw;
            return {u*u/denominator,
                kMobilityRatio*(2*u*nw + u*u*nonwetting_exponent*
                    std::pow(nonwetting,nonwetting_exponent-1))/(denominator*denominator)};
        }
        const double denominator = u * u + kMobilityRatio * nonwetting * nonwetting;
        const double flux = u * u / denominator;
        const double derivative = 2.0 * kMobilityRatio * u * nonwetting /
            (denominator * denominator);
        return {flux, derivative};
    }
};

}  // namespace twophasetransport::training
