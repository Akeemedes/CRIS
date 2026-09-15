#pragma once

#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace native_lm {

// Timing fields deliberately describe numerical phases rather than any
// governing equation. A problem adapter can populate only the phases it uses.
struct LmTimings {
    double normal_seconds{0.0};
    double jacobian_seconds{0.0};
    double normal_matrix_seconds{0.0};
    double host_convert_seconds{0.0};
    double host_to_device_seconds{0.0};
    double cuda_gemm_seconds{0.0};
    std::uint64_t host_to_device_bytes{0};
    double solve_seconds{0.0};
    double candidate_seconds{0.0};
    double candidate_train_seconds{0.0};
    double candidate_validation_seconds{0.0};
};

struct LmControl {
    double damping_decrease{2.0};
    double damping_increase{10.0};
    double damping_max{1.0e10};
    bool diagonal_damping{false};
    bool synchronize_cuda_after_solve{false};
    int max_iterations{0};
    int max_accepted_updates{0};
    int max_validation_failures{0};
};

struct LmState {
    int iteration{0};
    int accepted_updates{0};
    int validation_failures{0};
    double damping{0.0};
    double objective_loss{0.0};
    double training_mse{0.0};
    double validation_mse{0.0};
    double best_validation_mse{std::numeric_limits<double>::infinity()};
    double mean_objective_gradient_norm{0.0};
};

struct LmLinearization {
    torch::Tensor hessian;
    torch::Tensor gradient;
    std::size_t sample_count{0};
    LmTimings timings;
};

struct CandidateEvaluation {
    double objective_loss{std::numeric_limits<double>::infinity()};
    double training_mse{std::numeric_limits<double>::infinity()};
    double validation_mse{std::numeric_limits<double>::infinity()};
    LmTimings timings;
};

struct LmIteration {
    enum class Status { Accepted, Rejected };

    Status status{Status::Rejected};
    bool linearization_refreshed{false};
    bool validation_improved{false};
    LmState state;
    LmTimings timings;
};

template <class Model>
struct LmRunResult {
    Model model;
    LmState state;
    std::string stopping_reason{"max_iterations"};
};

// Problem-agnostic LM driver. The adapter supplies the model-specific
// Jacobian/normal-equation builder, candidate construction, and loss
// evaluation. This driver owns damping, acceptance, cached-linearization
// retries, validation-based early stopping, and the dense linear solve.
// All callbacks are templates, so none introduces a virtual call or Python
// transition in the training loop.
template <class Model, class RefreshLinearization, class MakeCandidate,
          class EvaluateCandidate, class ObserveIteration>
LmRunResult<Model> optimize(Model model, const LmControl& control,
                            LmState state,
                            RefreshLinearization&& refresh_linearization,
                            MakeCandidate&& make_candidate,
                            EvaluateCandidate&& evaluate_candidate,
                            ObserveIteration&& observe_iteration) {
    LmLinearization linearization;
    bool refresh_required = true;
    LmRunResult<Model> result{std::move(model), state, "max_iterations"};
    const int iteration_offset = state.iteration;

    // A resumed run can already have exhausted a total accepted/patience
    // budget. Finalize without taking an extra proposal in that case.
    if (control.max_validation_failures > 0 && state.validation_failures >= control.max_validation_failures) {
        result.stopping_reason = "max_validation_fail";
        return result;
    }
    if (control.max_accepted_updates > 0 && state.accepted_updates >= control.max_accepted_updates) {
        result.stopping_reason = "max_accepted_updates";
        return result;
    }

    for (int local_iteration = 1; local_iteration <= control.max_iterations; ++local_iteration) {
        result.state.iteration = iteration_offset + local_iteration;
        const bool linearization_refreshed = refresh_required;
        if (refresh_required) {
            linearization = refresh_linearization(result.model);
            if (linearization.sample_count == 0) {
                throw std::runtime_error("LM linearization requires at least one sample");
            }
            result.state.mean_objective_gradient_norm =
                2.0 * torch::linalg_vector_norm(linearization.gradient).item<double>() /
                static_cast<double>(linearization.sample_count);
        }

        LmTimings timings = linearization_refreshed ? linearization.timings : LmTimings{};
        torch::Tensor hessian = linearization.hessian.clone();
        if (control.diagonal_damping) {
            torch::Tensor diagonal = hessian.diag().clamp_min(1.0e-12);
            hessian.diagonal(0, 0, 1).add_(result.state.damping * diagonal);
        } else {
            // `damping` is expressed in sample-mean loss units while the
            // stored normal equations are sums of residual products.
            hessian.diagonal(0, 0, 1).add_(
                result.state.damping * static_cast<double>(linearization.sample_count));
        }
        const auto solve_started = std::chrono::steady_clock::now();
        torch::Tensor step = torch::linalg_solve(hessian, -linearization.gradient);
        if (control.synchronize_cuda_after_solve) torch::cuda::synchronize();
        timings.solve_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solve_started).count();

        Model candidate = make_candidate(result.model, step);
        CandidateEvaluation evaluation = evaluate_candidate(candidate);
        timings.candidate_seconds = evaluation.timings.candidate_seconds;
        timings.candidate_train_seconds = evaluation.timings.candidate_train_seconds;
        timings.candidate_validation_seconds = evaluation.timings.candidate_validation_seconds;

        LmIteration event;
        event.linearization_refreshed = linearization_refreshed;
        if (std::isfinite(evaluation.objective_loss) &&
            evaluation.objective_loss < result.state.objective_loss) {
            result.model = std::move(candidate);
            result.state.objective_loss = evaluation.objective_loss;
            result.state.training_mse = evaluation.training_mse;
            result.state.validation_mse = evaluation.validation_mse;
            result.state.damping = std::max(result.state.damping / control.damping_decrease, 1.0e-15);
            ++result.state.accepted_updates;
            event.status = LmIteration::Status::Accepted;
            if (result.state.validation_mse < result.state.best_validation_mse) {
                result.state.best_validation_mse = result.state.validation_mse;
                result.state.validation_failures = 0;
                event.validation_improved = true;
            } else {
                ++result.state.validation_failures;
            }
            refresh_required = true;
        } else {
            result.state.damping = std::min(
                result.state.damping * control.damping_increase, control.damping_max);
            event.status = LmIteration::Status::Rejected;
            refresh_required = false;
        }
        event.state = result.state;
        event.timings = timings;
        observe_iteration(result.model, event);

        if (control.max_validation_failures > 0 &&
            result.state.validation_failures >= control.max_validation_failures) {
            result.stopping_reason = "max_validation_fail";
            return result;
        }
        if (control.max_accepted_updates > 0 &&
            result.state.accepted_updates >= control.max_accepted_updates) {
            result.stopping_reason = "max_accepted_updates";
            return result;
        }
    }
    return result;
}

}  // namespace native_lm
