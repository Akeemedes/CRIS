#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {

struct Law {
    std::string name;
    double m;
    double n_w;
    double n_nw;
};

struct Options {
    Law law{"regular", 0.4, 2.0, 2.0};
    std::string split{"train"};
    std::string design{"uniform"};
    std::filesystem::path output;
    std::uint64_t samples{0};
    std::uint64_t seed{0};
    double beta_max{150000.0};
};

struct RootResult {
    double root;
    double derivative;
    double residual;
    double bracket_width;
    bool representation_limited;
};

double fractional_flow(double u, const Law& law) {
    if (u <= 0.0) return 0.0;
    if (u >= 1.0) return 1.0;
    const double krw = std::pow(u, law.n_w);
    const double krnw = std::pow(1.0 - u, law.n_nw);
    return krw / (krw + law.m * krnw);
}

double fractional_flow_derivative(double u, const Law& law) {
    const double safe_u = std::clamp(u, std::numeric_limits<double>::min(),
        1.0 - std::numeric_limits<double>::epsilon());
    const double krw = std::pow(safe_u, law.n_w);
    const double krnw = std::pow(1.0 - safe_u, law.n_nw);
    const double dkrw = law.n_w * krw / safe_u;
    const double dkrnw = -law.n_nw * krnw / (1.0 - safe_u);
    const double denominator = krw + law.m * krnw;
    return (dkrw * denominator - krw * (dkrw + law.m * dkrnw)) /
        (denominator * denominator);
}

double residual(double u, double a, double beta, double u_old, const Law& law) {
    return u - u_old + beta * (fractional_flow(u, law) - a);
}

RootResult exact_root(double a, double beta, double u_old, const Law& law) {
    double lo = 0.0;
    double hi = 1.0;
    if (residual(lo, a, beta, u_old, law) > 0.0 || residual(hi, a, beta, u_old, law) < 0.0) {
        throw std::runtime_error("The monotone root is not bracketed on [0, 1]");
    }
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double mid = 0.5 * (lo + hi);
        if (residual(mid, a, beta, u_old, law) > 0.0) hi = mid;
        else lo = mid;
    }
    const double residual_lo = residual(lo, a, beta, u_old, law);
    const double residual_hi = residual(hi, a, beta, u_old, law);
    const double root = std::abs(residual_lo) <= std::abs(residual_hi) ? lo : hi;
    const double root_residual = residual(root, a, beta, u_old, law);
    const double machine_spacing = std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(root));
    const bool representation_limited = (hi - lo) <= machine_spacing && std::abs(root_residual) > 1.0e-8;
    return {root, 1.0 + beta * fractional_flow_derivative(root, law),
        root_residual, hi - lo, representation_limited};
}

double edge_sample(std::mt19937_64& engine) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double x = uniform(engine);
    return x < 0.5 ? 0.5 * std::pow(2.0 * x, 3.0)
                   : 1.0 - 0.5 * std::pow(2.0 * (1.0 - x), 3.0);
}

void sample_target_tail(double& a, double& beta, double& u_old,
    std::mt19937_64& engine, const Law& law, double beta_max) {
    // Sample the physical target first.  The resulting controls are chosen
    // from the admissible interval that makes this target the exact root of
    // the recurrence, so the endpoint emphasis is not merely an input proxy.
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double target = edge_sample(engine);
    beta = std::max(beta_max * 1.0e-10, uniform(engine) * beta_max);
    const double flux = fractional_flow(target, law);
    const double lower = std::max(0.0, flux - (1.0 - target) / beta);
    const double upper = std::min(1.0, flux + target / beta);
    a = lower + uniform(engine) * (upper - lower);
    u_old = std::clamp(target + beta * (flux - a), 0.0, 1.0);
}

double sample_log_beta_bin(std::mt19937_64& engine, std::size_t bin, double beta_max) {
    // The bins are deliberately physical decades.  The final bin is extended
    // to beta_max, which is 1e6 for the balanced production campaign.
    static constexpr std::array<double, 7> kLower{
        1.0e-8, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5};
    if (bin >= kLower.size()) throw std::runtime_error("Invalid beta decade index");
    const double lower = kLower[bin];
    const double upper = bin + 1 == kLower.size() ? beta_max : kLower[bin + 1];
    if (upper <= lower) throw std::runtime_error("--beta-max must exceed 1e5 for balanced_transport");
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    return std::exp(std::log(lower) + uniform(engine) * (std::log(upper) - std::log(lower)));
}

void sample_target_tail_at_beta(double& a, double& u_old, std::mt19937_64& engine,
    const Law& law, double beta) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double target = edge_sample(engine);
    const double flux = fractional_flow(target, law);
    const double lower = std::max(0.0, flux - (1.0 - target) / beta);
    const double upper = std::min(1.0, flux + target / beta);
    a = lower + uniform(engine) * (upper - lower);
    u_old = std::clamp(target + beta * (flux - a), 0.0, 1.0);
}

void sample_transport_path(double& a, double& u_old, std::mt19937_64& engine,
    const Law& law, std::size_t mode) {
    // These are local states generated by the transport recurrence itself,
    // rather than three independently sampled coordinates.
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    if (mode == 0) {          // Inlet invasion of initially dry material.
        a = 1.0;
        u_old = 0.0;
    } else if (mode == 1) {   // Complementary retreat state.
        a = 0.0;
        u_old = 1.0;
    } else if (mode == 2) {   // A stationary local state exactly satisfies a=f(u_old).
        u_old = edge_sample(engine);
        a = fractional_flow(u_old, law);
    } else if (mode == 3) {   // Upwind flux coupled to a distinct local history.
        a = fractional_flow(edge_sample(engine), law);
        u_old = edge_sample(engine);
    } else {
        throw std::runtime_error("Invalid transport-path mode");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("Missing value for " + key);
            return argv[index];
        };
        if (key == "--law") {
            const std::string law = value();
            if (law == "regular") options.law = {"regular", 0.4, 2.0, 2.0};
            else if (law == "imp") options.law = {"imp", 0.4, 2.0, 0.2};
            else throw std::runtime_error("--law must be regular or imp");
        } else if (key == "--split") options.split = value();
        else if (key == "--design") options.design = value();
        else if (key == "--output") options.output = value();
        else if (key == "--samples") options.samples = std::stoull(value());
        else if (key == "--seed") options.seed = std::stoull(value());
        else if (key == "--beta-max") options.beta_max = std::stod(value());
        else if (key == "--help") {
            std::cout << "Usage: twophasetransport_label_oracle --law regular|imp --split NAME --samples N --seed N "
                         "--output FILE [--beta-max VALUE] [--design uniform|edge|log_beta|target_tail|stratified|stratified_target_tail|balanced_transport]\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown option: " + key);
    }
    if (options.output.empty()) throw std::runtime_error("--output is required");
    if (options.samples == 0) throw std::runtime_error("--samples must be positive");
    if (options.beta_max <= 0.0) throw std::runtime_error("--beta-max must be positive");
    if (options.design != "uniform" && options.design != "edge" && options.design != "log_beta" &&
        options.design != "target_tail" && options.design != "stratified" &&
        options.design != "stratified_target_tail" && options.design != "balanced_transport") {
        throw std::runtime_error("--design must be uniform, edge, log_beta, target_tail, stratified, stratified_target_tail, or balanced_transport");
    }
    return options;
}

void write_manifest(const Options& options, double max_residual, double max_representable_residual,
    double max_bracket_width, std::uint64_t representation_limited_count, double elapsed_seconds) {
    const std::filesystem::path manifest = options.output.string() + ".manifest.json";
    std::ofstream stream(manifest);
    if (!stream) throw std::runtime_error("Unable to write manifest: " + manifest.string());
    stream << std::setprecision(17);
    stream << "{\n"
           << "  \"artifact\": \"twophasetransport_active_advection_local_labels\",\n"
           << "  \"recurrence\": \"u - u_old + beta * (f(u) - a) = 0\",\n"
           << "  \"law\": \"" << options.law.name << "\",\n"
           << "  \"fractional_flow\": {\"M\": " << options.law.m << ", \"n_w\": " << options.law.n_w
           << ", \"n_nw\": " << options.law.n_nw << ", \"Ng\": 0},\n"
           << "  \"split\": \"" << options.split << "\",\n"
           << "  \"design\": \"" << options.design << "\",\n"
           << "  \"sampling_allocation\": \"stratified=60% uniform, 25% log_beta, 15% edge; stratified_target_tail=30% uniform, 20% log_beta, 10% edge, 40% target_tail; balanced_transport=8% beta_zero, 42% decade-balanced log-beta backbone, 25% exact target-tail, 20% transport-path, 5% independent edge\",\n"
           << "  \"samples\": " << options.samples << ",\n"
           << "  \"seed\": " << options.seed << ",\n"
           << "  \"beta_max\": " << options.beta_max << ",\n"
           << "  \"root_method\": \"bracketed_bisection\",\n"
           << "  \"bisection_iterations\": 80,\n"
           << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n"
           << "  \"samples_per_second\": " << (static_cast<double>(options.samples) / elapsed_seconds) << ",\n"
           << "  \"max_abs_residual\": " << max_residual << ",\n"
           << "  \"max_abs_residual_representable\": " << max_representable_residual << ",\n"
           << "  \"representation_limited_count\": " << representation_limited_count << ",\n"
           << "  \"max_bracket_width\": " << max_bracket_width << "\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (!options.output.parent_path().empty()) std::filesystem::create_directories(options.output.parent_path());
        std::ofstream stream(options.output);
        if (!stream) throw std::runtime_error("Unable to write dataset: " + options.output.string());
        stream << std::setprecision(17);
        stream << "sample_id,split,law,design,a,beta,u_old,root,dr_du,residual,bisection_iterations,bracket_width,representation_limited\n";

        std::mt19937_64 engine(options.seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const auto started = std::chrono::steady_clock::now();
        double max_residual = 0.0;
        double max_representable_residual = 0.0;
        double max_bracket_width = 0.0;
        std::uint64_t representation_limited_count = 0;
        for (std::uint64_t sample = 0; sample < options.samples; ++sample) {
            double a = uniform(engine);
            double u_old = uniform(engine);
            double beta = uniform(engine) * options.beta_max;
            std::string sample_design = options.design;
            if (options.design == "stratified") {
                // A deterministic 20-row cycle freezes the coverage allocation
                // exactly: 12 uniform, 5 log-beta, and 3 saturation-edge rows.
                const auto phase = sample % 20;
                sample_design = phase < 3 ? "edge" : (phase < 8 ? "log_beta" : "uniform");
            } else if (options.design == "stratified_target_tail") {
                // A deterministic 20-row cycle: 6 uniform, 4 log-beta,
                // 2 input-edge, and 8 exact target-tail rows.
                const auto phase = sample % 20;
                sample_design = phase < 8 ? "target_tail" : (phase < 10 ? "edge" :
                    (phase < 14 ? "log_beta" : "uniform"));
            } else if (options.design == "balanced_transport") {
                // The 200-row cycle is exact for every planned nested prefix:
                // 16 beta=0 anchors; 84 decade-balanced backbone rows; 50
                // target-tail rows; 40 recurrence-derived transport-path rows;
                // and 10 independent endpoint stress rows.
                constexpr std::array<std::size_t, 50> kTailBins{
                    0,0,0, 1,1,1,1,1, 2,2,2,2,2,2,
                    3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,
                    5,5,5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6,6,6,6};
                const auto phase = sample % 200;
                if (phase < 16) {
                    sample_design = "beta_zero";
                    beta = 0.0;
                    a = (phase % 2 == 0) ? uniform(engine) : edge_sample(engine);
                    u_old = (phase % 2 == 0) ? uniform(engine) : edge_sample(engine);
                } else if (phase < 100) {
                    sample_design = "log_beta_backbone";
                    beta = sample_log_beta_bin(engine, (phase - 16) % 7, options.beta_max);
                } else if (phase < 150) {
                    sample_design = "target_tail";
                    beta = sample_log_beta_bin(engine, kTailBins[phase - 100], options.beta_max);
                    sample_target_tail_at_beta(a, u_old, engine, options.law, beta);
                } else if (phase < 190) {
                    sample_design = "transport_path";
                    beta = sample_log_beta_bin(engine, (phase - 150) % 7, options.beta_max);
                    sample_transport_path(a, u_old, engine, options.law, ((phase - 150) / 10) % 4);
                } else {
                    sample_design = "independent_edge";
                    beta = sample_log_beta_bin(engine, (phase - 190) % 7, options.beta_max);
                    a = edge_sample(engine);
                    u_old = edge_sample(engine);
                }
            }
            if (sample_design == "edge") {
                a = edge_sample(engine);
                u_old = edge_sample(engine);
            } else if (sample_design == "log_beta") {
                const double beta_min = std::max(1.0e-8, options.beta_max * 1.0e-10);
                beta = std::exp(std::log(beta_min) + uniform(engine) * (std::log(options.beta_max) - std::log(beta_min)));
            } else if (sample_design == "target_tail") {
                sample_target_tail(a, beta, u_old, engine, options.law, options.beta_max);
            }
            const RootResult result = exact_root(a, beta, u_old, options.law);
            max_residual = std::max(max_residual, std::abs(result.residual));
            if (!result.representation_limited) {
                max_representable_residual = std::max(max_representable_residual, std::abs(result.residual));
            } else {
                ++representation_limited_count;
            }
            max_bracket_width = std::max(max_bracket_width, result.bracket_width);
            stream << sample << ',' << options.split << ',' << options.law.name << ',' << sample_design << ','
                   << a << ',' << beta << ',' << u_old << ',' << result.root << ',' << result.derivative << ','
                   << result.residual << ",80," << result.bracket_width << ',' << result.representation_limited << '\n';
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        write_manifest(options, max_residual, max_representable_residual, max_bracket_width, representation_limited_count, elapsed);
        std::cout << "wrote_dataset=" << options.output.string() << '\n'
                  << "wrote_manifest=" << options.output.string() + ".manifest.json" << '\n'
                  << "samples=" << options.samples << '\n'
                  << "elapsed_seconds=" << elapsed << '\n'
                  << "samples_per_second=" << (static_cast<double>(options.samples) / elapsed) << '\n'
                  << "max_abs_residual=" << max_residual << '\n'
                  << "representation_limited_count=" << representation_limited_count << '\n';
    } catch (const std::exception& error) {
        std::cerr << "twophasetransport_label_oracle error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
