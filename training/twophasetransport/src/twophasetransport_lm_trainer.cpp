#include <torch/torch.h>
#include <torch/cuda.h>
#include <native_lm/lm_optimizer.hpp>
#include <native_lm/recovery.hpp>
#include <native_lm/least_squares_objective.hpp>
#include "regular_advection_objective_terms.hpp"
#include "twophasetransport_model_package.hpp"

#include <ATen/Parallel.h>
#ifdef TWOPHASETRANSPORT_HAS_MKL
#include <mkl.h>
#endif
#ifdef TWOPHASETRANSPORT_HAS_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

constexpr int kInput = 3;
constexpr int kHidden = 20;
constexpr int kParameters = 1361;
constexpr std::array<char, 8> kMagic{'T', 'P', 'T', 'B', 'I', 'N', '0', '1'};

struct Header {
    std::array<char, 8> magic;
    std::uint64_t count;
    std::uint32_t fields;
    std::uint32_t reserved;
};

struct Record {
    double a;
    double beta;
    double u_old;
    double root;
    double dr_du;
    double representation_limited;
};

static_assert(sizeof(Header) == 24);
static_assert(sizeof(Record) == 48);

struct Dataset {
    std::vector<Record> values;
};

// This is the TwoPhaseTransport adapter to the generic native LM objective.
// The normalized target is intentionally local to the physics package; the
// generic objective layer only knows that a target functor maps a sample to a
// scalar supervised value.
struct NormalizedRootTarget {
    double operator()(const Record& record) const { return 2.0 * record.root - 1.0; }
};

using TrainingObjective = native_lm::StackedResiduals<
    native_lm::SupervisedMseTerm<NormalizedRootTarget>>;
using BoundedRecurrenceTerm = native_lm::SoftClippedResidualTerm<
    twophasetransport::training::RegularAdvectionRecurrenceTerm>;
using RecurrenceTrainingObjective = native_lm::StackedResiduals<
    native_lm::SupervisedMseTerm<NormalizedRootTarget>,
    native_lm::ScaledResidualTerm<BoundedRecurrenceTerm>>;
using ObjectiveVariant = std::variant<TrainingObjective, RecurrenceTrainingObjective>;

enum class Initialization { NguyenWidrow, Xavier };
enum class BetaCoordinate { Raw, Log1p };
enum class ObjectiveMode { SupervisedMse, MsePlusRecurrence };
enum class ComputeDevice { Cpu, Cuda };
enum class CudaPrecision { Float64, Float32 };
enum class NormalEquationKernel { TorchGemm, MklSyrk };
enum class ProfileMode { None, Jacobian, Candidate };

struct Options {
    std::filesystem::path train_path;
    std::filesystem::path validation_path;
    std::filesystem::path run_dir;
    // Optional directory of fixed, pre-shuffled packed batches.  Batching is
    // opt-in: no option changes the full-batch reference trajectory.
    std::filesystem::path batch_directory;
    std::uint64_t seed{20260904};
    int max_iterations{12};
    // Each block holds about 170 MiB of float64 Jacobian data for the
    // 1,361-parameter network, bounding temporary memory use.
    int block_size{16384};
    int threads{0};
    // CUDA offloads the dense normal-equation algebra. The analytic Jacobian
    // is assembled on the CPU and transferred blockwise. CPU float64 is the
    // default execution mode.
    ComputeDevice compute_device{ComputeDevice::Cpu};
    CudaPrecision cuda_precision{CudaPrecision::Float64};
    // DSYRK forms one triangle of the symmetric matrix J^T J. Builds without
    // oneMKL use the portable GEMM implementation.
#ifdef TWOPHASETRANSPORT_HAS_MKL
    NormalEquationKernel normal_equation_kernel{NormalEquationKernel::MklSyrk};
#else
    NormalEquationKernel normal_equation_kernel{NormalEquationKernel::TorchGemm};
#endif
    // Profiling modes measure one fixed-state phase and exit without updating
    // the model parameters.
    ProfileMode profile_mode{ProfileMode::None};
    double beta_max{150000.0};
    BetaCoordinate beta_coordinate{BetaCoordinate::Raw};
    ObjectiveMode objective_mode{ObjectiveMode::SupervisedMse};
    double recurrence_weight{0.0};
    double nonwetting_exponent{2.0};
    // This cap is measured in the root-Jacobian-normalized residual, whose
    // local scale equals normalized root error. It bounds only far-from-root
    // cold-start defects; tanh(z)=z+O(z^3) leaves late accuracy unchanged.
    double recurrence_cap{1.0};
    double lambda{1.0e-3};
    // Reduce damping after accepted steps and increase it after failed trials.
    double lambda_decrease{2.0};
    double lambda_increase{10.0};
    double lambda_max{1.0e10};
    bool diagonal_damping{false};
    Initialization initialization{Initialization::NguyenWidrow};
    std::vector<double> checkpoint_mse_thresholds{1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,
        1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12};
    // A recovery state is tiny (1361 float64 values), so retain it after every
    // accepted update.  The two alternating slots ensure that a process loss
    // during a write still leaves the preceding accepted model intact.
    int recovery_checkpoint_interval{1};
    int accepted_update_offset{0};
    // Proposal numbering continues across watchdog recovery or deliberate
    // extensions, so a single optimizer trace remains chronological.
    int iteration_offset{0};
    int max_accepted_updates{0};
    int max_validation_fail{500};
    bool gradient_check{false};
    std::filesystem::path initial_checkpoint;
    std::filesystem::path initial_best_validation_checkpoint;
    double initial_best_validation_mse{std::numeric_limits<double>::infinity()};
    int initial_best_validation_generation{0};
    std::filesystem::path export_package;
    bool portable_recovery{false};
    std::filesystem::path resume_state;
};

using Timings = native_lm::LmTimings;

std::size_t representation_limited_count(const Dataset& dataset) {
    return static_cast<std::size_t>(std::count_if(dataset.values.begin(), dataset.values.end(),
        [](const Record& row) { return row.representation_limited != 0.0; }));
}

void write_run_manifest(const std::filesystem::path& path, const Options& options,
    const Dataset& train, const Dataset& validation, std::size_t batch_count) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Unable to write run manifest: " + path.string());
    stream << std::setprecision(17);
    stream << "{\n"
           << "  \"artifact\": \"twophasetransport_native_lm_training_run\",\n"
           << "  \"architecture\": [3, 20, 20, 20, 20, 1],\n"
           << "  \"parameter_count\": " << kParameters << ",\n"
           << "  \"hidden_activation\": \"tanh\",\n"
           << "  \"output_activation\": \"linear\",\n"
           << "  \"optimizer\": \"Levenberg-Marquardt\",\n"
           << "  \"objective\": {\"composition\": \"stacked_residuals\", \"terms\": [{\"kind\": \"supervised_mse\", \"weight\": 1.0}";
    if (options.objective_mode == ObjectiveMode::MsePlusRecurrence) {
        stream << ", {\"kind\": \"soft_clipped_advection_root_jacobian_normalized_recurrence\", \"n_nw\": "
               << options.nonwetting_exponent << ", \"M\": 0.4, \"n_w\": 2, \"weight\": "
               << options.recurrence_weight << ", \"cap\": " << options.recurrence_cap << "}";
    }
    stream << "]},\n"
           << "  \"precision\": \"float64\",\n"
           << "  \"compute_device\": \"" << (options.compute_device == ComputeDevice::Cuda ? "cuda" : "cpu") << "\",\n"
           << "  \"cuda_accumulation_precision\": \"" << (options.cuda_precision == CudaPrecision::Float32 ? "float32" : "float64") << "\",\n"
           << "  \"normal_equation_kernel\": \"" << (options.normal_equation_kernel == NormalEquationKernel::MklSyrk ? "mkl_syrk" : "torch_gemm") << "\",\n"
           << "  \"profile_mode\": \"" << (options.profile_mode == ProfileMode::Jacobian ? "jacobian" : options.profile_mode == ProfileMode::Candidate ? "candidate" : "none") << "\",\n"
           << "  \"normalization\": {\"a\": [-1, 1], \"beta\": [-1, 1], \"u_old\": [-1, 1], \"root\": [-1, 1]},\n"
           << "  \"beta_max\": " << options.beta_max << ",\n"
           << "  \"beta_coordinate\": \"" << (options.beta_coordinate == BetaCoordinate::Log1p ? "log1p" : "raw") << "\",\n"
           << "  \"seed\": " << options.seed << ",\n"
           << "  \"initialization\": \"" << (options.initialization == Initialization::NguyenWidrow ? "nguyen_widrow" : "xavier") << "\",\n"
           << "  \"checkpoint_validation_mse_thresholds\": [";
    for (std::size_t index = 0; index < options.checkpoint_mse_thresholds.size(); ++index) {
        if (index != 0) stream << ", ";
        stream << options.checkpoint_mse_thresholds[index];
    }
    stream << "],\n"
           << "  \"recovery_checkpoint_interval\": " << options.recovery_checkpoint_interval << ",\n"
           << "  \"accepted_update_offset\": " << options.accepted_update_offset << ",\n"
           << "  \"iteration_offset\": " << options.iteration_offset << ",\n"
           << "  \"initial_lambda\": " << options.lambda << ",\n"
           << "  \"lambda_decrease_factor\": " << options.lambda_decrease << ",\n"
           << "  \"lambda_increase_factor\": " << options.lambda_increase << ",\n"
           << "  \"lambda_max\": " << options.lambda_max << ",\n"
           << "  \"damping\": \"" << (options.diagonal_damping ? "diagonal" : "identity") << "\",\n"
           << "  \"max_accepted_updates\": " << options.max_accepted_updates << ",\n"
           << "  \"max_validation_fail\": " << options.max_validation_fail << ",\n"
           << "  \"best_validation_snapshot\": \"two alternating slots; final export restores the best held-out model\",\n"
           << "  \"initial_checkpoint\": \"" << options.initial_checkpoint.generic_string() << "\",\n"
           << "  \"initial_best_validation_checkpoint\": \"" << options.initial_best_validation_checkpoint.generic_string() << "\",\n"
           << "  \"max_iterations\": " << options.max_iterations << ",\n"
           << "  \"jacobian_block_size\": " << options.block_size << ",\n"
           << "  \"requested_threads\": " << options.threads << ",\n"
           << "  \"batch_directory\": \"" << options.batch_directory.generic_string() << "\",\n"
           << "  \"batch_count\": " << batch_count << ",\n"
           << "  \"batch_schedule\": \""
           << (batch_count == 0 ? "full_batch" : "fixed_pre_shuffled; advance_after_accepted_update; batch_index=accepted_updates_modulo_batch_count")
           << "\",\n"
           << "  \"torch_version\": \"" << TORCH_VERSION << "\",\n"
           << "  \"checkpoint_format\": \"raw little-endian float64, fixed 1361-parameter layout\",\n"
           << "  \"train\": {\"path\": \"" << options.train_path.generic_string() << "\", \"rows\": " << train.values.size()
           << ", \"representation_limited_rows\": " << representation_limited_count(train) << "},\n"
           << "  \"validation\": {\"path\": \"" << options.validation_path.generic_string() << "\", \"rows\": " << validation.values.size()
           << ", \"representation_limited_rows\": " << representation_limited_count(validation) << "}\n"
           << "}\n";
}

Dataset load_dataset(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Unable to read dataset: " + path.string());
    Header header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream || header.magic != kMagic || header.fields != 6) {
        throw std::runtime_error("Invalid packed dataset: " + path.string());
    }
    Dataset dataset;
    dataset.values.resize(header.count);
    stream.read(reinterpret_cast<char*>(dataset.values.data()),
        static_cast<std::streamsize>(dataset.values.size() * sizeof(Record)));
    if (!stream) throw std::runtime_error("Truncated packed dataset: " + path.string());
    return dataset;
}

std::vector<Dataset> load_batches(const std::filesystem::path& directory, std::size_t expected_rows) {
    if (directory.empty()) return {};
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("--batch-directory is not a directory: " + directory.string());
    }
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".tptbin") continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("batch_", 0) == 0) paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.size() < 2) {
        throw std::runtime_error("--batch-directory must contain at least two batch_*.tptbin files");
    }
    std::vector<Dataset> batches;
    batches.reserve(paths.size());
    std::size_t rows = 0;
    for (const auto& path : paths) {
        batches.push_back(load_dataset(path));
        if (batches.back().values.empty()) throw std::runtime_error("Batch is empty: " + path.string());
        rows += batches.back().values.size();
    }
    if (rows != expected_rows) {
        throw std::runtime_error("Batch rows do not equal the full training rows");
    }
    return batches;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help") {
            std::cout << "Usage: twophasetransport_lm_trainer --train TRAIN.tptbin --validation VAL.tptbin --run-dir DIR "
                         "[--max-iterations N] [--block-size N] [--beta-max N] [--beta-coordinate raw|log1p] [--objective mse|mse+recurrence] [--recurrence-weight VALUE] [--recurrence-cap VALUE] [--lambda VALUE] [--seed N] [--threads N] "
                         "[--batch-directory DIR] [--device cpu|cuda] [--cuda-precision float64|float32] [--normal-equation-kernel torch-gemm|mkl-syrk] [--profile none|jacobian|candidate] "
                         "[--initial-checkpoint FILE] [--initial-best-validation-checkpoint FILE] [--initial-best-validation-mse VALUE] [--initial-best-validation-generation N] [--export-package FILE] [--lambda-decrease VALUE] "
                         "[--lambda-increase VALUE] [--lambda-max VALUE] [--initialization nguyen-widrow|xavier] [--damping diagonal|identity] "
                         "[--checkpoint-mse-thresholds LIST] [--recovery-checkpoint-interval N] [--accepted-update-offset N] [--iteration-offset N] [--max-accepted-updates N] [--max-validation-fail N] [--gradient-check]\n";
            std::exit(0);
        }
        if (key == "--gradient-check") {
            options.gradient_check = true;
            continue;
        }
        if (++index >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[index];
        if (key == "--train") options.train_path = value;
        else if (key == "--validation") options.validation_path = value;
        else if (key == "--run-dir") options.run_dir = value;
        else if (key == "--max-iterations") options.max_iterations = std::stoi(value);
        else if (key == "--block-size") options.block_size = std::stoi(value);
        else if (key == "--beta-max") options.beta_max = std::stod(value);
        else if (key == "--beta-coordinate") {
            if (value == "raw") options.beta_coordinate = BetaCoordinate::Raw;
            else if (value == "log1p") options.beta_coordinate = BetaCoordinate::Log1p;
            else throw std::runtime_error("--beta-coordinate must be 'raw' or 'log1p'");
        }
        else if (key == "--objective") {
            if (value == "mse") options.objective_mode = ObjectiveMode::SupervisedMse;
            else if (value == "mse+recurrence") options.objective_mode = ObjectiveMode::MsePlusRecurrence;
            else throw std::runtime_error("--objective must be 'mse' or 'mse+recurrence'");
        }
        else if (key == "--recurrence-weight") options.recurrence_weight = std::stod(value);
        else if (key == "--nonwetting-exponent") options.nonwetting_exponent = std::stod(value);
        else if (key == "--recurrence-cap") options.recurrence_cap = std::stod(value);
        else if (key == "--lambda") options.lambda = std::stod(value);
        else if (key == "--lambda-decrease") options.lambda_decrease = std::stod(value);
        else if (key == "--lambda-increase") options.lambda_increase = std::stod(value);
        else if (key == "--lambda-max") options.lambda_max = std::stod(value);
        else if (key == "--initialization") {
            if (value == "nguyen-widrow") options.initialization = Initialization::NguyenWidrow;
            else if (value == "xavier") options.initialization = Initialization::Xavier;
            else throw std::runtime_error("--initialization must be 'nguyen-widrow' or 'xavier'");
        }
        else if (key == "--checkpoint-mse-thresholds") {
            options.checkpoint_mse_thresholds.clear();
            std::stringstream values(value);
            std::string token;
            while (std::getline(values, token, ',')) {
                if (token.empty()) throw std::runtime_error("Empty value in --checkpoint-mse-thresholds");
                const double threshold = std::stod(token);
                if (!(threshold > 0.0) || !std::isfinite(threshold)) {
                    throw std::runtime_error("Checkpoint MSE thresholds must be finite and positive");
                }
                options.checkpoint_mse_thresholds.push_back(threshold);
            }
            if (options.checkpoint_mse_thresholds.empty()) throw std::runtime_error("--checkpoint-mse-thresholds cannot be empty");
            std::sort(options.checkpoint_mse_thresholds.begin(), options.checkpoint_mse_thresholds.end(), std::greater<double>());
            options.checkpoint_mse_thresholds.erase(std::unique(options.checkpoint_mse_thresholds.begin(),
                options.checkpoint_mse_thresholds.end()), options.checkpoint_mse_thresholds.end());
        }
        else if (key == "--recovery-checkpoint-interval") options.recovery_checkpoint_interval = std::stoi(value);
        else if (key == "--accepted-update-offset") options.accepted_update_offset = std::stoi(value);
        else if (key == "--iteration-offset") options.iteration_offset = std::stoi(value);
        else if (key == "--damping") {
            if (value == "diagonal") options.diagonal_damping = true;
            else if (value == "identity") options.diagonal_damping = false;
            else throw std::runtime_error("--damping must be 'diagonal' or 'identity'");
        }
        else if (key == "--max-accepted-updates") options.max_accepted_updates = std::stoi(value);
        else if (key == "--max-validation-fail") options.max_validation_fail = std::stoi(value);
        else if (key == "--seed") options.seed = std::stoull(value);
        else if (key == "--threads") options.threads = std::stoi(value);
        else if (key == "--batch-directory") options.batch_directory = value;
        else if (key == "--device") {
            if (value == "cpu") options.compute_device = ComputeDevice::Cpu;
            else if (value == "cuda") options.compute_device = ComputeDevice::Cuda;
            else throw std::runtime_error("--device must be 'cpu' or 'cuda'");
        }
        else if (key == "--cuda-precision") {
            if (value == "float64") options.cuda_precision = CudaPrecision::Float64;
            else if (value == "float32") options.cuda_precision = CudaPrecision::Float32;
            else throw std::runtime_error("--cuda-precision must be 'float64' or 'float32'");
        }
        else if (key == "--normal-equation-kernel") {
            if (value == "torch-gemm") options.normal_equation_kernel = NormalEquationKernel::TorchGemm;
            else if (value == "mkl-syrk") options.normal_equation_kernel = NormalEquationKernel::MklSyrk;
            else throw std::runtime_error("--normal-equation-kernel must be 'torch-gemm' or 'mkl-syrk'");
        }
        else if (key == "--profile") {
            if (value == "none") options.profile_mode = ProfileMode::None;
            else if (value == "jacobian") options.profile_mode = ProfileMode::Jacobian;
            else if (value == "candidate") options.profile_mode = ProfileMode::Candidate;
            else throw std::runtime_error("--profile must be 'none', 'jacobian', or 'candidate'");
        }
        else if (key == "--initial-checkpoint") options.initial_checkpoint = value;
        else if (key == "--portable-recovery") {
            if (value != "0" && value != "1") throw std::runtime_error("--portable-recovery must be 0 or 1");
            options.portable_recovery = value == "1";
        }
        else if (key == "--resume-state") options.resume_state = value;
        else if (key == "--initial-best-validation-checkpoint") options.initial_best_validation_checkpoint = value;
        else if (key == "--initial-best-validation-mse") options.initial_best_validation_mse = std::stod(value);
        else if (key == "--initial-best-validation-generation") options.initial_best_validation_generation = std::stoi(value);
        else if (key == "--export-package") options.export_package = value;
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (options.train_path.empty() || options.validation_path.empty() || options.run_dir.empty()) {
        throw std::runtime_error("--train, --validation, and --run-dir are required");
    }
    if (options.max_iterations <= 0 || options.block_size <= 0 || options.beta_max <= 0.0 || options.lambda <= 0.0) {
        throw std::runtime_error("Iteration, block-size, beta-max, and lambda options must be positive");
    }
    if (options.nonwetting_exponent != 2.0 && options.nonwetting_exponent != 0.2)
        throw std::runtime_error("--nonwetting-exponent must be 2 or 0.2");
    if (!(options.recurrence_weight >= 0.0) || !std::isfinite(options.recurrence_weight)) {
        throw std::runtime_error("--recurrence-weight must be finite and nonnegative");
    }
    if (!(options.recurrence_cap > 0.0) || !std::isfinite(options.recurrence_cap)) {
        throw std::runtime_error("--recurrence-cap must be finite and positive");
    }
    if (options.objective_mode == ObjectiveMode::SupervisedMse && options.recurrence_weight != 0.0) {
        throw std::runtime_error("--recurrence-weight requires --objective mse+recurrence");
    }
    if (options.objective_mode == ObjectiveMode::MsePlusRecurrence && !(options.recurrence_weight > 0.0)) {
        throw std::runtime_error("--objective mse+recurrence requires a positive --recurrence-weight");
    }
    if (!(options.lambda_decrease > 1.0 && options.lambda_increase > 1.0 && options.lambda_max >= options.lambda)) {
        throw std::runtime_error("Invalid LM damping parameters");
    }
    if (options.max_accepted_updates < 0) throw std::runtime_error("--max-accepted-updates must be nonnegative");
    if (options.recovery_checkpoint_interval < 0) throw std::runtime_error("--recovery-checkpoint-interval must be nonnegative");
    if (options.accepted_update_offset < 0) throw std::runtime_error("--accepted-update-offset must be nonnegative");
    if (options.iteration_offset < 0) throw std::runtime_error("--iteration-offset must be nonnegative");
    if (options.initial_best_validation_generation < 0) throw std::runtime_error("--initial-best-validation-generation must be nonnegative");
    if (!options.initial_best_validation_checkpoint.empty() &&
        (!(options.initial_best_validation_mse >= 0.0) || !std::isfinite(options.initial_best_validation_mse))) {
        throw std::runtime_error("A resumed best-validation checkpoint requires a finite nonnegative MSE");
    }
    if (options.max_validation_fail < 0) throw std::runtime_error("--max-validation-fail must be nonnegative");
    if (!options.resume_state.empty() && (!options.portable_recovery || !options.initial_checkpoint.empty() ||
        !options.initial_best_validation_checkpoint.empty() || options.accepted_update_offset || options.iteration_offset))
        throw std::runtime_error("--resume-state requires portable recovery and cannot be mixed with weight-only/legacy restart options");
    if (options.portable_recovery && (!options.batch_directory.empty() || options.profile_mode != ProfileMode::None))
        throw std::runtime_error("Portable recovery currently supports full-batch training only");
#ifndef TWOPHASETRANSPORT_HAS_MKL
    if (options.normal_equation_kernel == NormalEquationKernel::MklSyrk) {
        throw std::runtime_error("This build has no oneMKL DSYRK backend; use --normal-equation-kernel torch-gemm");
    }
#endif
    return options;
}

class Network {
public:
    explicit Network(std::uint64_t seed, Initialization initialization, BetaCoordinate beta_coordinate)
        : beta_coordinate_(beta_coordinate), weights_(kParameters) {
        std::mt19937_64 engine(seed);
        if (initialization == Initialization::NguyenWidrow) {
            std::uniform_real_distribution<double> uniform(-1.0, 1.0);
            initialize_nguyen_widrow_layer(kW1, kB1, kHidden, kInput, uniform, engine);
            initialize_nguyen_widrow_layer(kW2, kB2, kHidden, kHidden, uniform, engine);
            initialize_nguyen_widrow_layer(kW3, kB3, kHidden, kHidden, uniform, engine);
            initialize_nguyen_widrow_layer(kW4, kB4, kHidden, kHidden, uniform, engine);
            initialize_uniform_layer(kW5, kB5, 1, kHidden, uniform, engine);
        } else {
            std::normal_distribution<double> normal(0.0, 1.0);
            initialize_xavier_layer(kW1, kB1, kHidden, kInput, normal, engine);
            initialize_xavier_layer(kW2, kB2, kHidden, kHidden, normal, engine);
            initialize_xavier_layer(kW3, kB3, kHidden, kHidden, normal, engine);
            initialize_xavier_layer(kW4, kB4, kHidden, kHidden, normal, engine);
            initialize_xavier_layer(kW5, kB5, 1, kHidden, normal, engine);
        }
    }

    double forward(const Record& record, double beta_max, double* gradient = nullptr) const {
        const std::array<double, kInput> input{
            2.0 * record.a - 1.0,
            beta_coordinate_ == BetaCoordinate::Log1p
                ? 2.0 * std::log1p(record.beta) / std::log1p(beta_max) - 1.0
                : 2.0 * record.beta / beta_max - 1.0,
            2.0 * record.u_old - 1.0};
        std::array<double, kHidden> h1{}, h2{}, h3{}, h4{}, d1{}, d2{}, d3{}, d4{};
        hidden_layer(input.data(), kInput, h1.data(), 0, kB1);
        hidden_layer(h1.data(), kHidden, h2.data(), kW2, kB2);
        hidden_layer(h2.data(), kHidden, h3.data(), kW3, kB3);
        hidden_layer(h3.data(), kHidden, h4.data(), kW4, kB4);
        double output = weights_[kB5];
        for (int index = 0; index < kHidden; ++index) output += weights_[kW5 + index] * h4[index];
        if (gradient == nullptr) return output;

        for (int index = 0; index < kHidden; ++index) {
            gradient[kW5 + index] = h4[index];
            d4[index] = weights_[kW5 + index] * (1.0 - h4[index] * h4[index]);
        }
        gradient[kB5] = 1.0;
        for (int row = 0; row < kHidden; ++row) {
            for (int col = 0; col < kHidden; ++col) {
                gradient[kW4 + row * kHidden + col] = d4[row] * h3[col];
            }
            gradient[kB4 + row] = d4[row];
        }
        for (int col = 0; col < kHidden; ++col) {
            double sum = 0.0;
            for (int row = 0; row < kHidden; ++row) sum += weights_[kW4 + row * kHidden + col] * d4[row];
            d3[col] = sum * (1.0 - h3[col] * h3[col]);
        }
        for (int row = 0; row < kHidden; ++row) {
            for (int col = 0; col < kHidden; ++col) {
                gradient[kW3 + row * kHidden + col] = d3[row] * h2[col];
            }
            gradient[kB3 + row] = d3[row];
        }
        for (int col = 0; col < kHidden; ++col) {
            double sum = 0.0;
            for (int row = 0; row < kHidden; ++row) sum += weights_[kW3 + row * kHidden + col] * d3[row];
            d2[col] = sum * (1.0 - h2[col] * h2[col]);
        }
        for (int row = 0; row < kHidden; ++row) {
            for (int col = 0; col < kHidden; ++col) {
                gradient[kW2 + row * kHidden + col] = d2[row] * h1[col];
            }
            gradient[kB2 + row] = d2[row];
        }
        for (int col = 0; col < kHidden; ++col) {
            double sum = 0.0;
            for (int row = 0; row < kHidden; ++row) sum += weights_[kW2 + row * kHidden + col] * d2[row];
            d1[col] = sum * (1.0 - h1[col] * h1[col]);
        }
        for (int row = 0; row < kHidden; ++row) {
            for (int col = 0; col < kInput; ++col) gradient[row * kInput + col] = d1[row] * input[col];
            gradient[kB1 + row] = d1[row];
        }
        return output;
    }

    [[nodiscard]] double mse(const Dataset& dataset, double beta_max) const {
        // Fixed chunks make the final reduction independent of the requested
        // thread count while allowing the expensive forward evaluations to run
        // in parallel. Candidate MSE is used for LM acceptance, so this avoids
        // thread-count-dependent summation noise in the optimizer decision.
        constexpr std::int64_t kRowsPerChunk = 4096;
        const std::int64_t row_count = static_cast<std::int64_t>(dataset.values.size());
        const std::int64_t chunk_count = at::divup(row_count, kRowsPerChunk);
        std::vector<double> partial_sums(static_cast<std::size_t>(chunk_count), 0.0);
        at::parallel_for(0, chunk_count, 1, [&](std::int64_t first_chunk, std::int64_t end_chunk) {
            for (std::int64_t chunk = first_chunk; chunk < end_chunk; ++chunk) {
                const std::int64_t begin = chunk * kRowsPerChunk;
                const std::int64_t end = std::min(row_count, begin + kRowsPerChunk);
                double subtotal = 0.0;
                for (std::int64_t row = begin; row < end; ++row) {
                    const Record& record = dataset.values[static_cast<std::size_t>(row)];
                    const double error = forward(record, beta_max) - (2.0 * record.root - 1.0);
                    subtotal += error * error;
                }
                partial_sums[static_cast<std::size_t>(chunk)] = subtotal;
            }
        });
        double total = 0.0;
        for (const double partial : partial_sums) {
            total += partial;
        }
        return total / static_cast<double>(dataset.values.size());
    }

    void add_step(const torch::Tensor& step) {
        const auto* values = step.data_ptr<double>();
        for (int index = 0; index < kParameters; ++index) weights_[index] += values[index];
    }

    void add_to_parameter(int index, double value) { weights_.at(static_cast<std::size_t>(index)) += value; }
    const double* weight_data() const { return weights_.data(); }
    void restore_weights(const std::vector<double>& values) {
        if(values.size()!=kParameters) throw std::runtime_error("Recovery parameter mismatch");
        weights_=values;
    }

    void write_checkpoint(const std::filesystem::path& path) const {
        std::ofstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("Unable to write checkpoint: " + path.string());
        stream.write(reinterpret_cast<const char*>(weights_.data()), static_cast<std::streamsize>(weights_.size() * sizeof(double)));
    }

    void load_checkpoint(const std::filesystem::path& path) {
        weights_ = twophasetransport::model_package::read_checkpoint(path);
    }

    void write_model_package(const std::filesystem::path& path, double beta_max) const {
        twophasetransport::model_package::write(path, weights_, beta_max,
            beta_coordinate_ == BetaCoordinate::Log1p);
    }

private:
    BetaCoordinate beta_coordinate_;
    static constexpr int kW1 = 0;
    static constexpr int kB1 = kW1 + kInput * kHidden;
    static constexpr int kW2 = kB1 + kHidden;
    static constexpr int kB2 = kW2 + kHidden * kHidden;
    static constexpr int kW3 = kB2 + kHidden;
    static constexpr int kB3 = kW3 + kHidden * kHidden;
    static constexpr int kW4 = kB3 + kHidden;
    static constexpr int kB4 = kW4 + kHidden * kHidden;
    static constexpr int kW5 = kB4 + kHidden;
    static constexpr int kB5 = kW5 + kHidden;

    template <typename Distribution, typename Engine>
    void initialize_xavier_layer(int weight_offset, int bias_offset, int rows, int cols, Distribution& distribution, Engine& engine) {
        const double scale = std::sqrt(2.0 / static_cast<double>(rows + cols));
        for (int index = 0; index < rows * cols; ++index) weights_[weight_offset + index] = scale * distribution(engine);
        std::fill(weights_.begin() + bias_offset, weights_.begin() + bias_offset + rows, 0.0);
    }

    template <typename Distribution, typename Engine>
    void initialize_uniform_layer(int weight_offset, int bias_offset, int rows, int cols, Distribution& distribution, Engine& engine) {
        for (int index = 0; index < rows * cols; ++index) weights_[weight_offset + index] = distribution(engine);
        for (int row = 0; row < rows; ++row) weights_[bias_offset + row] = distribution(engine);
    }

    template <typename Distribution, typename Engine>
    void initialize_nguyen_widrow_layer(int weight_offset, int bias_offset, int rows, int cols, Distribution& distribution, Engine& engine) {
        const double magnitude = 1.4 * std::pow(static_cast<double>(rows), 1.0 / static_cast<double>(cols));
        // MATLAB's calcnw uses 0.7*rows^(1/cols), then rescales tansig's
        // active net-input interval [-2,2], giving the 1.4*rows^(1/cols) row norm.
        for (int row = 0; row < rows; ++row) {
            double squared_norm = 0.0;
            for (int col = 0; col < cols; ++col) {
                const double value = distribution(engine);
                weights_[weight_offset + row * cols + col] = value;
                squared_norm += value * value;
            }
            const double scale = magnitude / std::sqrt(squared_norm);
            for (int col = 0; col < cols; ++col) weights_[weight_offset + row * cols + col] *= scale;
            const double location = rows == 1 ? 0.0 : -1.0 + 2.0 * static_cast<double>(row) / static_cast<double>(rows - 1);
            const double sign = weights_[weight_offset + row * cols] < 0.0 ? -1.0 : 1.0;
            weights_[bias_offset + row] = magnitude * location * sign;
        }
    }

    void hidden_layer(const double* input, int input_size, double* output, int weight_offset, int bias_offset) const {
        for (int row = 0; row < kHidden; ++row) {
            double value = weights_[bias_offset + row];
            for (int col = 0; col < input_size; ++col) value += weights_[weight_offset + row * input_size + col] * input[col];
            output[row] = std::tanh(value);
        }
    }

    std::vector<double> weights_;
};

template <class Objective>
double objective_loss(const Dataset& dataset, const Network& network,
                      double beta_max, const Objective& objective) {
    // Use the same fixed reduction layout as Network::mse. The default
    // supervised-MSE objective consequently preserves the established
    // candidate-acceptance arithmetic and thread-count independence.
    constexpr std::int64_t kRowsPerChunk = 4096;
    constexpr std::size_t kResiduals = Objective::kResidualsPerSample;
    const std::int64_t row_count = static_cast<std::int64_t>(dataset.values.size());
    const std::int64_t chunk_count = at::divup(row_count, kRowsPerChunk);
    std::vector<double> partial_sums(static_cast<std::size_t>(chunk_count), 0.0);
    at::parallel_for(0, chunk_count, 1, [&](std::int64_t first_chunk, std::int64_t end_chunk) {
        for (std::int64_t chunk = first_chunk; chunk < end_chunk; ++chunk) {
            const std::int64_t begin = chunk * kRowsPerChunk;
            const std::int64_t end = std::min(row_count, begin + kRowsPerChunk);
            double subtotal = 0.0;
            for (std::int64_t row = begin; row < end; ++row) {
                const Record& record = dataset.values[static_cast<std::size_t>(row)];
                std::array<double, kResiduals> residuals{};
                std::array<double, kResiduals> derivatives_wrt_output{};
                objective.evaluate(record, network.forward(record, beta_max),
                    residuals.data(), derivatives_wrt_output.data());
                for (const double residual : residuals) subtotal += residual * residual;
            }
            partial_sums[static_cast<std::size_t>(chunk)] = subtotal;
        }
    });
    double total = 0.0;
    for (const double partial : partial_sums) total += partial;
    // Divide by samples, rather than residual components, so a stacked term
    // with scale sqrt(lambda) represents exactly lambda * mean(R^2).
    return total / static_cast<double>(dataset.values.size());
}

template <class Objective>
torch::Tensor normal_equations(const Dataset& data, const Network& network,
    const Objective& objective, const Options& options, torch::Tensor& gradient, Timings& timings) {
    constexpr std::size_t kResiduals = Objective::kResidualsPerSample;
    const torch::Device accumulation_device = options.compute_device == ComputeDevice::Cuda
        ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    auto host_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    const auto accumulation_dtype = options.compute_device == ComputeDevice::Cuda && options.cuda_precision == CudaPrecision::Float32
        ? torch::kFloat32 : torch::kFloat64;
    auto accumulation_options = torch::TensorOptions().dtype(accumulation_dtype).device(accumulation_device);
    torch::Tensor hessian = torch::zeros({kParameters, kParameters}, accumulation_options);
    gradient = torch::zeros({kParameters, 1}, accumulation_options);
    torch::Tensor jacobian_storage = torch::empty({options.block_size * static_cast<int>(kResiduals), kParameters}, host_options);
    torch::Tensor residual_storage = torch::empty({options.block_size * static_cast<int>(kResiduals), 1}, host_options);
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t first = 0; first < data.values.size(); first += options.block_size) {
        const int count = static_cast<int>(std::min<std::size_t>(options.block_size, data.values.size() - first));
        const int residual_count = count * static_cast<int>(kResiduals);
        auto jacobian = jacobian_storage.narrow(0, 0, residual_count);
        auto residuals = residual_storage.narrow(0, 0, residual_count);
        double* j = jacobian.data_ptr<double>();
        double* r = residuals.data_ptr<double>();
        // Each row writes to an exclusive, cache-contiguous Jacobian/residual
        // region. The immutable network weights make this safe to parallelize.
        // The reductions are serial in program order, but each matmul dispatches
        // to LibTorch's configured CPU kernel threads.
        const auto jacobian_started = std::chrono::steady_clock::now();
        at::parallel_for(0, count, 32, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t row = begin; row < end; ++row) {
                const Record& record = data.values[first + static_cast<std::size_t>(row)];
                double* sample_jacobian = j + static_cast<std::size_t>(row) * kResiduals * kParameters;
                const double output = network.forward(record, options.beta_max, sample_jacobian);
                std::array<double, kResiduals> sample_residuals{};
                std::array<double, kResiduals> derivatives_wrt_output{};
                objective.evaluate(record, output, sample_residuals.data(), derivatives_wrt_output.data());
                // Replicate the unscaled output Jacobian before any term
                // applies its own chain factor. This matters as soon as two
                // residual terms have different dr/du values.
                for (std::size_t residual_index = 1; residual_index < kResiduals; ++residual_index) {
                    std::copy(sample_jacobian, sample_jacobian + kParameters,
                        sample_jacobian + residual_index * kParameters);
                }
                for (std::size_t residual_index = 0; residual_index < kResiduals; ++residual_index) {
                    double* term_jacobian = sample_jacobian + residual_index * kParameters;
                    const double chain_factor = derivatives_wrt_output[residual_index];
                    if (chain_factor != 1.0) {
                        for (int parameter = 0; parameter < kParameters; ++parameter) {
                            term_jacobian[parameter] *= chain_factor;
                        }
                    }
                    r[static_cast<std::size_t>(row) * kResiduals + residual_index] = sample_residuals[residual_index];
                }
            }
        });
        timings.jacobian_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - jacobian_started).count();
        if (options.profile_mode == ProfileMode::Jacobian) continue;
        const auto normal_matrix_started = std::chrono::steady_clock::now();
        if (options.compute_device == ComputeDevice::Cuda) {
            // A 32k-row block is ~341 MiB for J.  Keeping it blockwise avoids
            // relying on the laptop GPU's 6 GiB memory while preserving the
            // exact float64 normal equations and CPU model representation.
            const auto conversion_started = std::chrono::steady_clock::now();
            const auto jacobian_host = accumulation_dtype == torch::kFloat64 ? jacobian : jacobian.to(accumulation_dtype);
            const auto residuals_host = accumulation_dtype == torch::kFloat64 ? residuals : residuals.to(accumulation_dtype);
            timings.host_convert_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - conversion_started).count();
            const auto transfer_started = std::chrono::steady_clock::now();
            const auto jacobian_device = jacobian_host.to(accumulation_device);
            const auto residuals_device = residuals_host.to(accumulation_device);
            torch::cuda::synchronize();
            timings.host_to_device_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - transfer_started).count();
            timings.host_to_device_bytes += static_cast<std::uint64_t>(jacobian_host.nbytes() + residuals_host.nbytes());
            const auto gemm_started = std::chrono::steady_clock::now();
            hessian.add_(jacobian_device.transpose(0, 1).matmul(jacobian_device));
            gradient.add_(jacobian_device.transpose(0, 1).matmul(residuals_device));
            torch::cuda::synchronize();
            timings.cuda_gemm_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - gemm_started).count();
        } else {
            if (options.normal_equation_kernel == NormalEquationKernel::MklSyrk) {
#ifdef TWOPHASETRANSPORT_HAS_MKL
                // H += J^T J is symmetric. DSYRK forms only the lower
                // triangle, avoiding the redundant half of a generic GEMM.
                // J and H are contiguous row-major float64 tensors.
                cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, kParameters, residual_count,
                    1.0, j, kParameters, 1.0, hessian.data_ptr<double>(), kParameters);
#endif
            } else {
                hessian.add_(jacobian.transpose(0, 1).matmul(jacobian));
            }
            gradient.add_(jacobian.transpose(0, 1).matmul(residuals));
        }
        timings.normal_matrix_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - normal_matrix_started).count();
    }
    if (options.compute_device == ComputeDevice::Cpu && options.normal_equation_kernel == NormalEquationKernel::MklSyrk) {
        // LAPACK expects the full symmetric coefficient matrix. Mirror the
        // triangle once after all rank-k block updates, not per block.
        hessian.add_(hessian.tril(-1).transpose(0, 1));
    }
    timings.normal_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return hessian;
}

void append_trace(std::ofstream& trace, int iteration, int accepted_updates, const std::string& status,
    bool linearization_refreshed, double mean_objective_gradient_norm, double lambda,
    double objective_loss_value, bool include_objective_loss, double train_mse, double validation_mse,
    double best_validation_mse, const Timings& timings,
    const std::string& checkpoint, const std::string& best_checkpoint) {
    trace << iteration << ',' << accepted_updates << ',' << status << ',' << linearization_refreshed << ',';
    if (std::isfinite(mean_objective_gradient_norm)) trace << mean_objective_gradient_norm;
    trace << ',' << lambda;
    // Append only the columns present in the trace header. The optional
    // stacked-objective value differs from supervised MSE when a residual
    // penalty is used.
    if (include_objective_loss) trace << ',' << objective_loss_value;
    trace << ',' << train_mse << ',' << validation_mse << ','
          << best_validation_mse << ','
          << timings.normal_seconds << ',' << timings.jacobian_seconds << ',' << timings.normal_matrix_seconds << ','
          << timings.solve_seconds << ',' << timings.candidate_seconds << ',' << timings.candidate_train_seconds << ','
          << timings.candidate_validation_seconds << ',' << checkpoint << ',' << best_checkpoint << '\n';
}

void write_best_validation_snapshot(const Network& network, const std::filesystem::path& run_dir,
    int iteration, int accepted_updates, int generation, double validation_mse) {
    // Unlike recovery slots, these hold the best *held-out* model.  Two slots
    // make replacement safe without retaining an archive of every improvement.
    const int slot = generation % 2;
    const std::string checkpoint_name = "best_validation_checkpoint_" + std::to_string(slot) + ".bin";
    const std::string state_name = "best_validation_state_" + std::to_string(slot) + ".json";
    network.write_checkpoint(run_dir / checkpoint_name);
    std::ofstream state(run_dir / state_name);
    if (!state) throw std::runtime_error("Unable to write best-validation state: " + state_name);
    state << std::setprecision(17)
          << "{\n"
          << "  \"checkpoint\": \"" << checkpoint_name << "\",\n"
          << "  \"iteration\": " << iteration << ",\n"
          << "  \"accepted_updates\": " << accepted_updates << ",\n"
          << "  \"generation\": " << generation << ",\n"
          << "  \"validation_mse_normalized\": " << validation_mse << "\n"
          << "}\n";
}

void write_recovery_snapshot(const Network& network, const std::filesystem::path& run_dir,
    int iteration, int accepted_updates, double lambda, double train_mse, double validation_mse) {
    // Write the model first, then its slot-specific state record.  The previous
    // slot remains a complete recovery point until both writes have succeeded.
    const int slot = accepted_updates % 2;
    const std::string checkpoint_name = "resume_checkpoint_" + std::to_string(slot) + ".bin";
    const std::string state_name = "resume_state_" + std::to_string(slot) + ".json";
    network.write_checkpoint(run_dir / checkpoint_name);
    std::ofstream state(run_dir / state_name);
    if (!state) throw std::runtime_error("Unable to write recovery state: " + state_name);
    state << std::setprecision(17)
          << "{\n"
          << "  \"checkpoint\": \"" << checkpoint_name << "\",\n"
          << "  \"iteration\": " << iteration << ",\n"
          << "  \"accepted_updates\": " << accepted_updates << ",\n"
          << "  \"lambda\": " << lambda << ",\n"
          << "  \"train_mse_normalized\": " << train_mse << ",\n"
          << "  \"validation_mse_normalized\": " << validation_mse << "\n"
          << "}\n";
}

void check_gradient(Network& network, const Record& record, double beta_max) {
    std::vector<double> analytic(kParameters);
    network.forward(record, beta_max, analytic.data());
    constexpr double perturbation = 1.0e-6;
    double max_relative_error = 0.0;
    for (int index = 0; index < kParameters; index += 29) {
        network.add_to_parameter(index, perturbation);
        const double plus = network.forward(record, beta_max);
        network.add_to_parameter(index, -2.0 * perturbation);
        const double minus = network.forward(record, beta_max);
        network.add_to_parameter(index, perturbation);
        const double finite_difference = (plus - minus) / (2.0 * perturbation);
        const double scale = std::max({1.0, std::abs(finite_difference), std::abs(analytic[index])});
        max_relative_error = std::max(max_relative_error, std::abs(finite_difference - analytic[index]) / scale);
    }
    std::cout << "gradient_check_max_relative_error=" << max_relative_error << '\n';
    if (max_relative_error > 1.0e-6) throw std::runtime_error("Manual network Jacobian failed the finite-difference check");
}

void check_recurrence_term_gradient(const Record& record, double cap, double exponent) {
    // Check the physics term independently of the network Jacobian.  The
    // chosen normalized output maps strictly inside the physical interval, so
    // this tests the analytic fractional-flow derivative rather than the
    // deliberately clipped extension used outside [0, 1].
    BoundedRecurrenceTerm term{twophasetransport::training::RegularAdvectionRecurrenceTerm{exponent}, cap};
    constexpr double normalized_prediction = 0.25;
    constexpr double perturbation = 1.0e-7;
    double residual = 0.0;
    double analytic_derivative = 0.0;
    term.evaluate(record, normalized_prediction, &residual, &analytic_derivative);
    double plus = 0.0;
    double unused_derivative = 0.0;
    term.evaluate(record, normalized_prediction + perturbation, &plus, &unused_derivative);
    double minus = 0.0;
    term.evaluate(record, normalized_prediction - perturbation, &minus, &unused_derivative);
    const double finite_difference = (plus - minus) / (2.0 * perturbation);
    const double relative_error = std::abs(finite_difference - analytic_derivative) /
        std::max({1.0, std::abs(finite_difference), std::abs(analytic_derivative)});
    std::cout << "recurrence_term_gradient_check_relative_error=" << relative_error << '\n';
    if (relative_error > 1.0e-6) {
        throw std::runtime_error("Regular recurrence term derivative failed the finite-difference check");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.threads > 0) at::set_num_threads(options.threads);
#ifdef TWOPHASETRANSPORT_HAS_OPENMP
        // at::parallel_for is compiled in this translation unit and therefore
        // uses this OpenMP runtime.  Configure it explicitly rather than
        // assuming LibTorch's separately built runtime propagates the setting.
        if (options.threads > 0) {
            omp_set_dynamic(0);
            omp_set_num_threads(options.threads);
        }
#endif
#ifdef TWOPHASETRANSPORT_HAS_MKL
        if (options.threads > 0) mkl_set_num_threads_local(options.threads);
#endif
        if (options.compute_device == ComputeDevice::Cuda && !torch::cuda::is_available()) {
            throw std::runtime_error("--device cuda was requested, but this LibTorch build cannot access a CUDA device");
        }
        const Dataset train = load_dataset(options.train_path);
        const Dataset validation = load_dataset(options.validation_path);
        const auto recovered = options.resume_state.empty() ? std::optional<native_lm::Recovery>{}
            : std::make_optional(native_lm::read_recovery(options.resume_state, kParameters));
        const std::vector<Dataset> batches = load_batches(options.batch_directory, train.values.size());
        std::filesystem::create_directories(options.run_dir);
        const auto manifest_path = options.initial_checkpoint.empty()
            ? options.run_dir / "run_manifest.json"
            : options.run_dir / ("continuation_manifest_" + std::to_string(options.iteration_offset) + ".json");
        write_run_manifest(manifest_path, options, train, validation, batches.size());
        const auto trace_path = options.run_dir / "optimizer_trace.csv";
        const bool trace_exists = std::filesystem::exists(trace_path) && std::filesystem::file_size(trace_path) > 0;
        bool trace_includes_objective_loss = true;
        if (trace_exists) {
            std::ifstream existing_trace(trace_path);
            std::string trace_header;
            std::getline(existing_trace, trace_header);
            trace_includes_objective_loss = trace_header.find(",objective_loss,") != std::string::npos;
        }
        std::ofstream trace(trace_path, std::ios::app);
        if (!trace) throw std::runtime_error("Unable to open optimizer trace: " + trace_path.string());
        if (!trace_exists) {
            trace << "iteration,accepted_updates,status,linearization_refreshed,mean_objective_gradient_norm,lambda,objective_loss,train_mse_normalized,validation_mse_normalized,best_validation_mse_normalized,normal_seconds,jacobian_seconds,normal_matrix_seconds,solve_seconds,candidate_seconds,candidate_train_seconds,candidate_validation_seconds,checkpoint,best_checkpoint\n";
        }
        const auto batch_trace_path = options.run_dir / "batch_trace.csv";
        const bool batch_trace_exists = std::filesystem::exists(batch_trace_path) && std::filesystem::file_size(batch_trace_path) > 0;
        std::ofstream batch_trace;
        if (!batches.empty()) {
            batch_trace.open(batch_trace_path, std::ios::app);
            if (!batch_trace) throw std::runtime_error("Unable to open batch trace: " + batch_trace_path.string());
            if (!batch_trace_exists) {
                batch_trace << "iteration,accepted_updates,status,batch_index,batch_rows,batch_objective_loss\n";
            }
        }
        Network network(options.seed, options.initialization, options.beta_coordinate);
        const native_lm::SupervisedMseTerm<NormalizedRootTarget> supervised_term{NormalizedRootTarget{}};
        const ObjectiveVariant objective = options.objective_mode == ObjectiveMode::MsePlusRecurrence
            ? ObjectiveVariant{RecurrenceTrainingObjective{
                supervised_term,
                native_lm::ScaledResidualTerm<BoundedRecurrenceTerm>{
                    BoundedRecurrenceTerm{twophasetransport::training::RegularAdvectionRecurrenceTerm{options.nonwetting_exponent},
                        options.recurrence_cap}, options.recurrence_weight}}}
            : ObjectiveVariant{TrainingObjective{supervised_term}};
        const bool objective_is_supervised_mse = options.objective_mode == ObjectiveMode::SupervisedMse;
        const auto evaluate_objective_loss = [&](const Dataset& dataset, const Network& active_network) {
            return std::visit([&](const auto& active_objective) {
                return objective_loss(dataset, active_network, options.beta_max, active_objective);
            }, objective);
        };
        const auto build_normal_equations = [&](const Dataset& dataset, const Network& active_network,
                                                torch::Tensor& gradient, Timings& timings) {
            return std::visit([&](const auto& active_objective) {
                return normal_equations(dataset, active_network, active_objective, options, gradient, timings);
            }, objective);
        };
        if (!options.initial_checkpoint.empty()) network.load_checkpoint(options.initial_checkpoint);
        if (recovered) network.restore_weights(recovered->current);
        if (options.gradient_check) {
            check_gradient(network, train.values.front(), options.beta_max);
            if (!objective_is_supervised_mse) check_recurrence_term_gradient(train.values.front(), options.recurrence_cap, options.nonwetting_exponent);
        }
        if (options.profile_mode == ProfileMode::Jacobian) {
            Timings timings;
            torch::Tensor gradient;
            build_normal_equations(train, network, gradient, timings);
            std::cout << "profile_mode=jacobian rows=" << train.values.size()
                      << " threads=" << options.threads
                      << " block_size=" << options.block_size
                      << " jacobian_seconds=" << timings.jacobian_seconds
                      << " total_seconds=" << timings.normal_seconds << '\n';
            return 0;
        }
        if (options.profile_mode == ProfileMode::Candidate) {
            const auto train_started = std::chrono::steady_clock::now();
            const double train_profile_mse = network.mse(train, options.beta_max);
            const double train_profile_objective = evaluate_objective_loss(train, network);
            const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - train_started).count();
            const auto validation_started = std::chrono::steady_clock::now();
            const double validation_profile_mse = network.mse(validation, options.beta_max);
            const double validation_profile_objective = evaluate_objective_loss(validation, network);
            const double validation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - validation_started).count();
            std::cout << "profile_mode=candidate train_rows=" << train.values.size()
                      << " validation_rows=" << validation.values.size()
                      << " threads=" << options.threads
                      << " train_mse=" << train_profile_mse
                      << " train_objective_loss=" << train_profile_objective
                      << " train_objective_minus_mse=" << (train_profile_objective - train_profile_mse)
                      << " validation_mse=" << validation_profile_mse
                      << " validation_objective_loss=" << validation_profile_objective
                      << " validation_objective_minus_mse=" << (validation_profile_objective - validation_profile_mse)
                      << " train_seconds=" << train_seconds
                      << " validation_seconds=" << validation_seconds << '\n';
            return 0;
        }
        double lambda = options.lambda;
        double train_mse = network.mse(train, options.beta_max);
        double validation_mse = network.mse(validation, options.beta_max);
        double best_validation_mse = validation_mse;
        int accepted_updates = options.accepted_update_offset;
        std::size_t batch_index = batches.empty() ? 0 :
            static_cast<std::size_t>(accepted_updates) % batches.size();
        double batch_objective_loss = batches.empty()
            ? evaluate_objective_loss(train, network)
            : evaluate_objective_loss(batches[batch_index], network);
        Network best_network = network;
        double cumulative_phase_seconds = recovered ? recovered->phase_seconds : 0;
        if(recovered) {
            lambda=recovered->state.damping; train_mse=recovered->state.training_mse;
            validation_mse=recovered->state.validation_mse; best_validation_mse=recovered->state.best_validation_mse;
            batch_objective_loss=recovered->state.objective_loss; accepted_updates=recovered->state.accepted_updates;
            best_network.restore_weights(recovered->best);
        }
        int best_validation_generation = 0;
        const bool resumed_best_validation = !options.initial_best_validation_checkpoint.empty();
        if (resumed_best_validation) {
            best_network.load_checkpoint(options.initial_best_validation_checkpoint);
            best_validation_mse = options.initial_best_validation_mse;
            best_validation_generation = options.initial_best_validation_generation;
        }
        const std::string initial_checkpoint_name = "checkpoint_000000.bin";
        if (!std::filesystem::exists(options.run_dir / initial_checkpoint_name)) {
            network.write_checkpoint(options.run_dir / initial_checkpoint_name);
        }
        write_recovery_snapshot(network, options.run_dir, 0, accepted_updates, lambda, train_mse, validation_mse);
        if (!resumed_best_validation) {
            write_best_validation_snapshot(best_network, options.run_dir, 0, accepted_updates,
                best_validation_generation, best_validation_mse);
        }
        std::vector<bool> checkpoint_threshold_reached(options.checkpoint_mse_thresholds.size(), false);
        for (std::size_t index = 0; index < options.checkpoint_mse_thresholds.size(); ++index) {
            checkpoint_threshold_reached[index] = (recovered ? best_validation_mse : validation_mse) <= options.checkpoint_mse_thresholds[index];
        }
        if (!recovered && (options.initial_checkpoint.empty() || options.portable_recovery)) {
            append_trace(trace, 0, accepted_updates, "initial", false, std::numeric_limits<double>::quiet_NaN(),
                lambda, batch_objective_loss, trace_includes_objective_loss, train_mse, validation_mse,
                best_validation_mse, {}, initial_checkpoint_name,
                resumed_best_validation ? "" : "best_validation_checkpoint_0.bin");
        }

        native_lm::LmControl lm_control;
        lm_control.damping_decrease = options.lambda_decrease;
        lm_control.damping_increase = options.lambda_increase;
        lm_control.damping_max = options.lambda_max;
        lm_control.diagonal_damping = options.diagonal_damping;
        lm_control.synchronize_cuda_after_solve = options.compute_device == ComputeDevice::Cuda;
        lm_control.max_iterations = recovered ? std::max(0, options.max_iterations - recovered->state.iteration) : options.max_iterations;
        lm_control.max_accepted_updates = options.max_accepted_updates;
        lm_control.max_validation_failures = options.max_validation_fail;

        native_lm::LmState initial_lm_state;
        initial_lm_state.iteration = options.iteration_offset;
        initial_lm_state.accepted_updates = accepted_updates;
        initial_lm_state.damping = lambda;
        initial_lm_state.objective_loss = batch_objective_loss;
        initial_lm_state.training_mse = train_mse;
        initial_lm_state.validation_mse = validation_mse;
        initial_lm_state.best_validation_mse = best_validation_mse;
        if(recovered) initial_lm_state=recovered->state;
        trace.flush();
        if(options.portable_recovery) native_lm::write_recovery(
            options.run_dir / ("optimizer_state_" + std::to_string(initial_lm_state.iteration % 2) + ".bin"),
            network.weight_data(), best_network.weight_data(), kParameters, initial_lm_state, cumulative_phase_seconds);

        const Dataset* linearization_data = nullptr;
        auto lm_result = native_lm::optimize(
            std::move(network), lm_control, initial_lm_state,
            [&](const Network& active_network) {
                linearization_data = batches.empty() ? &train : &batches[batch_index];
                if (!batches.empty()) {
                    batch_objective_loss = evaluate_objective_loss(*linearization_data, active_network);
                }
                Timings timings;
                torch::Tensor gradient;
                torch::Tensor hessian = build_normal_equations(*linearization_data, active_network,
                    gradient, timings);
                return native_lm::LmLinearization{std::move(hessian), std::move(gradient),
                    linearization_data->values.size(), timings};
            },
            [&](const Network& active_network, torch::Tensor step) {
                if (options.compute_device == ComputeDevice::Cuda) {
                    step = step.to(torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
                }
                Network candidate = active_network;
                candidate.add_step(step);
                return candidate;
            },
            [&](const Network& candidate) {
                const auto candidate_started = std::chrono::steady_clock::now();
                native_lm::CandidateEvaluation evaluation;
                evaluation.objective_loss = evaluate_objective_loss(*linearization_data, candidate);
                evaluation.timings.candidate_train_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - candidate_started).count();
                if (std::isfinite(evaluation.objective_loss) &&
                    evaluation.objective_loss < batch_objective_loss) {
                    // In stochastic LM a proposal is accepted on its fixed
                    // local batch. Global supervised training and validation
                    // MSE remain the model-selection safeguards.
                    if (objective_is_supervised_mse) {
                        evaluation.training_mse = batches.empty() ? evaluation.objective_loss :
                            candidate.mse(train, options.beta_max);
                    } else {
                        evaluation.training_mse = candidate.mse(train, options.beta_max);
                    }
                    evaluation.timings.candidate_train_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - candidate_started).count();
                    const auto validation_started = std::chrono::steady_clock::now();
                    evaluation.validation_mse = candidate.mse(validation, options.beta_max);
                    evaluation.timings.candidate_validation_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - validation_started).count();
                }
                evaluation.timings.candidate_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - candidate_started).count();
                return evaluation;
            },
            [&](const Network& active_network, const native_lm::LmIteration& event) {
                const bool accepted = event.status == native_lm::LmIteration::Status::Accepted;
                const auto& state = event.state;
                std::string checkpoint_name;
                std::string best_checkpoint_name;
                if (accepted) {
                    train_mse = state.training_mse;
                    validation_mse = state.validation_mse;
                    batch_objective_loss = state.objective_loss;
                    for (std::size_t index = 0; index < options.checkpoint_mse_thresholds.size(); ++index) {
                        if (!checkpoint_threshold_reached[index] &&
                            validation_mse <= options.checkpoint_mse_thresholds[index]) {
                            checkpoint_threshold_reached[index] = true;
                            checkpoint_name = "checkpoint_" + std::to_string(state.iteration) + ".bin";
                        }
                    }
                    if (!checkpoint_name.empty()) active_network.write_checkpoint(options.run_dir / checkpoint_name);
                    if (options.recovery_checkpoint_interval > 0 &&
                        state.accepted_updates % options.recovery_checkpoint_interval == 0) {
                        write_recovery_snapshot(active_network, options.run_dir, state.iteration,
                            state.accepted_updates, state.damping, state.training_mse, state.validation_mse);
                    }
                    if (event.validation_improved) {
                        best_network = active_network;
                        ++best_validation_generation;
                        write_best_validation_snapshot(best_network, options.run_dir, state.iteration,
                            state.accepted_updates, best_validation_generation, state.best_validation_mse);
                        best_checkpoint_name = "best_validation_checkpoint_" +
                            std::to_string(best_validation_generation % 2) + ".bin";
                    }
                }
                append_trace(trace, state.iteration, state.accepted_updates,
                    accepted ? "accepted" : "rejected", event.linearization_refreshed,
                    state.mean_objective_gradient_norm, state.damping, state.objective_loss,
                    trace_includes_objective_loss, state.training_mse,
                    state.validation_mse, state.best_validation_mse, event.timings,
                    checkpoint_name, best_checkpoint_name);
                if (batch_trace) {
                    batch_trace << state.iteration << ',' << state.accepted_updates << ','
                                << (accepted ? "accepted" : "rejected") << ',' << batch_index << ','
                                << linearization_data->values.size() << ',' << batch_objective_loss << '\n';
                }
                if (accepted && !batches.empty()) batch_index = (batch_index + 1) % batches.size();
                trace.flush();
                cumulative_phase_seconds += event.timings.normal_seconds + event.timings.solve_seconds + event.timings.candidate_seconds;
                if(options.portable_recovery) native_lm::write_recovery(
                    options.run_dir / ("optimizer_state_" + std::to_string(state.iteration % 2) + ".bin"),
                    active_network.weight_data(), best_network.weight_data(), kParameters, state, cumulative_phase_seconds);
                if (batch_trace) batch_trace.flush();
                std::cout << "iteration=" << state.iteration << " train_mse=" << state.training_mse
                          << " validation_mse=" << state.validation_mse
                          << " lambda=" << state.damping
                          << " mean_objective_gradient_norm=" << state.mean_objective_gradient_norm
                          << " normal_seconds=" << event.timings.normal_seconds;
                if (options.compute_device == ComputeDevice::Cuda) {
                    const double gib_per_second = event.timings.host_to_device_seconds > 0.0
                        ? static_cast<double>(event.timings.host_to_device_bytes) /
                            (1024.0 * 1024.0 * 1024.0) / event.timings.host_to_device_seconds : 0.0;
                    std::cout << " host_convert_seconds=" << event.timings.host_convert_seconds
                              << " host_to_device_seconds=" << event.timings.host_to_device_seconds
                              << " host_to_device_gib=" << static_cast<double>(event.timings.host_to_device_bytes) /
                                    (1024.0 * 1024.0 * 1024.0)
                              << " host_to_device_gib_per_second=" << gib_per_second
                              << " cuda_gemm_seconds=" << event.timings.cuda_gemm_seconds;
                }
                std::cout << '\n' << std::flush;
            });
        network = std::move(lm_result.model);
        best_validation_mse = lm_result.state.best_validation_mse;
        if (lm_result.stopping_reason != "max_iterations") {
            std::cout << "stopping_reason=" << lm_result.stopping_reason << '\n';
        }
        const auto final_model = options.run_dir / "final_checkpoint.bin";
        best_network.write_checkpoint(final_model);
        std::cout << "final_checkpoint=" << final_model.string() << '\n';
        std::cout << "best_validation_mse=" << best_validation_mse << '\n';
        if (!options.export_package.empty()) {
            best_network.write_model_package(options.export_package, options.beta_max);
            std::cout << "model_package=" << options.export_package.string() << '\n';
        }
    } catch (const c10::Error& error) {
        std::cerr << "Torch error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "twophasetransport_lm_trainer error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
