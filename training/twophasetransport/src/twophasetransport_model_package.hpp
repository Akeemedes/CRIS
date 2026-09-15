#pragma once

#include <torch/serialize.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace twophasetransport::model_package {

// Architecture: 3-20-20-20-20-1, with a bias on every affine layer.
inline constexpr std::size_t parameter_count = 1361;

inline std::vector<double> read_checkpoint(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Unable to read checkpoint: " + path.string());
    std::vector<double> parameters(parameter_count);
    stream.read(reinterpret_cast<char*>(parameters.data()),
        static_cast<std::streamsize>(parameters.size() * sizeof(double)));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Invalid checkpoint length: " + path.string());
    }
    return parameters;
}

inline void write(const std::filesystem::path& path, const std::vector<double>& parameters,
    double beta_max, bool beta_log1p = false) {
    if (parameters.size() != parameter_count) {
        throw std::runtime_error("Unexpected parameter count while writing model package");
    }
    if (!(beta_max > 0.0)) throw std::runtime_error("beta_max must be positive");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());

    const auto doubles = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    const auto integers = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::serialize::OutputArchive archive;
    archive.write("format_version", torch::tensor({beta_log1p ? 2 : 1}, integers), true);
    archive.write("parameter_layout_version", torch::tensor({1}, integers), true);
    archive.write("parameters", torch::from_blob(const_cast<double*>(parameters.data()),
        {static_cast<long long>(parameters.size())}, doubles).clone(), true);
    archive.write("input_min", torch::tensor({0.0, 0.0, 0.0}, doubles), true);
    archive.write("input_max", torch::tensor({1.0, beta_max, 1.0}, doubles), true);
    if (beta_log1p) {
        // Version 2 keeps physical input ranges while declaring the coordinate
        // applied before affine normalization: 0=identity, 1=log1p.
        archive.write("input_transform_codes", torch::tensor({0, 1, 0}, integers), true);
    }
    archive.write("output_min", torch::tensor({0.0}, doubles), true);
    archive.write("output_max", torch::tensor({1.0}, doubles), true);
    archive.write("layer_widths", torch::tensor({3, 20, 20, 20, 20, 1}, integers), true);
    archive.write("activation_codes", torch::tensor({2, 2, 2, 2, 3}, integers), true);
    archive.write("bias_flags", torch::tensor({1, 1, 1, 1, 1}, integers), true);
    archive.save_to(path.string());
}

}  // namespace twophasetransport::model_package
