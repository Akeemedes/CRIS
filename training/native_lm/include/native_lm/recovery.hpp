#pragma once

#include "lm_optimizer.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace native_lm {

// Opt-in, physics-independent recovery envelope. Physics, data and executable
// identity are checked by the launch manifest; this file protects optimizer
// state and both weight vectors against partial writes and accidental damage.
struct Recovery {
    LmState state;
    double phase_seconds{0};
    std::vector<double> current, best;
};

inline std::uint64_t recovery_checksum(const char* data, std::size_t n) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i=0; i<n; ++i) {
        hash ^= static_cast<unsigned char>(data[i]); hash *= 1099511628211ULL;
    }
    return hash; // FNV-1a detects damage; it is not an authenticity signature.
}

inline void write_recovery(const std::filesystem::path& path, const double* current,
                           const double* best, std::size_t parameters,
                           const LmState& s, double seconds) {
    std::vector<char> bytes(96 + 16 * parameters);
    std::memcpy(bytes.data(), "NLMR0001", 8);
    const std::uint64_t count=parameters;
    std::memcpy(bytes.data()+8, &count, 8);
    const double values[]{double(s.iteration), double(s.accepted_updates),
        double(s.validation_failures), s.damping, s.objective_loss,
        s.training_mse, s.validation_mse, s.best_validation_mse,
        s.mean_objective_gradient_norm, seconds};
    std::memcpy(bytes.data()+16, values, sizeof(values));
    std::memcpy(bytes.data()+96, current, parameters*8);
    std::memcpy(bytes.data()+96+parameters*8, best, parameters*8);
    const auto checksum=recovery_checksum(bytes.data(), bytes.size());
    auto temporary=path; temporary += ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary);
        out.write(bytes.data(), bytes.size());
        out.write(reinterpret_cast<const char*>(&checksum), 8);
        out.flush();
        if (!out) throw std::runtime_error("Recovery write failed");
    }
    // Only replace this slot; the other committed proposal remains recoverable.
    std::filesystem::remove(path);
    std::filesystem::rename(temporary, path);
}

inline Recovery read_recovery(const std::filesystem::path& path, std::size_t parameters) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    const auto size=104+16*parameters;
    if (!in || in.tellg()!=std::streamoff(size)) throw std::runtime_error("Recovery length mismatch");
    std::vector<char> bytes(size); in.seekg(0); in.read(bytes.data(), size);
    std::uint64_t count=0, checksum=0;
    std::memcpy(&count,bytes.data()+8,8); std::memcpy(&checksum,bytes.data()+size-8,8);
    if (!in || std::memcmp(bytes.data(),"NLMR0001",8) || count!=parameters ||
        checksum!=recovery_checksum(bytes.data(),size-8)) throw std::runtime_error("Invalid recovery envelope/checksum");
    double v[10]; std::memcpy(v,bytes.data()+16,sizeof(v));
    for(double x:v) if(!std::isfinite(x)||x<0) throw std::runtime_error("Invalid recovery state");
    for(int i=0;i<3;++i) if(v[i]!=std::floor(v[i])||v[i]>std::numeric_limits<int>::max())
        throw std::runtime_error("Invalid recovery counters");
    if(v[1]>v[0]||v[2]>v[1]||v[3]<=0) throw std::runtime_error("Inconsistent recovery state");
    Recovery r;
    r.state.iteration=int(v[0]); r.state.accepted_updates=int(v[1]); r.state.validation_failures=int(v[2]);
    r.state.damping=v[3]; r.state.objective_loss=v[4]; r.state.training_mse=v[5];
    r.state.validation_mse=v[6]; r.state.best_validation_mse=v[7];
    r.state.mean_objective_gradient_norm=v[8]; r.phase_seconds=v[9];
    r.current.resize(parameters); r.best.resize(parameters);
    std::memcpy(r.current.data(),bytes.data()+96,parameters*8);
    std::memcpy(r.best.data(),bytes.data()+96+parameters*8,parameters*8);
    for(const auto* weights:{&r.current,&r.best})
        for(double x:*weights) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite recovery weights");
    return r;
}
} // namespace native_lm
