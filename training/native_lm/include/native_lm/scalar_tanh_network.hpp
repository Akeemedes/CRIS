#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace native_lm {
// Fixed-width scalar MLP with native forward and parameter-Jacobian kernels.
// Input/output physical transforms belong to the physics adapter.
template<int Inputs, int Width = 20, int HiddenLayers = 4>
class ScalarTanhNetwork {
public:
    static constexpr int parameters = Width * (Inputs + 1)
        + (HiddenLayers - 1) * Width * (Width + 1) + Width + 1;
    std::vector<double> weights;
    explicit ScalarTanhNetwork(unsigned long long seed = 20260904) : weights(parameters) {
        std::mt19937_64 engine(seed);
        std::uniform_real_distribution<double> uniform(-1, 1);
        int offset = 0;
        for (int layer = 0; layer < HiddenLayers; ++layer) {
            const int cols = layer == 0 ? Inputs : Width;
            const double magnitude = 1.4 * std::pow(double(Width), 1.0 / cols);
            for (int row = 0; row < Width; ++row) {
                double norm = 0;
                for (int col = 0; col < cols; ++col) {
                    double& w = weights[offset + row * cols + col];
                    w = uniform(engine); norm += w * w;
                }
                for (int col = 0; col < cols; ++col) weights[offset + row * cols + col] *= magnitude / std::sqrt(norm);
                weights[offset + Width * cols + row] = magnitude * (-1.0 + 2.0 * row / (Width - 1))
                    * (weights[offset + row * cols] < 0 ? -1 : 1);
            }
            offset += Width * (cols + 1);
        }
        for (; offset < parameters; ++offset) weights[offset] = uniform(engine);
    }

    double forward(const std::array<double, Inputs>& input, double* gradient = nullptr) const {
        std::array<std::array<double, Width>, HiddenLayers> hidden{};
        std::array<int, HiddenLayers> offsets{};
        int offset = 0;
        for (int layer = 0; layer < HiddenLayers; ++layer) {
            offsets[layer] = offset;
            const int cols = layer == 0 ? Inputs : Width;
            const double* previous = layer == 0 ? input.data() : hidden[layer - 1].data();
            for (int row = 0; row < Width; ++row) {
                double value = weights[offset + Width * cols + row];
                for (int col = 0; col < cols; ++col) value += weights[offset + row * cols + col] * previous[col];
                hidden[layer][row] = std::tanh(value);
            }
            offset += Width * (cols + 1);
        }
        double output = weights[offset + Width];
        for (int col = 0; col < Width; ++col) output += weights[offset + col] * hidden.back()[col];
        if (!gradient) return output;
        std::array<double, Width> delta{};
        gradient[offset + Width] = 1;
        for (int col = 0; col < Width; ++col) {
            gradient[offset + col] = hidden.back()[col];
            delta[col] = weights[offset + col] * (1 - hidden.back()[col] * hidden.back()[col]);
        }
        for (int layer = HiddenLayers - 1; layer >= 0; --layer) {
            offset = offsets[layer];
            const int cols = layer == 0 ? Inputs : Width;
            const double* previous = layer == 0 ? input.data() : hidden[layer - 1].data();
            for (int row = 0; row < Width; ++row) {
                gradient[offset + Width * cols + row] = delta[row];
                for (int col = 0; col < cols; ++col) gradient[offset + row * cols + col] = delta[row] * previous[col];
            }
            if (layer > 0) {
                std::array<double, Width> next{};
                for (int col = 0; col < Width; ++col) {
                    for (int row = 0; row < Width; ++row) next[col] += weights[offset + row * cols + col] * delta[row];
                    next[col] *= 1 - previous[col] * previous[col];
                }
                delta = next;
            }
        }
        return output;
    }
};
} // namespace native_lm
