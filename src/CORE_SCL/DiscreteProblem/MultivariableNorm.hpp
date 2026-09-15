#ifndef __MULTIVARIABLENORM_HPP_
#define __MULTIVARIABLENORM_HPP_

#include <cstddef>
#include <cmath>
#include <limits>
#include <ostream>
#include <vector>
#include <algorithm>

// ---------------- Norm function policies ----------------

struct TwoNormFunct {
    template<class V>
    static double compute(const V& v, std::size_t offset, std::size_t stride) {
        long double sum = 0.0L;
        for (std::size_t i = offset; i < v.size(); i += stride) {
            const double x = static_cast<double>(v[i]);
            sum += static_cast<long double>(x) * static_cast<long double>(x);
        }
        return std::sqrt(static_cast<double>(sum));
    }
};

struct InfNormFunct {
    template<class V>
    static double compute(const V& v, std::size_t offset, std::size_t stride) {
        double m = 0.0;
        for (std::size_t i = offset; i < v.size(); i += stride) {
            const double x = std::abs(static_cast<double>(v[i]));
            if (x > m) m = x;
        }
        return m;
    }
};

// ---------------- MultivariableNorm ----------------
//
// Nvars = number of variables per cell (or per block).
// Evaluates one norm per variable component, using strided access:
//   component k uses indices: k, k+Nvars, k+2*Nvars, ...
template<std::size_t Nvars, class NormPolicy>
class MultivariableNorm {
public:
    MultivariableNorm() { values_.fill(0.0); }

    template<class V>
    void evaluate(const V& v) {
        static_assert(Nvars > 0, "Nvars must be > 0");
        for (std::size_t k = 0; k < Nvars; ++k)
            values_[k] = NormPolicy::compute(v, k, Nvars);
    }

    bool isLessEqual(double tol) const {
        for (double x : values_) {
            if (!(x <= tol)) return false; // handles NaN as false
        }
        return true;
    }

    double operator[](std::size_t k) const { return values_[k]; }

    static constexpr std::size_t nvars() { return Nvars; }

    // For your existing solver printouts
    std::size_t printout_width() const { return 16; }
    void print_preamble(std::ostream& os) const {
        for (std::size_t k = 0; k < Nvars; ++k) {
            os << " " << "nrm[" << k << "]";
        }
    }

private:
    std::array<double, Nvars> values_;
};

// Optional pretty-print for single variable (common case)
template<std::size_t Nvars, class NormPolicy>
inline std::ostream& operator<<(std::ostream& os, const MultivariableNorm<Nvars, NormPolicy>& n)
{
    os << " ";
    for (std::size_t k = 0; k < Nvars; ++k) {
        os << n[k];
        if (k + 1 < Nvars) os << " ";
    }
    return os;
}

#endif // __MULTIVARIABLENORM_HPP_