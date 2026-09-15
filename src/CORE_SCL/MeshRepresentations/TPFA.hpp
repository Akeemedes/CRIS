#ifndef __TPFA_HPP_
#define __TPFA_HPP_

#include <cmath>

// ============================== K policies ===================================

template<class T = double>
struct HomogeneousAnisotropicK {

    HomogeneousAnisotropicK() = default;
    explicit HomogeneousAnisotropicK(std::array<T, 3> k) : K(k) {}

    T operator()(uint8_t dir, std::size_t) const { return K[dir]; }
    T operator () (uint8_t dir) const { return K[dir]; }

private:
    std::array<T, 3> K{ {T(1),T(1),T(1)} };
};

template<class T = double>
struct HeterogeneousAnisotropicK {

    HeterogeneousAnisotropicK() = default;

    HeterogeneousAnisotropicK(std::vector<T> Kx, std::vector<T> Ky, std::vector<T> Kz)
        : Kdir{ std::move(Kx), std::move(Ky), std::move(Kz) }
    {
        assert(Kdir[0].size() == Kdir[1].size());
        assert(Kdir[1].size() == Kdir[2].size());
    }

    T operator()(int dir, std::size_t cell) const { return Kdir[dir][cell]; }

private:
    std::array<std::vector<T>, 3> Kdir;
};

// ============================== TPFA policies =================================
// Policy concept
//   interior(A, h0, K0, h1, K1) -> transmissibility magnitude
//   boundary(A, h, Kc)          -> transmissibility magnitude (sign handled by Conn)

struct HarmonicPolicy {
    static double interior(double A, double h0, double K0, double h1, double K1) {
        return A / (h0 / K0 + h1 / K1);
    }
    static double boundary(double A, double h, double Kc) {
        return A * Kc / h;
    }
};

struct ArithmeticPolicy {
    static double interior(double A, double h0, double K0, double h1, double K1) {
        // simple average permeability, total distance (h0+h1)
        const double Ka = 0.5 * (K0 + K1);
        return A * Ka / (h0 + h1);
    }
    static double boundary(double A, double h, double Kc) {
        return A * Kc / h;
    }
};

struct GeometricPolicy {
    static double interior(double A, double h0, double K0, double h1, double K1) {
        const double Kg = std::sqrt(K0 * K1);
        return A * Kg / (h0 + h1);
    }
    static double boundary(double A, double h, double Kc) {
        return A * Kc / h;
    }
};

// ============================== TPFA wrapper =================================
//
// Precomputes and stores coefficients once.
// Automatically compresses storage when:
//   Geom == UniformCartesianGeometry && KPolicy == HomogeneousAnisotropicK<T>

template<class AveragingPolicy, class Conn, class Geom, class KPolicy, class T = double>
class TPFA {
public:
    using value_t = T;
    using index_t = typename Conn::index_t;

    TPFA() = default;
    TPFA(const Conn& conn, const Geom& geom, const KPolicy& K) {
        build(conn, geom, K);
    }
    void build(const Conn& conn, const Geom& geom, const KPolicy& K) {


        if constexpr (compress_) {
            for (int dir = 0; dir < 3; ++dir) {
                const double h = geom.half_distance(dir);
                const double Kd = K(dir);
                const double A = geom.interior_face_area(dir);
                Tintr[dir] = static_cast<T>(AveragingPolicy::interior(A, h, Kd, h, Kd));
                Tbndr_mag[dir] = static_cast<T>(AveragingPolicy::boundary(A, h, Kd));
            }
        }
        else {
            intr_T.assign(conn.num_intr_faces(), T{ 0 });
            bndr_T.assign(conn.num_bndr_faces(), T{ 0 });

            for (index_t f = 0; f < conn.num_intr_faces(); ++f) {
                auto [c0, c1] = conn.interior_face_to_cells(f);
                const int dir = conn.interior_face_direction(f);

                const double A = geom.interior_face_area(f, conn);
                const double h0 = geom.half_distance(c0, dir);
                const double h1 = geom.half_distance(c1, dir);
                const double K0 = K(dir, c0);
                const double K1 = K(dir, c1);

                intr_T[f] = static_cast<T>(AveragingPolicy::interior(A, h0, K0, h1, K1));
            }

            for (index_t b = 0; b < conn.num_bndr_faces(); ++b) {
                auto [c, sdir] = conn.boundary_face_to_cell(b);
                //const int sign = (sdir > 0) ? +1 : -1;
                const int dir = std::abs(sdir) - 1;

                const double A = geom.boundary_face_area(b, conn);
                const double h = geom.half_distance(c, dir);
                const double Kc = K(dir, c);

                bndr_T[b] = static_cast<T>(AveragingPolicy::boundary(A, h, Kc));
            }
        }
        built = true;
    }

    T intr_face_coef(index_t f, const Conn& conn) const {
        assert(built && "Coefficient storage must be built first");
        if constexpr (compress_) return Tintr[conn.interior_face_direction(f)];
        else return intr_T[f];
    }

    T bndr_face_coef(index_t b, const Conn& conn) const {
        if constexpr (compress_) {
            auto [c, sdir] = conn.boundary_face_to_cell(b);
            (void)c;
            const int sign = (sdir > 0) ? +1 : -1;
            const int dir = std::abs(sdir) - 1;
            return Tbndr_mag[dir];
        }
        else {
            return bndr_T[b];
        }
    }

private:
    static constexpr bool compress_ =
        Geom::is_uniform&&
        std::is_same<KPolicy, HomogeneousAnisotropicK<T>>::value;
    bool built = false;

    // compressed storage
    std::array<T, 3> Tintr{ {0,0,0} };
    std::array<T, 3> Tbndr_mag{ {0,0,0} };

    // full storage
    std::vector<T> intr_T;
    std::vector<T> bndr_T;
};

// Variant aliases
template<class Conn, class Geom, class KPolicy, class T = double>
using HarmonicTPFA = TPFA<HarmonicPolicy, Conn, Geom, KPolicy, T>;

template<class Conn, class Geom, class KPolicy, class T = double>
using ArithmeticTPFA = TPFA<ArithmeticPolicy, Conn, Geom, KPolicy, T>;

template<class Conn, class Geom, class KPolicy, class T = double>
using GeometricTPFA = TPFA<GeometricPolicy, Conn, Geom, KPolicy, T>;

template<class AveragingPolicy, class Conn, class Geom, class CoefPolicy, class T = double>
class TPFAOperator {
public:
    using index_t = typename Conn::index_t;
    using value_t = T;
    using TPFA_t = TPFA<AveragingPolicy, Conn, Geom, CoefPolicy, T>;

    TPFAOperator() = default;

    TPFAOperator(const Conn& conn, const Geom& geom, const CoefPolicy& coef)
    {
        tpfa.build(conn, geom, coef);
    }

    void build(const Conn& conn, const Geom& geom, const CoefPolicy& coef)
    {
        tpfa.build(conn, geom, coef);
    }

    value_t intr_face_coef(index_t f, const Conn& conn) const { return tpfa.intr_face_coef(f, conn); }
    value_t bndr_face_coef(index_t b, const Conn& conn) const { return tpfa.bndr_face_coef(b, conn); }

private:
    TPFA_t tpfa;
};


#endif