#ifndef __STRUCTUREDCONNECTIVITY_HPP__
#define __STRUCTUREDCONNECTIVITY_HPP__

#include <cstddef>
#include <array>
#include <utility>
#include <cassert>


/* ================================================================== */
/* StructuredConnectivity provides indexing functionality for         */
/* a structured (i,j,k) grid topology. Establishes the way that cells */
/* are referenced using a flattened index l and how internal and      */
/* boundary faces are stored and referenced. StructureConnectivity is */
/* parameterized by a Layout policy allowing static polymorphism.     */ 
/* A collection of structured (i,j,k) to ( l ) layout policies.       */
/* Corresponding face layout policies	                              */
/*  fi | (i,j,k|loc)                                                  */
/*        --------                                                    */
/*           fj                                                       */
/* ================================================================== */

template<typename T = std::size_t>
class IJKSequentialFaces {
public:
    using index_t = T;

    IJKSequentialFaces() = default;
    IJKSequentialFaces(const IJKSequentialFaces&) = default;
    IJKSequentialFaces(IJKSequentialFaces&&) = default;

    void set_dimensions(index_t nx, index_t ny, index_t nz,
        const std::array<index_t, 3>& _NIF,
        const std::array<index_t, 3>& _NBF)
    {
        NC = { nx, ny, nz };
        s = { index_t{1}, nx, nx * ny };
        NIF = _NIF;
        NBF = _NBF;
    }

    int interior_face_direction(index_t f) const {
        if (f < NIF[0]) return 0;
        f -= NIF[0];
        if (f < NIF[1]) return 1;
        return 2;
    }

    int boundary_face_direction(index_t f) const {
        if (f < NBF[0]) return 0;
        f -= NBF[0];
        if (f < NBF[0]) return 1;
        f -= NBF[0];
        if (f < NBF[1]) return 2;
        f -= NBF[1];
        if (f < NBF[1]) return 3;
        f -= NBF[1];
        if (f < NBF[2]) return 4;
        return 5;
    }


    std::pair<index_t, index_t> interior_face_to_cells(index_t f) const {
        const index_t nx = NC[0], ny = NC[1];

        if (f < NIF[0]) {
            const index_t i = (f % (nx - 1)) + 1;
            const index_t t = f / (nx - 1);
            const index_t j = t % ny;
            const index_t k = t / ny;
            const index_t right = cell(i, j, k);
            return { right - s[0], right };
        }

        f -= NIF[0];

        if (f < NIF[1]) {
            const index_t i = f % nx;
            const index_t t = f / nx;
            const index_t j = (t % (ny - 1)) + 1;
            const index_t k = t / (ny - 1);
            const index_t right = cell(i, j, k);
            return { right - s[1], right };
        }

        f -= NIF[1];

        {
            const index_t i = f % nx;
            const index_t t = f / nx;
            const index_t j = t % ny;
            const index_t k = (t / ny) + 1;
            const index_t right = cell(i, j, k);
            return { right - s[2], right };
        }
    }

    std::pair<index_t, int> boundary_face_to_cell(index_t b) const {
        const index_t nx = NC[0], ny = NC[1], nz = NC[2];

        // dir0 block: minus-x then plus-x
        if (b < 2 * NBF[0]) {
            const bool plus = (b >= NBF[0]);
            const index_t r = plus ? (b - NBF[0]) : b; // r = j + ny*k
            const index_t j = r % ny;
            const index_t k = r / ny;
            const index_t i = plus ? (nx - 1) : index_t{ 0 };
            return { cell(i, j, k), plus ? +1 : -1 };
        }
        b -= 2 * NBF[0];

        // dir1 block: minus-y then plus-y
        if (b < 2 * NBF[1]) {
            const bool plus = (b >= NBF[1]);
            const index_t r = plus ? (b - NBF[1]) : b; // r = i + nx*k
            const index_t i = r % nx;
            const index_t k = r / nx;
            const index_t j = plus ? (ny - 1) : index_t{ 0 };
            return { cell(i, j, k), plus ? +2 : -2 };
        }
        b -= 2 * NBF[1];

        // dir2 block: minus-z then plus-z
        {
            const bool plus = (b >= NBF[2]);
            const index_t r = plus ? (b - NBF[2]) : b; // r = i + nx*j
            const index_t i = r % nx;
            const index_t j = r / nx;
            const index_t k = plus ? (nz - 1) : index_t{ 0 };
            return { cell(i, j, k), plus ? +3 : -3 };
        }
    }

private:
    index_t cell(index_t i, index_t j, index_t k) const {
        return i * s[0] + j * s[1] + k * s[2];
    }

    std::array<index_t, 3> NC{ {0,0,0} };
    std::array<index_t, 3> s{ {0,0,0} };
    std::array<index_t, 3> NIF{ {0,0,0} }; // provided, not computed
    std::array<index_t, 3> NBF{ {0,0,0} }; // per-side, provided, not computed
};

template<template<typename> class Layout, typename T = std::size_t>
class StructuredConnectivity {
public:
    using index_t = T;
    using LayoutT = Layout<index_t>;

    StructuredConnectivity() = default;
    StructuredConnectivity(const StructuredConnectivity&) = default;
    StructuredConnectivity(StructuredConnectivity&&) = default;

    StructuredConnectivity(index_t nx, index_t ny, index_t nz) { set_dimensions(nx, ny, nz); }

    void set_dimensions(index_t nx, index_t ny, index_t nz) {
        assert(nx > 0 && ny > 0 && nz > 0);

        NC_dim = { nx, ny, nz };
        NC_total = nx * ny * nz;

        NBF = { (nx > 0 ? ny * nz : index_t{0}),
                (ny > 0 ? nx * nz : index_t{0}),
                (nz > 0 ? nx * ny : index_t{0}) };

        NIF = { (nx > 1 ? (nx - 1) * ny * nz : index_t{0}),
                (ny > 1 ? (ny - 1) * nx * nz : index_t{0}),
                (nz > 1 ? (nz - 1) * nx * ny : index_t{0}) };

        max_conn = index_t{ 1 } + index_t{ 2 } *
            (index_t(nx > 1) + index_t(ny > 1) + index_t(nz > 1));

        layout.set_dimensions(nx, ny, nz, NIF, NBF);
    }

    index_t num_cells() const { return NC_total; }
    index_t max_num_neighbors() const { return max_conn; }

    index_t num_intr_faces() const { return NIF[0] + NIF[1] + NIF[2]; }
    index_t num_bndr_faces() const { return 2 * (NBF[0] + NBF[1] + NBF[2]); }

    int interior_face_direction(index_t f) const {
        assert(f < num_intr_faces());
        return layout.interior_face_direction(f);
    }

    int boundary_face_direction(index_t f) const {
        return layout.boundary_face_direction(f);
    }

    std::pair<index_t, index_t> interior_face_to_cells(index_t f) const {
        assert(f < num_intr_faces());
        return layout.interior_face_to_cells(f);
    }

    std::pair<index_t, int> boundary_face_to_cell(index_t b) const {
        assert(b < num_bndr_faces());
        return layout.boundary_face_to_cell(b);
    }

    const std::array<index_t, 3>& dims() const { return NC_dim; }

    index_t ijk_to_cell(std::size_t i, std::size_t j, std::size_t k) const
    {
        const index_t nx = NC_dim[0];
        const index_t ny = NC_dim[1];
        const index_t nz = NC_dim[2];
        assert(i < nx && j < ny && k < nz);
        return i + j * nx + k * nx * ny;
    
    }
    std::array<index_t, 3> cell_to_ijk(index_t cell) const
    {
        assert(cell < NC_total);
        const index_t nx = NC_dim[0];
        const index_t ny = NC_dim[1];

        const index_t i = cell % nx;
        const index_t t = cell / nx;
        const index_t j = t % ny;
        const index_t k = t / ny;

        return { i, j, k };
    }

    const LayoutT& layout_ref () const { return layout; }


private:
    LayoutT layout{};

    index_t NC_total{ 0 };
    index_t max_conn{ 0 };

    std::array<index_t, 3> NC_dim{ {0,0,0} };
    std::array<index_t, 3> NBF{ {0,0,0} }; // per-side
    std::array<index_t, 3> NIF{ {0,0,0} };
};

#endif // __STRUCTUREDCONNECTIVITY_HPP__