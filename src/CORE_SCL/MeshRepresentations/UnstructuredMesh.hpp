#ifndef __UNSTRUCTUREDMESH_HPP_
#define __UNSTRCUTUREDMESH_HPP_

#include <unordered_map>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include "SimUtils.hpp"
#include "StructuredConnectivity.hpp"
#include "TPFA.hpp"



class UnstructuredGrid
{
public:
    using index_t = std::size_t;
    using value_t = double;
    using LayoutT = IJKSequentialFaces<>;
    using CoefPolicy = HomogeneousAnisotropicK<double>;
    static constexpr bool is_structured = false;
    struct IFace { index_t c0, c1; };

    UnstructuredGrid() = default;

    UnstructuredGrid(std::vector<std::size_t> intr_faces_vec, 
        std::vector<value_t> _vol,
        std::vector<value_t> _depth,
        std::size_t _nx, std::size_t _ny, std::size_t _nz,
        const std::vector<std::uint64_t>& _indexMap)
        : vol(std::move(_vol))
        , depth(std::move(_depth))
        , nx(_nx), ny(_ny), nz(_nz)
    {
        parse_intr_faces(intr_faces_vec);
        validate_mesh();
        build_ijk_lookup(_indexMap);
        set_max_num_neighbors();
    }

    UnstructuredGrid(std::vector<std::size_t> intr_faces_vec, 
        std::vector<value_t> _vol,
        std::vector<value_t> _depth,
        std::size_t _nx, std::size_t _ny, std::size_t _nz)
        : vol(std::move(_vol))
        , depth(std::move(_depth))
        , nx(_nx), ny(_ny), nz(_nz)
    {
        parse_intr_faces(intr_faces_vec);
        validate_mesh();
        set_max_num_neighbors();
    }

    index_t num_cells() const { return vol.size(); }
    index_t num_intr_faces() const { return intr_faces.size(); }
    index_t num_bndr_faces() const { return 0; }
    index_t max_num_neighbors() const { return max_conn; }

    std::pair<index_t, index_t> intr_face_to_cells(index_t f) const {
        ASSERT_WITH_MSG(f < intr_faces.size(), "intr_face_to_cells: f out of range");
        const auto& e = intr_faces[f];
        return { e.c0, e.c1 };
    }

    std::pair<index_t, int> bndr_face_to_intr_cell(index_t) const {
        ASSERT_WITH_MSG(false, "bndr_face_to_intr_cell called but mesh has no boundary faces");
        return { 0, 0 };
    }

    value_t cell_volume(index_t c) const {
        ASSERT_WITH_MSG(c < vol.size(), "cell_volume: cell out of range");
        return vol[c];
    }

    value_t cell_depth(index_t c) const {
        ASSERT_WITH_MSG(c < depth.size(), "cell_depth: cell out of range");
        return depth[c];
    }

    value_t bndr_face_coef(index_t) const {
        ASSERT_WITH_MSG(false, "bndr_face_coef called but mesh has no boundary faces");
        return value_t(0);
    }

    const std::array<std::size_t, 3> dims() const { return { nx, ny, nz }; }

    bool try_ijk_to_cell(std::size_t i, std::size_t j, std::size_t k, index_t& out_cell) const {
        if (i >= nx || j >= ny || k >= nz) return false;

        const std::uint64_t g0 =
            static_cast<std::uint64_t>(i)
            + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx)
            + static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny);

        auto it = global1_to_active.find(g0);
        if (it == global1_to_active.end()) return false;

        out_cell = it->second;
        return true;
    }

    index_t ijk_to_cell(std::size_t i, std::size_t j, std::size_t k) const {
        index_t c = 0;
        const bool ok = try_ijk_to_cell(i, j, k, c);
        ASSERT_WITH_MSG(ok, "ijk_to_cell maps to inactive/out-of-component cell");
        return c;
    }

    const CoefPolicy& coeffs() const { return dummy_coef; }
    const LayoutT& layout() const { return dummy_layout; }

private:
    void parse_intr_faces(const std::vector<std::size_t>& intr_faces_vec) {
        const std::size_t size = intr_faces_vec.size();
        ASSERT_WITH_MSG(size % 2 == 0, "intr_faces_vec size must be divisible by 2");

        intr_faces.clear();
        intr_faces.reserve(size / 2);

        for (std::size_t i = 0; i < size / 2; ++i) {
            const std::size_t c0 = intr_faces_vec[2 * i];
            const std::size_t c1 = intr_faces_vec[2 * i + 1];
            intr_faces.push_back(IFace{ static_cast<index_t>(c0), static_cast<index_t>(c1) });
        }
    }

    void build_ijk_lookup(const std::vector<std::uint64_t>& indexMap) {
        ASSERT_WITH_MSG(indexMap.size() == vol.size(), "indexMap size mismatch with vol");

        global1_to_active.clear();
        global1_to_active.reserve(indexMap.size() * 2);

        for (index_t a = 0; a < indexMap.size(); ++a) {
            const std::uint64_t g1 = indexMap[a];
            global1_to_active[g1] = a;
        }
    }

    void set_max_num_neighbors() {
        std::vector<std::size_t> num_neighbors(num_cells(), 0);
        for (const auto& face : intr_faces) {
            ++num_neighbors[face.c0];
            ++num_neighbors[face.c1];
        }
        max_conn = static_cast<index_t>(*std::max_element(num_neighbors.begin(), num_neighbors.end()));
        if (max_conn == 0) max_conn = 1;
    }

    void validate_mesh() const {
        ASSERT_WITH_MSG(nx > 0 && ny > 0 && nz > 0, "nx, ny, nz must be > 0");
        ASSERT_WITH_MSG(vol.size() == depth.size(), "vol/depth size mismatch");
        

        const index_t N = vol.size();
        for (index_t f = 0; f < intr_faces.size(); ++f) {
            ASSERT_WITH_MSG(intr_faces[f].c0 < N, "intr_faces[c0] out of range");
            ASSERT_WITH_MSG(intr_faces[f].c1 < N, "intr_faces[c1] out of range");
        }
    }

private:
    std::vector<value_t> vol;
    std::vector<value_t> depth;

    std::vector<IFace> intr_faces;

    const LayoutT dummy_layout{};
    const CoefPolicy dummy_coef{};

    std::size_t nx = 0, ny = 0, nz = 0;
    index_t max_conn = 64;

    std::unordered_map<std::uint64_t, index_t> global1_to_active;
};

template <typename face_coeff_t>
struct UnstructuredMeshView {
    using Grid = UnstructuredGrid;
    using index_t = typename Grid::index_t;
    using value_t = typename Grid::value_t;
    using LayoutT = typename Grid::LayoutT;
    using CoefPolicy = typename Grid::CoefPolicy;

    static constexpr bool is_structured = false;

    
    UnstructuredMeshView(const Grid& _grid, face_coeff_t _coeff) : grid(_grid), coeff(std::move(_coeff)) {
        ASSERT_WITH_MSG(coeff.size() == grid.num_intr_faces(), "intr_faces/Tintr size mismatch");
    }

    index_t num_cells() const { return grid.num_cells(); }
    index_t max_num_neighbors() const { return grid.max_num_neighbors(); }
    index_t num_intr_faces() const { return grid.num_intr_faces(); }
    index_t num_bndr_faces() const { return grid.num_bndr_faces(); }

    const std::array<std::size_t, 3> dims() const { return grid.dims(); }

    std::pair<index_t, index_t> intr_face_to_cells(index_t f) const { return grid.intr_face_to_cells(f); }
    std::pair<index_t, int> bndr_face_to_intr_cell(index_t b) const { return grid.bndr_face_to_intr_cell(b); }

    value_t cell_volume(index_t c) const { return grid.cell_volume(c); }
    value_t cell_depth(index_t c)  const { return grid.cell_depth(c); }

    value_t intr_face_coef(index_t f) const {
        ASSERT_WITH_MSG(f < coeff.size(), "intr_face_coef: face out of range");
        return coeff[f];
    }
    value_t bndr_face_coef(index_t b) const { return grid.bndr_face_coef(b); }

    index_t ijk_to_cell(std::size_t i, std::size_t j, std::size_t k) const { return grid.ijk_to_cell(i, j, k); }
    bool try_ijk_to_cell(std::size_t i, std::size_t j, std::size_t k, index_t& out_cell) const {
        return grid.try_ijk_to_cell(i, j, k, out_cell);
    }

    const CoefPolicy& coeffs() const { return grid.coeffs(); }
    const LayoutT& layout() const { return grid.layout(); }

private:
    const Grid& grid;
    const face_coeff_t coeff;
};
#endif