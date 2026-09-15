#ifndef __CARTESIANMESH_HPP_
#define __CARTESIANMESH_HPP_

#include <array>
#include <vector>
#include <cstddef>
#include <utility>
#include <cmath>
#include <cassert>
#include <type_traits>
#include "TPFA.hpp"


// ============================== Geometry =====================================

struct UniformCartesianGeometry {
private:
    const std::array<double, 3> A{0.0, 0.0, 0.0}; // face areas per dir
    double V{ 0.0 };
    const std::array<double, 3> dim{0.0, 0.0, 0.0};
public:
    const static int is_uniform = true;
    const static int is_cartesian = true;
    UniformCartesianGeometry() = default;
    UniformCartesianGeometry(const UniformCartesianGeometry&) = default;
    UniformCartesianGeometry(UniformCartesianGeometry&&) = default;
    UniformCartesianGeometry(double dx, double dy, double dz) : dim{ dx, dy, dz }, A{ dy * dz, dx * dz, dx * dy }, V{ dx * dy * dz }
    {
    }

    const std::array<double, 3>& get_dims (std::size_t /*c*/) const
    {
        return dim;
    }

    double cell_volume(std::size_t) const { return V; }

    double k_to_depth(std::size_t k ) const {
        return (static_cast<double>(k) + 0.5) * dim[2];
    }

    //double face_delta_depth(std::size_t /*face*/) {
    //    return dim[2];
    //}

    template<class Conn>
    double interior_face_area(std::size_t f, const Conn& conn) const {
        return A[conn.interior_face_direction(f)];
    }

    double interior_face_area(uint8_t dir) const {

        return A[dir];
    }

    template<class Conn>
    double boundary_face_area(std::size_t b, const Conn& conn) const {
        auto [c, sdir] = conn.boundary_face_to_cell(b);
        (void)c;
        const int dir = std::abs(sdir) - 1;
        return A[dir];
    }

    double half_distance(std::size_t, uint8_t dir) const
    {
        return dir ? (dir == 1 ? 0.5 * dim[1] : 0.5 * dim[2]) : 0.5 * dim[0];
    }
    double half_distance(uint8_t dir) const
    {
        return dir ? (dir == 1 ? 0.5 * dim[1] : 0.5 * dim[2]) : 0.5 * dim[0];
    }

    template<class Conn>
    void build_face_areas(const Conn&) {}
};

template<class Index = std::size_t>
struct NonuniformCartesianGeometry {
    using index_t = Index;
    const static int is_uniform = false;
    const static int is_cartesian = true;
    NonuniformCartesianGeometry() = default;

    NonuniformCartesianGeometry(std::vector<double> _dx,
        std::vector<double> _dy,
        std::vector<double> _dz)
        : dx(std::move(_dx)), dy(std::move(_dy)), dz(std::move(_dz)), vol(dx.size())
    {
        assert(dx.size() == dy.size());
        assert(dy.size() == dz.size());
        for (std::size_t c = 0; c < dx.size(); ++c) {   
            vol[c] = dx[c] * dy[c] * dz[c];
        }
    }
    std::array<double, 3> get_dims(std::size_t c) const
    {
        return { dx[c], dy[c], dz[c] };
    }

    std::size_t num_cells() const { return dx.size(); }

    double cell_volume(index_t c) const { return vol[c]; }

    double cell_depth(std::size_t c) const {
        return 0.0;//todo
    }

    double k_to_depth(std::size_t k) const {
        return 0.0;
    }

    //double face_delta_depth(std::size_t f) const {
    //    auto [c0, c1] = conn.interior_face_to_cells(f);
    //    const double dz0 = dz[c0];
    //    const double dz1 = dz[c1];
    //    return 0.5 * (dz0 + dz1);
    //}


    double half_distance(index_t c, uint8_t dir) const {
        if (dir == 0) return 0.5 * dx[c];
        if (dir == 1) return 0.5 * dy[c];
        return 0.5 * dz[c];
    }

    // Precompute face-based areas from dx/dy/dz + connectivity.
    template<class Conn>
    void build_face_areas(const Conn& conn) {
        assert(!dx.empty() && !dy.empty() && !dz.empty());
        A_intr.assign(conn.num_intr_faces(), 0.0);
        A_bndr.assign(conn.num_bndr_faces(), 0.0);

        for (index_t f = 0; f < conn.num_intr_faces(); ++f) {
            auto [c0, c1] = conn.interior_face_to_cells(f);
            const int dir = conn.interior_face_direction(f);
            //A_intr[f] = 0.5 * (cell_face_area(c0, dir) + cell_face_area(c1, dir)); //
            A_intr[f] = cell_face_area(c0, dir);
        }

        for (index_t b = 0; b < conn.num_bndr_faces(); ++b) {
            auto [c, sdir] = conn.boundary_face_to_cell(b);
            const int dir = std::abs(sdir) - 1;
            A_bndr[b] = cell_face_area(c, dir);
        }
    }

    template<class Conn>
    double interior_face_area(index_t f, const Conn&) const { return A_intr[f]; }

    template<class Conn>
    double boundary_face_area(index_t b, const Conn&) const { return A_bndr[b]; }

private:
    double cell_face_area(index_t c, int dir) const {
        if (dir == 0) return dy[c] * dz[c]; // normal x
        if (dir == 1) return dx[c] * dz[c]; // normal y
        return dx[c] * dy[c];               // normal z
    }

    std::vector<double> dx, dy, dz;
    //::vector<double> hx, hy, hz;
    std::vector<double> vol;

    std::vector<double> A_intr;
    std::vector<double> A_bndr;
};



// ============================== Mesh =========================================
//
// TPFAmodel is a template template parameter so you can choose Harmonic/Arithmetic/Geometric.

template<class Conn, class Geom>
class CartesianGrid {
public:
    using index_t = typename Conn::index_t;
    using LayoutT = typename Conn::LayoutT;
    using ConnT = Conn;
    using GeomT = Geom;
    static constexpr bool is_structured = true;
    CartesianGrid() = default;

    CartesianGrid(Conn _conn, Geom _geom)
        : conn(std::move(_conn)), geom(std::move(_geom))
    {
        geom.build_face_areas(conn);
    }

    index_t num_cells() const { return conn.num_cells(); }
    index_t max_num_neighbors() const { return conn.max_num_neighbors(); }
    index_t num_intr_faces() const { return conn.num_intr_faces(); }
    index_t num_bndr_faces() const { return conn.num_bndr_faces(); }

    const std::array<index_t, 3>& dims() const { return conn.dims(); }
    std::pair<index_t, index_t> intr_face_to_cells(index_t f) const { return conn.interior_face_to_cells(f); }
    std::pair<index_t, int> bndr_face_to_intr_cell(index_t b) const { return conn.boundary_face_to_cell(b); }

    index_t ijk_to_cell(std::size_t i, std::size_t j, std::size_t k) const{ return conn.ijk_to_cell(i, j, k);}

    double cell_volume(index_t c) const { return geom.cell_volume(c); }

    // Nonuniform geometry computes dimensions by value; never return a reference
    // to that temporary through the common grid interface.
    std::array<double, 3> get_cell_dims(index_t c) const { return geom.get_dims(c); }

    double cell_depth(index_t c) const { 
        auto ijk = conn.cell_to_ijk(c);

        return geom.k_to_depth(ijk[2]); }
    double k_to_depth(std::size_t k) const {
        return geom.k_to_depth(k);
    }

    //double face_delta_depth(std::size_t f) const { return geom.face_delta_depth(f); }

    const Conn& connectivity() const { return conn; }
    const Geom& geometry() const { return geom; }

    const LayoutT& layout() const { return conn.layout_ref(); }

private:
    Conn conn;
    Geom geom;
};


template<
  class Grid,
  class CoefPolicy,
  template<class,class,class,class,class> class TPFAOperatorT,
  class AveragingPolicy,                 // harmonic/arithmetic/...
  class T = double
>
class TPFAMeshView {
public:
    using index_t = typename Grid::index_t;
    using value_t = T;
    using LayoutT = typename Grid::LayoutT;
    using ConnT = typename Grid::ConnT;
    using GeomT = typename Grid::GeomT;
    static constexpr bool is_structured = true;
    TPFAMeshView() = delete;
    TPFAMeshView(const TPFAMeshView&) = default;
    TPFAMeshView(TPFAMeshView&&) = default;
    

    TPFAMeshView(const Grid& _grid, CoefPolicy _coef)
        : grid(_grid), coef(std::move(_coef))
    {
        op.build(grid.connectivity(), grid.geometry(), coef);
    }

    index_t num_cells() const { return grid.num_cells(); }
    index_t max_num_neighbors() const { return grid.max_num_neighbors(); }
    index_t num_intr_faces() const { return grid.num_intr_faces(); }
    index_t num_bndr_faces() const { return grid.num_bndr_faces(); }

    const std::array<index_t, 3>& dims() const { return grid.dims(); }

    std::pair<index_t, index_t> intr_face_to_cells(index_t f) const { return grid.intr_face_to_cells(f); }
    std::pair<index_t, int> bndr_face_to_intr_cell(index_t b) const { return grid.bndr_face_to_intr_cell(b); }

    bool try_ijk_to_cell(std::size_t i, std::size_t j, std::size_t k, index_t& out_cell) const {
        out_cell = grid.ijk_to_cell(i, j, k);
        return true;
    }

    index_t ijk_to_cell(std::size_t i, std::size_t j, std::size_t k) const { return grid.ijk_to_cell(i, j, k); }
    value_t cell_volume(index_t c) const { return static_cast<value_t>(grid.cell_volume(c)); }
    std::array<double, 3> get_cell_dims(index_t c) const { return grid.get_cell_dims(c); }

    double cell_depth(index_t c) const { return grid.cell_depth(c); }
    
    //double face_delta_depth(std::size_t f) const { return grid.face_delta_depth(f); }

    value_t intr_face_coef(index_t f) const { return op.intr_face_coef(f, grid.connectivity()); }
    value_t bndr_face_coef(index_t b) const { return op.bndr_face_coef(b, grid.connectivity()); }

    const CoefPolicy& coeffs() const { return coef; }
    const LayoutT& layout() const { return grid.layout(); }

private:
    const Grid& grid;   // shared infrastructure
    const CoefPolicy coef;    // owned by this view. Possibly take off const qualifier for assignment operator
    TPFAOperatorT<AveragingPolicy, ConnT, GeomT, CoefPolicy, T> op;
};


#endif // __CARTESIANMESH_HPP_ included
