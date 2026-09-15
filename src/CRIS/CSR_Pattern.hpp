#ifndef __CSR_PATTERN_HPP_
#define __CSR_PATTERN_HPP_
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cassert>

namespace csr_pattern {

    // Binary search to find column index in sorted row (with bounds checking)
    static inline std::size_t find_vidx(const std::vector<std::size_t>& row, std::size_t j) {
        auto it = std::lower_bound(row.begin(), row.end(), j);
        assert(it != row.end() && *it == j && "Column index not found in row");
        return std::distance(row.begin(), it);
    }

    template <typename Matrix_type, typename TPFA_mesh>
    void build_pattern(Matrix_type& J, const TPFA_mesh& mesh,
        std::vector<std::size_t>& diag,
        std::vector<std::size_t>& k_12,
        std::vector<std::size_t>& k_21) {
        using index_t = typename Matrix_type::index_t;
        using index_vector_t = typename Matrix_type::index_vector_t;
        using value_vector_t = typename Matrix_type::value_vector_t;

        std::size_t n_rows = mesh.num_cells();
        std::size_t n_intr_faces = mesh.num_intr_faces();

        // Pre-size output vectors
        assert(diag.size() == n_rows && "diag vector pre-sized incorrectly");
        assert(k_12.size() == n_intr_faces && "k_12 vector pre-sized incorrectly");
        assert(k_21.size() == n_intr_faces && "k_21 vector pre-sized incorrectly");

        // Build adjacency structure: cols[i] = list of column indices in row i
        std::vector<std::vector<std::size_t>> cols(n_rows);

        // Add diagonal entries
        for (std::size_t l = 0; l < n_rows; ++l)
            cols[l].push_back(l);

        // Add off-diagonal entries from interior faces
        for (std::size_t f = 0; f < n_intr_faces; ++f) {
            auto [l1, l2] = mesh.intr_face_to_cells(f);
            assert(l1 < n_rows && l2 < n_rows && "Mesh returned invalid cell indices");
            cols[l1].push_back(l2);
            cols[l2].push_back(l1);
        }

        // Sort and remove duplicates from each row
        for (std::size_t l = 0; l < n_rows; ++l) {
            auto& row = cols[l];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }

        // Build CSR structure: rowptr, colind, value
        index_vector_t& row_ptr = J.rowptr();
        index_vector_t& col_ind = J.colind();
        value_vector_t& value = J.value();

        row_ptr.resize(n_rows + 1); 
        row_ptr[0] = 0;
        std::size_t nnz = 0;

        for (std::size_t l = 0; l < n_rows; ++l) {
            std::size_t col_size = cols[l].size();
            row_ptr[l + 1] = row_ptr[l] + col_size;
            nnz += col_size;
        }

        col_ind.resize(nnz, 0);
        value.resize(nnz, 0.0);

        // Fill column indices
        for (std::size_t l = 0; l < n_rows; ++l) {
            std::size_t k = row_ptr[l];
            for (std::size_t col : cols[l]) {
                col_ind[k++] = col;
            }
        }

        // Record positions of face connections and diagonals
        for (std::size_t f = 0; f < n_intr_faces; ++f) {
            auto [l1, l2] = mesh.intr_face_to_cells(f);
            k_12[f] = row_ptr[l1] + find_vidx(cols[l1], l2);
            k_21[f] = row_ptr[l2] + find_vidx(cols[l2], l1);
        }

        for (std::size_t l = 0; l < n_rows; ++l) {
            diag[l] = row_ptr[l] + find_vidx(cols[l], l);
        }
    }

} // namespace csr_pattern

#endif // __CSR_PATTERN_HPP_