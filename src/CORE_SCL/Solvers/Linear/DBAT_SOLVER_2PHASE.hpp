/**************************************SOMTO VERSION 9 START... *****************************************************/

#include <vector>
#include <cmath>
#include <stdexcept> 
#include <iostream>
#include <iomanip>    
#include <chrono>
#include <map>
#include <utility>    
#include <initializer_list> 
#include <algorithm>  
#include <string>     
#include <mkl.h>      
#include <type_traits> 
#include "Solvers/Linear/CSR_Matrix.hpp" // Assuming this is the correct path to GENSOL::CSR_Matrix

// --- Helper for 2x1 Vector/Block ---
struct Vector2x1 {
    double val[2]; // Stored as [row 0; row 1]
    Vector2x1() : val{ 0.0, 0.0 } {}
    double& operator()(int r) { return val[r]; }
    const double& operator()(int r) const { return val[r]; }

    static Vector2x1 subtract(const Vector2x1& A, const Vector2x1& B) {
        Vector2x1 C; C.val[0] = A.val[0] - B.val[0]; C.val[1] = A.val[1] - B.val[1]; return C;
    }
    static Vector2x1 add(const Vector2x1& A, const Vector2x1& B) {
        Vector2x1 C; C.val[0] = A.val[0] + B.val[0]; C.val[1] = A.val[1] + B.val[1]; return C;
    }

    Vector2x1(std::initializer_list<double> il) {
        if (il.size() == 2) { std::copy(il.begin(), il.end(), val); }
        else if (il.size() == 0) { val[0] = 0.0; val[1] = 0.0; }
        else { throw std::invalid_argument("Initializer list must have 0 or 2 elements for Vector2x1"); }
    }
};

// --- Helper for 2x2 Matrix Operations (ROW-MAJOR) ---
struct Matrix2x2 {
    double val[4]; // Stored row-major: [0]=(0,0), [1]=(0,1), [2]=(1,0), [3]=(1,1)
    Matrix2x2() : val{ 0.0, 0.0, 0.0, 0.0 } {}
    Matrix2x2(std::initializer_list<double> il) {
        if (il.size() == 4) { std::copy(il.begin(), il.end(), val); }
        else if (il.size() == 0) { std::fill(val, val + 4, 0.0); }
        else { throw std::invalid_argument("Initializer list must have 0 or 4 elements for Matrix2x2"); }
    }
    double& operator()(int r, int c) { return val[r * 2 + c]; }
    const double& operator()(int r, int c) const { return val[r * 2 + c]; }
    Matrix2x2 transpose() const { Matrix2x2 T; T.val[0] = val[0]; T.val[1] = val[2]; T.val[2] = val[1]; T.val[3] = val[3]; return T; }
    Matrix2x2 inverse() const {
        double det = val[0] * val[3] - val[1] * val[2];
        if (std::abs(det) < 1e-12) { return Matrix2x2(); }
        double invDet = 1.0 / det; Matrix2x2 inv;
        inv.val[0] = val[3] * invDet; inv.val[1] = -val[1] * invDet;
        inv.val[2] = -val[2] * invDet; inv.val[3] = val[0] * invDet;
        return inv;
    }
    static Matrix2x2 multiply(const Matrix2x2& A, const Matrix2x2& B) { Matrix2x2 C; C.val[0] = A.val[0] * B.val[0] + A.val[1] * B.val[2]; C.val[1] = A.val[0] * B.val[1] + A.val[1] * B.val[3]; C.val[2] = A.val[2] * B.val[0] + A.val[3] * B.val[2]; C.val[3] = A.val[2] * B.val[1] + A.val[3] * B.val[3]; return C; }
    static Matrix2x2 subtract(const Matrix2x2& A, const Matrix2x2& B) { Matrix2x2 C; C.val[0] = A.val[0] - B.val[0]; C.val[1] = A.val[1] - B.val[1]; C.val[2] = A.val[2] - B.val[2]; C.val[3] = A.val[3] - B.val[3]; return C; }
    static Vector2x1 multiply(const Matrix2x2& A, const Vector2x1& B_vec) {
        Vector2x1 C_vec;
        C_vec.val[0] = A.val[0] * B_vec.val[0] + A.val[1] * B_vec.val[1];
        C_vec.val[1] = A.val[2] * B_vec.val[0] + A.val[3] * B_vec.val[1];
        return C_vec;
    }
};

// --- Structure to represent the sparse B matrix segments ---
struct B_Segment_Info {
    MKL_INT block_row_L; Matrix2x2 value_L;
    MKL_INT block_row_R; Matrix2x2 value_R;
};

// --- Structure to represent the sparse E matrix segments ---
struct E_Segment_Info {
    MKL_INT block_col_L; std::vector<Matrix2x2> values_L;
    MKL_INT block_col_R; std::vector<Matrix2x2> values_R;
    E_Segment_Info(MKL_INT nc_sys = 0) : block_col_L(-1), values_L(nc_sys), block_col_R(-1), values_R(nc_sys) {}
};

// --- Structure to represent the Z matrix (sparse block form) ---
struct Z_Matrix {
    MKL_INT NP_blocks;
    std::vector<Matrix2x2> diagonal_blocks;
    std::map<std::pair<MKL_INT, MKL_INT>, Matrix2x2> off_diagonal_blocks;
    Z_Matrix(MKL_INT np = 0) : NP_blocks(np), diagonal_blocks(np) {}
    Matrix2x2& get_block_for_accumulation(MKL_INT r, MKL_INT c) {
        if (r < 0 || r >= NP_blocks || c < 0 || c >= NP_blocks) throw std::out_of_range("Z_Matrix block indices out of range");
        if (r == c) return diagonal_blocks[r];
        else return off_diagonal_blocks[{r, c}];
    }
    void print() const {
        std::cout << "Z Matrix (NP_blocks = " << NP_blocks << "): Showing non-zero blocks." << std::endl;
        std::cout << std::fixed << std::setprecision(4); bool printed_any = false;
        for (MKL_INT i = 0; i < NP_blocks; ++i) { const auto& db = diagonal_blocks[i]; bool is_zero = true; for (int k = 0; k < 4; ++k)if (std::abs(db.val[k]) > 1e-9)is_zero = false; if (!is_zero) { std::cout << "  Diag[" << i << "]: {{" << db(0, 0) << "," << db(0, 1) << "},{" << db(1, 0) << "," << db(1, 1) << "}}\n"; printed_any = true; } }
        if (!printed_any && NP_blocks > 0) std::cout << "  (All diagonal blocks are zero or close to zero)" << std::endl;
        bool has_printed_offdiag_header = false;
        for (const auto& e : off_diagonal_blocks) { const auto& odb = e.second; bool is_zero = true; for (int k = 0; k < 4; ++k)if (std::abs(odb.val[k]) > 1e-9)is_zero = false; if (!is_zero) { if (!has_printed_offdiag_header) { std::cout << "  Off-Diagonal Blocks:\n"; has_printed_offdiag_header = true; } std::cout << "    Block[" << e.first.first << "," << e.first.second << "]: {{" << odb(0, 0) << "," << odb(0, 1) << "},{" << odb(1, 0) << "," << odb(1, 1) << "}}\n"; printed_any = true; } }
        if (!printed_any && off_diagonal_blocks.empty()) std::cout << "  (All blocks appear zero or Z is empty)" << std::endl;
        std::cout << "--- End of Z Matrix ---" << std::endl;
    }
};

// --- Struct to Store Factorization Results for Block Tridiagonal T_i Systems ---
struct BlockTridiagonalSystemFactorization {
    MKL_INT N_sys_blocks; std::vector<Matrix2x2> L_sys; std::vector<Matrix2x2> InvD_prime_sys; std::vector<Matrix2x2> U_prime_sys; bool factorization_succeeded;
    BlockTridiagonalSystemFactorization(MKL_INT n_sys_blocks = 0) : N_sys_blocks(n_sys_blocks), L_sys(n_sys_blocks), InvD_prime_sys(n_sys_blocks), U_prime_sys(n_sys_blocks), factorization_succeeded(false) {}
};

// --- Templated function to print a vector of Matrix2x2 or Vector2x1 objects ---
template <typename BlockType>
void print_block_vector(const std::string& label, const std::vector<BlockType>& vec) {
    std::cout << label << ":" << std::endl; if (vec.empty()) { std::cout << "  (empty vector)" << std::endl; return; }
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < vec.size(); ++i) {
        const BlockType& block = vec[i];
        if constexpr (std::is_same_v<BlockType, Matrix2x2>) {
            std::cout << "  Block[" << i << "] (Matrix2x2):" << std::endl;
            std::cout << "    [ " << std::setw(10) << block(0, 0) << ", " << std::setw(10) << block(0, 1) << " ]" << std::endl;
            std::cout << "    [ " << std::setw(10) << block(1, 0) << ", " << std::setw(10) << block(1, 1) << " ]" << std::endl;
        }
        else if constexpr (std::is_same_v<BlockType, Vector2x1>) {
            std::cout << "  Block[" << i << "] (Vector2x1):" << std::endl;
            std::cout << "    [ " << std::setw(10) << block(0) << " ]" << std::endl;
            std::cout << "    [ " << std::setw(10) << block(1) << " ]" << std::endl;
        }
        if (i < vec.size() - 1) { std::cout << std::endl; }
    }
    std::cout << "--- End of " << label << " ---" << std::endl;
}

// --- Function to print B segments ---
void print_B_segments(const std::string& label, const std::vector<B_Segment_Info>& segments) {
    std::cout << label << ":" << std::endl; if (segments.empty()) { std::cout << "  (No B segments)" << std::endl; return; }
    std::cout << std::fixed << std::setprecision(4);
    for (size_t s = 0; s < segments.size(); ++s) {
        const auto& seg = segments[s];
        std::cout << "  Segment " << s << ":" << std::endl;
        std::cout << "    L: row_idx=" << seg.block_row_L << ", val={{" << std::setw(10) << seg.value_L(0, 0) << ", " << std::setw(10) << seg.value_L(0, 1) << "}, {" << std::setw(10) << seg.value_L(1, 0) << ", " << std::setw(10) << seg.value_L(1, 1) << "}}" << std::endl;
        std::cout << "    R: row_idx=" << seg.block_row_R << ", val={{" << std::setw(10) << seg.value_R(0, 0) << ", " << std::setw(10) << seg.value_R(0, 1) << "}, {" << std::setw(10) << seg.value_R(1, 0) << ", " << std::setw(10) << seg.value_R(1, 1) << "}}" << std::endl;
    } std::cout << "--- End of B Segments ---" << std::endl;
}

// --- Function to print E segments ---
void print_E_segments(const std::string& label, const std::vector<E_Segment_Info>& segments, MKL_INT NC_sys) {
    std::cout << label << ":" << std::endl; if (segments.empty()) { std::cout << "  (No E segments)" << std::endl; return; }
    std::cout << std::fixed << std::setprecision(4);
    for (size_t s = 0; s < segments.size(); ++s) {
        const auto& seg = segments[s]; std::cout << "  Segment " << s << ":" << std::endl;
        std::cout << "    Column L (at E's block col " << seg.block_col_L << "):" << std::endl;
        if (seg.values_L.size() == static_cast<size_t>(NC_sys)) {
            for (MKL_INT r = 0; r < NC_sys; ++r) { std::cout << "      Block row " << r << " of column: {{" << std::setw(10) << seg.values_L[r](0, 0) << ", " << std::setw(10) << seg.values_L[r](0, 1) << "}, {" << std::setw(10) << seg.values_L[r](1, 0) << ", " << std::setw(10) << seg.values_L[r](1, 1) << "}}" << std::endl; }
        }
        else { std::cout << "      (Incorrect size: " << seg.values_L.size() << " != NC_sys=" << NC_sys << ")" << std::endl; }
        std::cout << "    Column R (at E's block col " << seg.block_col_R << "):" << std::endl;
        if (seg.values_R.size() == static_cast<size_t>(NC_sys)) {
            for (MKL_INT r = 0; r < NC_sys; ++r) { std::cout << "      Block row " << r << " of column: {{" << std::setw(10) << seg.values_R[r](0, 0) << ", " << std::setw(10) << seg.values_R[r](0, 1) << "}, {" << std::setw(10) << seg.values_R[r](1, 0) << ", " << std::setw(10) << seg.values_R[r](1, 1) << "}}" << std::endl; }
        }
        else { std::cout << "      (Incorrect size: " << seg.values_R.size() << " != NC_sys=" << NC_sys << ")" << std::endl; }
    } std::cout << "--- End of E Segments ---" << std::endl;
}

// --- Solver Class ---
class DBAT_SOLVER_2PHASE {
public:
public:
    struct report_t
    {
        bool     is_converged;
        int      ln_error_code;
        int      niter;
        std::chrono::milliseconds timing;
        std::chrono::milliseconds extraction_timing{ 0 };
    }                                                  ;

    typedef double                                     double_type;
    typedef MKL_INT                                    int_type;

    typedef GENSOL::CSR_Matrix< double_type, int_type >        A_type;
    typedef std::vector< double_type >                         x_type;
    static std::size_t offset() { return 0; }

    DBAT_SOLVER_2PHASE(MKL_INT np_main, MKL_INT nc_sys, MKL_INT ne_main);
    ~DBAT_SOLVER_2PHASE();

    //bool solve(
    //    const MKL_INT* J_rowptr, const MKL_INT* J_colind, const double* J_val,
    //    MKL_INT J_scalar_rows, MKL_INT J_scalar_cols,
    //    const std::vector<double>& r_global
    //);

    report_t solve(A_type& _A, x_type& _x_, x_type& _b, bool rebuild);

    void test_print_extracted_data() const;
    const std::vector<Vector2x1>& get_Xp_solution() const { return Xp_solution_blocks_out_; }
    const std::vector<Vector2x1>& get_Xe_solution() const { return Xe_result_blocks_out_; }

private:
    report_t report;
    // Dimensions
    const MKL_INT NP_main_; MKL_INT NC_sys_; MKL_INT NE_main_; MKL_INT NCE_main_;

    // Member vectors for extracted and computed data
    std::vector<std::vector<Matrix2x2>> L_systems_, D_T_systems_, U_systems_, A_rhs_systems_;
    std::vector<std::vector<Vector2x1>> Re_rhs_systems_, gs_segmented_;
    std::vector<B_Segment_Info> B_segments_global_;
    std::vector<Matrix2x2> D_global_blocks_;
    std::vector<Vector2x1> Rp_global_blocks_, g_flat_for_h_, h_computed_blocks_, Xp_solution_blocks_out_, Xe_result_blocks_out_;
    std::vector<E_Segment_Info> E_segment_data_;
    Z_Matrix Z_result_;
    void* pt_[64];

    // Private Helper Methods
    bool extract_all_data_from_J_and_r(
        const MKL_INT* J_rowptr, const MKL_INT* J_colind, const double* J_val,
        const std::vector<double>& r_global
    );
    BlockTridiagonalSystemFactorization factorize_block_tridiagonal_system(
        MKL_INT N_sys_blocks_local, const std::vector<Matrix2x2>& L_sys_local,
        const std::vector<Matrix2x2>& D_sys_local, const std::vector<Matrix2x2>& U_sys_local
    );
    template <typename RhsBlockType, typename SolBlockType>
    bool solve_system_with_factorization(
        const BlockTridiagonalSystemFactorization& fact, const std::vector<RhsBlockType>& rhs_sys_blocks,
        bool rhs_is_only_at_bottom, std::vector<SolBlockType>& sol_sys_blocks
    );
    Z_Matrix compute_Z_matrix_optimized(
        const std::vector<Matrix2x2>& D_diag_blocks, const std::vector<B_Segment_Info>& B_seg_data,
        const std::vector<E_Segment_Info>& E_seg_data, MKL_INT current_NP, MKL_INT current_NC, MKL_INT current_NE
    );
    bool compute_h_Rp_minus_Bg(
        const std::vector<Vector2x1>& Rp_blks, const std::vector<B_Segment_Info>& B_seg_data,
        const std::vector<Vector2x1>& g_flat_blks, MKL_INT current_NP, MKL_INT current_NC, MKL_INT current_NE,
        std::vector<Vector2x1>& h_result_blks
    );
    void Z_Matrix_to_CSR(
        const Z_Matrix& Z_in, MKL_INT current_NP, std::vector<MKL_INT>& csr_rowptr_out,
        std::vector<MKL_INT>& csr_colind_out, std::vector<double>& csr_val_out
    );
    bool solve_Z_Xp_eq_h_pardiso(
        const Z_Matrix& Z_in, const std::vector<Vector2x1>& h_blks,
        MKL_INT current_NP, std::vector<Vector2x1>& Xp_sol_blks
    );
    bool compute_Xe_g_minus_EXp(
        const std::vector<Vector2x1>& g_blks, const std::vector<E_Segment_Info>& E_seg_data,
        const std::vector<Vector2x1>& Xp_blks, MKL_INT current_NP, MKL_INT current_NC, MKL_INT current_NE,
        std::vector<Vector2x1>& Xe_result_blks
    );
};

// --- Implementations of DBAT_SOLVER_2PHASE Methods ---

DBAT_SOLVER_2PHASE::DBAT_SOLVER_2PHASE(MKL_INT np_main, MKL_INT nc_sys, MKL_INT ne_main) :
    NP_main_(np_main), NC_sys_(nc_sys), NE_main_(ne_main), NCE_main_(nc_sys* ne_main),
    L_systems_(ne_main, std::vector<Matrix2x2>(nc_sys)),
    D_T_systems_(ne_main, std::vector<Matrix2x2>(nc_sys)),
    U_systems_(ne_main, std::vector<Matrix2x2>(nc_sys)),
    A_rhs_systems_(2 * ne_main, std::vector<Matrix2x2>(nc_sys)),
    Re_rhs_systems_(ne_main, std::vector<Vector2x1>(nc_sys)),
    B_segments_global_(ne_main),
    D_global_blocks_(np_main),
    Rp_global_blocks_(np_main),
    E_segment_data_(ne_main, E_Segment_Info(nc_sys)),
    gs_segmented_(ne_main, std::vector<Vector2x1>(nc_sys)),
    g_flat_for_h_(nc_sys* ne_main),
    Z_result_(np_main),
    h_computed_blocks_(np_main),
    Xp_solution_blocks_out_(np_main),
    Xe_result_blocks_out_(nc_sys* ne_main)
{
    if (NP_main_ <= 0 || NC_sys_ <= 0 || NE_main_ <= 0) {
        throw std::invalid_argument("Solver dimensions NP, NC, NE must be positive.");
    }
    for (int i = 0; i < 64; i++) pt_[i] = 0;
    std::cout << "DBAT_SOLVER_2PHASE initialized with NP_main=" << NP_main_
        << ", NC_sys=" << NC_sys_ << ", NE_main=" << NE_main_
        << ", NCE_main=" << NCE_main_ << std::endl;
}

DBAT_SOLVER_2PHASE::~DBAT_SOLVER_2PHASE() {
    if (pt_[0] != 0) {
        MKL_INT N_scalar_dummy = 2 * NP_main_; MKL_INT mtype_dummy = 11; MKL_INT nrhs_dummy = 1;
        MKL_INT iparm_dummy[64]; for (int i = 0; i < 64; ++i)iparm_dummy[i] = 0;
        MKL_INT maxfct_dummy = 1, mnum_dummy = 1, phase_dummy = -1, error_dummy = 0, msglvl_dummy = 0;
        PARDISO(pt_, &maxfct_dummy, &mnum_dummy, &mtype_dummy, &phase_dummy, &N_scalar_dummy, nullptr, nullptr, nullptr, nullptr, &nrhs_dummy, iparm_dummy, &msglvl_dummy, nullptr, nullptr, &error_dummy);
    }
}

DBAT_SOLVER_2PHASE::report_t DBAT_SOLVER_2PHASE::solve(A_type& J, x_type& _x_, x_type& r, bool rebuild = false)
{
    //std::cout << "Solver::solve invoked." << std::endl;

    report.is_converged = 0; //i.e. NOt converged at start ERROR: ;
	report.ln_error_code = -999; //error;

    //std::cout << "  Solver: Extracting all data from Jacobian and RHS..." << std::endl;
    auto start_extract = std::chrono::high_resolution_clock::now();
    if (!extract_all_data_from_J_and_r(J.rowptr().data(), J.colind().data(), J.value().data(), r)) {
        std::cerr << "Error: Failed during data extraction from J and r." << std::endl;
        return report;
    }

    auto end_extract = std::chrono::high_resolution_clock::now();
    //auto extraction_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_extract - start_extract);

    auto start = std::chrono::high_resolution_clock::now();
    std::size_t expected_J_dim = 2 * NP_main_ + 2 * NCE_main_;
    if (J.N() != expected_J_dim || r.size() != expected_J_dim) {
        std::cerr << "Error: Jacobian J or RHS r dimensions mismatch." << std::endl; 
        return report;
    }

    //std::cout << "  Solver: Populating E and gs by solving T_i systems..." << std::endl;
    for (MKL_INT i = 0; i < NE_main_; ++i) {
        BlockTridiagonalSystemFactorization fact = factorize_block_tridiagonal_system(NC_sys_, L_systems_[i], D_T_systems_[i], U_systems_[i]);
        if (!fact.factorization_succeeded) { 
            std::cerr << "Error: Factorization failed for T_system " << i << std::endl; 
            return report; }
        E_segment_data_[i].block_col_L = B_segments_global_[i].block_row_L;
        E_segment_data_[i].block_col_R = B_segments_global_[i].block_row_R;
        if (!solve_system_with_factorization<Matrix2x2, Matrix2x2>(fact, A_rhs_systems_[2 * i], false, E_segment_data_[i].values_L)) 
            return report;
        if (!solve_system_with_factorization<Matrix2x2, Matrix2x2>(fact, A_rhs_systems_[2 * i + 1], true, E_segment_data_[i].values_R)) 
            return report;
        if (!solve_system_with_factorization<Vector2x1, Vector2x1>(fact, Re_rhs_systems_[i], false, gs_segmented_[i])) 
            return report;
    }

    //std::cout << "  Solver: Flattening gs..." << std::endl;
    g_flat_for_h_.assign(NCE_main_, Vector2x1()); MKL_INT current_flat_idx = 0;
    for (MKL_INT i = 0; i < NE_main_; ++i) {
        for (MKL_INT j = 0; j < NC_sys_; ++j) g_flat_for_h_[current_flat_idx++] = gs_segmented_[i][j];
    }

    //std::cout << "  Solver: Computing Z = D - B*E..." << std::endl;
    Z_result_ = compute_Z_matrix_optimized(D_global_blocks_, B_segments_global_, E_segment_data_, NP_main_, NC_sys_, NE_main_);

    //std::cout << "  Solver: Computing h = Rp - B*g..." << std::endl;
    if (!compute_h_Rp_minus_Bg(Rp_global_blocks_, B_segments_global_, g_flat_for_h_, NP_main_, NC_sys_, NE_main_, h_computed_blocks_)) 
        return report;

    //std::cout << "  Solver: Solving Z*Xp = h using PARDISO..." << std::endl;
    if (!solve_Z_Xp_eq_h_pardiso(Z_result_, h_computed_blocks_, NP_main_, Xp_solution_blocks_out_)) 
        return report;

    //std::cout << "  Solver: Computing Xe = g - E*Xp..." << std::endl;
    if (!compute_Xe_g_minus_EXp(g_flat_for_h_, E_segment_data_, Xp_solution_blocks_out_, NP_main_, NC_sys_, NE_main_, Xe_result_blocks_out_)) 
        return report;

    for (MKL_INT i = 0; i < NP_main_; ++i) {
        _x_[2 * i] = Xp_solution_blocks_out_[i](0); // Assuming _x_ is a flat vector and we only need the first element of each Vector2x1
        _x_[2 * i + 1] = Xp_solution_blocks_out_[i](1); // Assuming _x_ is a flat vector and we only need the first element of each Vector2x1
	}

    for (MKL_INT i = 0; i < NCE_main_; ++i) {
        _x_[2 * NP_main_ + 2 * i] = Xe_result_blocks_out_[i](0); // Assuming _x_ is a flat vector and we only need the first element of each Vector2x1
        _x_[2 * NP_main_ + 2 * i + 1] = Xe_result_blocks_out_[i](1); // Assuming _x_ is a flat vector and we only need the first element of each Vector2x1
	}
    //std::cout << "  Solver: All steps completed successfully." << std::endl;

    auto end = std::chrono::high_resolution_clock::now();

    report.timing = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    report.extraction_timing = std::chrono::duration_cast<std::chrono::milliseconds>(end_extract - start_extract);
    report.is_converged = 1; //i.e. NO ERROR: Solver: All steps completed successfully (error == 0);
    report.ln_error_code = 0; //error;
    report.niter = 0;
    return report;
}

void DBAT_SOLVER_2PHASE::test_print_extracted_data() const {
    std::cout << "\n\n========================================================" << std::endl;
    std::cout << "--- VERIFICATION: PRINTING EXTRACTED DATA ---" << std::endl;
    std::cout << "========================================================" << std::endl;

    std::cout << "\n--- GLOBAL D BLOCKS ---" << std::endl;
    print_block_vector("D_global_blocks_", D_global_blocks_);

    std::cout << "\n--- GLOBAL B SEGMENTS ---" << std::endl;
    print_B_segments("B_segments_global_", B_segments_global_);

    std::cout << "\n--- GLOBAL RP BLOCKS (from r) ---" << std::endl;
    print_block_vector("Rp_global_blocks_", Rp_global_blocks_);

    for (MKL_INT i = 0; i < NE_main_; ++i) {
        std::string header = "\n--- T_SYSTEM " + std::to_string(i) + " ---";
        std::cout << header << std::endl;
        print_block_vector("L_systems_[" + std::to_string(i) + "]", L_systems_[i]);
        print_block_vector("D_T_systems_[" + std::to_string(i) + "]", D_T_systems_[i]);
        print_block_vector("U_systems_[" + std::to_string(i) + "]", U_systems_[i]);
    }

    for (MKL_INT i = 0; i < NE_main_; ++i) {
        std::string header_top = "\n--- A_TOP RHS for T_SYSTEM " + std::to_string(i) + " ---";
        std::cout << header_top << std::endl;
        print_block_vector("A_rhs_systems_[" + std::to_string(2 * i) + "]", A_rhs_systems_[2 * i]);

        std::string header_bot = "\n--- A_BOT RHS for T_SYSTEM " + std::to_string(i) + " ---";
        std::cout << header_bot << std::endl;
        print_block_vector("A_rhs_systems_[" + std::to_string(2 * i + 1) + "]", A_rhs_systems_[2 * i + 1]);
    }

    for (MKL_INT i = 0; i < NE_main_; ++i) {
        std::string header = "\n--- RE RHS for T_SYSTEM " + std::to_string(i) + " ---";
        std::cout << header << std::endl;
        print_block_vector("Re_rhs_systems_[" + std::to_string(i) + "]", Re_rhs_systems_[i]);
    }

    std::cout << "\n========================================================" << std::endl;
    std::cout << "--- END OF EXTRACTED DATA VERIFICATION ---" << std::endl;
    std::cout << "========================================================" << std::endl;
}
template <typename T>
void printvector(T vector, const std::string& name);


bool DBAT_SOLVER_2PHASE::extract_all_data_from_J_and_r(
    const MKL_INT* J_rowptr, const MKL_INT* J_colind, const double* J_val,
    const std::vector<double>& r_global
) {
    //printvector(J_rowptr, "Inner J_rowptr");
    D_global_blocks_.assign(NP_main_, Matrix2x2());
    B_segments_global_.assign(NE_main_, B_Segment_Info());
    L_systems_.assign(NE_main_, std::vector<Matrix2x2>(NC_sys_));
    D_T_systems_.assign(NE_main_, std::vector<Matrix2x2>(NC_sys_));
    U_systems_.assign(NE_main_, std::vector<Matrix2x2>(NC_sys_));
    A_rhs_systems_.assign(2 * NE_main_, std::vector<Matrix2x2>(NC_sys_));
    Rp_global_blocks_.assign(NP_main_, Vector2x1());
    Re_rhs_systems_.assign(NE_main_, std::vector<Vector2x1>(NC_sys_));

    MKL_INT D_B_scalar_row_limit = 2 * NP_main_;
    MKL_INT A_T_scalar_row_start = 2 * NP_main_;
    MKL_INT J_total_scalar_dim = 2 * NP_main_ + 2 * NCE_main_;

    // PASS 1: Extract D and B to fully populate B_segments_global_
    for (MKL_INT srJ = 0; srJ < D_B_scalar_row_limit; ++srJ) {
        for (MKL_INT k = J_rowptr[srJ]; k < J_rowptr[srJ + 1]; ++k) {
            MKL_INT scJ = J_colind[k]; double value = J_val[k];
            MKL_INT block_r_NP = srJ / 2; int local_r_2x2 = srJ % 2;
            if (scJ < D_B_scalar_row_limit) { // D quadrant
                MKL_INT block_c_NP = scJ / 2; int local_c_2x2 = scJ % 2;
                if (block_r_NP == block_c_NP) D_global_blocks_[block_r_NP](local_r_2x2, local_c_2x2) = value;
            }
            else { // B quadrant
                MKL_INT sc_relative = scJ - D_B_scalar_row_limit;
                MKL_INT segment_idx = (sc_relative / 2) / NC_sys_;
                MKL_INT block_col_in_seg = (sc_relative / 2) % NC_sys_;
                int local_c_2x2 = sc_relative % 2;
                if (segment_idx >= 0 && segment_idx < NE_main_) {
                    if (block_col_in_seg == 0) {
                        B_segments_global_[segment_idx].block_row_L = block_r_NP;
                        B_segments_global_[segment_idx].value_L(local_r_2x2, local_c_2x2) = value;
                    }
                    else if (block_col_in_seg == NC_sys_ - 1) {
                        B_segments_global_[segment_idx].block_row_R = block_r_NP;
                        B_segments_global_[segment_idx].value_R(local_r_2x2, local_c_2x2) = value;
                    }
                }
            }
        }
    }
    //print_B_segments("Inner B_segs_global_", B_segments_global_);
    // PASS 2: Extract A and T, now that B_segments_global_ is known
    for (MKL_INT srJ = A_T_scalar_row_start; srJ < J_total_scalar_dim; ++srJ) {
        for (MKL_INT k = J_rowptr[srJ]; k < J_rowptr[srJ + 1]; ++k) {
            MKL_INT scJ = J_colind[k]; double value = J_val[k];
            MKL_INT sr_relative = srJ - A_T_scalar_row_start;
            MKL_INT segment_idx = (sr_relative / 2) / NC_sys_;
            MKL_INT block_r_in_seg = (sr_relative / 2) % NC_sys_;
            int local_r_2x2 = sr_relative % 2;

            if (scJ < D_B_scalar_row_limit) { // A quadrant
                MKL_INT target_block_col_NP = scJ / 2; int local_c_2x2 = scJ % 2;
                if (segment_idx >= 0 && segment_idx < NE_main_) {
                    if (block_r_in_seg == 0 && target_block_col_NP == B_segments_global_[segment_idx].block_row_L) {
                        A_rhs_systems_[2 * segment_idx][0](local_r_2x2, local_c_2x2) = value;
                    }
                    if (block_r_in_seg == NC_sys_ - 1 && target_block_col_NP == B_segments_global_[segment_idx].block_row_R) {
                        A_rhs_systems_[2 * segment_idx + 1][NC_sys_ - 1](local_r_2x2, local_c_2x2) = value;
                    }
                }
            }
            else { // T quadrant
                MKL_INT sc_relative = scJ - A_T_scalar_row_start;
                MKL_INT target_seg_idx = (sc_relative / 2) / NC_sys_;
                if (segment_idx == target_seg_idx && segment_idx < NE_main_) {
                    MKL_INT block_c_in_Ti = (sc_relative / 2) % NC_sys_; int local_c_2x2 = sc_relative % 2;
                    if (block_c_in_Ti == block_r_in_seg) D_T_systems_[segment_idx][block_r_in_seg](local_r_2x2, local_c_2x2) = value;
                    else if (block_c_in_Ti == block_r_in_seg - 1) L_systems_[segment_idx][block_r_in_seg](local_r_2x2, local_c_2x2) = value;
                    else if (block_c_in_Ti == block_r_in_seg + 1) U_systems_[segment_idx][block_r_in_seg](local_r_2x2, local_c_2x2) = value;
                }
            }
        }
    }

    // Extract Rp and Re from r_global
    if (r_global.size() < 2 * NP_main_) { std::cerr << "Error: r_global too small for Rp." << std::endl; return false; }
    for (MKL_INT i = 0; i < NP_main_; ++i) {
        Rp_global_blocks_[i](0) = r_global[2 * i + 0]; Rp_global_blocks_[i](1) = r_global[2 * i + 1];
    }
    MKL_INT Re_offset = 2 * NP_main_;
    if (r_global.size() < Re_offset + 2 * NCE_main_) { std::cerr << "Error: r_global too small for Re." << std::endl; return false; }
    for (MKL_INT s = 0; s < NE_main_; ++s) {
        for (MKL_INT br_Re = 0; br_Re < NC_sys_; ++br_Re) {
            MKL_INT idx0 = Re_offset + s * (2 * NC_sys_) + 2 * br_Re; MKL_INT idx1 = idx0 + 1;
            if (idx1 >= r_global.size()) { std::cerr << "Error: r_global index out of bounds for Re." << std::endl; return false; }
            Re_rhs_systems_[s][br_Re](0) = r_global[idx0]; Re_rhs_systems_[s][br_Re](1) = r_global[idx1];
        }
    }
    return true;
}

// --- Implementations of helper methods (condensed for brevity) ---
BlockTridiagonalSystemFactorization DBAT_SOLVER_2PHASE::factorize_block_tridiagonal_system(MKL_INT N, const std::vector<Matrix2x2>& L, const std::vector<Matrix2x2>& D, const std::vector<Matrix2x2>& U) { BlockTridiagonalSystemFactorization fact(N); if (N <= 0) { fact.factorization_succeeded = false; return fact; } if (L.size() != (size_t)N || D.size() != (size_t)N || U.size() != (size_t)N)throw std::invalid_argument("LDU size mismatch"); fact.L_sys = L; std::vector<Matrix2x2> Dp(N); Dp[0] = D[0]; fact.InvD_prime_sys[0] = Dp[0].inverse(); if (std::abs(Dp[0].val[0] * Dp[0].val[3] - Dp[0].val[1] * Dp[0].val[2]) < 1e-12) { fact.factorization_succeeded = false; return fact; } if (N > 1)fact.U_prime_sys[0] = Matrix2x2::multiply(fact.InvD_prime_sys[0], U[0]); for (MKL_INT i = 1; i < N; ++i) { Matrix2x2 LU = Matrix2x2::multiply(fact.L_sys[i], fact.U_prime_sys[i - 1]); Dp[i] = Matrix2x2::subtract(D[i], LU); fact.InvD_prime_sys[i] = Dp[i].inverse(); if (std::abs(Dp[i].val[0] * Dp[i].val[3] - Dp[i].val[1] * Dp[i].val[2]) < 1e-12) { fact.factorization_succeeded = false; return fact; } if (i < N - 1)fact.U_prime_sys[i] = Matrix2x2::multiply(fact.InvD_prime_sys[i], U[i]); } fact.factorization_succeeded = true; return fact; }
template<typename RBT, typename SBT> bool DBAT_SOLVER_2PHASE::solve_system_with_factorization(const BlockTridiagonalSystemFactorization& f, const std::vector<RBT>& r, bool b, std::vector<SBT>& s) { if (!f.factorization_succeeded || f.N_sys_blocks != r.size())return false; MKL_INT N = f.N_sys_blocks; if (N <= 0) { s.clear(); return true; }s.assign(N, SBT()); std::vector<SBT>Ap(N); Ap[0] = Matrix2x2::multiply(f.InvD_prime_sys[0], r[0]); for (MKL_INT i = 1; i < N; ++i) { if (b && i < N - 1)Ap[i] = SBT(); else { SBT LA = Matrix2x2::multiply(f.L_sys[i], Ap[i - 1]); SBT t = SBT::subtract(r[i], LA); Ap[i] = Matrix2x2::multiply(f.InvD_prime_sys[i], t); } }s[N - 1] = Ap[N - 1]; for (MKL_INT i = N - 2; i >= 0; --i) { SBT UX = Matrix2x2::multiply(f.U_prime_sys[i], s[i + 1]); s[i] = SBT::subtract(Ap[i], UX); }return true; }
Z_Matrix DBAT_SOLVER_2PHASE::compute_Z_matrix_optimized(const std::vector<Matrix2x2>& Dd, const std::vector<B_Segment_Info>& Bd, const std::vector<E_Segment_Info>& Ed, MKL_INT cNP, MKL_INT cNC, MKL_INT cNE) { Z_Matrix Zt(cNP); for (MKL_INT i = 0; i < cNP; ++i)Zt.diagonal_blocks[i] = Dd[i]; for (MKL_INT s = 0; s < cNE; ++s) { const auto& bs = Bd[s]; const auto& es = Ed[s]; if (es.block_col_L != bs.block_row_L || es.block_col_R != bs.block_row_R || es.values_L.size() != (size_t)cNC || es.values_R.size() != (size_t)cNC || bs.block_row_L < 0 || bs.block_row_L >= cNP || bs.block_row_R < 0 || bs.block_row_R >= cNP)continue; const Matrix2x2& vLB = bs.value_L; const Matrix2x2& vRB = bs.value_R; const Matrix2x2& ETL = es.values_L[0]; const Matrix2x2& ETR = es.values_R[0]; const Matrix2x2& EBL = es.values_L[cNC - 1]; const Matrix2x2& EBR = es.values_R[cNC - 1]; Matrix2x2 PLL = Matrix2x2::multiply(vLB, ETL); Matrix2x2 PLR = Matrix2x2::multiply(vLB, ETR); Matrix2x2 PRL = Matrix2x2::multiply(vRB, EBL); Matrix2x2 PRR = Matrix2x2::multiply(vRB, EBR); Matrix2x2& ZLL = Zt.get_block_for_accumulation(bs.block_row_L, es.block_col_L); ZLL = Matrix2x2::subtract(ZLL, PLL); Matrix2x2& ZRR = Zt.get_block_for_accumulation(bs.block_row_R, es.block_col_R); ZRR = Matrix2x2::subtract(ZRR, PRR); if (bs.block_row_L != es.block_col_R) { Matrix2x2& ZLR = Zt.get_block_for_accumulation(bs.block_row_L, es.block_col_R); ZLR = Matrix2x2::subtract(ZLR, PLR); } else { ZLL = Matrix2x2::subtract(ZLL, PLR); }if (bs.block_row_R != es.block_col_L) { Matrix2x2& ZRL = Zt.get_block_for_accumulation(bs.block_row_R, es.block_col_L); ZRL = Matrix2x2::subtract(ZRL, PRL); } else { ZRR = Matrix2x2::subtract(ZRR, PRL); } }return Zt; }
bool DBAT_SOLVER_2PHASE::compute_h_Rp_minus_Bg(const std::vector<Vector2x1>& Rb, const std::vector<B_Segment_Info>& Bsd, const std::vector<Vector2x1>& gfb, MKL_INT cNP, MKL_INT cNC, MKL_INT cNE, std::vector<Vector2x1>& hrb) { MKL_INT cNCE = cNC * cNE; if (Rb.size() != (size_t)cNP || gfb.size() != (size_t)cNCE || Bsd.size() != (size_t)cNE || cNC < 1)return false; hrb = Rb; for (MKL_INT s = 0; s < cNE; ++s) { const auto& sg = Bsd[s]; MKL_INT brL = sg.block_row_L; const Matrix2x2& vLB = sg.value_L; MKL_INT gLi = s * cNC; if (gLi < 0 || gLi >= cNCE)return false; const Vector2x1& gLt = gfb[gLi]; Vector2x1 pL = Matrix2x2::multiply(vLB, gLt); if (brL < 0 || brL >= cNP)return false; hrb[brL] = Vector2x1::subtract(hrb[brL], pL); MKL_INT brR = sg.block_row_R; const Matrix2x2& vRB = sg.value_R; MKL_INT gRi = s * cNC + cNC - 1; if (gRi < 0 || gRi >= cNCE)return false; const Vector2x1& gRt = gfb[gRi]; Vector2x1 pR = Matrix2x2::multiply(vRB, gRt); if (brR < 0 || brR >= cNP)return false; hrb[brR] = Vector2x1::subtract(hrb[brR], pR); }return true; }
void DBAT_SOLVER_2PHASE::Z_Matrix_to_CSR(const Z_Matrix& Zi, MKL_INT cNP, std::vector<MKL_INT>& csr_r, std::vector<MKL_INT>& csr_c, std::vector<double>& csr_v) { MKL_INT Ns = 2 * cNP; csr_r.assign(Ns + 1, 0); csr_c.clear(); csr_v.clear(); std::vector<std::map<MKL_INT, double>>tr(Ns); for (MKL_INT br = 0; br < cNP; ++br) { const Matrix2x2& db = Zi.diagonal_blocks[br]; MKL_INT r0 = 2 * br, c0 = 2 * br; if (std::abs(db(0, 0)) > 1e-12)tr[r0][c0] += db(0, 0); if (std::abs(db(0, 1)) > 1e-12)tr[r0][c0 + 1] += db(0, 1); if (std::abs(db(1, 0)) > 1e-12)tr[r0 + 1][c0] += db(1, 0); if (std::abs(db(1, 1)) > 1e-12)tr[r0 + 1][c0 + 1] += db(1, 1); }for (const auto& e : Zi.off_diagonal_blocks) { MKL_INT br = e.first.first; MKL_INT bc = e.first.second; const Matrix2x2& odb = e.second; MKL_INT r0 = 2 * br, c0 = 2 * bc; if (std::abs(odb(0, 0)) > 1e-12)tr[r0][c0] += odb(0, 0); if (std::abs(odb(0, 1)) > 1e-12)tr[r0][c0 + 1] += odb(0, 1); if (std::abs(odb(1, 0)) > 1e-12)tr[r0 + 1][c0] += odb(1, 0); if (std::abs(odb(1, 1)) > 1e-12)tr[r0 + 1][c0 + 1] += odb(1, 1); }MKL_INT nnz = 0; csr_r[0] = 0; for (MKL_INT r = 0; r < Ns; ++r) { for (const auto& pcv : tr[r]) { csr_c.push_back(pcv.first); csr_v.push_back(pcv.second); nnz++; }csr_r[r + 1] = nnz; } }
bool DBAT_SOLVER_2PHASE::solve_Z_Xp_eq_h_pardiso(const Z_Matrix& Zi, const std::vector<Vector2x1>& hb, MKL_INT cNP, std::vector<Vector2x1>& Xsb) { MKL_INT Ns = 2 * cNP; std::vector<MKL_INT>Zr, Zc; std::vector<double>Zv; Z_Matrix_to_CSR(Zi, cNP, Zr, Zc, Zv); std::vector<MKL_INT>ia(Ns + 1); std::vector<MKL_INT>ja = Zc; std::vector<double>a = Zv; for (MKL_INT i = 0; i <= Ns; ++i)ia[i] = Zr[i] + 1; for (MKL_INT& ci : ja)ci += 1; std::vector<double>hf(Ns); for (MKL_INT i = 0; i < cNP; ++i) { hf[2 * i] = hb[i](0); hf[2 * i + 1] = hb[i](1); }std::vector<double>Xf(Ns, 0.0); MKL_INT mt = 11, nr = 1, ip[64], mxf, mn, ph, er, ms; for (int i = 0; i < 64; i++)ip[i] = 0; ip[0] = 1; ip[1] = 2; ip[3] = 0; ip[4] = 0; ip[5] = 0; ip[7] = 2; ip[9] = 13; ip[10] = 1; ip[12] = 1; ip[17] = -1; ip[18] = -1; ip[34] = 0; mxf = 1; mn = 1; ms = 0; er = 0; ph = 11; PARDISO(pt_, &mxf, &mn, &mt, &ph, &Ns, a.data(), ia.data(), ja.data(), nullptr, &nr, ip, &ms, nullptr, nullptr, &er); if (er != 0)return false; ph = 22; PARDISO(pt_, &mxf, &mn, &mt, &ph, &Ns, a.data(), ia.data(), ja.data(), nullptr, &nr, ip, &ms, nullptr, nullptr, &er); if (er != 0) { ph = -1; PARDISO(pt_, &mxf, &mn, &mt, &ph, &Ns, nullptr, ia.data(), ja.data(), nullptr, &nr, ip, &ms, nullptr, nullptr, &er); return false; }ph = 33; PARDISO(pt_, &mxf, &mn, &mt, &ph, &Ns, a.data(), ia.data(), ja.data(), nullptr, &nr, ip, &ms, hf.data(), Xf.data(), &er); if (er != 0) { ph = -1; PARDISO(pt_, &mxf, &mn, &mt, &ph, &Ns, nullptr, ia.data(), ja.data(), nullptr, &nr, ip, &ms, nullptr, nullptr, &er); return false; }ph = -1; PARDISO(pt_, &mxf, &mn, &mt, &ph, &Ns, nullptr, ia.data(), ja.data(), nullptr, &nr, ip, &ms, nullptr, nullptr, &er); if (er != 0) {/*std::cerr<<"PARDISO Err(-1): "<<er<<std::endl;*/ }Xsb.assign(cNP, Vector2x1()); for (MKL_INT i = 0; i < cNP; ++i) { Xsb[i](0) = Xf[2 * i]; Xsb[i](1) = Xf[2 * i + 1]; }return true; }

bool DBAT_SOLVER_2PHASE::compute_Xe_g_minus_EXp(const std::vector<Vector2x1>& gb, const std::vector<E_Segment_Info>& Esd, 
    const std::vector<Vector2x1>& Xpb, MKL_INT cNP, MKL_INT cNC, MKL_INT cNE, std::vector<Vector2x1>& Xrb) 
 { 
        MKL_INT cNCE = cNC * cNE; 
        if (gb.size() != (size_t)cNCE || Esd.size() != (size_t)cNE || Xpb.size() != (size_t)cNP || cNC < 1)
            return false; 
        Xrb = gb; 
        for (MKL_INT s = 0; s < cNE; ++s) { 
            const auto& es = Esd[s]; 
            if (es.block_col_L < 0 || es.block_col_L >= cNP || es.block_col_R < 0 || es.block_col_R >= cNP || es.values_L.size() != (size_t)cNC || es.values_R.size() != (size_t)cNC)
                continue; 
            const Vector2x1& XpL = Xpb[es.block_col_L]; 
            const Vector2x1& XpR = Xpb[es.block_col_R]; 
            for (MKL_INT r = 0; r < cNC; ++r) { 
                MKL_INT kX = s * cNC + r; if (kX < 0 || kX >= cNCE)
                    return false; 
                Vector2x1 t1 = Matrix2x2::multiply(es.values_L[r], XpL); 
                Vector2x1 t2 = Matrix2x2::multiply(es.values_R[r], XpR); 
                Vector2x1 EX = Vector2x1::add(t1, t2); 
                Xrb[kX] = Vector2x1::subtract(Xrb[kX], EX); 
            } 
        }
        return true; 
}






template <typename T>
void printvector(T vector, const std::string& name) {
    std::cout << name << ": [ ";
    for (const auto& val : vector) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}




















// --- Main Example (using the Solver class) ---
//int main() {
//    MKL_INT NP_main = 4;
//    MKL_INT NC_sys = 4;
//    MKL_INT NE_main = 5;
//
//    std::cout << "--- Solver Class Example ---" << std::endl;
//    std::cout << "NP_main: " << NP_main << ", NC_sys (for T_i): " << NC_sys << ", NE_main: " << NE_main << std::endl;
//    std::cout << "--------------------------" << std::endl;
//
//    DBAT_SOLVER_2PHASE solver(NP_main, NC_sys, NE_main);
//
//    MKL_INT J_scalar_dim = 2 * NP_main + 2 * NC_sys * NE_main;
//
//    std::vector<double> r_global(J_scalar_dim);
//
//    //std::cout << "  Main: Populating example global Jacobian J (CSR) and RHS r..." << std::endl;
//    //// --- Simplified but structured CSR generation for J ---
//    //// (This populates J_rowptr, J_colind, J_val, and r_global with example data)
//    //{
//    //    std::vector<std::map<MKL_INT, double>> J_temp_rows(J_scalar_dim);
//    //    std::vector<Matrix2x2> D_glob_example(NP_main);
//    //    D_glob_example[0] = { 1,0.1,0.2,1.5 }; D_glob_example[1] = { 11,10.1,10.2,11.5 }; D_glob_example[2] = { 21,20.1,20.2,21.5 }; D_glob_example[3] = { 3.1,6,17,15 };
//    //    std::vector<B_Segment_Info> B_seg_example(NE_main);
//    //    if (NE_main > 0) { B_seg_example[0].block_row_L = 0; B_seg_example[0].value_L = { 1,0.5,0.2,1.1 }; B_seg_example[0].block_row_R = 1; B_seg_example[0].value_R = { 0.3,-1.2,1.3,0.4 }; }
//    //    if (NE_main > 1) { B_seg_example[1].block_row_L = 1; B_seg_example[1].value_L = { 2,0.1,-0.1,2.1 }; B_seg_example[1].block_row_R = 2; B_seg_example[1].value_R = { 0.1,3,3.1,0.1 }; }
//    //    // ... (populate remaining B_seg_example as before) ...
//    //    std::vector<std::vector<Matrix2x2>> A_rhs_example(2 * NE_main, std::vector<Matrix2x2>(NC_sys));
//    //    if (NE_main > 0) { A_rhs_example[0][0] = { 6,-5,8,7 }; A_rhs_example[1][NC_sys - 1] = { -3,8,-7,12 }; }
//    //    if (NE_main > 1) { A_rhs_example[2][0] = { 16,-15,18,17 }; A_rhs_example[3][NC_sys - 1] = { -13,18,-17,112 }; }
//    //    // ... (populate remaining A_rhs_example as before) ...
//    //    std::vector<std::vector<Matrix2x2>> L_sys_example(NE_main, std::vector<Matrix2x2>(NC_sys));
//    //    std::vector<std::vector<Matrix2x2>> D_T_sys_example(NE_main, std::vector<Matrix2x2>(NC_sys));
//    //    std::vector<std::vector<Matrix2x2>> U_sys_example(NE_main, std::vector<Matrix2x2>(NC_sys));
//    //    if (NE_main > 0) { L_sys_example[0] = { {0,0,0,0},{6,7,-2,4},{-4,9,-7,2},{3,-1,7,-9} }; D_T_sys_example[0] = { {5,3,2,4},{-5,7,9,8},{5,7,2,4},{5,2,4,-6} }; U_sys_example[0] = { {9,3,-3,1},{3,-1,-2,5},{-5,3,7,6},{0,0,0,0} }; }
//    //    // ... (populate remaining T_i component examples as before) ...
//
//    //    for (MKL_INT br = 0; br < NP_main; ++br) { for (int lr = 0; lr < 2; ++lr)for (int lc = 0; lc < 2; ++lc) { if (std::abs(D_glob_example[br](lr, lc)) > 1e-12)J_temp_rows[2 * br + lr][2 * br + lc] += D_glob_example[br](lr, lc); } }
//    //    for (MKL_INT s = 0; s < NE_main; ++s) { const auto& bs = B_seg_example[s]; MKL_INT scl = 2 * NP_main + s * (2 * NC_sys); MKL_INT scr = 2 * NP_main + s * (2 * NC_sys) + 2 * (NC_sys - 1); for (int lr = 0; lr < 2; ++lr)for (int lc = 0; lc < 2; ++lc) { if (std::abs(bs.value_L(lr, lc)) > 1e-12)J_temp_rows[2 * bs.block_row_L + lr][scl + lc] += bs.value_L(lr, lc); if (std::abs(bs.value_R(lr, lc)) > 1e-12)J_temp_rows[2 * bs.block_row_R + lr][scr + lc] += bs.value_R(lr, lc); } }
//    //    MKL_INT A_J_row_offset = 2 * NP_main; for (MKL_INT s = 0; s < NE_main; ++s) { MKL_INT Jtcl = B_seg_example[s].block_row_L; MKL_INT Jtcr = B_seg_example[s].block_row_R; MKL_INT srJb = A_J_row_offset + s * (2 * NC_sys); const auto& At = A_rhs_example[2 * s][0]; const auto& Ab = A_rhs_example[2 * s + 1][NC_sys - 1]; for (int lr = 0; lr < 2; ++lr)for (int lc = 0; lc < 2; ++lc) { if (std::abs(At(lr, lc)) > 1e-12)J_temp_rows[srJb + lr][2 * Jtcl + lc] += At(lr, lc); if (std::abs(Ab(lr, lc)) > 1e-12)J_temp_rows[srJb + 2 * (NC_sys - 1) + lr][2 * Jtcr + lc] += Ab(lr, lc); } }
//    //    MKL_INT T_J_offset = 2 * NP_main; for (MKL_INT s = 0; s < NE_main; ++s) { const auto& Ls = L_sys_example[s]; const auto& Ds = D_T_sys_example[s]; const auto& Us = U_sys_example[s]; for (MKL_INT br_Ti = 0; br_Ti < NC_sys; ++br_Ti) { MKL_INT srJb = T_J_offset + s * (2 * NC_sys) + 2 * br_Ti; MKL_INT scJbd = T_J_offset + s * (2 * NC_sys) + 2 * br_Ti; for (int lr = 0; lr < 2; ++lr)for (int lc = 0; lc < 2; ++lc) { if (std::abs(Ds[br_Ti](lr, lc)) > 1e-12)J_temp_rows[srJb + lr][scJbd + lc] += Ds[br_Ti](lr, lc); }if (br_Ti > 0) { MKL_INT scJbL = T_J_offset + s * (2 * NC_sys) + 2 * (br_Ti - 1); for (int lr = 0; lr < 2; ++lr)for (int lc = 0; lc < 2; ++lc)if (std::abs(Ls[br_Ti](lr, lc)) > 1e-12)J_temp_rows[srJb + lr][scJbL + lc] += Ls[br_Ti](lr, lc); }if (br_Ti < NC_sys - 1) { MKL_INT scJbU = T_J_offset + s * (2 * NC_sys) + 2 * (br_Ti + 1); for (int lr = 0; lr < 2; ++lr)for (int lc = 0; lc < 2; ++lc)if (std::abs(Us[br_Ti](lr, lc)) > 1e-12)J_temp_rows[srJb + lr][scJbU + lc] += Us[br_Ti](lr, lc); } } }
//    //    MKL_INT J_nnz = 0; J_rowptr[0] = 0; for (MKL_INT r = 0; r < J_scalar_dim; ++r) { for (const auto& pcv : J_temp_rows[r]) { J_colind.push_back(pcv.first); J_val.push_back(pcv.second); J_nnz++; }J_rowptr[r + 1] = J_nnz; }
//    //    std::vector<Vector2x1>Rp_ex(NP_main); for (MKL_INT i = 0; i < NP_main; ++i)Rp_ex[i] = { (double)i * 10.0 + 100.0,(double)i * 10.0 + 101.0 }; for (MKL_INT i = 0; i < NP_main; ++i) { r_global[2 * i] = Rp_ex[i](0); r_global[2 * i + 1] = Rp_ex[i](1); }
//    //    MKL_INT Re_r_offset = 2 * NP_main; MKL_INT NCE_main_local = NC_sys * NE_main; std::vector<std::vector<Vector2x1>>Re_ex(NE_main, std::vector<Vector2x1>(NC_sys)); if (NE_main > 0)Re_ex[0] = { {-3,7},{5,8},{-2,9},{1,6} };
//    //    for (MKL_INT s = 0; s < NE_main; ++s) { for (MKL_INT br = 0; br < NC_sys; ++br) { r_global[Re_r_offset + s * (2 * NC_sys) + 2 * br] = Re_ex[s][br](0); r_global[Re_r_offset + s * (2 * NC_sys) + 2 * br + 1] = Re_ex[s][br](1); } }
//    //}
//    //std::cout << "  Solver: Example J and r populated." << std::endl;
//
//    std::vector<Vector2x1> Xp_final_solution;
//    std::vector<Vector2x1> Xe_final_solution;
//
//    std::cout << "\n--- Invoking Solver::solve() ---" << std::endl;
//
//
//    std::vector<double> J_val = { 1, 0.1, 1, 0.5, 3.3, 8.2, 9.2, 7.1, 0.2, 1.5, 0.2, 1.1, 1.4, 5.1, 2.9, 7.7, 11, 10.1, 0.3, -1.2, 2, 0.1, 10.2, 11.5, 1.3, 0.4, -0.1, 2.1, 21, 20.1, 0.1, 3, -0.7, 3.5, -0.6, 5.9, 20.2, 21.5, 3.1, 0.1, 5.5, 4.2, 8.1, 0.9, 3.1, 6, 0.2, 6, 0.6, -0.3, 17, 15, -4.3, 0.1, 2.3, 0.9, 6, -5, 5, 3, 9, 3, 8, 7, 2, 4, -3, 1, 6, 7, -5, 7, 3, -1, -2, 4, 9, 8, -2, 5, -4, 9, 5, 7, -5, 3, -7, 2, 2, 4, 7, 6, -3, 8, 3, -1, 5, 2, -7, 12, 7, -9, 4, -6, 16, -15, 15, 13, 19, 13, 18, 17, 12, 14, -13, 11, 16, 17, -15, 17, 13, -11, -12, 14, 19, 18, -12, 15, -14, 19, 15, 17, -15, 13, -17, 2, 12, 14, 17, 16, -13, 18, 13, -11, 15, 12, -17, 112, 17, -19, 14, -16, 26, -25, 25, 23, 29, 23, 28, 27, 22, 24, -23, 21, 26, 7, -25, 27, 23, -21, -2, 4, 29, 28, -22, 25, -4, 9, 25, 27, -25, 23, -7, 2, 22, 24, 27, 26, -23, 28, 3, -1, 25, 22, -27, 212, 7, -9, 24, -26, 36, -35, 35, 33, 39, 33, 38, 37, 32, 34, -33, 31, 36, 37, -35, 37, 33, -31, -32, 34, 39, 38, -32, 35, -34, 39, 35, 37, -35, 33, -37, 32, 32, 34, 37, 36, -33, 38, 33, -31, 35, 32, -37, 312, 37, -39, 34, -36, 46, -45, 45, 43, 49, 43, 48, 47, 42, 44, -43, 41, 46, 47, -45, 47, 43, -41, -42, 44, 49, 48, -42, 45, -44, 49, 45, 47, -45, 43, -47, 42, 42, 44, 47, 46, -43, 48, 43, -41, 45, 42, -47, 412, 47, -49, 44, -46 };
//
//    // The 0-indexed column of each non-zero value
//    std::vector<int> J_colind = { 0, 1, 8, 9, 38, 39, 40, 41, 0, 1, 8, 9, 38, 39, 40, 41, 2, 3, 14, 15, 16, 17, 2, 3, 14, 15, 16, 17, 4, 5, 22, 23, 24, 25, 46, 47, 4, 5, 22, 23, 24, 25, 46, 47, 6, 7, 30, 31, 32, 33, 6, 7, 30, 31, 32, 33, 0, 1, 8, 9, 10, 11, 0, 1, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 8, 9, 10, 11, 12, 13, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 2, 3, 12, 13, 14, 15, 2, 3, 12, 13, 14, 15, 2, 3, 16, 17, 18, 19, 2, 3, 16, 17, 18, 19, 16, 17, 18, 19, 20, 21, 16, 17, 18, 19, 20, 21, 18, 19, 20, 21, 22, 23, 18, 19, 20, 21, 22, 23, 4, 5, 20, 21, 22, 23, 4, 5, 20, 21, 22, 23, 4, 5, 24, 25, 26, 27, 4, 5, 24, 25, 26, 27, 24, 25, 26, 27, 28, 29, 24, 25, 26, 27, 28, 29, 26, 27, 28, 29, 30, 31, 26, 27, 28, 29, 30, 31, 6, 7, 28, 29, 30, 31, 6, 7, 28, 29, 30, 31, 6, 7, 32, 33, 34, 35, 6, 7, 32, 33, 34, 35, 32, 33, 34, 35, 36, 37, 32, 33, 34, 35, 36, 37, 34, 35, 36, 37, 38, 39, 34, 35, 36, 37, 38, 39, 0, 1, 36, 37, 38, 39, 0, 1, 36, 37, 38, 39, 0, 1, 40, 41, 42, 43, 0, 1, 40, 41, 42, 43, 40, 41, 42, 43, 44, 45, 40, 41, 42, 43, 44, 45, 42, 43, 44, 45, 46, 47, 42, 43, 44, 45, 46, 47, 4, 5, 44, 45, 46, 47, 4, 5, 44, 45, 46, 47 };
//
//    // The 0-indexed pointer to the start of each row
//    std::vector<int> J_rowptr = { 0, 8, 16, 22, 28, 36, 44, 50, 56, 62, 68, 74, 80, 86, 92, 98, 104, 110, 116, 122, 128, 134, 140, 146, 152, 158, 164, 170, 176, 182, 188, 194, 200, 206, 212, 218, 224, 230, 236, 242, 248, 254, 260, 266, 272, 278, 284, 290, 296 };
//    solver.test_print_extracted_data();
//    std::cout << r_global.size() << std::endl;
//    r_global = { 100, 101, 110, 111, 120, 121, 130, 131, -3, 7, 5, 8, -2, 9, 1, 6, -13, 17, 15, 18, -12, 19, 11, 16, -23, 27, 25, 28, -22, 29, 21, 26, -33, 37, 35, 38, -32, 39, 31, 36, -43, 47, 45, 48, -42, 49, 41, 46 };
//    if (solver.solve(J_rowptr.data(), J_colind.data(), J_val.data(), J_scalar_dim, J_scalar_dim, r_global)) {
//
//
//        std::cout << "\n--- Solver Succeeded ---" << std::endl;
//        Xp_final_solution = solver.get_Xp_solution();
//        Xe_final_solution = solver.get_Xe_solution();
//
//        print_block_vector("Final Xp Solution", Xp_final_solution);
//        std::cout << std::endl;
//        print_block_vector("Final Xe Solution", Xe_final_solution);
//    }
//    else {
//        std::cerr << "--- Solver Failed ---" << std::endl;
//        return 1;
//    }
//
//    return 0;
//}

/**************************************SOMTO VERSION 9... GEMINI'S OWN CORRECTED CODE*****************************************************/