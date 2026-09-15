#pragma once
//#pragma once

#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_rci.h"
#include "mkl_service.h"
#include <vector>
#include "CPR_CSR_Matrix.hpp"

namespace GENSOL {

    class Intel_ILU0
    {
    public:
        typedef double                                     double_type;
        typedef MKL_INT                                    int_type;

        //typedef CSR_Matrix< double_type, int_type >        A_type;
        typedef CPR_CSR_Matrix< int_type, double_type > A_type;
        typedef std::vector< double_type >                 x_type;

    public:
        //.............................  LIFECYCLE  ..........................//
        Intel_ILU0(std::size_t _N_MAX, std::size_t _NNZ_MAX) :
            transA(SPARSE_OPERATION_NON_TRANSPOSE),
            trvec(_N_MAX, 0.0), bilu0(_NNZ_MAX, 0.0)
        {
        };

        ~Intel_ILU0()
        {
            // mkl_sparse_destroy(csrL);
        };

        static int_type offset() { return 1; }

        //.............................  OPERATORS  .........................//
        bool setup_A(A_type& _A, int_type* ipar, double* dpar, int_type& RCIreq_)
        {
            bool is_good = true;
            mN = _A.N();
            trvec.resize(_A.value().size(), 0.0);
            ipar[1] = 6;       // output of error messages to the screen,
            ipar[5] = 1;       //  allow output of errors,
            ipar[30] = 1;      // relax the zero diagonal requirement
            dpar[30] = 1.E-20; // any diag entry smaller than this is set to...
            dpar[31] = 1.E-16; // this for preconditioning purposes

            dcsrilu0(&mN, _A.value().data(), _A.rowptr().data(), _A.colind().data(),
                bilu0.data(), ipar, dpar, &RCIreq_);
            mkl_sparse_d_create_csr(&csrL, SPARSE_INDEX_BASE_ONE, mN, mN,
                _A.rowptr().data(), _A.rowptr().data() + 1, _A.colind().data(),
                bilu0.data());
            if (RCIreq_ != 0) is_good = false;
            return is_good;
        }

        bool apply(double_type* _x, double_type* _b)
        {
            descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
            descrL.mode = SPARSE_FILL_MODE_LOWER;
            descrL.diag = SPARSE_DIAG_UNIT;
            mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, _b, trvec.data());

            descrL.mode = SPARSE_FILL_MODE_UPPER;
            descrL.diag = SPARSE_DIAG_NON_UNIT;
            mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, trvec.data(), _x);

            return true;
        }

    private:
        sparse_operation_t  transA;
        std::vector<double> trvec;
        std::vector<double> bilu0;
        sparse_matrix_t     csrL;
        int_type            mN;
        struct matrix_descr descrL;
    };

};






//#pragma once
//
//#include "mkl.h"
//#include "mkl_rci.h"
//#include "CSR_Matrix.hpp"
//#include <vector>
//#include <iostream> // For error message
//
//namespace GENSOL {
//
//    class Intel_ILU0
//    {
//    public:
//        typedef double      double_type;
//        typedef MKL_INT     int_type;
//        typedef CSR_Matrix<double_type, int_type> A_type;
//        typedef std::vector<double_type> x_type;
//
//    public:
//        //.............................  LIFECYCLE  ..........................//
//        Intel_ILU0(std::size_t _N_MAX, std::size_t _NNZ_MAX) :
//            bilu0(_NNZ_MAX),
//            tmp_vec(_N_MAX),
//            mN(_N_MAX),
//            bilu0_handle(nullptr) // ADDED: Initialize handle to null
//        {
//            for (std::size_t i = 0; i < 128; ++i)
//            {
//                ipar[i] = 0;
//                dpar[i] = 0.0;
//            }
//            ipar[1] = 6;      // if error messages allowed display to screen
//            ipar[5] = 0;      // no error messages
//            ipar[30] = 1;     // check for zero pivot
//            dpar[30] = 1.0e-16; // this is considered zero
//            dpar[31] = 1.0e-10; // replace zero with this number
//        };
//
//        ~Intel_ILU0()
//        {
//            // ADDED: Clean up the matrix handle to prevent memory leaks
//            if (bilu0_handle) {
//                mkl_sparse_destroy(bilu0_handle);
//            }
//        };
//
//        static int_type offset() { return 1; }
//
//        //.............................  OPERATORS  .........................//
//        int setup_A(A_type& _A)
//        {
//            int ierr = 0;
//            mN = _A.N();
//            tmp_vec.resize(mN);
//            bilu0.resize(_A.value().size());
//
//            // Note: dcsrilu0 computes the factors but does not respect the base (0 or 1).
//            // It uses the raw pointers. The CSR matrix handle created later will define the base.
//            dcsrilu0(&mN,
//                _A.value().data(), _A.rowptr().data(), _A.colind().data(),
//                bilu0.data(), ipar, dpar, &ierr);
//
//            if (ierr != 0)
//            {
//                std::cout << "dcsrilu0 gave error= " << ierr << std::endl;
//                return -1;
//            }
//
//            // --- CHANGED SECTION: Create a handle for the ILU0 factors ---
//
//            // 1. Destroy previous handle if it exists
//            if (bilu0_handle) {
//                mkl_sparse_destroy(bilu0_handle);
//                bilu0_handle = nullptr;
//            }
//
//            // 2. Create CSR handle for the newly computed ILU0 matrix
//            // We use SPARSE_INDEX_BASE_ONE because your GMRES solver does.
//            mkl_sparse_d_create_csr(&bilu0_handle, SPARSE_INDEX_BASE_ONE, mN, mN,
//                _A.rowptr().data(), _A.rowptr().data() + 1,
//                _A.colind().data(), bilu0.data());
//
//            // 3. Optimize the handle for subsequent triangular solves (the "Inspector" step)
//            mkl_sparse_optimize(bilu0_handle);
//            // --- END CHANGED SECTION ---
//
//            return ierr;
//        }
//
//        // The 'apply' method is completely rewritten to use the modern API.
//        int apply(double_type* _x, double_type* _b)
//        {
//            // --- REWRITTEN METHOD ---
//            struct matrix_descr bilu0_descr;
//
//            // Step 1: Solve L*y = b (Forward solve)
//            // L is lower triangular with a unit diagonal.
//            bilu0_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
//            bilu0_descr.mode = SPARSE_FILL_MODE_LOWER;
//            bilu0_descr.diag = SPARSE_DIAG_UNIT;
//
//            mkl_sparse_d_trsv(
//                SPARSE_OPERATION_NON_TRANSPOSE, // No transpose
//                1.0,                            // alpha (ignored for trsv)
//                bilu0_handle,                   // The ILU0 matrix handle
//                bilu0_descr,                    // Matrix descriptor
//                _b,                             // Input vector
//                tmp_vec.data()                  // Output vector 'y'
//            );
//
//            // Step 2: Solve U*x = y (Backward solve)
//            // U is upper triangular with a non-unit diagonal.
//            bilu0_descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
//            bilu0_descr.mode = SPARSE_FILL_MODE_UPPER;
//            bilu0_descr.diag = SPARSE_DIAG_NON_UNIT;
//
//            mkl_sparse_d_trsv(
//                SPARSE_OPERATION_NON_TRANSPOSE, // No transpose
//                1.0,                            // alpha
//                bilu0_handle,                   // The ILU0 matrix handle
//                bilu0_descr,                    // Matrix descriptor
//                tmp_vec.data(),                 // Input vector 'y'
//                _x                              // Final output vector 'x'
//            );
//
//            return 0; // Success
//        }
//
//        int solve(A_type& _A, x_type& _x, x_type& _b)
//        {
//            int ierr = setup_A(_A);
//            if (ierr == 0)
//            {
//                apply(_x.data(), _b.data());
//            }
//            return ierr;
//        }
//
//    private:
//        std::vector<double_type> bilu0;
//        // bilu0_rowptr and bilu0_colind were removed as they were redundant.
//        // The original matrix's structure is used directly to create the handle.
//        std::vector<double_type> tmp_vec;
//        int_type mN;
//        int_type ipar[128];
//        double_type dpar[128];
//
//        // ADDED: Member variables for the modern API
//        sparse_matrix_t bilu0_handle;
//    };
//
//};











//#pragma once
//
//#include "mkl.h"
//#include "mkl_rci.h"
//#include "CSR_Matrix.hpp"
//#include <vector>
//
//namespace GENSOL{
//
//  class Intel_ILU0
//  {
//  public:
//    typedef double                                     double_type;
//    typedef MKL_INT                                    int_type;
//
//    typedef CSR_Matrix< double_type, int_type >        A_type;
//    typedef std::vector< double_type >                 x_type;
//   
//  public:
//    //.............................  LIFECYCLE  ..........................//
//    Intel_ILU0( std::size_t _N_MAX, std::size_t _NNZ_MAX ) :
//      bilu0( _NNZ_MAX ),
//      bilu0_rowptr( _N_MAX + 1 ),
//      bilu0_colind( _NNZ_MAX ),
//      tmp_vec( _N_MAX ),
//      mN( _N_MAX )
//    {
//      for ( std::size_t i=0; i<128; ++i )
//	{
//	  ipar[i] = 0;
//	  dpar[i] = 0.0;
//	}
//      ipar[ 1] = 6;       // if error messages allowed display to screen
//      ipar[ 5] = 0;       // no error messages
//      ipar[30] = 1;       // check for zero pivot
//      dpar[30] = 1.0e-16; // this is condered zero
//      dpar[31] = 1.0e-10; // replace zero with this number
//    };
//   
//    ~Intel_ILU0( )
//    {
//    };
//
//    static int_type offset( ) { return 1; }
//
//    //.............................  OPERATORS  .........................//
//    int setup_A( A_type &_A )
//    {
//      int ierr = 0;
//
//      mN       = _A.N();
//
//      tmp_vec.resize( mN );
//      bilu0.resize( _A.value().size()  );
//      bilu0_rowptr = _A.rowptr();
//      bilu0_colind = _A.colind();
//
//      dcsrilu0 (&mN, 
//		_A.value().data(), bilu0_rowptr.data(), bilu0_colind.data(), 
//		bilu0.data(), ipar, dpar, &ierr);
//
//      if (ierr != 0) 
//	{
//	  std::cout << "dcsrilu0 gave error= " << ierr <<std::endl;
//	  ierr = -1;
//	}
//
//      return ierr;
//    }
//
//    int apply( double_type *_x, double_type *_b )
//    {
//      int  ierr = 0;
//      char cvar = 'N'; // no transpose
//
//      char cvar1 = 'L'; // invert lower triangular part
//      char cvar2 = 'U'; // assume unit lower traingular
//      mkl_dcsrtrsv ( &cvar1, &cvar, &cvar2, 
//		     &mN, bilu0.data(), bilu0_rowptr.data(), bilu0_colind.data(),
//		     _b, tmp_vec.data());
//
//      cvar1 = 'U'; // take upper traingular part
//      cvar2 = 'N'; // it is not unit triangular
//      mkl_dcsrtrsv ( &cvar1, &cvar, &cvar2, 
//		     &mN, bilu0.data(), bilu0_rowptr.data(), bilu0_colind.data(),
//		     tmp_vec.data(), _x );
//      return ierr;
//    }
//
//    int solve( A_type &_A, x_type &_x, x_type & _b )
//    {
//      int ierr = setup_A( _A );
//      apply( _x.data( ), _b.data( ) );
//      return ierr;
//    }
//
//  private:
//    std::vector<double_type> bilu0;
//    std::vector<int_type>    bilu0_rowptr;
//    std::vector<int_type>    bilu0_colind;
//    std::vector<double_type> tmp_vec;
//    int_type      mN;
//    int_type      ipar [ 128  ];
//    double_type   dpar [ 128  ];
//  };
//
//};

