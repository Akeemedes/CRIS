#pragma once

#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_rci.h"
#include "mkl_service.h"
#include <vector>
#include <cmath>
#include "CPR_CSR_Matrix.hpp"

namespace GENSOL {

    template< typename T_PRECOND >
    class Intel_Prec_GMRES
    {

    public:
        typedef double                                     double_type;
        typedef MKL_INT                                    int_type;
        typedef CPR_CSR_Matrix< int_type, double_type > A_type;
        //typedef CSR_Matrix< double_type, int_type >        A_type;
        typedef std::vector< double_type >                 x_type;

        typedef struct
        {
            bool     is_converged;
            bool     is_bad_solution;
            int      niter;
            int_type ln_error_code[5]; // init, prec, check, solve, get
            std::chrono::milliseconds timing;
        }                                                  report_t;

    public:
        //.............................  LIFECYCLE  ..........................//
        Intel_Prec_GMRES(std::size_t _N_MAX, std::size_t _NNZ_MAX) :
            mPreconditioner(_N_MAX, _NNZ_MAX),
            transA(SPARSE_OPERATION_NON_TRANSPOSE)
        {
            descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
            descrA.mode = SPARSE_FILL_MODE_UPPER;
            descrA.diag = SPARSE_DIAG_NON_UNIT;

            NRESTART = 150;
            double sz = (2.0 * NRESTART + 1.0) * _N_MAX + 0.5 * NRESTART * (NRESTART + 9.0) + 1.0;
            if (sz > sizeof(double) * 1.0e9) // 1GB
            {
                const double b = 4.0 * _N_MAX + 9.0;
                const double c = _N_MAX + 1.0 + sizeof(double) * 1.0e9;
                NRESTART = std::max(1, static_cast<int>(0.5 * (sqrt(b * b - 4.0 * c) - b)));
                sz = (2.0 * NRESTART + 1.0) * _N_MAX + 0.5 * NRESTART * (NRESTART + 9.0) + 1.0;
            }
            tmp_vec.resize(static_cast<std::size_t>(sz), 0.0);

            //std::cout << "Restart GMRES every " << NRESTART << " iterations" << std::endl;
        };

        ~Intel_Prec_GMRES()
        {
            //        mkl_sparse_destroy(csrA);
            MKL_Free_Buffers();
        };

        static int_type offset() { return 1; }

        //.............................  OPERATORS  .........................//
        // //SOMTO MODEIFIED PARAMETER LIST TO EXCLUDE ELLIPTIC_INFO
        //report_t solve(A_type& _A, x_type& _x, x_type& _b, const EllipticInfo)
        report_t solve(A_type& _A, x_type& _x, x_type& _b, bool rebuild = false )
        {
            auto start = std::chrono::high_resolution_clock::now();
			// Initialize report
            lnreport.niter = 0;
            lnreport.is_converged = false;
            N = _A.N();
            for (std::size_t i = 0; i < tmp_vec.size(); ++i) tmp_vec[i] = 0.0;
            bool is_good = set_initial_guess(_x);
            if (is_good) {
                is_good = initialize_solver(_A, _x, _b);
                if (is_good) {
                    is_good = setup_preconditioner(_A);
                    if (is_good) {
                        setup_dfgmres_options();
                        is_good = check_parameter_setup(_x, _b);
                        if (is_good) {
                            bool keep_iterating = true;
                            while (keep_iterating)
                            {
                                //		  std::cout << compute_residual_norm(_x,_b) << " " << std::flush;
                                dfgmres(&N, _x.data(), _b.data(), &RCIreq, ipar, dpar, tmp_vec.data());
                                switch (RCIreq)
                                {
                                case 0: // convereged
                                    keep_iterating = false;
                                    lnreport.is_converged = true;
                                    break;
                                case 1: // do M-V mutiply
                                    mkl_sparse_d_mv(transA, 1.0, csrA, descrA,
                                        &tmp_vec[ipar[21] - 1], 0.0, &tmp_vec[ipar[22] - 1]);
                                    break;
                                case 2: // performing extra convergence criteria
                                    std::cout << "Problem: RCI gmres should not ask for extra stopping criteria" << std::endl;
                                    break;
                                case 3: // apply preconditioner
                                    keep_iterating = mPreconditioner.apply(&tmp_vec[ipar[22] - 1], &tmp_vec[ipar[21] - 1]);
                                    break;
                                case 4: // check for near zero orthogonal vector up to rounding err
                                    if (dpar[6] < 1.0E-10) {
                                        keep_iterating = false;
                                        lnreport.is_converged = true;
                                    }
                                    break;
                                default: // failed
                                    keep_iterating = false;
                                    lnreport.is_converged = false;
                                    break;
                                } // RCI switch
                            } // end fgmres while

                            lnreport.ln_error_code[3] = RCIreq;
                            if (lnreport.is_converged)
                            {
                                ipar[12] = 0;
                                dfgmres_get(&N, _x.data(), _b.data(), &RCIreq, ipar, dpar, tmp_vec.data(),
                                    &lnreport.niter);
                                if (RCIreq == 0) {
                                    is_good = true;
                                    lnreport.is_converged = true;
                                }
                                else {
                                    is_good = false;
                                    lnreport.is_converged = false;
                                }
                                lnreport.ln_error_code[4] = RCIreq;
                            }
                            else
                            {
                                is_good = false;
                            }
                        }
                    } //option check
                } // preconditioner
            } // setup
            //        mkl_sparse_destroy(csrA);

            auto end = std::chrono::high_resolution_clock::now();

            lnreport.timing = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            return lnreport;
        }

    protected:

        bool set_initial_guess(x_type& _x)
        {
            bool is_success = true;
            for (int i = 0; i < _x.size(); ++i) _x[i] = 0.0;
            return is_success;
        }

        bool initialize_solver(A_type& _A, x_type& _x, x_type& _b)
        {
            bool is_success = true;
            csrA = NULL;
            mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, N, N,
                _A.rowptr().data(), _A.rowptr().data() + 1,
                _A.colind().data(), _A.value().data());
            dfgmres_init(&N, _x.data(), _b.data(), &RCIreq, ipar, dpar, tmp_vec.data());
            if (RCIreq != 0) is_success = false;
            lnreport.ln_error_code[0] = RCIreq;
            return is_success;
        }

        bool setup_preconditioner(A_type& _A)
        {
            bool is_success = mPreconditioner.setup_A(_A, ipar, dpar, lnreport.ln_error_code[1]);
            return is_success;
        }

        bool check_parameter_setup(x_type& _x, x_type& _b)
        {
            bool is_success = true;
            dfgmres_check(&N, _x.data(), _b.data(), &RCIreq, ipar, dpar, tmp_vec.data());
            if (RCIreq != 0) is_success = false;
            lnreport.ln_error_code[2] = RCIreq;
            return is_success;
        }

        void setup_dfgmres_options()
        {
            ipar[1] = 6; // out to screen

            //SOMTO ADDED ipar[4] FOR MAX #ITERATIONS FROM DEFAULT OF MIN(150,n) TO 500
            ipar[4] = 500;
            //ipar[4] = 1000;

            ipar[5] = 1; // no error output
            ipar[6] = 0; // 0 means no warnings output, otherwise 1

            ipar[7] = 1; // auto test for max iter
            ipar[8] = 1; // auto test for residual criterion
            ipar[9] = 0; // no manual tests
            ipar[10] = 1; // use a preconditioner
            ipar[11] = 0; // ask to test for zero orthogonal vector norm to eps_mach
            ipar[14] = NRESTART; // restart every NRESTART iterations

            //SOMTO changed
            //dpar[0] = 1.0E-6; // EndIF ||A x_i - b|| <= dpar[0] * ||A x_0 - b ||
            dpar[0] = 1.0E-10; // EndIF ||A x_i - b|| <= dpar[0] * ||A x_0 - b ||
            //dpar[0] = 1.0E-12; // EndIF ||A x_i - b|| <= dpar[0] * ||A x_0 - b ||
            //dpar[0] = 1.0E-8; // EndIF ||A x_i - b|| <= dpar[0] * ||A x_0 - b ||
        }

        double compute_residual_norm(x_type& _x, x_type& _rhs)
        {
            // resize iterate and residual vectors and set zero
            iterate.resize(N, 0.0);
            residual.resize(N, 0.0);
            int_type iterno;
            ipar[12] = 1;
            /* Get the current FGMRES solution in the vector b[N] */
            dfgmres_get(&N, _x.data(), iterate.data(), &RCIreq, ipar, dpar, tmp_vec.data(), &iterno);
            /* Compute the current true residual via Intel(R) MKL (Sparse) BLAS routines */
            mkl_sparse_d_mv(transA, 1.0, csrA, descrA, iterate.data(), 0.0, residual.data());
            double dvar = -1.0E0;
            int i = 1;
            daxpy(&N, &dvar, _rhs.data(), &i, residual.data(), &i);
            double rnorm = dnrm2(&N, residual.data(), &i);
            return rnorm;
        }

    private:
        T_PRECOND     mPreconditioner;
        sparse_operation_t    transA;
        int_type      ipar[128];
        double_type   dpar[128];
        struct matrix_descr   descrA;
        sparse_matrix_t       csrA;
        report_t      lnreport;
        int_type      RCIreq;
        x_type        tmp_vec;
        int_type      N; // matsize
        std::size_t   NRESTART;
        std::vector<double> iterate;
        std::vector<double> residual;
    };

};











//#pragma once
//
//#include "mkl_rci.h"
//
//#include <vector>
//#include "CSR_Matrix.hpp"
//
//namespace GENSOL{
//
//  template< typename T_PRECOND >
//  class Intel_Prec_GMRES
//  {
//
//  public:
//    typedef double                                     double_type;
//    typedef MKL_INT                                    int_type;
//    typedef CSR_Matrix< double_type, int_type >        A_type;
//    typedef std::vector< double_type >                 x_type;
//   
//    typedef struct 
//    {
//      bool     is_converged;
//      int      ln_error_code;
//      int      niter;
//      std::chrono::milliseconds timing;
//    }                                                  report_t;
//
//  public:
//    //.............................  LIFECYCLE  ..........................//
//    Intel_Prec_GMRES( std::size_t _N_MAX, std::size_t _NNZ_MAX ) : 
//      mPreconditioner( _N_MAX, _NNZ_MAX ),
//      N( _N_MAX ),
//      RCIreq( ),
//      CVAR('N')
//    { 
//      if ( _N_MAX > 100000 ) 
//	{
//	  do_restart = true;
//	  double NTMP = _N_MAX;
//	  NRESTART   = 2 * ( sqrt( NTMP*(NTMP+4) + 15055967.56 ) - NTMP ) - 4.5;
//	  NRESTART   = (NRESTART < 3 ? 3 : NRESTART );
//	}
//      else
//	{
//	  NRESTART = ( N > 150 ? 150 : N );
//	}
//      tmp_vec.resize( (2*NRESTART+1)*N+NRESTART*(NRESTART+9)/2+1  );
//    };
//   
//    ~Intel_Prec_GMRES( )
//    {
//    };
//
//    static int_type offset( ) { return 1; }
//    
//    //.............................  OPERATORS  .........................//
//    report_t solve( A_type &_A, x_type &_x, x_type & _b, bool rebuild = false, const EllipticInfo &ainfo=EllipticInfo() )
//    {
//      N = _A.N();
//
//      for ( std::size_t i=0; i<_x.size(); ++i ) _x[i]= 0;
//
//      int result = initialize_solver( _x, _b );
//      if (result >= 0 )
//	{
//	  result = mPreconditioner.setup_A( _A  );
//	  if ( result >= 0 )
//	    {
//	      bool keep_going = true;
//	      while (keep_going)
//		{
//
//		  dfgmres (&N, _x.data(), _b.data(), &RCIreq, ipar, dpar, tmp_vec.data() );
//		  switch (RCIreq)
//		    {
//		    case 0:
//		      keep_going = false;
//		      break;
//		    case 1: 
//		      mkl_dcsrgemv (&CVAR, 
//				    &N, 
//				    _A.value().data(), _A.rowptr().data(), _A.colind().data(), 
//				    &tmp_vec[ipar[21] - 1 ], &tmp_vec[ipar[22] - 1 ]);
//		      break;
//		    case 3:
//		      int temp=mPreconditioner.apply( &tmp_vec[ipar[22]-1], &tmp_vec[ipar[21]-1] );
//		      break;
//		    default:
//		      keep_going = false;
//		      break;
//		    } //switch
//		} // while
//	      result = RCIreq;
//	    }// ilu0 ok
//	} // gmres ok
//
//      int_type itercount;
//      ipar[12] = 0;
//      dfgmres_get (&N, _x.data(), _b.data(), &RCIreq, ipar, dpar, tmp_vec.data(), &itercount);
//      
//      lnreport.is_converged  = (result >= 0);
//      lnreport.ln_error_code = result;
//      lnreport.niter         = itercount;
//      return lnreport;
//    }
//
//  protected:
//
//    int initialize_solver( x_type &_x, x_type & _b  )
//    {
//      int result = -1;
//      dfgmres_init ( &N, _x.data(), _b.data(), &RCIreq, ipar, dpar, tmp_vec.data() );
//      if (RCIreq == 0)
//	{
//	  setup_options( );
//	  dfgmres_check (&N, _x.data(), _b.data(), &RCIreq, ipar, dpar, tmp_vec.data() );
//	  if (RCIreq != -1100) result = 0;
//	}
//      return result;
//    }
//
//    void setup_options( )
//    {
//      
//      ipar[ 1] = 6; // out to screen
//      ipar[ 5] = 1; // no error output
//      ipar[ 6] = 1; // no error output
//
//      ipar[ 7] = 1; // auto test for max iter
//      ipar[ 8] = 1; // auto test for residual criterion
//      ipar[ 9] = 0; // no manual tests
//      ipar[10] = 1; // use a preconditioner
//      ipar[11] = 1; // auto test for zero orthogonal vector norm
//      ipar[14] = NRESTART;
//    }
//      
//  private:
//    T_PRECOND     mPreconditioner;
//    report_t      lnreport;
//    int_type      N;
//    int_type      RCIreq;
//    char          CVAR;
//    x_type        tmp_vec;
//    int_type      ipar [ 128 ];
//    double_type   dpar [ 128 ];
//    std::size_t   NRESTART;
//    bool          do_restart;
//  };
//
//};
//
