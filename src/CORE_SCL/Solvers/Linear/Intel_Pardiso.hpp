#ifndef __INTEL_PARDISO_HPP__
#define __INTEL_PARDISO_HPP__

#include <cstring>
#include <vector>
#include <chrono>
#include <iostream>
#include "mkl_pardiso.h"
#include "CPR_CSR_Matrix.hpp"

namespace GENSOL {

  class Intel_Pardiso
  {
  public:
    typedef double                                     double_type;
    typedef MKL_INT                                    int_type;

    //typedef CSR_Matrix< int_type, double_type >        A_type;
    typedef CPR_CSR_Matrix< int_type, double_type > A_type;
    typedef std::vector< double_type >                 x_type;

    struct report_t
    {
        double   xnrm;
        bool     is_converged;
        bool     is_badsolution;
        int      ln_error_code;
        int      niter;
        std::chrono::milliseconds timing{ 0 };
        std::chrono::milliseconds extraction_timing{ 0 };
    };


  public:

    Intel_Pardiso( std::size_t _N, std::size_t _NNZ_MAX ) :
      pt(),
      maxfct(1),
      mnum(1),
      mtype(11),
      phase(13),
      n(int(_N)),
      idummy(0),
      nrhs(1),
      iparm(),
      msglvl(0),
      error(0),
      report{}
    {
      std::memset( pt, 0, sizeof(pt) );
      std::memset( iparm, 0, sizeof(iparm) );
      iparm[0]  = 1;        /* No default values for solver */
      iparm[1]  = 3;        /* 3: Fill-in reducing ordering */
      iparm[2]  = 0;        /* Reserved set to zero */
      iparm[3]  = 0;        /* 0: No iterative-direct algorithm */
      iparm[4]  = 0;        /* 0: No user fill-in reducing permutation */
      iparm[5]  = 0;        /* 0: Write solution on x */
      iparm[6]  = 0;        /* No output of iterative refinement progress */
      iparm[7]  = 2;        /* Maximum number of iterative refinement steps */
      iparm[9]  = 13;       /* Perturb the pivot elements with 1E-13 */
      iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
      iparm[11] = 0;        /* Do not solve transposed matrix*/
      iparm[12] = 1;        /* Maximum weighted matching (default for nonsymmetric) */
      iparm[17] = 1;        /* No Output: Number of non-zero values in the factor LU */
      iparm[18] = 1;        /* No Output: Mflops for LU factorization */
      iparm[19] = 0;        /* No Output: Numbers of CG Iterations */
      iparm[26] = 1;        /* 1: enable matrix checking */
      iparm[34] = 1;        /* 0: Zero based indexing */
    }

    ~Intel_Pardiso( )
    {
      phase = -1;
      double dsentinel = 0.0;
      PARDISO ( pt, &maxfct, &mnum, &mtype, &phase, 
		&n, &dsentinel, &idummy, &idummy, &idummy, 
		&nrhs, iparm, &msglvl, &dsentinel, &dsentinel, &error );
    }

    static int_type offset( ) { return 1; }

    report_t solve( A_type & _A, x_type & _x_, x_type & _b, bool rebuild = false, double dummy = 0)
    {
        (void)dummy;
        try {

            
            
            auto start = std::chrono::high_resolution_clock::now();
            n = _A.N();
            const int_type* p_rowptr = static_cast<const int_type*>(_A.rowptr().data());
            const int_type* p_colind = static_cast<const int_type*>(_A.colind().data());
            const auto& values = _A.value();
            _x_.resize(n, 0.0);
            PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
                &n, _A.value().data(), p_rowptr, p_colind,
                &idummy, &nrhs, iparm, &msglvl, _b.data(), _x_.data(), &error);
            //auto end = std::chrono::high_resolution_clock::now();

            //report.timing = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            //
            report.is_converged = (error == 0);
            report.ln_error_code = error;
            report.niter = 0;
            report.is_badsolution = false;
            for (std::size_t i = 0; i < _x_.size(); ++i) {
                report.xnrm += _x_[i] * _x_[i];
                if (!std::isfinite(_x_[i])) report.is_badsolution = true;
            }
            return report;
        }
      catch (const std::exception& ex) {
          std::cerr << "Fatal: " << ex.what() << std::endl;
      }
    }

  private:
    void     * pt[64];
    int_type   maxfct;
    int_type   mnum;
    int_type   mtype;
    int_type   phase;
    int_type   n;
    int_type   idummy;
    int_type   nrhs;
    int_type   iparm[64];
    int_type   msglvl;
    int_type   error;
    report_t   report;
  };

};


#endif
