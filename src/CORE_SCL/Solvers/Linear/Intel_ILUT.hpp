#pragma once

#include "mkl.h"
#include "mkl_rci.h"
#include "CSR_Matrix.hpp"
#include <vector>

namespace GENSOL{

  class Intel_ILUT
  {
  public:
    typedef double                                     double_type;
    typedef MKL_INT                                    int_type;

    typedef CSR_Matrix< double_type, int_type >        A_type;
    typedef std::vector< double_type >                 x_type;
   
  public:
    //.............................  LIFECYCLE  ..........................//
    Intel_ILUT( std::size_t _N_MAX, std::size_t _NNZ_MAX ) :
      p_tmp_vec( new double_type[ _N_MAX ] ),
      mN( _N_MAX )
    {
      ILUTOL = 1.0e-6; // threshold for ILUT algorithm
      MAXFIL = 7;      // half bandwidth of desired preconditioner (minimum 3!!)

      const std::size_t NFIL = (2*MAXFIL+1)*mN - MAXFIL*(MAXFIL+1) + 1;
      p_bilut        = new double_type[ NFIL ];
      p_bilut_rowptr = new int_type [ _N_MAX + 1 ];
      p_bilut_colind = new int_type [ NFIL ];

      for ( std::size_t i=0; i<128; ++i )
	{
	  ipar[i] = 0;
	  dpar[i] = 0.0;
	}
      ipar[ 1] = 6;       // if error messages allowed display to screen
      ipar[ 5] = 0;       // no error messages
      ipar[30] = 1;       // whether to check for |d_i| < TOL * ||A(i,:)||
      dpar[30] = ILUTOL;  // if diag is small, set it to d_i = dpar[30] * ||A(i,:)||
    };
   
    ~Intel_ILUT( )
    {
      delete [] p_bilut_colind;
      delete [] p_bilut_rowptr;
      delete [] p_bilut;
      delete [] p_tmp_vec;    
    };

    static int_type offset( ) { return 1; }

    //.............................  OPERATORS  .........................//
    int setup_A( A_type &_A )
    {
      int ierr = 0;

      mN       = _A.N();

      dcsrilut (&mN, 
		_A.value().data(), _A.rowptr().data(), _A.colind().data(), 
		p_bilut, p_bilut_rowptr, p_bilut_colind, 
		&ILUTOL, &MAXFIL,
		ipar, dpar, &ierr);
      if (ierr != 0) ierr = -1;

      return ierr;
    }

    int apply( double_type *_x, double_type *_b )
    {
      int  ierr = 0;
      char cvar = 'N'; // no transpose

      char cvar1 = 'L'; // invert lower triangular part
      char cvar2 = 'U'; // assume unit lower traingular
      mkl_dcsrtrsv ( &cvar1, &cvar, &cvar2, 
		     &mN, p_bilut, p_bilut_rowptr, p_bilut_colind,
		     _b, p_tmp_vec);

      cvar1 = 'U'; // take upper traingular part
      cvar2 = 'N'; // it is not unit triangular
      mkl_dcsrtrsv ( &cvar1, &cvar, &cvar2, 
		     &mN, p_bilut, p_bilut_rowptr, p_bilut_colind,
		     p_tmp_vec, _x );
      return ierr;
    }

    int solve( A_type &_A, x_type &_x, x_type & _b )
    {
      int ierr = setup_A( _A );
      apply( _x.data( ), _b.data( ) );
      return ierr;
    }

  private:
    double_type * p_tmp_vec;
    int_type      mN;
    double_type   ILUTOL;
    int_type      MAXFIL;
    double_type * p_bilut;
    int_type    * p_bilut_rowptr;
    int_type    * p_bilut_colind;

    int_type      ipar [ 128  ];
    double_type   dpar [ 128  ];
  };

};

