#pragma once

#include "CSR_Matrix.hpp"
#include <vector>

namespace GENSOL{

  template< typename SOL >
  class SYMD_Precond_Solver
  {
  public:
    typedef double                                              double_type;
    typedef int                                                 int_type;

    typedef CSR_Matrix< double_type, int_type >                 A_type;
    typedef std::vector< double_type >                          x_type;
    typedef std::vector< std::size_t >                          i_type;

  public:
    //.............................  LIFECYCLE  ..........................//
    SYMD_Precond_Solver( std::size_t _N_MAX, std::size_t _NNZ_MAX ) : solver( _N_MAX, _NNZ_MAX )
    {
    };
   
    ~SYMD_Precond_Solver( )
    {
    };

    static int_type offset( ) { return 1; }

    //.............................  OPERATORS  .........................//
    int solve( const A_type &_A, x_type &_x, const x_type & _b )
    {
      mA = _A;
      mb = _b;
      mx = _x;
      N = _A.N();
      calculate_diags( );
      apply_scaling ( );
      int ierr = solver.solve( mA, mx, mb );
      unscale( );
      _x = mx;
      return ierr;
    }

  protected:

    void
    apply_scaling ( )
    {
      // apply row and column scaling
      for ( std::size_t r=0; r<N; ++r ) // for each row
	{
	  mb[ r ] /= mDiags[ r ];

	  const std::size_t inz1 = mA.rowptr()[r]  -mA.offset();
	  const std::size_t inz2 = mA.rowptr()[r+1]-mA.offset();
	  for ( std::size_t inz=inz1; inz<inz2; ++inz ) // for each nonzero in this row
	    {
	      const std::size_t c = mA.colind()[inz] - mA.offset();
	      mA.value()[inz] /= mDiags[r];
	      mA.value()[inz] /= mDiags[c];
	    }
	}
    }

    void
    calculate_diags( )
    {
      // initialize arrays
      mDiags.resize( N );
      for ( std::size_t r=0; r<N; ++r )
	{
	  const double dg = std::sqrt( std::fabs( mA( r, r ) ) );
	  if ( dg != 0 )
	    mDiags[r] =  dg;
	  else
	    mDiags[r] =  1.0;
	}
    }
    
    void
    unscale( )
    {
      for ( std::size_t r=0; r<N; ++r )
	mx[r] /= mDiags[r];
    }

  private:
    SOL                 solver;
    A_type              mA;
    x_type              mb;
    x_type              mx;
    x_type              mDiags;
    std::size_t         N;
  };

};

