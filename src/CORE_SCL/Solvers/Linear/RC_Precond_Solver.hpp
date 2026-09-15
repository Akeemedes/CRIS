#pragma once

#include "CSR_Matrix.hpp"
#include <vector>

namespace GENSOL{

  template< typename SOL >
  class RC_Precond_Solver
  {
  public:
    typedef double                                              double_type;
    typedef int                                                 int_type;

    typedef CSR_Matrix< double_type, int_type >                 A_type;
    typedef std::vector< double_type >                          x_type;
    typedef std::vector< std::size_t >                          i_type;

  public:
    //.............................  LIFECYCLE  ..........................//
    RC_Precond_Solver( std::size_t _N_MAX, std::size_t _NNZ_MAX ) : solver( _N_MAX, _NNZ_MAX )
    {
    };
   
    ~RC_Precond_Solver( )
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

      rc_algorithm( );
      int ierr = solver.solve( mA, mx, mb );
      c_unscale( );
      _x = mx;
      return ierr;
    }

  protected:

    void
    rc_algorithm( )
    {

      calculate_rc_norms( );
      calculate_rc_errors( );

      mDR.resize( N );
      mDL.resize( N );
      for ( std::size_t i=0; i<N; ++i ) 
	mDL[i] = mDR[i] = 1.0;

      std::size_t   ITERCOUNT = 0;
      const double  EPS = 0.1;

      std::cout << ITERCOUNT << "\t" << row_err << "\t" << col_err << std::endl;
      while ( (ITERCOUNT < 2) && ( (row_err > EPS) || (col_err > EPS) ) )
	{
	  apply_scaling ( );
	  calculate_rc_norms( );
	  calculate_rc_errors( );
	  ++ITERCOUNT;
	  std::cout << ITERCOUNT << "\t" << row_err << "\t" << col_err << std::endl;
	}
    }

    void
    apply_scaling ( )
    {
      // apply row and column scaling
      for ( std::size_t r=0; r<N; ++r ) // for each row
	{
	  mb[ r ] /= std::sqrt( nrm_row[ r ] );

	  const std::size_t inz1 = mA.rowptr()[r]  -mA.offset();
	  const std::size_t inz2 = mA.rowptr()[r+1]-mA.offset();
	  for ( std::size_t inz=inz1; inz<inz2; ++inz ) // for each nonzero in this row
	    {
	      const std::size_t c = mA.colind()[inz] - mA.offset();
	      mA.value()[inz] /= std::sqrt( nrm_row[r] );
	      mA.value()[inz] /= std::sqrt( nrm_col[c] );
	    }
	  mDL[r] *= std::sqrt( nrm_row[r] );
	  mDR[r] *= std::sqrt( nrm_col[r] );
	}
    }

    void
    calculate_rc_norms( )
    {
      // initialize arrays
      nrm_row.resize( N );
      nrm_col.resize( N );
      for ( std::size_t r=0; r<N; ++r ) 
	nrm_row[r] = nrm_col[r] = 0.0;

      // calculate row and column infinity norms
      for ( std::size_t r=0; r<N; ++r ) // for each row
	{
	  double max_val = 0.0;
	  const std::size_t inz1 = mA.rowptr()[r]  -mA.offset();
	  const std::size_t inz2 = mA.rowptr()[r+1]-mA.offset();
	  for ( std::size_t inz=inz1; inz<inz2; ++inz ) // for each nonzero in this row
	    {
	      const std::size_t c = mA.colind()[inz] - mA.offset();
	      const double      v = mA.value()[inz];
		  
	      if ( std::fabs(v) > max_val )    max_val = std::fabs(v); // build row inf norm
	      if ( std::fabs(v) > nrm_col[c] ) nrm_col[c] = std::fabs(v); // build col inf norm
	    }
	  nrm_row[r] = max_val;
	}
    }
    
    void
    calculate_rc_errors( )
    {
      // calculate errors
      row_err   = std::fabs( 1.0 - nrm_row[0] );
      col_err   = std::fabs( 1.0 - nrm_col[0] );
      for ( std::size_t r=1; r<N; ++r ) // for each row
	{
	  double rerr   = std::fabs( 1.0 - nrm_row[r] );
	  if (rerr > row_err) row_err = rerr;
	  double cerr   = std::fabs( 1.0 - nrm_col[r] );
	  if (cerr > col_err) col_err = cerr;
	}
    }

    void
    c_unscale( )
    {
      for ( std::size_t r=0; r<N; ++r )
	mx[r] /= mDR[r];
    }

  private:
    SOL                 solver;
    A_type              mA;
    x_type              mb;
    x_type              mx;
    x_type              nrm_row;
    x_type              nrm_col;
    x_type              mDR;
    x_type              mDL;
    std::size_t         N;
    double              row_err;
    double              col_err;
  };

};

