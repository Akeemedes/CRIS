#pragma once

#include "CSR_Matrix.hpp"
#include <vector>

namespace GENSOL{

  template< typename SOL >
  class DRS_Precond_Solver
  {
  public:
    typedef double                                              double_type;
    typedef int                                                 int_type;
    typedef typename SOL::report_t                              report_t;
    typedef CSR_Matrix< double_type, int_type >                 A_type;
    typedef std::vector< double_type >                          x_type;
    typedef std::vector< std::size_t >                          i_type;

  public:
    //.............................  LIFECYCLE  ..........................//
    DRS_Precond_Solver( std::size_t _N_MAX, std::size_t _NNZ_MAX ) : solver( _N_MAX, _NNZ_MAX )
    {
    };
   
    ~DRS_Precond_Solver( )
    {
    };

    static int_type offset( ) { return 1; }

    //.............................  OPERATORS  .........................//
    report_t solve( A_type &_A, x_type &_x, x_type & _b )
    {
      int ierr = left_precondition( _A, _b, _A.ainfo );
      return solver.solve( LA, _x, Lb );
    }

  protected:

    int left_precondition( A_type &A, x_type &b, const GENSOL::EllipticInfo &ainfo  )
    {
      // implement Dynamic Row Sum method of SPE-163608-PA
      setup_LeftPrecond( A, ainfo );       // Produces left preconditiong matrix L

      //      std::ofstream strm( "Matrix_3_L.dat");
      //      HB_dump( strm, L );
      //      strm.close();

      char TRANS   = 'N';      // no transpose
      int  REQUEST = 1;        // calculates rowptr in first mult and provides C[NNZ]
      int  SORT    = 7;        // sort column indices of nonzeros in each row in LA
      int  NL      = L.N();    // dimension of square matrices
      int  NA      = A.N();    // dimension of square matrices
      int LA_NNZ   = LA.NNZ(); // NNZ in result L*A
      int INFO     = 0;        // error code

      // LA = L * A
      LA.resize( NA, 0 );
      mkl_dcsrmultcsr( &TRANS, &REQUEST, &SORT, &NL, &NL, &NA,
		       L.value().data(),  L.colind().data(),  L.rowptr().data(),
		       A.value().data(),  A.colind().data(),  A.rowptr().data(),
		       LA.value().data(), LA.colind().data(), LA.rowptr().data(),
		       &LA_NNZ,
		       &INFO );
      SORT    = 9;
      REQUEST = 2;
      LA_NNZ = LA.rowptr().back() - 1;
      LA.resize( NA, LA_NNZ );
      mkl_dcsrmultcsr( &TRANS, &REQUEST, &SORT, &NL, &NL, &NA,
		       L.value().data(),  L.colind().data(),  L.rowptr().data(),
		       A.value().data(),  A.colind().data(),  A.rowptr().data(),
		       LA.value().data(), LA.colind().data(), LA.rowptr().data(),
		       &LA_NNZ,
		       &INFO );
      LA.ainfo = A.ainfo;

      // Lb = L * b
      Lb.resize( NA );
      mkl_dcsrgemv (&TRANS, 
		    &NA, 
		    L.value().data(), L.rowptr().data(), L.colind().data(), 
		    b.data(), 
		    Lb.data() );

      return INFO; // 0 success; +ve need more memory; -ve calculates NNZ and puts in rowptr
    }

    void setup_LeftPrecond( const A_type &A, const GENSOL::EllipticInfo & ainfo )
    {
      // form L using Dynamic Row Sum procedure in SPE-163608-PA

      L.value().clear();
      L.colind().clear();
      L.rowptr().clear();
      L.rowptr().push_back(0+A.offset());

      std::size_t blk_r, r, c;
      for ( blk_r = 0; blk_r < ainfo.n_blocks; ++blk_r )
	{
	  calc_blk_deltas( A, ainfo, blk_r );
	  bool is_singular_block = false;
	  if ( delta[ainfo.elliptic_var_id] == 0.0 )
	    {
	      is_singular_block   = true;
	      delta[mVarDeltaMax] = 1.0;
	    }

	  for ( r = 0; r < ainfo.elliptic_var_id; ++r )
	    {
	      L.value().push_back( 1.0 );
	      if ( (is_singular_block) && (r == mVarDeltaMax) )
		L.colind().push_back( blk_r * ainfo.n_vars_per_block + ainfo.elliptic_var_id + A.offset() );
	      else
		L.colind().push_back( blk_r * ainfo.n_vars_per_block + r + A.offset() );
	      L.rowptr().push_back( L.rowptr().back() + 1 );
	    }

	  for ( c = 0; c < ainfo.n_vars_per_block; ++c )
	    {
	      L.value().push_back( delta[ c ] );
	      L.colind().push_back( blk_r * ainfo.n_vars_per_block + c + A.offset() );
	    }
	  L.rowptr().push_back( L.rowptr().back() + c );	  

	  for ( ++r ; r < ainfo.n_vars_per_block; ++r )
	    {
	      L.value().push_back( 1.0 );
	      if ( (is_singular_block) && (r == mVarDeltaMax) )
		L.colind().push_back( blk_r * ainfo.n_vars_per_block +  ainfo.elliptic_var_id + A.offset() );
	      else
		L.colind().push_back( blk_r * ainfo.n_vars_per_block + r + A.offset() );
	      L.rowptr().push_back( L.rowptr().back() + 1 );
	    }
	  
	}
      for (blk_r = ainfo.n_blocks * ainfo.n_vars_per_block  ; blk_r < A.N(); ++blk_r )
	{
	  L.value().push_back( 1.0 );
	  L.colind().push_back( blk_r + A.offset() );
	  L.rowptr().push_back( L.rowptr().back() + 1 );
	}
      L.check_size();
    }

    void
    calc_blk_deltas( const A_type &A, const GENSOL::EllipticInfo & ainfo, std::size_t blk )
    {
      calc_delta_1( A, ainfo, blk );
      calc_delta_2( A, ainfo, blk );
      //for ( std::size_t i = 0; i< delta.size(); ++i ) delta[i] *= delta2[i];
    }

    void
    calc_delta_1( const A_type &A, const GENSOL::EllipticInfo & ainfo, std::size_t blk )
    {
      const double EPS = 0.2;

      delta.clear();
      diag_dom.clear();
      std::size_t c = blk * ainfo.n_vars_per_block + ainfo.elliptic_var_id;
      for ( std::size_t v = 0; v < ainfo.n_vars_per_block; ++v )
	{
	  std::size_t r = blk * ainfo.n_vars_per_block + v;
	  const double diagonal = A( r, c );

	  double offdiags = 0.0;
	  for ( std::size_t inz = A.rowptr()[r] - A.offset(); 
		inz < A.rowptr()[r+1] - A.offset(); 
		++inz )
	    {
	      if ( ( A.colind()[inz] % ainfo.n_vars_per_block - A.offset() == ainfo.elliptic_var_id) && 
		   (A.colind()[inz]-A.offset()< ainfo.n_blocks*ainfo.n_vars_per_block ) )
		{
		  offdiags += std::fabs( A.value()[inz] );
		}
	    }
	  offdiags -= std::fabs(diagonal);
	  diag_dom.push_back(diagonal/offdiags);
	  if ( diag_dom[v] >= EPS )
	    delta.push_back(1.0);
	  else
	    delta.push_back(0.0);
	}
      
      std::size_t v = 0;
      mVarDeltaMax    = ainfo.elliptic_var_id;
      double maxDelta = diag_dom[mVarDeltaMax];
      for ( ; v<ainfo.n_vars_per_block; ++v ) 
	if ( A(blk*ainfo.n_vars_per_block + ainfo.elliptic_var_id, blk*ainfo.n_vars_per_block + v) != 0.0 )
	  if (diag_dom[v]>maxDelta)
	    {
	      mVarDeltaMax = v;
	      maxDelta     = diag_dom[v];
	    }
    }

    void
    calc_delta_2( const A_type &A, const GENSOL::EllipticInfo & ainfo, std::size_t blk )
    {
      const double EPS = 0.02;

      std::size_t row = blk * ainfo.n_vars_per_block + ainfo.elliptic_var_id;
      const double diag = EPS * std::fabs( A( row, row ) );

      delta2.resize( ainfo.n_vars_per_block );
      for ( std::size_t v = 0; v < ainfo.n_vars_per_block; ++v )
	{
	  if (v == ainfo.elliptic_var_id )
	    {
	      delta2[v] = 1.0;
	    }
	  else
	    {
	      double sum = 0.0;
	      for ( std::size_t inz = A.rowptr()[row] - A.offset();
		    inz <  A.rowptr()[row+1] - A.offset();
		    ++inz )
		{
		  if ( A.colind()[inz] % ainfo.n_vars_per_block - A.offset() == v)
		    sum += std::fabs(  A.value()[inz] );
		}
	      if ( sum < diag ) 
		delta2[v] = 0.0;
	      else
		delta2[v] = 1.0;
	    }
	}
    }
  
  private:
    SOL                 solver;
    A_type              L;
    A_type              LA;
    x_type              Lb;
    x_type              delta;
    x_type              delta2;
    x_type              diag_dom;
    std::size_t         mVarDeltaMax;
  };

};

