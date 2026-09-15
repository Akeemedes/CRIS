#pragma once

#include "samg.h"
#include "CSR_Matrix.hpp"
#include <vector>

namespace GENSOL{
  
  class SAMGwrapper
  {
  public:
    typedef double                                     double_type;
    typedef int                                        int_type;
    
    typedef CSR_Matrix< double_type, int_type >        A_type;
    typedef std::vector< double_type >                 x_type;
   
  public:
    //.............................  LIFECYCLE  ..........................//
    SAMGwrapper( std::size_t _N_MAX, std::size_t _NNZ_MAX )
    {
      ierr_solve       = 0;
      nnu              = _N_MAX;
      nna              = 0;
      matrix           = 220;
      ifirst           = 0;
      eps              = 1.0e-6;
      nsys             = 1;
      iscale           = new int[1];
      iscale[0]        = 0;
      iu               = new int[1];
      ndiu             = 1;
      ip               = new int[1];
      ndip             = 1;
      nsolve           = 100010;
      ncyc             = 12050;
      iswtch           = 41400;
      chktol           = -1.0;
      iout             = -1;
      idump            = -1;
      a_cmplx          = 0.0;
      g_cmplx          = 0.0;
      p_cmplx          = 0.0;
      w_avrge          = 0.0;
      is_first         = true;
      int tmp = -3;
      double tmp2 = 21.50;
      int   tmp3 = true;

      SAMG_SET_MODE_MESS(&tmp);
      tmp = 2;
      SAMG_SET_NTYP_GALERKIN(&tmp);
      //      SAMG_SET_ECG(&tmp2);
      //      tmp2 = 0.0;
      //      SAMG_SET_EWT(&tmp2);
      //      tmp = 510;
      //      SAMG_SET_NCG(&tmp);
      //      SAMG_SET_GLK_MULT_ZEROS( &tmp3 );
    };
   
    ~SAMGwrapper( )
    {
      delete [] ip;
      delete [] iu;
      delete [] iscale;
      int err1;
      SAMG_LEAVE( &err1 );
    };

    static int_type offset( ) { return 1; }

    //.............................  OPERATORS  .........................//
    int setup_A( A_type &_A )
    {
      int ierr = 0;
      iswtch = 41400;
      is_first = true;

      nnu              = _A.N();
      nna              = _A.NNZ();

      ia = _A.rowptr();
      ja = _A.colind();
      a  = _A.value();
      negatives.resize( nnu);
      std::size_t inz_diag;
      bool        diag_found;
      for ( std::size_t r=0; r<nnu; ++r)
	{
	  std::size_t inz=ia[r]-_A.offset();
	  for ( ; inz < ia[r+1] - _A.offset(); ++inz )
	    {
	      if ( ja[inz]-_A.offset() == r )
		{
		  diag_found = true;
		  inz_diag   = inz;
		}
	    }
	  if (diag_found)
	    {
	      int     j_swap        = ja[inz_diag];
	      double  v_swap        = a[inz_diag];
	      ja[inz_diag]          = ja[ia[r]-_A.offset()];
	      a[inz_diag]           = a[ ia[r]-_A.offset()];
	      ja[ia[r]-_A.offset()] = j_swap;
	      a[ ia[r]-_A.offset()] = v_swap;
	    }
	  if ( a[ ia[r]-_A.offset()] < 0 )
	    {
	      negatives[r] = -1.0;
	      inz=ia[r]-_A.offset();
	      for ( ; inz < ia[r+1] - _A.offset(); ++inz )
		{
		  a[inz] *= -1.0;
		}
	    }
	  else
	    negatives[r] = 1.0;	      
	}
      return ierr;
    }

    int apply( double_type *_x, double_type *_b )
    {
      mb.resize( nnu );
      for ( std::size_t r=0; r<nnu; ++r ) mb[r] = _b[r] * negatives[r];

      SAMG(&nnu,&nna,&nsys,
           &ia[0],&ja[0],&a[0],&mb[0],&_x[0],&iu[0],&ndiu,&ip[0],&ndip,&matrix,&iscale[0],
           &res_in,&res_out,&ncyc_done,&ierr_solve,
           &nsolve,&ifirst,&eps,&ncyc,&iswtch,
           &a_cmplx,&g_cmplx,&p_cmplx,&w_avrge,
           &chktol,&idump,&iout);

      if ( (is_first ) && (ierr_solve <= 0) )
	{
	  is_first  = false;
	  iswtch   -= 30000;
	}

      if (ierr_solve<0)
	std::cout << "SAMG WARNING ierr = " << ierr_solve << std::endl;
      else if (ierr_solve>0)
	std::cout << "SAMG FATAL ERROR ierr = " << ierr_solve << std::endl;

      return -1 * ( ierr_solve > 0 );
    }

    int solve( A_type &_A, x_type &_x, x_type & _b )
    {
      setup_A( _A );
      int ierr = apply( _x.data( ), _b.data( ) );
      return ierr;
    }

  protected:

  private:
    int     ierr_solve;
    int     nnu;
    int     nna;
    int     nsys;
    std::vector<int>    ia;
    std::vector<int>    ja;
    std::vector<double> a;
    std::vector<double> negatives;
    std::vector<double> mb;
    int    *iu;
    int     ndiu;
    int    *ip;
    int     ndip;
    int     matrix;
    int    *iscale;
    double  res_in;
    double  res_out;
    int     ncyc_done;
    int     ierr;
    int     nsolve;
    int     ifirst;
    double  eps;
    int     ncyc;
    int     iswtch;
    double  a_cmplx;
    double  g_cmplx;
    double  p_cmplx;
    double  w_avrge;
    double  chktol;
    int     idump;
    int     iout;
    bool    is_first;
  };

};

