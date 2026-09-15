#ifndef __CPR_CSR_MATRIX_HPP_INCLUDED_
#define __CPR_CSR_MATRIX_HPP_INCLUDED_

#include <vector>
#include <ostream>
#include "CSR_Matrix.hpp"

  // ----------------------------  CPR_CSR_Matrix  ------------------------------ //
  /** \class CPR_CSR_Matrix
   *  
   *  Compressed Sparse Row Storage Matrix datatype with a "row" for the 
   * pressure equation
   **/
  // ------------------------------------------------------------------------- //
template< typename index_t = std::size_t, typename value_t = double >
class CPR_CSR_Matrix : public CSR_Matrix<index_t, value_t>
{
public:
    
        //.............................  LIFECYCLE  ..........................//
    index_t block_size;

    CPR_CSR_Matrix(index_t _N, index_t _NNZ) : CSR_Matrix<index_t, value_t>(_N, _NNZ) {};

    CPR_CSR_Matrix() : CSR_Matrix<index_t, value_t>() {};

}; // CPR_CSR_Matrix


#endif // __CPR_CSR_Matrix_HPP_INCLUDED_

//---------------------------------------------------------------------------//
//                           EOF CPR_CSR_Matrix.hpp
//---------------------------------------------------------------------------//

