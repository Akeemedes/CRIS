#ifndef __CSR_MATRIX_HPP_INCLUDED_
#define __CSR_MATRIX_HPP_INCLUDED_

#include <vector>
#include <ostream>
#include <utility>

  // ----------------------------  CSR_Matrix  ------------------------------ //
  /** \class CSR_Matrix
   *  
   *  Compressed Sparse Row Storage Matrix datatype
   **/
  // ------------------------------------------------------------------------- //
template< typename Index = std::size_t, typename Value = double >
class CSR_Matrix
  {
  public:
      using index_t = Index;
      using value_t = Value;
      using value_vector_t = std::vector<value_t >;
      using index_vector_t = std::vector<index_t >;
  private:
      index_t mN;
      index_t mNNZ;
      value_vector_t m_val;
      index_vector_t m_colind;
      index_vector_t m_rowptr;

  public:
    //.............................  LIFECYCLE  ..........................//
    CSR_Matrix(  ) : mN(0), mNNZ(0), m_val( ), m_colind( ), m_rowptr( ) {  }

    CSR_Matrix(  index_t _N, index_t _NNZ ) { resize(_N,_NNZ); }

    //...........................  ASSIGNMENT  ...........................//      
    CSR_Matrix& operator= (const CSR_Matrix&) = default;
    
    //.............................  ACCESS  .............................//
    index_t N() const { return mN; }
    index_t NNZ() const { return mNNZ; }
    index_t offset() const { 
        if (mN>0) return m_rowptr[0]; 
        return 0;
    }

    const value_vector_t& value()  const { return m_val; }
    const index_vector_t& colind() const { return m_colind; }
    const index_vector_t& rowptr() const { return m_rowptr; }
    typename value_vector_t::const_iterator begin_row(index_t r) const {
        return m_val.begin() + m_rowptr[r];
    }

    std::pair< bool, index_t > find_vidx(index_t i, index_t j) const {
        if ((i < mN) && (j < mN))
        {
            const index_t r1 = m_rowptr[i];
            const index_t r2 = m_rowptr[i + 1];
            if ((j >= m_colind[r1]) && (j <= m_colind[r2-1]) )
            {
                index_t v = r1;
                while (j > m_colind[v]) ++v;
                if (j == m_colind[v]) return { true, v };
            }
        }
        return {false, 0};
    }

    value_t operator()( index_t i, index_t j ) const
    {
      double    value  = 0;
      if ( (i<mN) && (j<mN) ) 
	{
          const index_t offset = m_rowptr[0];
          const index_t r1 = m_rowptr[i] - offset;
	  const index_t r2 = m_rowptr[i+1] - offset - 1;
	  if ( ( j >= (m_colind[r1]-offset) ) && ( j <= ( m_colind[r2]-offset ) ) )
	    {
	      index_t r = r1;
	      while ( j > (m_colind[r]-offset) ) ++r;
	      if ( j == ( m_colind[r]-offset) ) value = m_val[r];
	    }
	}
      return value;
    };

    value_vector_t& value() { return m_val; }
    index_vector_t& colind() { return m_colind; }
    index_vector_t& rowptr() { return m_rowptr; }   
    typename value_vector_t::iterator begin_row(index_t r) {
        return m_val.begin() + m_rowptr[r];
    }

    //.............................  OPERATORS  .........................//
    void resize( index_t _N, index_t _NNZ )
    {
      mN = _N; mNNZ = _NNZ;
      m_val.resize( mNNZ, 0.0 );
      m_colind.resize( mNNZ, 0 );
      m_rowptr.resize( mN + 1, 0 );
    }

    void reserve(index_t _N, index_t _NNZ)
    {
        m_val.reserve(_NNZ);
        m_colind.reserve(_NNZ);
        m_rowptr.reserve(_N + 1);
    }

    void reset_size( )
    {
      mN   = m_rowptr.size()-1;
      mNNZ = m_val.size();
    }

    // ........................... PRINTING   ..............................//
    std::ostream& spy(std::ostream& out) {
        for (index_t r = 0; r < mN; ++r) {
            index_t i1 = m_rowptr[r], i2 = m_rowptr[r + 1] - 1;
            for (index_t k = 0; k < m_colind[i1]; ++k) out << std::setw(2) << ".";
            for (index_t j = i1; j < i2; ++j) {
                out << std::setw(2) << "X";
                for (index_t k = m_colind[j] + 1; k < m_colind[j + 1]; ++k) out << std::setw(2) << ".";
            }
            out << std::setw(2) << "X";
            for (index_t k = m_colind[i2] + 1; k < mN; ++k) out << std::setw(2) << ".";
            out << std::endl;
        }
        return out;
    }

    //.............................  OPERATORS  .........................//
    void multiply_v(const std::vector<value_t>& _x, std::vector<value_t>& y_) const {
        assert(mN == _x.size());
        y_.resize(mN, 0.0);
        for (index_t r{ 0 }; r < mN; ++r)
        {
            for (index_t l{ m_rowptr[r] }; l < m_rowptr[r + 1]; ++l)
                y_[r] += m_val[l] * _x[m_colind[l]];
        }
    }

    void transpose_multiply_v(const std::vector<value_t>& _x, std::vector<value_t>& y_) const {
        //std::cout << _x.size() << " " << mN << std::endl;
        assert("Size Mismatch " && mN == _x.size());
        y_.resize(mN, 0.0);
        for (index_t r{ 0 }; r < mN; ++r)
        {
            y_[r] = 0.0;
            for (index_t l{ m_rowptr[r] }; l < m_rowptr[r + 1]; ++l)
                y_[m_colind[l]] += m_val[l] * _x[r];
        }
    }

  }; // CSR_Matrix


#endif // __CSR_Matrix_HPP_INCLUDED_

//---------------------------------------------------------------------------//
//                           EOF CSR_Matrix.hpp
//---------------------------------------------------------------------------//
