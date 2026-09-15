#ifndef __UNIFORMFIELD_HPP_
#define __UNIFORMFIELD_HPP_

#include <cstddef>
#include <array>
#include <type_traits>

/* ******************************************************** */
// A simple scalar object acting as a collection
// models the conspet of a homogeneous field
/* ******************************************************** */

enum FaceType {Internal, Boundary};

template< typename T, typename index_t = std::size_t>
class uniform_field {
public:
	using value_t = T;
	using reference = T&;
	using const_reference = const T&;
	using pointer = T*;

	uniform_field(std::size_t _N, const T& _clone = T()) : a(_clone), N(_N) {};
	uniform_field() = default;

	void resize(index_t _N) { N = _N; }
	
	std::size_t size() const { return N; }

	reference operator[](index_t ) { return a; }
	const_reference operator[](index_t ) const { return a; }

private:
	T a;
	std::size_t N;
};

template<typename T,
	FaceType FT,
	template<typename> class Layout,
	typename index_t = std::size_t>
class uniform_cartesian_faces
{
public:
	using value_t = T;
	using reference = T&;
	using const_reference = const T&;
	using pointer = T*;

	using IndexingPolicy = Layout<index_t>;

	static constexpr int nDim = (FT == FaceType::Internal) ? 3 : 6;

	uniform_cartesian_faces() = default;
	uniform_cartesian_faces(const uniform_cartesian_faces&) = default;
	uniform_cartesian_faces(uniform_cartesian_faces&&) = default;

	uniform_cartesian_faces(index_t N_,
		const std::array<T, nDim>& a_,
		const IndexingPolicy& fIndex_)
		: fIndex(fIndex_), a(a_), N(N_)
	{
	}

	index_t size() const { return N; }

	int direc(index_t f) const
	{
		if constexpr (FT == FaceType::Internal)
			return fIndex.interior_face_direction(f);
		else
			return fIndex.boundary_face_direction(f); // you implemented this
	}

	reference operator[](index_t f) { return a[static_cast<std::size_t>(direc(f))]; }
	const_reference operator[](index_t f) const { return a[static_cast<std::size_t>(direc(f))]; }

private:
	const IndexingPolicy& fIndex;
	std::array<T, nDim> a{};
	index_t N{ 0 };
};
// aliases
template <typename T, template <typename> class Layout, typename index_t = std::size_t> 
using UCartesianBFaces = 
uniform_cartesian_faces<T, FaceType::Boundary, Layout, index_t>;
template <typename T, template <typename> class Layout, typename index_t = std::size_t> 
using UCartesianIFaces = 
uniform_cartesian_faces<T, FaceType::Internal, Layout, index_t>;


#endif // __UNIFORMFIELD_HPP_ included
