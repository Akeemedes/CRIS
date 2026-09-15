#ifndef _TRANSMISSIBILITYFUNCTORS_HPP_
#define _TRANSMISSIBILITYFUNCTORS_HPP_
#include <cassert>

template< typename index_t >
class UniformHarmonic {
public:
	UniformHarmonic(double dx, double dy, double dz) :
		Cx(0.00112712 * 2.0 * dy * dz / dx),
		Cy(0.00112712 * 2.0 * dx * dz / dy),
		Cz(0.00112712 * 2.0 * dy * dx / dz)
	{
		assert((((Cx > 0.0) && (Cy > 0.0)) && (Cz > 0.0)));
	};
	// k1 k2 i1 i2 j k
	double i_face(double K1, double K2, index_t, index_t, index_t, index_t) const
	{
		return Cx * K1 * K2 / (K1 + K2);
	}
	// k1 k2 i j1 j2 k
	double j_face(double K1, double K2, index_t, index_t, index_t, index_t) const
	{
		return Cy * K1 * K2 / (K1 + K2);
	}
	// k1 k2 i j k1 k2
	double k_face(double K1, double K2, index_t, index_t, index_t, index_t) const
	{
		return Cz * K1 * K2 / (K1 + K2);
	}
private:
	const double Cx, Cy, Cz;
};

template< typename index_t >
class NonUniformHarmonic {
public:
	NonUniformHarmonic(const double* _pdx, const double* _pdy, const double* _pdz, index_t nx, index_t ny, index_t nz) :
		pdx(_pdx), pdy(_pdy), pdz(_pdz)
#ifdef DEBUG
		, mnx(nx), mny(ny), mnz(nz)
#endif // DEBUG
	{
	};
	double i_face(double K1, double K2, index_t i1, index_t i2, index_t j, index_t k) const
	{
		assert(((i1 < mnx) && (i2 < mnx)));
		assert(((i1 >= 0) && (i2 >= 0)));
		const double area = 0.00112712 * 2.0 * pdy[j] * pdz[k];
		return area * K1 * K2 / (K1 * pdx[i2] + K2 * pdx[i1]);
	}
	double j_face(double K1, double K2, index_t i, index_t j1, index_t j2, index_t k) const
	{
		assert(((j1 < mny) && (j2 < mny)));
		assert(((j1 >= 0) && (j2 >= 0)));
		const double area = 0.00112712 * 2.0 * pdx[i] * pdz[k];
		return area * K1 * K2 / (K1 * pdy[j2] + K2 * pdx[j1]);
	}
	double k_face(double K1, double K2, index_t i, index_t j, index_t k1, index_t k2) const
	{
		assert(((k1 < mnz) && (k2 < mnz)));
		assert(((k1 >= 0) && (k2 >= 0)));
		const double area = 0.00112712 * 2.0 * pdx[i] * pdy[j];
		return area * K1 * K2 / (K1 * pdz[k2] + K2 * pdz[k1]);
	}
private:
	const double* pdx;
	const double* pdy;
	const double* pdz;
#ifdef DEBUG
	double mnx, mny, mnz;
#endif // DEBUG
};
#endif
