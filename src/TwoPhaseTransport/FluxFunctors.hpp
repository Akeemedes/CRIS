#ifndef __FLUXFUNCTORSHPP_
#define __FLUXFUNCTORSHPP_

#include <cmath>
#include <cfloat>
#include <tuple>

struct CocurrentFracFlux
{
  CocurrentFracFlux(double _M, double _nw, double _nnw, double _Ng)
    : M(_M), nw(_nw), nnw(_nnw), Ng(_Ng) {  }
  std::tuple<double,double> val_grad( double u) const {
    //const double omu = 1.0 - u;
    const double krw = std::pow(u, nw); 
	const double krw_grad = nw * krw/(u + DBL_MIN);
    const double krnw = std::pow(1.0 - u, nnw); 
	const double krnw_grad = -nnw * krnw/(1 - u + DBL_MIN);
    const double N  = krw * (1.0 - Ng * krnw);
    const double D  = krw + M * krnw;
    const double Np = krw_grad * (1.0 - Ng * krnw) - krw * (Ng * krnw_grad);
    const double Dp = krw_grad + M *krnw_grad;

    if (u < 0)
        return { 0.0, 0.0 };
    if (u > 1.0)
        return { 1.0, 0.0 };
    //std::cout << "M: " << M << ", nw: " << nw << ", nnw: " << nnw << ", Ng: " << Ng << std::endl;
    //std::cout << "krw: " << krw << ", krnw: " << krnw << std::endl;
    //std::cout << "Num: " << N << ", Denom: " << D << std::endl;
    return { N / D, (Np * D - N * Dp) / (D * D) };
  }
  template <typename Ty>
  Ty operator () (const Ty& u) const {
      //Ty u_nw = 1.0 - u;
      //auto krw = pow(u, nw);
      //auto krnw = pow(1.0 - u, nnw);
      //Ty N = pow(u, nw) * (1.0 - Ng * pow(1.0 - u, nnw));
      //Ty D = pow(u, nw) + M * pow(1.0 - u, nnw);
      //auto flux = N / D ;
      //std::cout << N<<" "<<D <<" "<<flux<< std::endl;
      return pow(u, nw) * (1.0 - Ng * pow(1.0 - u, nnw))/(pow(u, nw) + M * pow(1.0 - u, nnw));
  }
  
  double operator()(double u) const { return std::get<0>(val_grad(u)); }
  double grad(double u) const       { return std::get<1>(val_grad(u)); }

private:
  double M, nw, nnw, Ng;
};



struct ExponentialFlux
{
  ExponentialFlux(double _lambda) :  lambda(_lambda) {}

  std::tuple<double,double> val_grad(double u) const
  {
    const double val = std::exp((u - 1.0) * lambda);
    const double grad = lambda * val;
    return {val, grad};
  }

  double operator()(double u) const { return std::get<0>(val_grad(u)); }
  double grad(double u) const       { return std::get<1>(val_grad(u)); }

  template <typename Ty>
  Ty operator () (const Ty& u) const {
      return exp((u- 1.0) * lambda);
  }

private:
  double lambda;
};

//struct ExponentialFlux
//{
//    ExponentialFlux(double _lambda) : exp(- 1.0/_lambda) {}
//    std::tuple<double, double> val_grad(double u) const
//    {
//        const double val = -std::pow(u + eps, exp);
//        const double grad = exp * val/(u+ eps);
//        return { val, grad };
//    }
//
//    double operator()(double u) const { return std::get<0>(val_grad(u)); }
//    double grad(double u) const { return std::get<1>(val_grad(u)); }
//
//    template <typename Ty>
//    Ty operator () (const Ty& u) const {
//        //Ty partial = pow(u + 1e-5, exp);
//        //std::cout << "exponent: " << exp << " u: " << u << " exp: "<<partial << std::endl;
//        return -pow(u+ eps, exp);
//    }
//
//private:
//    double exp;
//    double eps{ 1e-12 };
//};


#endif // __FLUXFUNCTORSHPP_ included
