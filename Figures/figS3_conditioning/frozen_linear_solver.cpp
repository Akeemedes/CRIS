#include <amgcl/backend/builtin.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/preconditioner/dummy.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <string>

using Backend=amgcl::backend::builtin<double>;
using Matrix=Backend::matrix;
using Clock=std::chrono::steady_clock;
template<class T> void read(std::ifstream& in,T* x,std::size_t n) {
    in.read(reinterpret_cast<char*>(x),sizeof(T)*n);
    if(!in) throw std::runtime_error("Truncated frozen linear system");
}
template<class Prec> void solve(const Matrix& A,const std::vector<double>& b,
                              double tol,int cap) {
    const auto start=Clock::now();
    Prec P(A);
    const double setup=std::chrono::duration<double>(Clock::now()-start).count();
    amgcl::solver::gmres<Backend>::params prm;
    prm.M=50;prm.maxiter=cap;prm.tol=tol;
    prm.pside=amgcl::preconditioner::side::right;
    amgcl::solver::gmres<Backend> gmres(b.size(),prm);
    std::vector<double> x(b.size(),0),r(b.size());
    const auto begin=Clock::now();
    const auto result=gmres(A,P,b,x);
    const double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
    amgcl::backend::residual(b,A,x,r);
    double bn=0,rn=0;for(std::size_t i=0;i<b.size();++i){bn+=b[i]*b[i];rn+=r[i]*r[i];}
    const double relative=bn>0?std::sqrt(rn/bn):std::sqrt(rn);
    if(!std::isfinite(relative)) throw std::runtime_error("Nonfinite GMRES residual");
    std::cout<<std::setprecision(17)<<"{\"iterations\":"<<std::get<0>(result)
        <<",\"reported_relative_residual\":"<<std::get<1>(result)
        <<",\"true_relative_residual\":"<<relative
        <<",\"converged\":"<<(relative<=tol?"true":"false")
        <<",\"setup_seconds\":"<<setup<<",\"solve_seconds\":"<<elapsed<<"}\n";
}
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::runtime_error("Usage: frozen_linear_solver system.bin none|ilu0 tolerance maxiter");
        std::ifstream in(argv[1],std::ios::binary);char magic[8];std::uint64_t n,nnz;
        read(in,magic,8);read(in,&n,1);read(in,&nnz,1);
        if(std::string(magic,8)!="FROZEN01"||!n||n>2000000||nnz>100000000) throw std::runtime_error("Invalid header");
        std::vector<std::int64_t> ptr(n+1),col(nnz);std::vector<double> val(nnz),b(n);
        read(in,ptr.data(),n+1);read(in,col.data(),nnz);read(in,val.data(),nnz);read(in,b.data(),n);
        if(in.peek()!=EOF||ptr.front()!=0||ptr.back()!=std::int64_t(nnz)) throw std::runtime_error("Invalid CSR length");
        for(std::size_t i=0;i<n;++i) {
            if(ptr[i]<0||ptr[i]>ptr[i+1]||ptr[i+1]>std::int64_t(nnz)) throw std::runtime_error("Invalid row pointers");
            for(auto j=ptr[i];j<ptr[i+1];++j)
                if(col[j]<0||col[j]>=std::int64_t(n)||!std::isfinite(val[j])||(j>ptr[i]&&col[j]<=col[j-1]))
                    throw std::runtime_error("Invalid sorted CSR values/columns");
            if(!std::isfinite(b[i])) throw std::runtime_error("Invalid RHS");
        }
        const double tol=std::stod(argv[3]);const int cap=std::stoi(argv[4]);
        if(!(tol>0&&tol<1)||cap<1) throw std::runtime_error("Invalid solver limit");
        Matrix A(n,n,ptr,col,val);
        if(std::string(argv[2])=="ilu0") solve<amgcl::relaxation::as_preconditioner<Backend,amgcl::relaxation::ilu0>>(A,b,tol,cap);
        else if(std::string(argv[2])=="none") solve<amgcl::preconditioner::dummy<Backend>>(A,b,tol,cap);
        else throw std::runtime_error("Unknown preconditioner");
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
