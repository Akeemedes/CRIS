#include <torch/torch.h>
#include <ATen/Parallel.h>
#include <mkl.h>
#include <omp.h>
#include <native_lm/lm_optimizer.hpp>
#include <native_lm/recovery.hpp>
#include <native_lm/least_squares_objective.hpp>
#include <native_lm/scalar_tanh_network.hpp>
#include "bratu_objective.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

namespace fs = std::filesystem;
using Model = native_lm::ScalarTanhNetwork<2>;
using Data = std::vector<bratu::Sample>;
using Clock = std::chrono::steady_clock;
constexpr int P = Model::parameters;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
Data load(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[8]{}; std::uint64_t n=0;
    in.read(magic,8); in.read(reinterpret_cast<char*>(&n),8);
    if (!in || std::string(magic,8)!="BRATUD01" || n==0 || n>10000000) throw std::runtime_error("Invalid dataset");
    Data data(n); in.read(reinterpret_cast<char*>(data.data()), n*sizeof(bratu::Sample));
    if (!in || in.peek()!=EOF) throw std::runtime_error("Dataset size mismatch");
    for (const auto& s:data) if (!std::isfinite(s.root) || s.a < -2 || s.a>6 || s.beta<0 || s.beta>1 ||
         !(s.root_slope>0) || s.root<s.a || s.root>s.a+1+1e-12) throw std::runtime_error("Invalid branch label");
    return data;
}
void weights(const fs::path& path,const Model& model) {
    std::ofstream out(path,std::ios::binary); out.write(reinterpret_cast<const char*>(model.weights.data()),P*8);
    if (!out) throw std::runtime_error("Checkpoint write failed");
}
void recovery(const fs::path& path,const Model& model,const Model& best,const native_lm::LmState& s,double cost) {
    auto temp=path; temp += ".tmp";
    { std::ofstream out(temp,std::ios::binary); out.write("BRATUR01",8);
      std::uint64_t p=P; out.write(reinterpret_cast<char*>(&p),8);
      const double state[]{double(s.iteration),double(s.accepted_updates),double(s.validation_failures),s.damping,
          s.objective_loss,s.training_mse,s.validation_mse,s.best_validation_mse,cost};
      out.write(reinterpret_cast<const char*>(state),sizeof(state));
      out.write(reinterpret_cast<const char*>(model.weights.data()),P*8);
      out.write(reinterpret_cast<const char*>(best.weights.data()),P*8);
      if (!out) throw std::runtime_error("Recovery write failed"); }
    fs::remove(path); fs::rename(temp,path);
}
void restore(const fs::path& path,Model& model,Model& best,native_lm::LmState& s,double& cost) {
    std::ifstream in(path,std::ios::binary); char magic[8]{}; std::uint64_t p=0; double state[9]{};
    in.read(magic,8); in.read(reinterpret_cast<char*>(&p),8); in.read(reinterpret_cast<char*>(state),sizeof(state));
    in.read(reinterpret_cast<char*>(model.weights.data()),P*8); in.read(reinterpret_cast<char*>(best.weights.data()),P*8);
    if (!in || in.peek()!=EOF || p!=P || std::string(magic,8)!="BRATUR01") throw std::runtime_error("Invalid recovery");
    s.iteration=int(state[0]); s.accepted_updates=int(state[1]); s.validation_failures=int(state[2]); s.damping=state[3];
    s.objective_loss=state[4]; s.training_mse=state[5]; s.validation_mse=state[6]; s.best_validation_mse=state[7]; cost=state[8];
}
template<class Objective> double loss(const Data& data,const Model& model,const Objective& objective) {
    constexpr int R=int(Objective::kResidualsPerSample);
    constexpr int chunk=4096;
    const int chunks=int((data.size()+chunk-1)/chunk);
    std::vector<double> sums(chunks,0);
    at::parallel_for(0,chunks,1,[&](std::int64_t first,std::int64_t end){
        for(auto c=first;c<end;++c) for(std::size_t i=c*chunk;i<std::min(data.size(),std::size_t((c+1)*chunk));++i) {
            double r[R]{},d[R]{}; objective.evaluate(data[i],model.forward(bratu::inputs(data[i])),r,d);
            for(double value:r) sums[c]+=value*value;
        }
    });
    double total=0; for(double value:sums) total+=value;
    return total/data.size();
}
template<class Objective> native_lm::LmLinearization linearize(const Data& data,const Model& model,const Objective& objective,int block) {
    constexpr int R=int(Objective::kResidualsPerSample);
    auto opt=torch::TensorOptions().dtype(torch::kFloat64);
    native_lm::LmLinearization result{torch::zeros({P,P},opt),torch::zeros({P,1},opt),data.size(),{}};
    auto jstore=torch::empty({block*R,P},opt),rstore=torch::empty({block*R,1},opt);
    const auto started=Clock::now();
    for(std::size_t first=0;first<data.size();first+=block) {
        int count=int(std::min(std::size_t(block),data.size()-first));
        double* j=jstore.data_ptr<double>(); double* r=rstore.data_ptr<double>();
        auto phase=Clock::now();
        at::parallel_for(0,count,32,[&](std::int64_t begin,std::int64_t end){
            for(auto row=begin;row<end;++row) {
                double* jac=j+row*R*P; double residual[R]{},derivative[R]{};
                const auto& sample=data[first+row];
                const double y=model.forward(bratu::inputs(sample),jac);
                objective.evaluate(sample,y,residual,derivative);
                for(int k=1;k<R;++k) std::copy(jac,jac+P,jac+k*P);
                for(int k=0;k<R;++k) {
                    r[row*R+k]=residual[k];
                    for(int p=0;p<P;++p) jac[k*P+p]*=derivative[k];
                }
            }
        });
        result.timings.jacobian_seconds+=seconds(phase); phase=Clock::now();
        cblas_dsyrk(CblasRowMajor,CblasLower,CblasTrans,P,count*R,1,j,P,1,result.hessian.data_ptr<double>(),P);
        cblas_dgemv(CblasRowMajor,CblasTrans,count*R,P,1,j,P,r,1,1,result.gradient.data_ptr<double>(),1);
        result.timings.normal_matrix_seconds+=seconds(phase);
    }
    auto* h=result.hessian.data_ptr<double>();
    for(int i=0;i<P;++i) for(int j=i+1;j<P;++j) h[i*P+j]=h[j*P+i];
    result.timings.normal_seconds=seconds(started);
    return result;
}
void test() {
    Model model; bratu::Sample s{.9,.1,1.24,.65}; std::vector<double> g(P);
    model.forward(bratu::inputs(s),g.data()); double worst=0;
    for(int p=0;p<P;++p) {
        const double saved=model.weights[p],eps=1e-6;
        model.weights[p]=saved+eps; const double plus=model.forward(bratu::inputs(s));
        model.weights[p]=saved-eps; const double minus=model.forward(bratu::inputs(s)); model.weights[p]=saved;
        worst=std::max(worst,std::abs((plus-minus)/(2*eps)-g[p]));
    }
    using Term=native_lm::SoftClippedResidualTerm<bratu::Recurrence>;
    Term term({},1); double r[1],d[1],rp[1],rm[1],unused[1];
    for(double u:{-2.0,0.0,1.24,2.5,6.0}) {
        const double y=(u+2)/4.5-1,eps=1e-6;
        term.evaluate(s,y,r,d); term.evaluate(s,y+eps,rp,unused); term.evaluate(s,y-eps,rm,unused);
        worst=std::max(worst,std::abs((rp[0]-rm[0])/(2*eps)-d[0]));
    }
    if(worst>1e-6) throw std::runtime_error("Gradient test failed");
    native_lm::SupervisedMseTerm<bratu::Target> mse;
    auto objective=native_lm::StackedResiduals(mse,native_lm::ScaledResidualTerm<Term>(term,.1));
    Data data{s,bratu::Sample{.2,.05,.3,.9},bratu::Sample{2,.001,2.1,.99}};
    auto normal=linearize(data,model,objective,2);
    auto expected=torch::zeros({P,P},torch::kFloat64),eg=torch::zeros({P,1},torch::kFloat64);
    for(const auto& sample:data) {
        double residual[2],derivative[2]; model.forward(bratu::inputs(sample),g.data());
        objective.evaluate(sample,model.forward(bratu::inputs(sample)),residual,derivative);
        auto jac=torch::from_blob(g.data(),{P,1},torch::kFloat64);
        for(int k=0;k<2;++k) {auto j=jac*derivative[k];expected+=j.matmul(j.transpose(0,1));eg+=j*residual[k];}
    }
    if((normal.hessian-expected).abs().max().item<double>()>1e-9 ||
       (normal.gradient-eg).abs().max().item<double>()>1e-9) throw std::runtime_error("Normal-equation test failed");
    std::cout<<"gradient_check_max_abs_error="<<worst<<" parameters="<<P<<std::endl;
}
template<class Objective> int train(const Data& data,const Data& validation,const fs::path& dir,const Objective& objective,
                                   int maxaccepted,int maxproposals,int patience,int block,const std::string& resume,
                                   bool portable, const std::string& resume_state, const std::string& initial) {
    fs::create_directories(dir);
    Model model,best; native_lm::LmState state; state.damping=1e-3; double cumulative=0;
    native_lm::SupervisedMseTerm<bratu::Target> mse;
    if(!resume_state.empty()) {
        const auto r=native_lm::read_recovery(resume_state,P);
        state=r.state; cumulative=r.phase_seconds;
        std::copy(r.current.begin(),r.current.end(),model.weights.begin());
        std::copy(r.best.begin(),r.best.end(),best.weights.begin());
    } else if(resume.empty()) {
        if(fs::exists(dir/"optimizer_trace.csv")) throw std::runtime_error("Refusing to overwrite existing trace without recovery");
        if(!initial.empty()) {
            std::ifstream in(initial,std::ios::binary);
            in.read(reinterpret_cast<char*>(model.weights.data()),P*8);
            if(!in || in.peek()!=EOF) throw std::runtime_error("Invalid initial checkpoint");
            for(double x:model.weights) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite initial weights");
        }
        state.objective_loss=loss(data,model,objective); state.training_mse=loss(data,model,mse);
        state.validation_mse=loss(validation,model,mse); state.best_validation_mse=state.validation_mse; best=model;
        weights(dir/"checkpoint_000000.bin",model);
    } else restore(resume,model,best,state,cumulative);
    const bool has_trace=fs::exists(dir/"optimizer_trace.csv") && fs::file_size(dir/"optimizer_trace.csv")>0;
    std::ofstream trace(dir/"optimizer_trace.csv",std::ios::app); trace<<std::setprecision(17);
    if(!has_trace) trace<<"iteration,accepted_updates,status,validation_failures,lambda,objective_loss,train_mse_normalized,validation_mse_normalized,best_validation_mse_normalized,mean_objective_gradient_norm,normal_seconds,jacobian_seconds,normal_matrix_seconds,solve_seconds,candidate_seconds,candidate_train_seconds,candidate_validation_seconds,cumulative_phase_seconds,checkpoint\n";
    trace.flush();
    if(portable) native_lm::write_recovery(dir/("optimizer_state_"+std::to_string(state.iteration%2)+".bin"),
        model.weights.data(),best.weights.data(),P,state,cumulative);
    const double thresholds[]{1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12};
    native_lm::LmControl control;
    control.max_iterations=std::max(0,maxproposals-state.iteration);
    control.max_accepted_updates=maxaccepted; control.max_validation_failures=patience;
    if(state.accepted_updates>=maxaccepted || state.validation_failures>=patience) control.max_iterations=0;
    double current_objective=state.objective_loss;
    std::cout<<std::setprecision(10)<<"initial train_mse="<<state.training_mse<<" validation_mse="<<state.validation_mse<<std::endl;
    auto result=native_lm::optimize(model,control,state,
        [&](const Model& m){return linearize(data,m,objective,block);},
        [&](const Model& m,const torch::Tensor& step){Model candidate=m; const auto* p=step.data_ptr<double>(); for(int i=0;i<P;++i) candidate.weights[i]+=p[i];return candidate;},
        [&](const Model& m){native_lm::CandidateEvaluation e;auto t=Clock::now(); e.objective_loss=loss(data,m,objective);
            if(std::isfinite(e.objective_loss)&&e.objective_loss<current_objective) {
                if constexpr(Objective::kObjectiveEqualsSupervisedMse) e.training_mse=e.objective_loss;
                else e.training_mse=loss(data,m,mse);
                e.timings.candidate_train_seconds=seconds(t); auto v=Clock::now(); e.validation_mse=loss(validation,m,mse);
                e.timings.candidate_validation_seconds=seconds(v);
            } else e.timings.candidate_train_seconds=seconds(t);
            e.timings.candidate_seconds=seconds(t);return e;},
        [&](const Model& m,const native_lm::LmIteration& e){
            const auto& s=e.state;const auto& t=e.timings;current_objective=s.objective_loss;
            cumulative+=t.normal_seconds+t.solve_seconds+t.candidate_seconds;
            std::string checkpoint;
            if(e.validation_improved) best=m;
            if(e.status==native_lm::LmIteration::Status::Accepted) {
                for(int i=0;i<11;++i) if(s.validation_mse<=thresholds[i]&&!fs::exists(dir/("milestone_"+std::to_string(i)+".bin"))) {
                    checkpoint="milestone_"+std::to_string(i)+".bin"; weights(dir/checkpoint,m);
                }
                recovery(dir/("recovery_"+std::to_string(s.accepted_updates%2)+".bin"),m,best,s,cumulative);
            }
            trace<<s.iteration<<','<<s.accepted_updates<<','<<(e.status==native_lm::LmIteration::Status::Accepted?"accepted":"rejected")<<','
                <<s.validation_failures<<','<<s.damping<<','<<s.objective_loss<<','<<s.training_mse<<','<<s.validation_mse<<','
                <<s.best_validation_mse<<','<<s.mean_objective_gradient_norm<<','<<t.normal_seconds<<','<<t.jacobian_seconds<<','
                <<t.normal_matrix_seconds<<','<<t.solve_seconds<<','<<t.candidate_seconds<<','<<t.candidate_train_seconds<<','
                <<t.candidate_validation_seconds<<','<<cumulative<<','<<checkpoint<<'\n';trace.flush();
            if(portable) native_lm::write_recovery(dir/("optimizer_state_"+std::to_string(s.iteration%2)+".bin"),
                m.weights.data(),best.weights.data(),P,s,cumulative);
            std::cout<<"iteration="<<s.iteration<<" accepted="<<s.accepted_updates<<" train_mse="<<s.training_mse
                <<" validation_mse="<<s.validation_mse<<" best="<<s.best_validation_mse<<" validation_failures="<<s.validation_failures<<std::endl;
        });
    if(state.accepted_updates>=maxaccepted) result.stopping_reason="max_accepted_updates";
    else if(state.validation_failures>=patience) result.stopping_reason="max_validation_fail";
    weights(dir/"final_checkpoint.bin",best);
    std::ofstream done(dir/"result.json");done<<std::setprecision(17)<<"{\n\"stopping_reason\":\""<<result.stopping_reason
        <<"\",\n\"accepted_updates\":"<<result.state.accepted_updates<<",\n\"iteration\":"<<result.state.iteration
        <<",\n\"best_validation_mse_normalized\":"<<result.state.best_validation_mse
        <<",\n\"best_validation_mse_physical\":"<<20.25*result.state.best_validation_mse
        <<",\n\"cumulative_phase_seconds\":"<<cumulative<<"\n}\n";
    std::cout<<"stopping_reason="<<result.stopping_reason<<std::endl;return 0;
}
int main(int argc,char** argv) {
    try {
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;++i) {std::string key=argv[i];if(key=="--self-test"){test();return 0;}if(i+1==argc)throw std::runtime_error("Missing argument");args[key]=argv[++i];}
        auto value=[&](const std::string& key,const std::string& fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
        int threads=std::stoi(value("--threads","16")); at::set_num_threads(threads);omp_set_dynamic(0);omp_set_num_threads(threads);mkl_set_num_threads_local(threads);
        auto data=load(args.at("--train")),validation=load(args.at("--validation"));const fs::path dir=args.at("--run-dir");
        double weight=std::stod(value("--weight","0")); if(!std::isfinite(weight)||weight<0)throw std::runtime_error("Invalid weight");
        int accepted=std::stoi(value("--max-accepted","20000")),proposals=std::stoi(value("--max-proposals","50000"));
        int patience=std::stoi(value("--patience","500")),block=std::stoi(value("--block-size","16384"));
        if(threads<1||block<1||accepted<1||proposals<1||patience<1)throw std::runtime_error("Invalid limits");
        const auto resume=value("--resume","");
        const auto portable_value=value("--portable-recovery","0"), resume_state=value("--resume-state",""), initial=value("--initial-checkpoint","");
        if(portable_value!="0" && portable_value!="1") throw std::runtime_error("Invalid portable recovery option");
        const bool portable=portable_value=="1";
        if((!resume_state.empty() && (!portable || !resume.empty() || !initial.empty())) || (!initial.empty() && !resume.empty()))
            throw std::runtime_error("Cannot mix complete recovery with weight-only/legacy restart");
        native_lm::SupervisedMseTerm<bratu::Target> mse;
        if(weight==0)return train(data,validation,dir,mse,accepted,proposals,patience,block,resume,portable,resume_state,initial);
        using Physics=native_lm::ScaledResidualTerm<native_lm::SoftClippedResidualTerm<bratu::Recurrence>>;
        auto objective=native_lm::StackedResiduals(mse,Physics(native_lm::SoftClippedResidualTerm<bratu::Recurrence>(bratu::Recurrence{},1),weight));
        return train(data,validation,dir,objective,accepted,proposals,patience,block,resume,portable,resume_state,initial);
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
