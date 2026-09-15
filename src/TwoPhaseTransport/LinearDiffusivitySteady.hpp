#ifndef __LINEARDIFFUSIVITYSTEADY_HPP__
#define __LINEARDIFFUSIVITYSTEADY_HPP__

#include <cstddef>
#include <vector>
#include <utility>
#include <cassert>
#include <algorithm>

#include "DiscreteProblem/MultivariableNorm.hpp"
#include "CRIS/CSR_Pattern.hpp"
#include "WellsManager.hpp"

// Steady TPFA pressure with gravity + BHP-controlled wells, no-flow boundaries.
//
// PDE (depth D positive downward):
//   -div( K (grad p - rho g_vec) ) = q
//
// Interior face flux (two-point):
//   F_ij = T_ij * [ (p_i - p_j) + rho*g*(D_i - D_j) ]
//

template<typename TPFA_Mesh>
class LinearDiffusivitySteady {
public:
  using State        = std::vector<double>;
  using ResidualNorm = MultivariableNorm<1, TwoNormFunct>;
  using UpdateNorm   = MultivariableNorm<1, InfNormFunct>;

  LinearDiffusivitySteady(const TPFA_Mesh& _mesh,
						  WellsManager<TPFA_Mesh>& _wm,
                          //const std::vector<Well>& _wells,
                          double _rho,
                          double _g,
                          double _RTOL  = 1.0e-8,
                          double _DUTOL = 1.0e-7)
  : mesh(_mesh)
  , wm(_wm)
  , rho(_rho)
  , g(_g)
  , RTOL(_RTOL)
  , DUTOL(_DUTOL)
  , diag(_mesh.num_cells())
  , k_12(_mesh.num_intr_faces())
  , k_21(_mesh.num_intr_faces())
  {}

  //void set_schedule_time(double t) { t_eval = t; }

  std::size_t neqs() const { return mesh.num_cells(); }
  std::size_t neq_per_cell() const { return 1; }
  std::size_t max_nnz() const { return mesh.max_num_neighbors(); }

  template<typename V1, typename V2, typename M>
  void allocate_containers(V1& du, V2& r, M& J) const {
    J.block_size = neq_per_cell();
    J.reserve(neqs(), max_nnz());
    csr_pattern::build_pattern(J, mesh, diag, k_12, k_21);
    J.reset_size();
    r.resize(neqs(), 0.0);
    du.resize(neqs(), 0.0);

    std::cout << "Containers Allocated Successfully" << std::endl;
  }

  template<typename V1, typename V2, typename M>
  void initialize_containers(V1& du, V2&, M&) const {
    std::fill(du.begin(), du.end(), 0.0);
  }

  template <typename Vresid>
  void evaluate(const State& p, Vresid& resid) 
  {
      const std::size_t nc = mesh.num_cells();
      const std::size_t nif = mesh.num_intr_faces();
      assert(p.size() == nc);

      
      for (std::size_t c = 0; c < nc; ++c) resid[c] = 0.0;

      const double rho_g = rho * g;
      //const auto& layout = mesh.layout();

      // ---------------- interior faces ----------------


      for (std::size_t f = 0; f < nif; ++f) {
          const auto [c0, c1] = mesh.intr_face_to_cells(f);
          const double T = mesh.intr_face_coef(f);
          const double dz = mesh.cell_depth(c1) - mesh.cell_depth(c0);
          const double p0 = p[c0];
          const double p1 = p[c1];
          //const auto F = T * ((p[c0] - p[c1]) + rho_g * dD);

          resid[c0] += T * ((p0 - p1) + rho_g * dz);
          resid[c1] -= T * ((p0 - p1) + rho_g * dz);
          // Scalar pressure flux is assembled directly above.
      }
      //std::cout << wm.wells().size() << " wells" << std::endl;
      for (auto const& w : wm.wells()) {
          //std::cout << "well: " << w.name << std::endl;
          const auto cmd = w.schedule.command(wm.current_event_time());
          if (!cmd.on) continue;
          for (std::size_t ci = 0; ci < w.comps.size(); ++ci) {
              
              const auto& comp = w.comps[ci];
              assert(comp.cell < nc);
              const double p_well_comp = cmd.bhp + rho_g * comp.d_depth_from_datum;
              //std::cout << p_well_comp << std::endl;
              const double p_cell = p[comp.cell];
              resid[comp.cell] += comp.WI * p_cell - comp.WI * p_well_comp;
          }
      }
  }

  template<typename Vresid, typename M>
  bool discretize(const State& p, Vresid& resid, M& J)
  {
      assert(resid.size() == mesh.num_cells());
      evaluate(p, resid);
      std::fill(J.value().begin(), J.value().end(), 0.0);

      for (std::size_t f = 0; f < mesh.num_intr_faces(); ++f) {
          const auto [c0, c1] = mesh.intr_face_to_cells(f);
          const double T = mesh.intr_face_coef(f);
          J.value()[diag[c0]] += T;
          J.value()[k_12[f]] -= T;
          J.value()[k_21[f]] -= T;
          J.value()[diag[c1]] += T;
      }
      for (const auto& w : wm.wells()) {
          const auto cmd = w.schedule.command(wm.current_event_time());
          if (!cmd.on) continue;
          for (const auto& comp : w.comps)
              J.value()[diag[comp.cell]] += comp.WI;
      }
      J.reset_size();
      return true;
  }

  template<typename V>
  std::pair<bool, ResidualNorm> is_residual_norm_converged(const V& resid) const {
    ResidualNorm rnrm;
    rnrm.evaluate(resid);
    return {rnrm.isLessEqual(RTOL), rnrm};
  }

  template<typename V>
  std::pair<bool, UpdateNorm> is_update_norm_converged(const V& upd) const {
    UpdateNorm dunrm;
    dunrm.evaluate(upd);
    return {dunrm.isLessEqual(DUTOL), dunrm};
  }
  
  void update_flow (const State& p, std::vector<double>& intr_face_vel)
  {
      //std::cout << "Updating Fluxes" << std::endl;
      std::vector<double> vel_sum(p.size(), 0.0);
      assert(p.size() == mesh.num_cells());
      assert(intr_face_vel.size() == mesh.num_intr_faces());
      //const auto& layout = mesh.layout();
      double max_vel = 0.0;
      double ave_vel = 0.0;
      const double rho_g = rho * g;
      std::size_t n_faces = 0;
      //std::cout << "Updating Fluxes" << std::endl;
      for (std::size_t f = 0; f < mesh.num_intr_faces(); ++f) {
          const auto [c0, c1] = mesh.intr_face_to_cells(f);
          const double T = mesh.intr_face_coef(f);
          const double dz = mesh.cell_depth(c1) - mesh.cell_depth(c0);
          double face_vel = 0.001127 * T * (p[c0] - p[c1] + rho_g * dz);
          intr_face_vel[f] = face_vel;
          vel_sum[c0] += face_vel;
          vel_sum[c1] -= face_vel;
          ave_vel += std::abs(face_vel);
          ++n_faces;
          if (std::abs(face_vel) > max_vel)
              max_vel = std::abs(face_vel);
          //std::cout << intr_face_vel[f] << std::endl;
      }
	  
	  for (auto& w: wm.wells()) {
		  const auto cmd = w.schedule.command(wm.current_event_time());
		  
		  w.rates.q_total = 0.0;
		  std::fill(w.rates.q_comp.begin(), w.rates.q_comp.end(), 0.0);
		  if (!cmd.on) continue;
		  
		  for (std::size_t ci = 0; ci < w.comps.size(); ++ci){
			  const auto& comp = w.comps[ci];
			  const double p_well_comp = cmd.bhp + rho * g * comp.d_depth_from_datum;
              const double q = 0.001127 * comp.WI * (p_well_comp - p[comp.cell]);
			  
			  w.rates.q_comp[ci] = q;
			  w.rates.q_total += q;
              vel_sum[comp.cell] -= q;
              ave_vel += std::abs(q);
              ++n_faces;
              if (std::abs(q) > max_vel)
                  max_vel = std::abs(q);
		  }
		  
	  }
      std::cout << "Max face velocity: " << max_vel << std::endl;
      std::cout << "Mean face velocity: " << ave_vel / static_cast<double>(n_faces) << std::endl;
      max_vel = 0;
      for (auto vel : vel_sum) {
          if (std::abs(vel) > max_vel)
              max_vel = std::abs(vel);
      }
      std::cout << "Total Material balance Error: " << max_vel << std::endl;
  
  }

  void initial_guess(State& /*unew*/) const {

  }

  State initialize_state() const 
  {
      std::size_t size = neqs();
      return State(size, 0.0);

  }
private:
  const TPFA_Mesh& mesh;
  WellsManager<TPFA_Mesh>& wm;
  double rho = 0.0;
  double g   = 0.0;
  //double t_eval = 0.0;
  double RTOL  = 1.0e-8;
  double DUTOL = 1.0e-7;

  // Filled once while the solver allocates its CSR storage.
  mutable std::vector<std::size_t> diag, k_12, k_21;
};

#endif // __LINEARDIFFUSIVITYSTEADY_HPP__
