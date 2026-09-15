#ifndef __SIMCONFIG_HPP_
#define __SIMCONFIG_HPP_

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include "Solvers/Nonlinear/LinearSolvePolicy.hpp"

#include "SimUtils.hpp"   // provides ASSERT_WITH_MSG
// #include "SimUtils"     // (your original include looked off; keep only the .hpp)

namespace fs = std::filesystem;

namespace {
    inline std::string to_upper(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return s;
    }
} // namespace

struct SimConfig {
    std::size_t nx = 10000, ny = 1, nz = 1;
    enum class MeshType {Cartesian, Unstructured} mesh_type = MeshType::Cartesian;
    enum class GeometryKind { Uniform, Nonuniform } geom_kind = GeometryKind::Uniform;
    enum class FieldType { Homogeneous, Heterogeneous };

    FieldType k_kind = FieldType::Homogeneous;
    FieldType disp_kind = FieldType::Homogeneous;
    FieldType bc_type = FieldType::Homogeneous;
    FieldType ic_type = FieldType::Homogeneous;
    FieldType IVel_type = FieldType::Homogeneous;
    FieldType BVel_type = FieldType::Homogeneous;

    double dx = 1.0, dy = 1.0, dz = 1.0;
    std::string dx_file, dy_file, dz_file;

    double kx = 1.0, ky = 1.0, kz = 1.0;
    std::string kx_file, ky_file, kz_file;

    double dispx = 0.0, dispy = 0.0, dispz = 0.0;
    std::string dispx_file, dispy_file, dispz_file;

    std::array<double, 6> bndry_val{ {0.0, 0.0, 0.0, 0.0, 0.0, 0.0} };
    std::array<bool, 6>   drchlet_bndry{ {false, false, false, false, false, false} };

    std::array<double, 3> intr_vel_val{ {0.0, 0.0, 0.0} };
    std::array<double, 6> bndr_vel_val{ {0.0, 0.0, 0.0, 0.0, 0.0, 0.0} };

    std::string bc_file;
    

    double init_val = 0.0;
    std::string init_file;

    std::string intr_vel_file;
    std::string bndr_vel_file;

    std::string vol_file;
    std::string cl_file;
    std::string T_file;
    std::string depth_file;
    std::string global_idx_file;

    double rho = 62.4;
    double g = 1.0 / 144.0;
    double mu_w = 1.0;
    double mu_nw = 5.0;

    double Ng = 0.0;
    double n_w = 2.0;
    double n_nw = 2.0;
    double lambda = 1.0;

    double dt_init = 1.0;
    double dt_min = 1e-3;
    double dt_max = 100.0;
    double dt_incr = 1.5;
    double dt_decr = 0.5;

    double t_end = 1000.0;
    double report_freq = 200.0;

    double DUTOL = 1e-8;
    double RTOL = 1e-7;
    double eta = 0.1;
    double linear_tolerance_floor = 1e-12;
    LinearSolvePolicy linear_solve_policy = LinearSolvePolicy::ContinueOnBudget;

    int max_steps_newton = 100;
    int LINMAX = 1000;
    // Zero is the internal inheritance sentinel, never a valid input value.
    int pressure_linmax = 0;
    int pressure_linear_budget() const { return pressure_linmax ? pressure_linmax : LINMAX; }
    int max_backtrack = 10;
    int verbosity = 1;
    // Transport Jacobians evolve at every nonlinear update.  Reusing the
    // previous preconditioner is legacy behavior and is not a valid default
    // for reproducible transport comparisons.
    bool refresh_transport_preconditioner = true;
    bool write_jacobian_snapshot = false;

    enum class NewtonUpdater { Standard, Linesearch } newton_updater = NewtonUpdater::Standard;

    std::string wells_file;

    // CRIS_LEARNED is the deployed surrogate.  CRIS_EXACT routes the same
    // global CRIS formulation through a numerical local inverse and is used
    // only as an accuracy/mechanism reference in paired experiments.
    enum class RunMode { SRDM, CRIS_LEARNED, CRIS_EXACT } mode = RunMode::SRDM;
    std::string cris_model_path;
    // Optional CSV of the raw local coordinates supplied to CRIS.  It is
    // intentionally opt-in: a production simulation otherwise performs no
    // embedding I/O.
    std::string embedding_trace_file;

    std::string output_folder;
};
    template<typename value_t = double>
    inline std::vector<value_t> read_vector_file(const std::string& path)
    {
        std::ifstream in(path);
        std::cout << path << std::endl;
        ASSERT_WITH_MSG(in.is_open(), "Cannot open file: " + path);

        std::vector<value_t> v;
        value_t x;
        while (in >> x) v.push_back(x);

        ASSERT_WITH_MSG(!in.bad(), "Error reading file: " + path);
        ASSERT_WITH_MSG(!v.empty(), "File is empty: " + path);
        return v;
    }

inline SimConfig::FieldType parse_field_type(const std::string& s, const std::string& what)
{
    const std::string s_upper = to_upper(s);
    if (s_upper == "HOMOGENEOUS") return SimConfig::FieldType::Homogeneous;
    if (s_upper == "HETEROGENEOUS") return SimConfig::FieldType::Heterogeneous;
    ASSERT_WITH_MSG(false, "Unknown " + what + " kind: " + s);
    return SimConfig::FieldType::Homogeneous; // unreachable
}

inline SimConfig::NewtonUpdater parse_newton_updater(const std::string& s)
{
    const std::string s_upper = to_upper(s);
    if (s_upper == "STANDARD") return SimConfig::NewtonUpdater::Standard;
    if (s_upper == "LINESEARCH") return SimConfig::NewtonUpdater::Linesearch;
    ASSERT_WITH_MSG(false, "NewtonUpdater must be STANDARD or LINESEARCH");
    return SimConfig::NewtonUpdater::Standard; // unreachable
}

inline SimConfig parse_sim_config(const std::string& folder)
{
    const std::string path = folder + "/sim.txt";
    std::ifstream in(path);
    ASSERT_WITH_MSG(in.is_open(), "Cannot open config: " + path);

    std::cout << "Reading simulation Config" << std::endl;

    SimConfig cfg;
    std::string key;

    cfg.output_folder = folder + "/output";

    // Create output folder, and clear previous results
    try {
        fs::create_directories(cfg.output_folder);
        for (const auto& entry : fs::directory_iterator(cfg.output_folder)) {
            if (entry.is_regular_file()) {
                fs::remove(entry.path());
                std::cout << "Removed: " << entry.path().filename() << "\n";
            }
        }
        std::cout << "Output directory cleared and ready: " << cfg.output_folder << "\n";
    }
    catch (const fs::filesystem_error& e) {
        ASSERT_WITH_MSG(false, std::string("Filesystem error: ") + e.what());
    }

    while (in >> key) {
        key = to_upper(key);
        std::cout << key << std::endl;

        if (key == "NX") {
            ASSERT_WITH_MSG(bool(in >> cfg.nx), "Failed reading NX");
        }
        else if (key == "NY") {
            ASSERT_WITH_MSG(bool(in >> cfg.ny), "Failed reading NY");
        }
        else if (key == "NZ") {
            ASSERT_WITH_MSG(bool(in >> cfg.nz), "Failed reading NZ");
        }
        else if (key == "GEOMETRY") {
            std::string kind;
            ASSERT_WITH_MSG(bool(in >> kind), "Failed reading GEOMETRY kind");
            kind = to_upper(kind);
            if (kind == "UNIFORM") cfg.geom_kind = SimConfig::GeometryKind::Uniform;
            else if (kind == "NONUNIFORM") cfg.geom_kind = SimConfig::GeometryKind::Nonuniform;
            else ASSERT_WITH_MSG(false, "Unknown GEOMETRY: " + kind);

        }
        else if (key == "MESH" || key == "GRID" || key == "TOPOLOGY") {
            std::string kind;
            ASSERT_WITH_MSG(bool(in >> kind), "Failed reading Mesh kind");
            kind = to_upper(kind);
            if (kind == "CARTESIAN") cfg.mesh_type = SimConfig::MeshType::Cartesian;
            else if (kind == "UNSTRUCTURED") cfg.mesh_type = SimConfig::MeshType::Unstructured;
            else ASSERT_WITH_MSG(false, "Unknown MESH: " + kind);

        }
        else if (key == "DX") {
            ASSERT_WITH_MSG(bool(in >> cfg.dx), "Failed reading DX");
        }
        else if (key == "DY") {
            ASSERT_WITH_MSG(bool(in >> cfg.dy), "Failed reading DY");
        }
        else if (key == "DZ") {
            ASSERT_WITH_MSG(bool(in >> cfg.dz), "Failed reading DZ");

        }
        else if (key == "DX_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.dx_file), "Failed reading DX_FILE");
        }
        else if (key == "DY_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.dy_file), "Failed reading DY_FILE");
        }
        else if (key == "DZ_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.dz_file), "Failed reading DZ_FILE");

        }
        else if (key == "K") {
            std::string kind;
            ASSERT_WITH_MSG(bool(in >> kind), "Failed reading K kind");
            cfg.k_kind = parse_field_type(kind, "K");
        }
        else if (key == "KX") {
            ASSERT_WITH_MSG(bool(in >> cfg.kx), "Failed reading KX");
        }
        else if (key == "KY") {
            ASSERT_WITH_MSG(bool(in >> cfg.ky), "Failed reading KY");
        }
        else if (key == "KZ") {
            ASSERT_WITH_MSG(bool(in >> cfg.kz), "Failed reading KZ");
        }
        else if (key == "KX_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.kx_file), "Failed reading KX_FILE");
        }
        else if (key == "KY_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.ky_file), "Failed reading KY_FILE");
        }
        else if (key == "KZ_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.kz_file), "Failed reading KZ_FILE");

        }
        else if (key == "DISPERSION") {
            std::string kind;
            ASSERT_WITH_MSG(bool(in >> kind), "Failed reading DISPERSION kind");
            cfg.disp_kind = parse_field_type(kind, "Dispersion");
        }
        else if (key == "DISPX") {
            ASSERT_WITH_MSG(bool(in >> cfg.dispx), "Failed reading DISPX");
        }
        else if (key == "DISPY") {
            ASSERT_WITH_MSG(bool(in >> cfg.dispy), "Failed reading DISPY");
        }
        else if (key == "DISPZ") {
            ASSERT_WITH_MSG(bool(in >> cfg.dispz), "Failed reading DISPZ");
        }
        else if (key == "DISPX_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.dispx_file), "Failed reading DISPX_FILE");
        }
        else if (key == "DISPY_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.dispy_file), "Failed reading DISPY_FILE");
        }
        else if (key == "DISPZ_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.dispz_file), "Failed reading DISPZ_FILE");

        }
        else if (key == "BOUNDARY_FIELD") {
            std::string kind;
            ASSERT_WITH_MSG(bool(in >> kind), "Failed reading BOUNDARY_FIELD kind");
            cfg.bc_type = parse_field_type(kind, "Boundary");
        }
        else if (key == "BC_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.bc_file), "Failed reading BC_FILE");
        }
        else if (key == "BC_VALUE") {
            double a, b, c, d, e, f;
            ASSERT_WITH_MSG(bool(in >> a >> b >> c >> d >> e >> f), "Failed reading BC_VALUE (6 numbers)");
            cfg.bndry_val = { a,b,c,d,e,f };
        }
        else if (key == "DIRICHLET_BC") {
            int a, b, c, d, e, f;
            ASSERT_WITH_MSG(bool(in >> a >> b >> c >> d >> e >> f), "Failed reading DIRICHLET_BC (6 ints)");
            cfg.drchlet_bndry = { bool(a), bool(b), bool(c), bool(d), bool(e), bool(f) };

        }
        else if (key == "INIT_FIELD") {
            std::string kind;
            ASSERT_WITH_MSG(bool(in >> kind), "Failed reading INIT_FIELD kind");
            cfg.ic_type = parse_field_type(kind, "InitialCondition");
        }
        else if (key == "INIT_VALUE") {
            ASSERT_WITH_MSG(bool(in >> cfg.init_val), "Failed reading IC_VALUE");
        }
        else if (key == "INIT_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.init_file), "Failed reading IC_FILE");

        }
        else if (key == "INTR_VEL_FIELD") {
            std::string kind;
            ASSERT_WITH_MSG(bool(in >> kind), "Failed reading INTR_VEL_FIELD kind");
            cfg.IVel_type = parse_field_type(kind, "InteriorVelocity");
        }
        else if (key == "INTR_VEL_VALS") {
            double vx, vy, vz;
            ASSERT_WITH_MSG(bool(in >> vx >> vy >> vz), "Failed reading INTR_VEL_VALS (3 numbers)");
            cfg.intr_vel_val = { vx, vy, vz };
        }
        else if (key == "INTR_VEL_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.intr_vel_file), "Failed reading INTR_VEL_FILE");

        }
        else if (key == "BNDR_VEL_FIELD") {
            std::string kind;
            ASSERT_WITH_MSG(bool(in >> kind), "Failed reading BNDR_VEL_FIELD kind");
            cfg.BVel_type = parse_field_type(kind, "BoundaryVelocity");
        }
        else if (key == "BNDR_VEL_VALS") {
            double vxm, vxp, vym, vyp, vzm, vzp;
            ASSERT_WITH_MSG(bool(in >> vxm >> vxp >> vym >> vyp >> vzm >> vzp),
                "Failed reading BNDR_VEL_VALS (6 numbers)");
            cfg.bndr_vel_val = { vxm, vxp, vym, vyp, vzm, vzp };
        }
        else if (key == "BNDR_VEL_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.bndr_vel_file), "Failed reading BNDR_VEL_FILE");

        }
        else if (key == "CL_FILE" || key == "CLIST_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.cl_file), "Failed reading connection list file");
        }

        else if (key == "T_FILE" || key == "TRANS_FILE"|| key == "TRANSMISSIBILITIES_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.T_file), "Failed reading connection list file");
        }

        else if (key == "DEPTH_FILE")
        {
            ASSERT_WITH_MSG(bool(in >> cfg.depth_file), "Failed reading depth file");
        }

        else if (key == "VOL_FILE" || key == "VOLUME_FILE")
        {
            ASSERT_WITH_MSG(bool(in >> cfg.vol_file), "Failed reading cell volumes file");

        }

        else if (key == "GLOBAL_IDX_FILE" || key == "GLOBAL_INDEX_FILE")
        {
            ASSERT_WITH_MSG(bool(in >> cfg.global_idx_file), "Failed reading global cell index file");

        }




        else if (key == "RHO") {
            ASSERT_WITH_MSG(bool(in >> cfg.rho), "Failed reading RHO");
        }
        else if (key == "G") {
            ASSERT_WITH_MSG(bool(in >> cfg.g), "Failed reading G");
        }
        else if (key == "MU_W") {
            ASSERT_WITH_MSG(bool(in >> cfg.mu_w), "Failed reading MU_W");
        }
        else if (key == "MU_NW") {
            ASSERT_WITH_MSG(bool(in >> cfg.mu_nw), "Failed reading MU_NW");
        }
        else if (key == "N_W") {
            ASSERT_WITH_MSG(bool(in >> cfg.n_w), "Failed reading N_W");
        }
        else if (key == "N_NW") {
            ASSERT_WITH_MSG(bool(in >> cfg.n_nw), "Failed reading N_NW");
        }
        else if (key == "NG") {
            ASSERT_WITH_MSG(bool(in >> cfg.Ng), "Failed reading NG");
        }
        else if (key == "LAMBDA") {
            ASSERT_WITH_MSG(bool(in >> cfg.lambda), "Failed reading LAMBDA");

        }
        else if (key == "DT_INIT") {
            ASSERT_WITH_MSG(bool(in >> cfg.dt_init), "Failed reading DT_INIT");
        }
        else if (key == "DT_MIN") {
            ASSERT_WITH_MSG(bool(in >> cfg.dt_min), "Failed reading DT_MIN");
        }
        else if (key == "DT_MAX") {
            ASSERT_WITH_MSG(bool(in >> cfg.dt_max), "Failed reading DT_MAX");
        }
        else if (key == "DT_INCR") {
            ASSERT_WITH_MSG(bool(in >> cfg.dt_incr), "Failed reading DT_INCR");
        }
        else if (key == "DT_DECR") {
            ASSERT_WITH_MSG(bool(in >> cfg.dt_decr), "Failed reading DT_DECR");
        }
        else if (key == "T_END") {
            ASSERT_WITH_MSG(bool(in >> cfg.t_end), "Failed reading T_END");
        }
        else if (key == "REPORT_FREQ") {
            ASSERT_WITH_MSG(bool(in >> cfg.report_freq), "Failed reading REPORT_FREQ");

        }
        else if (key == "DUTOL") {
            ASSERT_WITH_MSG(bool(in >> cfg.DUTOL), "Failed reading DUTOL");
        }
        else if (key == "RTOL") {
            ASSERT_WITH_MSG(bool(in >> cfg.RTOL), "Failed reading RTOL");
        }
        else if (key == "MAX_STEPS_NEWTON") {
            ASSERT_WITH_MSG(bool(in >> cfg.max_steps_newton), "Failed reading MAX_STEPS_NEWTON");
        }
        else if (key == "MAX_BACKTRACK") {
            ASSERT_WITH_MSG(bool(in >> cfg.max_backtrack), "Failed reading MAX_BACKTRACK");
            }
        else if (key == "REFRESH_TRANSPORT_PRECONDITIONER") {
            int enabled = 0;
            ASSERT_WITH_MSG(bool(in >> enabled), "Failed reading REFRESH_TRANSPORT_PRECONDITIONER");
            ASSERT_WITH_MSG(enabled == 0 || enabled == 1,
                "REFRESH_TRANSPORT_PRECONDITIONER must be 0 or 1");
            cfg.refresh_transport_preconditioner = (enabled == 1);
        }
        else if (key == "WRITE_JACOBIAN_SNAPSHOT") {
            int enabled = 0;
            ASSERT_WITH_MSG(bool(in >> enabled), "Failed reading WRITE_JACOBIAN_SNAPSHOT");
            ASSERT_WITH_MSG(enabled == 0 || enabled == 1,
                "WRITE_JACOBIAN_SNAPSHOT must be 0 or 1");
            cfg.write_jacobian_snapshot = (enabled == 1);
        }
        else if (key == "LINTOL") {
            ASSERT_WITH_MSG(bool(in >> cfg.eta), "Failed reading LINTOL");
        }
        else if (key == "LINTOL_MIN") {
            ASSERT_WITH_MSG(bool(in >> cfg.linear_tolerance_floor), "Failed reading LINTOL_MIN");
        }
        else if (key == "LINEAR_SOLVE_POLICY") {
            std::string policy;
            ASSERT_WITH_MSG(bool(in >> policy), "Failed reading LINEAR_SOLVE_POLICY");
            policy = to_upper(policy);
            ASSERT_WITH_MSG(policy == "STRICT" || policy == "CONTINUE_ON_BUDGET",
                "LINEAR_SOLVE_POLICY must be STRICT or CONTINUE_ON_BUDGET");
            cfg.linear_solve_policy = policy == "STRICT" ? LinearSolvePolicy::Strict
                : LinearSolvePolicy::ContinueOnBudget;
        }

        else if (key == "LINMAX") {
            ASSERT_WITH_MSG(bool(in >> cfg.LINMAX), "Failed reading LINMAX");
            }
        else if (key == "PRESSURE_LINMAX") {
            ASSERT_WITH_MSG(bool(in >> cfg.pressure_linmax) && cfg.pressure_linmax > 0,
                "PRESSURE_LINMAX must be positive");
        }

        else if (key == "NEWTONUPDATER") {
            std::string v;
            ASSERT_WITH_MSG(bool(in >> v), "Failed reading NEWTONUPDATER value");
            cfg.newton_updater = parse_newton_updater(v);

        }
        else if (key == "WELLS_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.wells_file), "Failed reading WELLS_FILE");

        }
        else if (key == "MODE") {
            std::string m;
            ASSERT_WITH_MSG(bool(in >> m), "Failed reading MODE value");
            m = to_upper(m);
            if (m == "SRDM") cfg.mode = SimConfig::RunMode::SRDM;
            // CRIS remains a backwards-compatible spelling for pre-package
            // case folders. New, reproducible cases should say CRIS_LEARNED.
            else if (m == "CRIS" || m == "CRIS_LEARNED") cfg.mode = SimConfig::RunMode::CRIS_LEARNED;
            else if (m == "CRIS_EXACT") cfg.mode = SimConfig::RunMode::CRIS_EXACT;
            else ASSERT_WITH_MSG(false, "MODE must be SRDM, CRIS_LEARNED, or CRIS_EXACT, got: " + m);

        }
        else if (key == "CRIS_MODEL") {
            ASSERT_WITH_MSG(bool(in >> cfg.cris_model_path), "Failed reading CRIS_MODEL path");

        }
        else if (key == "EMBEDDING_TRACE_FILE") {
            ASSERT_WITH_MSG(bool(in >> cfg.embedding_trace_file),
                "Failed reading EMBEDDING_TRACE_FILE path");
        }
        else if (key == "VERBOSITY" || key == "VERBOSE") {
            ASSERT_WITH_MSG(bool(in >> cfg.verbosity), "Failed reading verbosity option");
        }

        // else {
        //     throw std::runtime_error("Unknown token in config: " + key);
        // }
    }

    // -------------------- validation --------------------

    if (cfg.mesh_type == SimConfig::MeshType::Unstructured) {
        ASSERT_WITH_MSG(!(cfg.cl_file.empty() || cfg.T_file.empty() || cfg.vol_file.empty() || cfg.depth_file.empty()), "Unstructured grid must provide Connection list, transmissibilities, volume and depth files");
    }

    ASSERT_WITH_MSG(cfg.nx > 0 && cfg.ny > 0 && cfg.nz > 0, "NX, NY, NZ must be > 0");

    if (cfg.geom_kind == SimConfig::GeometryKind::Uniform) {
        ASSERT_WITH_MSG(cfg.dx > 0 && cfg.dy > 0 && cfg.dz > 0,
            "Uniform geometry requires DX DY DZ > 0");
        ASSERT_WITH_MSG(cfg.dx_file.empty() && cfg.dy_file.empty() && cfg.dz_file.empty(),
            "Uniform geometry cannot specify DX_FILE/DY_FILE/DZ_FILE");
    }
    else {
        ASSERT_WITH_MSG(!cfg.dx_file.empty() && !cfg.dy_file.empty() && !cfg.dz_file.empty(),
            "Nonuniform geometry requires DX_FILE DY_FILE DZ_FILE");
    }

    if (cfg.k_kind == SimConfig::FieldType::Heterogeneous) {
        ASSERT_WITH_MSG(!cfg.kx_file.empty() && !cfg.ky_file.empty() && !cfg.kz_file.empty(),
            "Heterogeneous K requires KX_FILE KY_FILE KZ_FILE");
    }

    if (cfg.disp_kind == SimConfig::FieldType::Heterogeneous) {
        ASSERT_WITH_MSG(!cfg.dispx_file.empty() && !cfg.dispy_file.empty() && !cfg.dispz_file.empty(),
            "Heterogeneous Dispersion requires DISPX_FILE DISPY_FILE DISPZ_FILE");
    }

    if (cfg.bc_type == SimConfig::FieldType::Heterogeneous) {
        ASSERT_WITH_MSG(!cfg.bc_file.empty(), "Heterogeneous Boundary requires BC_FILE");
    }

    if (cfg.ic_type == SimConfig::FieldType::Heterogeneous) {
        ASSERT_WITH_MSG(!cfg.init_file.empty(), "Heterogeneous Initial Condition requires IC_FILE");
    }

    ASSERT_WITH_MSG(!(cfg.IVel_type == SimConfig::FieldType::Heterogeneous && cfg.intr_vel_file.empty() && cfg.wells_file.empty()),
        "INTR_VEL_FIELD HETEROGENEOUS requires INTR_VEL_FILE or WELLS_FILE");
    ASSERT_WITH_MSG(!(cfg.BVel_type == SimConfig::FieldType::Heterogeneous && cfg.bndr_vel_file.empty() && cfg.wells_file.empty()),
        "BNDR_VEL_FIELD HETEROGENEOUS requires BNDR_VEL_FILE or WELLS_FILE");

    ASSERT_WITH_MSG(cfg.dt_init > 0.0, "DT_INIT must be > 0");
    ASSERT_WITH_MSG(cfg.dt_min > 0.0 && cfg.dt_max > 0.0 && cfg.dt_max >= cfg.dt_min,
        "Require 0 < dt_min <= dt_max");
    ASSERT_WITH_MSG(cfg.dt_incr > 1.0, "DT_INCR must be > 1");
    ASSERT_WITH_MSG(cfg.dt_decr > 0.0 && cfg.dt_decr < 1.0, "DT_DECR must be in (0, 1)");

    ASSERT_WITH_MSG(cfg.rho > 0.0, "RHO must be > 0");
    ASSERT_WITH_MSG(cfg.g >= 0.0, "G must be >= 0");
    ASSERT_WITH_MSG(cfg.mu_w > 0.0, "MU_W must be > 0");
    ASSERT_WITH_MSG(cfg.mu_nw > 0.0, "MU_NW must be > 0");
    ASSERT_WITH_MSG(cfg.n_w > 0.0, "N_W must be > 0");
    ASSERT_WITH_MSG(cfg.n_nw > 0.0, "N_NW must be > 0");

    ASSERT_WITH_MSG(!(cfg.mode == SimConfig::RunMode::CRIS_LEARNED && cfg.cris_model_path.empty()),
        "MODE CRIS_LEARNED requires CRIS_MODEL <path>");


    ASSERT_WITH_MSG(std::isfinite(cfg.eta) && cfg.eta > 0.0 && cfg.eta < 1.0,
        "Require finite 0 < LINTOL < 1");
    ASSERT_WITH_MSG(std::isfinite(cfg.linear_tolerance_floor) && cfg.linear_tolerance_floor > 0.0
        && cfg.linear_tolerance_floor <= cfg.eta, "Require 0 < LINTOL_MIN <= LINTOL");
    ASSERT_WITH_MSG(cfg.LINMAX > 0, "LINMAX must be positive");

    // -------------------- prefix file paths with folder (only if non-empty) --------------------
    auto prefix = [&](std::string& s) {
        if (!s.empty()) s = folder + "/" + s;
        };

    prefix(cfg.cl_file);
    prefix(cfg.T_file);
    prefix(cfg.vol_file);
    prefix(cfg.depth_file);
    prefix(cfg.global_idx_file);

    prefix(cfg.dx_file);
    prefix(cfg.dy_file);
    prefix(cfg.dz_file);

    prefix(cfg.kx_file);
    prefix(cfg.ky_file);
    prefix(cfg.kz_file);

    prefix(cfg.dispx_file);
    prefix(cfg.dispy_file);
    prefix(cfg.dispz_file);

    prefix(cfg.bc_file);
    prefix(cfg.init_file);
    prefix(cfg.intr_vel_file);
    prefix(cfg.bndr_vel_file);
    prefix(cfg.wells_file);
    prefix(cfg.cris_model_path);
    prefix(cfg.embedding_trace_file);


    return cfg;
}

#endif // __SIMCONFIG_HPP_
