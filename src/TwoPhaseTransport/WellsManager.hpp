#ifndef __WELLSMANAGERHPP_
#define __WELLSMANAGERHPP_

#include <cstddef>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "SimUtils.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif
// ============================ Schedule =======================================
struct WellCommandBHP {
    bool on = false;
    double bhp = 10000; 
};

class PiecewiseConstantBHP {
public:
    void add_event(double t, WellCommandBHP cmd) { events_.push_back({ t, cmd }); }

    void finalize() {
        std::sort(events_.begin(), events_.end(),
            [](auto const& a, auto const& b) { return a.t < b.t; });
        //if (events_.empty()) throw std::runtime_error("Well schedule has no events");

        // Check for duplicates
        auto it = std::adjacent_find(events_.begin(), events_.end(),
            [](auto const& a, auto const& b) { return a.t == b.t; });
        if (it != events_.end()) {
            throw std::runtime_error("Duplicate event times in schedule");
        }
    }

    WellCommandBHP command(double t) const {
        assert(!events_.empty());
        auto it = std::upper_bound(events_.begin(), events_.end(), t,
            [](double x, Event const& e) { return x < e.t; });
        if (it == events_.begin()) return WellCommandBHP();
        return std::prev(it)->cmd;
    }

    double next_event_time(double t) const {
        assert(!events_.empty());
        auto it = std::upper_bound(events_.begin(), events_.end(), t,
            [](double x, Event const& e) { return x < e.t; });
        if (it == events_.end()) return std::numeric_limits<double>::infinity();
        return it->t;
    }

private:
    struct Event { double t; WellCommandBHP cmd; };
    std::vector<Event> events_;
};

// ============================ Well spec/types ================================
enum class WellType { Producer, Injector };


struct GenericWellSpec {
    struct Segment {
        std::vector<std::size_t> cell_ids; // 0-based
        std::vector<double> WI;            // optional; if present, must match cell_ids.size()
    };

    std::string name;
    WellType type = WellType::Producer;

    std::vector<Segment> segments; // must be non-empty
    std::size_t datum_cell = 0;    // 0-based cell id (BHP reference)
    double rw = 0.1;
    double skin = 0.0;
    PiecewiseConstantBHP schedule;
};


struct VerticalWellSpec {
    struct Segment {
        std::size_t k_top = 0;
        std::size_t k_bot = 0;
        std::vector<double> WI; // optional; if present, must match (k_bot-k_top+1)
    };

    std::string name;
    WellType type = WellType::Producer;
    std::size_t i = 0, j = 0;

    std::vector<Segment> segments;   // NEW
    std::size_t k_datum = 0;

    double rw = 0.1;
    double skin = 0.0;

    PiecewiseConstantBHP schedule;
};

struct WellCompletion {
    std::size_t cell = 0;
    double WI = 0.0;
    // depth difference (meters, positive downward) between completion center and datum center:
    double d_depth_from_datum = 0.0;
};

struct WellRates {
    std::vector<double> q_comp;  // per completion
    double q_total = 0.0;
};

struct Well {
    std::string name;
    WellType type = WellType::Producer;
    std::vector<WellCompletion> comps;
    WellRates rates;
    PiecewiseConstantBHP schedule;
};

// ============================ Peaceman WI ====================================
inline double peaceman_re(double dx, double dy) {
    return 0.14 * std::sqrt(dx * dx + dy * dy);
}

inline double peaceman_WI(double kxy, double dz, double dx, double dy,
    double rw, double skin)
{
    const double re = peaceman_re(dx, dy);
    return (2.0 * M_PI * kxy * dz) / (std::log(re / rw) + skin);
}

// q positive into reservoir:
inline double bhp_completion_rate(double WI, double p_cell, double p_well_comp) {
    return WI * (p_well_comp - p_cell);
}

// ============================ WellsManager ====================================
// Mesh requirements:
//  - using index_t
//  - dims() -> {nx,ny,nz}
//  - ijk_to_cell(i,j,k)
//  - get_cell_dims(cell) -> {dx,dy,dz}
//  - coeffs() -> KPolicy where K(dir,cell)
template<class Mesh>
class WellsManager {
public:
    using index_t = typename Mesh::index_t;

    WellsManager (const Mesh& _mesh):mesh(_mesh) {}

    void add_vertical_well(VerticalWellSpec spec)
    {
        const auto dims = mesh.dims();
        const std::size_t nx = dims[0], ny = dims[1], nz = dims[2];

        ASSERT_WITH_MSG(spec.i < nx && spec.j < ny, "Well IJ out of bounds");
        ASSERT_WITH_MSG(spec.k_datum < nz, "Well DATUMK out of bounds");
        ASSERT_WITH_MSG(!spec.segments.empty(), "Well must have at least one K segment");

        // Validate segments + WI sizing (if provided)
        std::size_t nperf_total = 0;
        for (std::size_t s = 0; s < spec.segments.size(); ++s) {
            const auto& seg = spec.segments[s];
            ASSERT_WITH_MSG(seg.k_top <= seg.k_bot, "Well segment has k_top > k_bot");
            ASSERT_WITH_MSG(seg.k_bot < nz, "Well segment out of bounds (k_bot >= nz)");

            const std::size_t nseg = seg.k_bot - seg.k_top + 1;
            nperf_total += nseg;

            if (!seg.WI.empty()) {
                ASSERT_WITH_MSG(seg.WI.size() == nseg,
                    "Well segment WI size mismatch. Expected (k_bot-k_top+1) values.");
            }
        }

        // For unstructured meshes, require WI per segment
        if constexpr (!Mesh::is_structured) {
            for (const auto& seg : spec.segments) {
                ASSERT_WITH_MSG(!seg.WI.empty(),
                    "Unstructured mesh requires WI per segment (provide WI block after each K)");
            }
        }

        spec.schedule.finalize();

        Well w;
        //w.spec = std::move(spec);
        w.name = spec.name;
        w.type = spec.type;
        w.schedule = spec.schedule;

        w.comps.reserve(nperf_total);
        w.rates.q_comp.assign(nperf_total, 0.0);

        const auto datum_cell = mesh.ijk_to_cell(spec.i, spec.j, spec.k_datum);
        const double depth_datum = mesh.cell_depth(datum_cell);

        std::size_t perf_idx = 0;
        for (std::size_t s = 0; s < spec.segments.size(); ++s) {
            const auto& seg = spec.segments[s];

            for (std::size_t kk = seg.k_top; kk <= seg.k_bot; ++kk, ++perf_idx) {
                const index_t cell = mesh.ijk_to_cell(spec.i, spec.j, kk);

                WellCompletion c;
                c.cell = static_cast<std::size_t>(cell);
                c.d_depth_from_datum = mesh.cell_depth(cell) - depth_datum;
                //std::cout << "k: " << kk << "\tDepth: " << c.d_depth_from_datum << std::endl;

                if constexpr (Mesh::is_structured) {
                    if (!seg.WI.empty()) {
                        c.WI = seg.WI[kk - seg.k_top];
                    }
                    else {
                        const auto& K = mesh.coeffs();
                        const auto cell_dims = mesh.get_cell_dims(cell);

                        const double dx = cell_dims[0];
                        const double dy = cell_dims[1];
                        const double dz = cell_dims[2];

                        const double Kx = K(0, cell);
                        const double Ky = K(1, cell);
                        const double kxy = std::sqrt(Kx * Ky);

                        c.WI = peaceman_WI(kxy, dz, dx, dy, spec.rw, spec.skin);
                    }
                }
                else {
                    c.WI = seg.WI[kk - seg.k_top];
                }

                w.comps.push_back(c);
            }
        }

        ASSERT_WITH_MSG(w.comps.size() == nperf_total, "Internal error: completion count mismatch");
        ASSERT_WITH_MSG(w.rates.q_comp.size() == nperf_total, "Internal error: q_comp size mismatch");

        mWells.push_back(std::move(w));
    }

    void add_generic_well(GenericWellSpec spec)
    {
        const std::size_t nc = mesh.num_cells();

        ASSERT_WITH_MSG(!spec.segments.empty(), "Well must have at least one segment");
        ASSERT_WITH_MSG(spec.datum_cell < nc, "Well DATUMCELL out of bounds");

        // Validate segments + WI sizing
        std::size_t nperf_total = 0;
        for (std::size_t s = 0; s < spec.segments.size(); ++s) {
            const auto& seg = spec.segments[s];
            ASSERT_WITH_MSG(!seg.cell_ids.empty(), "Well segment must contain at least one cell id");
            nperf_total += seg.cell_ids.size();

            for (auto cid : seg.cell_ids) {
                ASSERT_WITH_MSG(cid < nc, "Well completion cell id out of bounds");
            }

            if (!seg.WI.empty()) {
                ASSERT_WITH_MSG(seg.WI.size() == seg.cell_ids.size(),
                    "Well segment WI size mismatch. Expected one WI per completion cell.");
            }
            else {
                // For generic/unstructured wells, require WI explicitly unless you implement a fallback
                ASSERT_WITH_MSG(!seg.WI.empty(),
                    "Generic well requires WI per completion (provide WI list matching CELLS)");
            }
        }

        spec.schedule.finalize();

        Well w;
        //w.spec = std::move(spec);
        w.name = spec.name;
        w.type = spec.type;
        w.schedule = spec.schedule;
        w.comps.reserve(nperf_total);
        w.rates.q_comp.assign(nperf_total, 0.0);

        const index_t datum_cell = static_cast<index_t>(spec.datum_cell);
        const double depth_datum = mesh.cell_depth(datum_cell);

        std::size_t perf_idx = 0;
        for (std::size_t s = 0; s < spec.segments.size(); ++s) {
            const auto& seg = spec.segments[s];

            for (std::size_t p = 0; p < seg.cell_ids.size(); ++p, ++perf_idx) {
                const index_t cell = static_cast<index_t>(seg.cell_ids[p]);

                WellCompletion c;
                c.cell = static_cast<std::size_t>(cell);

                c.d_depth_from_datum = mesh.cell_depth(cell) - depth_datum;

                c.WI = seg.WI[p];
                std::cout << "Segment: " << p << ", WI: " << c.WI << std::endl;

                w.comps.push_back(c);
            }
        }

        ASSERT_WITH_MSG(w.comps.size() == nperf_total, "Internal error: completion count mismatch");
        ASSERT_WITH_MSG(w.rates.q_comp.size() == nperf_total, "Internal error: q_comp size mismatch");

        mWells.push_back(std::move(w));
    }
    
    std::size_t num_wells() const { return mWells.size(); }
    std::vector<Well>& wells() { return mWells; }
    const std::vector<Well>& wells() const { return mWells; }

    double current_event_time() const { return t_current; }

    double update_event_time() {
        double next = std::numeric_limits<double>::infinity();
        for (auto const& w : mWells) next = std::min(next, w.schedule.next_event_time(t_current));
        t_current = next;
        return t_current;
    }

private:
    const Mesh& mesh;
    std::vector<Well> mWells;
    double t_current{ 0 };
};

#endif // __WELLSMANAGERHPP_ included
