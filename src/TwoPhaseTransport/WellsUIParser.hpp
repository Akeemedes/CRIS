#ifndef __WELLSUIPARSER_HPP__
#define __WELLSUIPARSER_HPP__

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <variant>
#include <type_traits>

#include "WellsManager.hpp"  // WellType, VerticalWellSpec, GenericWellSpec,
// PiecewiseConstantBHP, WellCommandBHP, ASSERT_WITH_MSG

namespace wellparser {

    inline std::string to_upper(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return s;
    }

    inline std::string trim_comment(const std::string& line) {
        auto pos = line.find('#');
        return (pos == std::string::npos) ? line : line.substr(0, pos);
    }

    inline bool is_blank(const std::string& s) {
        return !std::any_of(s.begin(), s.end(),
            [](unsigned char c) { return !std::isspace(c); });
    }

    inline bool read_nonempty_line(std::ifstream& in, std::string& out) {
        while (std::getline(in, out)) {
            out = trim_comment(out);
            if (!is_blank(out)) return true;
        }
        return false;
    }

    inline void validate_positive(double val, const std::string& name) {
        ASSERT_WITH_MSG(val > 0.0, name + " must be positive");
    }

    inline void validate_non_negative(long long val, const std::string& name) {
        ASSERT_WITH_MSG(val >= 0, name + " must be non-negative");
    }

    inline std::string first_token_upper(const std::string& line) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        return to_upper(key);
    }

    // Unified spec type
    using WellSpec = std::variant<VerticalWellSpec, GenericWellSpec>;

    // ---------------------------------------------------------------------
    // Read a WELL block, respecting nested END blocks.
    // Nested blocks are introduced by: WI, CELLS, SCHEDULE
    // Each of those is terminated by an END.
    // The WELL itself is terminated by an END that occurs when nesting depth==0.
    // ---------------------------------------------------------------------
    static inline std::vector<std::string> read_well_block(std::ifstream& in,
        const std::string& firstLine)
    {
        std::vector<std::string> lines;
        lines.push_back(firstLine);

        int depth = 0; // nesting level for WI/CELLS/SCHEDULE blocks
        std::string line;

        while (read_nonempty_line(in, line)) {
            lines.push_back(line);

            const std::string key = first_token_upper(line);

            if (key == "WI" || key == "WELLINDEX" || key == "WELL_INDEX" ||
                key == "SCHEDULE" || key == "CELLS")
            {
                ++depth;
                continue;
            }

            if (key == "END") {
                if (depth > 0) {
                    --depth;     // closes WI/CELLS/SCHEDULE
                }
                else {
                    break;       // closes WELL
                }
            }
        }

        return lines;
    }

    // Decide well type from keywords (and forbid mixing)
    static inline bool block_is_generic(const std::vector<std::string>& lines)
    {
        bool saw_generic = false;
        bool saw_vertical = false;

        for (const auto& l : lines) {
            const std::string key = first_token_upper(l);
            if (key == "CELLS" || key == "DATUMCELL") saw_generic = true;
            if (key == "IJ" || key == "K" || key == "DATUMK")     saw_vertical = true;
        }

        ASSERT_WITH_MSG(!(saw_generic && saw_vertical),
            "Well block mixes vertical (IJ/K/DATUMK) and generic (CELLS/DATUMCELL) keywords");

        return saw_generic;
    }

    // Parse a numeric block terminated by END (multiple numbers per line ok).
    static inline std::vector<double> parse_double_block(const std::vector<std::string>& lines,
        std::size_t& idx)
    {
        std::vector<double> vals;
        for (; idx < lines.size(); ++idx) {
            const std::string key = first_token_upper(lines[idx]);
            if (key == "END") { ++idx; break; } // consume END

            std::istringstream ls(lines[idx]);
            double x;
            bool got_any = false;
            while (ls >> x) { vals.push_back(x); got_any = true; }
            ASSERT_WITH_MSG(got_any, "Numeric block line has no numeric values: " + lines[idx]);
        }
        ASSERT_WITH_MSG(!vals.empty(), "Expected numeric block with at least one value");
        return vals;
    }

    static inline std::vector<std::size_t> parse_cells_block(const std::vector<std::string>& lines,
        std::size_t& idx)
    {
        std::vector<std::size_t> vals;
        for (; idx < lines.size(); ++idx) {
            const std::string key = first_token_upper(lines[idx]);
            if (key == "END") { ++idx; break; } // consume END

            std::istringstream ls(lines[idx]);
            long long x;
            bool got_any = false;
            while (ls >> x) {
                validate_non_negative(x, "cell_id");
                vals.push_back(static_cast<std::size_t>(x));
                got_any = true;
            }
            ASSERT_WITH_MSG(got_any, "CELLS line has no integer values: " + lines[idx]);
        }
        ASSERT_WITH_MSG(!vals.empty(), "Expected CELLS block with at least one cell id");
        return vals;
    }

    template <class SpecT>
    static inline void parse_schedule_block(SpecT& w,
        const std::vector<std::string>& lines,
        std::size_t& idx)
    {
        // idx currently points to first schedule line after "SCHEDULE"
        for (; idx < lines.size(); ++idx) {
            const std::string key = first_token_upper(lines[idx]);
            if (key == "END") { ++idx; break; } // consume END

            double t = 0.0, val = 0.0;
            std::string onoff, mode;

            std::istringstream ss(lines[idx]);
            ASSERT_WITH_MSG(bool(ss >> t >> onoff >> mode >> val),
                "Schedule line malformed in well '" + w.name + "': " + lines[idx]);

            ASSERT_WITH_MSG(t >= 0.0, "Schedule time must be non-negative");
            onoff = to_upper(onoff);
            mode = to_upper(mode);

            ASSERT_WITH_MSG(mode == "BHP", "Only BHP mode supported (got " + mode + ")");
            ASSERT_WITH_MSG(onoff == "ON" || onoff == "OFF", "Schedule state must be ON or OFF");

            WellCommandBHP cmd;
            cmd.on = (onoff == "ON");
            cmd.bhp = val;
            w.schedule.add_event(t, cmd);
        }
    }

    // -------------------- Vertical parsing --------------------
    static inline VerticalWellSpec parse_vertical_from_lines(const std::vector<std::string>& lines)
    {
        ASSERT_WITH_MSG(!lines.empty(), "Empty well block");

        VerticalWellSpec w;
        {
            std::istringstream iss(lines[0]);
            std::string tok;
            ASSERT_WITH_MSG(bool(iss >> tok >> w.name), "Malformed WELL header line: " + lines[0]);
            ASSERT_WITH_MSG(to_upper(tok) == "WELL" && !w.name.empty(), "Expected: WELL <name>");
        }

        bool has_type = false, has_ij = false, has_k = false, has_datum = false, has_rw = false, has_skin = false, has_sched = false;
        bool has_WI = false;

        for (std::size_t idx = 1; idx < lines.size(); ) {
            std::istringstream ls(lines[idx]);
            std::string key;
            ASSERT_WITH_MSG(bool(ls >> key), "Malformed line in well '" + w.name + "'");
            key = to_upper(key);

            if (key == "END") { ++idx; break; } // end WELL

            if (key == "TYPE") {
                std::string t;
                ASSERT_WITH_MSG(bool(ls >> t), "TYPE expects: TYPE <PRODUCER|INJECTOR>");
                t = to_upper(t);
                if (t == "PRODUCER") w.type = WellType::Producer;
                else if (t == "INJECTOR") w.type = WellType::Injector;
                else ASSERT_WITH_MSG(false, "TYPE must be PRODUCER or INJECTOR, got: " + t);
                has_type = true; ++idx;
            }
            else if (key == "IJ") {
                long long i, j;
                ASSERT_WITH_MSG(bool(ls >> i >> j), "IJ expects: IJ <i> <j>");
                validate_non_negative(i, "i");
                validate_non_negative(j, "j");
                w.i = static_cast<std::size_t>(i);
                w.j = static_cast<std::size_t>(j);
                has_ij = true; ++idx;
            }
            else if (key == "K") {
                long long k_top, k_bot;
                ASSERT_WITH_MSG(bool(ls >> k_top >> k_bot), "K expects: K <k_top> <k_bot>");
                validate_non_negative(k_top, "k_top");
                validate_non_negative(k_bot, "k_bot");
                ASSERT_WITH_MSG(k_top <= k_bot, "k_top must be <= k_bot");

                VerticalWellSpec::Segment seg;
                seg.k_top = static_cast<std::size_t>(k_top);
                seg.k_bot = static_cast<std::size_t>(k_bot);
                w.segments.push_back(std::move(seg));
                has_k = true; ++idx;
            }
            else if (key == "DATUMK") {
                long long k;
                ASSERT_WITH_MSG(bool(ls >> k), "DATUMK expects: DATUMK <k>");
                validate_non_negative(k, "k_datum");
                w.k_datum = static_cast<std::size_t>(k);
                has_datum = true; ++idx;
            }
            else if (key == "RW") {
                ASSERT_WITH_MSG(bool(ls >> w.rw), "RW expects: RW <radius>");
                validate_positive(w.rw, "RW");
                has_rw = true; ++idx;
            }
            else if (key == "SKIN") {
                ASSERT_WITH_MSG(bool(ls >> w.skin), "SKIN expects: SKIN <value>");
                has_skin = true; ++idx;
            }
            else if (key == "WI" || key == "WELLINDEX" || key == "WELL_INDEX") {
                ASSERT_WITH_MSG(!w.segments.empty(), "WI specified before any K segment in well '" + w.name + "'");
                auto& seg = w.segments.back();

                double wi;
                while (ls >> wi) seg.WI.push_back(wi);

                ++idx;
                if (seg.WI.empty()) {
                    seg.WI = parse_double_block(lines, idx);
                }

                has_WI = true;
            }
            else if (key == "SCHEDULE") {
                has_sched = true;
                ++idx;
                parse_schedule_block(w, lines, idx);
            }
            else {
                ASSERT_WITH_MSG(false, "Unknown keyword in vertical well '" + w.name + "': " + key);
            }
        }

        // Updated required-fields logic:
        // If WI was provided anywhere, RW/SKIN are optional.
        const bool ok_geom = has_ij && has_k && has_datum;
        const bool ok_aux = (has_WI) ? true : (has_rw && has_skin);

        ASSERT_WITH_MSG(has_type && ok_geom && ok_aux && has_sched,
            "Vertical well '" + w.name + "' missing required fields");

        // WI length check (if provided)
        for (std::size_t s = 0; s < w.segments.size(); ++s) {
            const auto& seg = w.segments[s];
            const std::size_t nperf = seg.k_bot - seg.k_top + 1;
            if (!seg.WI.empty()) {
                ASSERT_WITH_MSG(seg.WI.size() == nperf,
                    "Vertical well '" + w.name + "': WI count mismatch for segment " + std::to_string(s));
            }
        }

        w.schedule.finalize();
        return w;
    }

    // -------------------- Generic parsing --------------------
    static inline GenericWellSpec parse_generic_from_lines(const std::vector<std::string>& lines)
    {
        ASSERT_WITH_MSG(!lines.empty(), "Empty well block");

        for (auto& line : lines)
            std::cout << line << std::endl;

        GenericWellSpec w;
        {
            std::istringstream iss(lines[0]);
            std::string tok;
            ASSERT_WITH_MSG(bool(iss >> tok >> w.name), "Malformed WELL header line: " + lines[0]);
            ASSERT_WITH_MSG(to_upper(tok) == "WELL" && !w.name.empty(), "Expected: WELL <name>");
        }

        bool has_type = false, has_datum = false, has_rw = false, has_skin = false, has_sched = false, has_cells = false;
        bool has_WI = false;

        for (std::size_t idx = 1; idx < lines.size(); ) {
            std::istringstream ls(lines[idx]);
            std::string key;
            ASSERT_WITH_MSG(bool(ls >> key), "Malformed line in well '" + w.name + "'");
            key = to_upper(key);

            if (key == "END") { ++idx; break; } // end WELL

            if (key == "TYPE") {
                std::string t;
                ASSERT_WITH_MSG(bool(ls >> t), "TYPE expects: TYPE <PRODUCER|INJECTOR>");
                t = to_upper(t);
                if (t == "PRODUCER") w.type = WellType::Producer;
                else if (t == "INJECTOR") w.type = WellType::Injector;
                else ASSERT_WITH_MSG(false, "TYPE must be PRODUCER or INJECTOR, got: " + t);
                has_type = true; ++idx;
            }
            else if (key == "DATUMCELL") {
                long long c;
                ASSERT_WITH_MSG(bool(ls >> c), "DATUMCELL expects: DATUMCELL <cell_id>");
                validate_non_negative(c, "datum_cell");
                w.datum_cell = static_cast<std::size_t>(c);
                has_datum = true; ++idx;
            }
            else if (key == "RW") {
                ASSERT_WITH_MSG(bool(ls >> w.rw), "RW expects: RW <radius>");
                validate_positive(w.rw, "RW");
                has_rw = true; ++idx;
            }
            else if (key == "SKIN") {
                ASSERT_WITH_MSG(bool(ls >> w.skin), "SKIN expects: SKIN <value>");
                has_skin = true; ++idx;
            }
            else if (key == "CELLS") {
                ++idx; // first line after CELLS
                GenericWellSpec::Segment seg;
                seg.cell_ids = parse_cells_block(lines, idx);
                w.segments.push_back(std::move(seg));
                has_cells = true;
            }
            else if (key == "WI" || key == "WELLINDEX" || key == "WELL_INDEX") {
                ASSERT_WITH_MSG(!w.segments.empty(), "WI specified before any CELLS segment in well '" + w.name + "'");
                auto& seg = w.segments.back();

                double wi;
                while (ls >> wi) seg.WI.push_back(wi);

                ++idx;
                if (seg.WI.empty()) {
                    seg.WI = parse_double_block(lines, idx);
                }

                has_WI = true;
            }
            else if (key == "SCHEDULE") {
                has_sched = true;
                ++idx;
                parse_schedule_block(w, lines, idx);
            }
            else {
                ASSERT_WITH_MSG(false, "Unknown keyword in generic well '" + w.name + "': " + key);
            }
        }

        // Updated required-fields logic:
        // If WI was provided anywhere, RW/SKIN are optional.
        const bool ok_geom = has_datum && has_cells;
        const bool ok_aux = (has_WI) ? true : (has_rw && has_skin);

        ASSERT_WITH_MSG(has_type && ok_geom && ok_aux && has_sched,
            "Generic well '" + w.name + "' missing required fields");

        // WI length check (if provided)
        for (std::size_t s = 0; s < w.segments.size(); ++s) {
            const auto& seg = w.segments[s];
            ASSERT_WITH_MSG(!seg.cell_ids.empty(), "Generic well '" + w.name + "': empty CELLS segment");
            if (!seg.WI.empty()) {
                ASSERT_WITH_MSG(seg.WI.size() == seg.cell_ids.size(),
                    "Generic well '" + w.name + "': WI count mismatch for segment " + std::to_string(s));
            }
        }

        w.schedule.finalize();
        return w;
    }

    static inline WellSpec parse_one_well_any(std::ifstream& in, const std::string& firstLine)
    {
        auto lines = read_well_block(in, firstLine);
        if (block_is_generic(lines)) {
            return WellSpec{ parse_generic_from_lines(lines) };
        }
        else {
            return WellSpec{ parse_vertical_from_lines(lines) };
        }
    }

    static inline std::vector<WellSpec> parse_wells_file(const std::string& path)
    {
        std::ifstream in(path);
        ASSERT_WITH_MSG(in.is_open(), "Could not open wells file: " + path);

        std::vector<WellSpec> wells;
        std::string line;

        while (read_nonempty_line(in, line)) {
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            ASSERT_WITH_MSG(to_upper(key) == "WELL", "Expected WELL keyword, got: " + key);
            wells.push_back(parse_one_well_any(in, line));
        }

        return wells;
    }

} // namespace wellparser

#endif // __WELLSUIPARSER_HPP__