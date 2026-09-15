#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> kMagic{'T', 'P', 'T', 'B', 'I', 'N', '0', '1'};

struct Header {
    std::array<char, 8> magic{kMagic};
    std::uint64_t count{0};
    std::uint32_t fields{6};
    std::uint32_t reserved{0};
};

struct Record {
    double a;
    double beta;
    double u_old;
    double root;
    double dr_du;
    double representation_limited;
};

static_assert(sizeof(Header) == 24, "Unexpected binary header layout");
static_assert(sizeof(Record) == 48, "Unexpected binary record layout");

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    std::uint64_t max_records{0};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help") {
            std::cout << "Usage: twophasetransport_dataset_packer --input DATASET.csv --output DATASET.tptbin "
                         "[--max-records N]\n";
            std::exit(0);
        }
        if (++index >= argc) throw std::runtime_error("Missing value for " + key);
        if (key == "--input") options.input = argv[index];
        else if (key == "--output") options.output = argv[index];
        else if (key == "--max-records") options.max_records = std::stoull(argv[index]);
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (options.input.empty() || options.output.empty()) {
        throw std::runtime_error("--input and --output are required");
    }
    return options;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> values;
    std::stringstream stream(line);
    std::string value;
    while (std::getline(stream, value, ',')) values.push_back(value);
    return values;
}

std::size_t require_column(const std::vector<std::string>& header, const std::string& name) {
    for (std::size_t index = 0; index < header.size(); ++index) {
        if (header[index] == name) return index;
    }
    throw std::runtime_error("Input CSV lacks required column: " + name);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::ifstream input(options.input);
        if (!input) throw std::runtime_error("Unable to read CSV: " + options.input.string());
        std::string line;
        if (!std::getline(input, line)) throw std::runtime_error("CSV is empty");
        const auto header = split_csv(line);
        const std::size_t a_column = require_column(header, "a");
        const std::size_t beta_column = require_column(header, "beta");
        const std::size_t u_old_column = require_column(header, "u_old");
        const std::size_t root_column = require_column(header, "root");
        const std::size_t derivative_column = require_column(header, "dr_du");
        const std::size_t limited_column = require_column(header, "representation_limited");

        std::vector<Record> records;
        records.reserve(200'000);
        while (std::getline(input, line)) {
            if (options.max_records != 0 && records.size() >= options.max_records) break;
            const auto fields = split_csv(line);
            if (fields.size() != header.size()) throw std::runtime_error("Malformed CSV row");
            records.push_back({
                std::stod(fields[a_column]), std::stod(fields[beta_column]), std::stod(fields[u_old_column]),
                std::stod(fields[root_column]), std::stod(fields[derivative_column]), std::stod(fields[limited_column])});
        }

        if (!options.output.parent_path().empty()) std::filesystem::create_directories(options.output.parent_path());
        std::ofstream output(options.output, std::ios::binary);
        if (!output) throw std::runtime_error("Unable to write packed dataset: " + options.output.string());
        Header binary_header;
        binary_header.count = records.size();
        output.write(reinterpret_cast<const char*>(&binary_header), sizeof(binary_header));
        output.write(reinterpret_cast<const char*>(records.data()), static_cast<std::streamsize>(records.size() * sizeof(Record)));
        if (!output) throw std::runtime_error("Failed while writing packed dataset");

        std::cout << "input_rows=" << records.size() << '\n'
                  << "record_bytes=" << sizeof(Record) << '\n'
                  << "packed_bytes=" << (sizeof(binary_header) + records.size() * sizeof(Record)) << '\n'
                  << "wrote_dataset=" << options.output.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "twophasetransport_dataset_packer error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
