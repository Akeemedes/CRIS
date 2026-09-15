#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> kMagic{'T', 'P', 'T', 'B', 'I', 'N', '0', '1'};

struct Header {
    std::array<char, 8> magic;
    std::uint64_t count;
    std::uint32_t fields;
    std::uint32_t reserved;
};

struct Record {
    double a;
    double beta;
    double u_old;
    double root;
    double dr_du;
    double representation_limited;
};

static_assert(sizeof(Header) == 24);
static_assert(sizeof(Record) == 48);

struct Options {
    std::filesystem::path input;
    std::filesystem::path output_directory;
    int batch_rows{0};
    std::uint64_t seed{20260904};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help") {
            std::cout << "Usage: twophasetransport_batch_partitioner --input DATA.tptbin --output-directory DIR --batch-rows N [--seed N]\n";
            std::exit(0);
        }
        if (++index >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[index];
        if (key == "--input") options.input = value;
        else if (key == "--output-directory") options.output_directory = value;
        else if (key == "--batch-rows") options.batch_rows = std::stoi(value);
        else if (key == "--seed") options.seed = std::stoull(value);
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (options.input.empty() || options.output_directory.empty() || options.batch_rows <= 0) {
        throw std::runtime_error("--input, --output-directory, and positive --batch-rows are required");
    }
    return options;
}

std::vector<Record> load_dataset(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Unable to read packed dataset: " + path.string());
    Header header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream || header.magic != kMagic || header.fields != 6) {
        throw std::runtime_error("Invalid packed dataset: " + path.string());
    }
    std::vector<Record> records(header.count);
    stream.read(reinterpret_cast<char*>(records.data()), static_cast<std::streamsize>(records.size() * sizeof(Record)));
    if (!stream) throw std::runtime_error("Truncated packed dataset: " + path.string());
    return records;
}

void write_batch(const std::filesystem::path& path, const std::vector<Record>& source,
    const std::vector<std::size_t>& permutation, std::size_t begin, std::size_t end) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Unable to write batch: " + path.string());
    Header header{kMagic, static_cast<std::uint64_t>(end - begin), 6, 0};
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (std::size_t index = begin; index < end; ++index) {
        const Record& row = source[permutation[index]];
        stream.write(reinterpret_cast<const char*>(&row), sizeof(row));
    }
    if (!stream) throw std::runtime_error("Failed while writing batch: " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (std::filesystem::exists(options.output_directory)) {
            throw std::runtime_error("Output directory already exists; refuse to replace a batch partition: " + options.output_directory.string());
        }
        const std::vector<Record> records = load_dataset(options.input);
        std::vector<std::size_t> permutation(records.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::mt19937_64 engine(options.seed);
        std::shuffle(permutation.begin(), permutation.end(), engine);
        std::filesystem::create_directories(options.output_directory);
        const std::size_t batch_rows = static_cast<std::size_t>(options.batch_rows);
        const std::size_t batch_count = (records.size() + batch_rows - 1) / batch_rows;
        std::ofstream manifest(options.output_directory / "batch_manifest.json");
        if (!manifest) throw std::runtime_error("Unable to write batch manifest");
        manifest << std::setprecision(17)
                 << "{\n  \"artifact\": \"fixed_pre_shuffled_packed_batches\",\n"
                 << "  \"source\": \"" << options.input.generic_string() << "\",\n"
                 << "  \"source_rows\": " << records.size() << ",\n"
                 << "  \"seed\": " << options.seed << ",\n"
                 << "  \"requested_batch_rows\": " << batch_rows << ",\n"
                 << "  \"batch_count\": " << batch_count << ",\n"
                 << "  \"batches\": [\n";
        for (std::size_t batch = 0; batch < batch_count; ++batch) {
            const std::size_t begin = batch * batch_rows;
            const std::size_t end = std::min(begin + batch_rows, records.size());
            const std::string name = "batch_" + std::to_string(batch) + ".tptbin";
            write_batch(options.output_directory / name, records, permutation, begin, end);
            manifest << "    {\"file\": \"" << name << "\", \"rows\": " << (end - begin) << "}"
                     << (batch + 1 == batch_count ? "\n" : ",\n");
        }
        manifest << "  ]\n}\n";
        std::cout << "source_rows=" << records.size() << '\n'
                  << "batch_count=" << batch_count << '\n'
                  << "batch_rows=" << batch_rows << '\n'
                  << "wrote_batch_directory=" << options.output_directory.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "twophasetransport_batch_partitioner error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
