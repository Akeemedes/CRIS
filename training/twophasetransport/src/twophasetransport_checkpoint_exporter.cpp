#include "twophasetransport_model_package.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path checkpoint;
    std::filesystem::path output;
    double beta_max{150000.0};
    bool beta_log1p{false};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help") {
            std::cout << "Usage: twophasetransport_checkpoint_exporter --checkpoint FILE --output MODEL.pt "
                         "[--beta-max VALUE] [--beta-coordinate raw|log1p]\n";
            std::exit(0);
        }
        if (index + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--checkpoint") options.checkpoint = value;
        else if (key == "--output") options.output = value;
        else if (key == "--beta-max") options.beta_max = std::stod(value);
        else if (key == "--beta-coordinate") {
            if (value == "raw") options.beta_log1p = false;
            else if (value == "log1p") options.beta_log1p = true;
            else throw std::runtime_error("--beta-coordinate must be raw or log1p");
        }
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (options.checkpoint.empty() || options.output.empty()) {
        throw std::runtime_error("--checkpoint and --output are required");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto parameters = twophasetransport::model_package::read_checkpoint(options.checkpoint);
        twophasetransport::model_package::write(options.output, parameters, options.beta_max, options.beta_log1p);
        std::cout << "checkpoint=" << options.checkpoint.string() << '\n'
                  << "model_package=" << options.output.string() << '\n'
                  << "parameters=" << parameters.size() << '\n'
                  << "beta_coordinate=" << (options.beta_log1p ? "log1p" : "raw") << '\n';
    } catch (const c10::Error& error) {
        std::cerr << "Torch error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "twophasetransport_checkpoint_exporter error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
