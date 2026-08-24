#include "huygens_cli.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/huygens_sweep.hpp>

#include <chrono>
#include <exception>
#include <iostream>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr float kPi = 3.14159265358979323846f;
}

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        huygens_cli::initialize(argc, argv);

        const std::string velocity = huygens_cli::required_string("velocity");
        const std::string block_file = huygens_cli::required_string("block_file");
        const std::string table_prefix = huygens_cli::required_string("table_prefix");
        const std::string output_real = huygens_cli::optional_string(
            "output_real", "wave_frequency_direct_real.rsf");
        const std::string output_imag = huygens_cli::optional_string(
            "output_imag", "wave_frequency_direct_imag.rsf");
        const std::string timing_file = huygens_cli::optional_string(
            "timing", "wave_frequency_direct_timing.rsf");

        const float frequency = huygens_cli::optional_float("frequency", 25.0f);
        const float source_amplitude = huygens_cli::optional_float(
            "source_amplitude", 1.0f);
        const float source_radius = huygens_cli::optional_float("source_radius", 0.0f);
        const int source_ix_parameter = huygens_cli::optional_int("source_ix", -1);
        const int source_iz = huygens_cli::optional_int("source_iz", 0);
        const int source_stride = huygens_cli::optional_int("source_stride", 1);
        const int target_x_stride = huygens_cli::optional_int("target_x_stride", 1);
        const int target_z_stride = huygens_cli::optional_int("target_z_stride", 1);
        const int threads = huygens_cli::optional_int("threads", 0);

        if (!(frequency > 0.0f) || source_stride < 1) {
            throw std::invalid_argument("frequency and source_stride must be positive");
        }
        if (target_x_stride != 1 || target_z_stride != 1) {
            throw std::invalid_argument(
                "minimal recursive prototype requires target_x_stride=1 and target_z_stride=1");
        }
        huygens_cli::set_openmp_threads(threads);

        const se::huygens::Model2D model = se::huygens::read_velocity_model(velocity);
        const se::huygens::BlockInfo info = se::huygens::read_block_info(block_file);
        se::huygens::validate_block_info(info, &model);

        const int source_ix = source_ix_parameter >= 0
            ? source_ix_parameter
            : model.nx / 2;
        if (source_ix < 0 || source_ix >= model.nx || source_iz != 0) {
            throw std::invalid_argument(
                "source must satisfy 0<=source_ix<nx and source_iz=0 in this prototype");
        }

        std::vector<se::huygens::Complex> wavefield;
        se::huygens::initialize_first_block_hankel(
            model, info, frequency, source_ix, source_iz,
            source_amplitude, source_radius, wavefield);

        const int propagation_blocks = static_cast<int>(info.blocks.size()) - 1;
        const int metrics = 3;
        std::vector<float> timing_values(
            static_cast<std::size_t>(metrics) * propagation_blocks, 0.0f);
        double total_apply_seconds = 0.0;
        const float omega = 2.0f * kPi * frequency;

        for (std::size_t iblock = 1; iblock < info.blocks.size(); ++iblock) {
            const se::huygens::Block& block = info.blocks[iblock];
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block, source_stride,
                    target_x_stride, target_z_stride);
            const se::huygens::LayerTables tables =
                se::huygens::read_layer_tables(table_prefix, block.id);
            se::huygens::validate_layer_tables(tables, geometry, block);

            const std::vector<float> weights =
                se::huygens::trapezoidal_weights(model, geometry.source_ix);
            const std::vector<se::huygens::Complex> boundary_input =
                se::huygens::gather_boundary_cauchy_data(
                    model, block, geometry.source_ix, wavefield);

            se::huygens::KernelData kernel;
            kernel.rows = static_cast<int>(geometry.targets.size());
            kernel.sources = static_cast<int>(geometry.source_ix.size());
            kernel.omega = omega;
            kernel.quadrature_weights = &weights;
            kernel.tables = &tables;

            std::cout << "Direct block " << block.id
                      << ": matrix " << kernel.rows << " x "
                      << 2 * kernel.sources << '\n';

            const auto started = Clock::now();
            const std::vector<se::huygens::Complex> target_values =
                se::huygens::direct_apply(kernel, boundary_input);
            const double seconds = std::chrono::duration<double>(
                Clock::now() - started).count();
            total_apply_seconds += seconds;

            se::huygens::inject_target_values(
                model, geometry.targets, target_values, wavefield);

            const int column = static_cast<int>(iblock) - 1;
            timing_values[static_cast<std::size_t>(column) * metrics + 0] =
                static_cast<float>(seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 1] =
                static_cast<float>(kernel.sources);
            timing_values[static_cast<std::size_t>(column) * metrics + 2] =
                static_cast<float>(kernel.rows);
        }

        se::huygens::write_wavefield_component_rsf(
            output_real, model, wavefield, frequency, "direct_huygens_sweep", false);
        se::huygens::write_wavefield_component_rsf(
            output_imag, model, wavefield, frequency, "direct_huygens_sweep", true);
        se::huygens::write_timing_rsf(
            timing_file, timing_values, metrics, propagation_blocks,
            "direct_huygens_sweep",
            {"direct_apply_seconds", "source_points", "target_points"},
            {{"total_direct_apply_seconds",
              static_cast<float>(total_apply_seconds)},
             {"frequency_hz", frequency}});

        std::cout << "Direct layered Huygens sweep completed.\n"
                  << "Total direct integration time (table I/O excluded): "
                  << total_apply_seconds << " s\n"
                  << "Real: " << output_real << '\n'
                  << "Imag: " << output_imag << '\n'
                  << "Timing: " << timing_file << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "wave_frequency_direct: " << error.what() << '\n';
        return 1;
    }
}
