#include "huygens_cli.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/huygens_sweep.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

std::vector<int> make_rhs_sources(int nx,
                                  int rhs_count,
                                  int requested_source_ix)
{
    if (rhs_count < 1) throw std::invalid_argument("rhs_count must be positive");
    if (rhs_count == 1) {
        const int source = requested_source_ix >= 0 ? requested_source_ix : nx / 2;
        if (source < 0 || source >= nx) throw std::invalid_argument("source_ix is outside model");
        return {source};
    }
    std::vector<int> sources(static_cast<std::size_t>(rhs_count));
    for (int rhs = 0; rhs < rhs_count; ++rhs) {
        sources[static_cast<std::size_t>(rhs)] = static_cast<int>(
            std::llround((nx - 1.0) * rhs / (rhs_count - 1.0)));
    }
    return sources;
}

} // namespace

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
        const float filter_dt = huygens_cli::optional_float("filter_dt", 0.001f);
        const float filter_length = huygens_cli::optional_float("filter_length", 0.025f);
        const int filter_lookup_subsamples =
            huygens_cli::optional_int("filter_lookup_subsamples", 64);
        const float source_amplitude = huygens_cli::optional_float(
            "source_amplitude", 1.0f);
        const float source_radius = huygens_cli::optional_float("source_radius", 0.0f);
        const int source_ix_parameter = huygens_cli::optional_int("source_ix", -1);
        const int source_iz = huygens_cli::optional_int("source_iz", 0);
        const int rhs_count = huygens_cli::optional_int("rhs_count", 1);
        const int output_rhs_parameter = huygens_cli::optional_int("output_rhs", -1);
        const int source_stride = huygens_cli::optional_int("source_stride", 1);
        const int target_x_stride = huygens_cli::optional_int("target_x_stride", 1);
        const int target_z_stride = huygens_cli::optional_int("target_z_stride", 1);
        const int threads = huygens_cli::optional_int("threads", 0);

        if (!(frequency > 0.0f) || !(filter_dt > 0.0f) ||
            !(filter_length > 0.0f) || filter_lookup_subsamples < 1 ||
            source_stride < 1 || rhs_count < 1) {
            throw std::invalid_argument("invalid direct one-way Kirchhoff parameters");
        }
        if (target_x_stride != 1 || target_z_stride != 1) {
            throw std::invalid_argument(
                "recursive prototype requires target_x_stride=1 and target_z_stride=1");
        }
        huygens_cli::set_openmp_threads(threads);

        const se::huygens::Model2D model = se::huygens::read_velocity_model(velocity);
        const se::huygens::BlockInfo info = se::huygens::read_block_info(block_file);
        se::huygens::validate_block_info(info, &model);
        if (source_iz != 0) {
            throw std::invalid_argument("source_iz must be zero in this prototype");
        }

        const std::vector<int> source_positions =
            make_rhs_sources(model.nx, rhs_count, source_ix_parameter);
        const int output_rhs = output_rhs_parameter >= 0
            ? output_rhs_parameter
            : rhs_count / 2;
        if (output_rhs < 0 || output_rhs >= rhs_count) {
            throw std::invalid_argument("output_rhs is outside [0,rhs_count)");
        }

        std::cout
            << "One-way Fourier convention: launch exp(-i*k*r), "
               "propagation exp(-i*omega*tau).\n";
        std::vector<std::vector<se::huygens::Complex>> wavefields(
            static_cast<std::size_t>(rhs_count));
        for (int rhs = 0; rhs < rhs_count; ++rhs) {
            se::huygens::initialize_first_block_hankel_one_way(
                model, info, frequency,
                source_positions[static_cast<std::size_t>(rhs)], source_iz,
                source_amplitude, source_radius,
                wavefields[static_cast<std::size_t>(rhs)]);
        }

        const int propagation_blocks = static_cast<int>(info.blocks.size()) - 1;
        const int metrics = 4;
        std::vector<float> timing_values(
            static_cast<std::size_t>(metrics) * propagation_blocks, 0.0f);
        double total_apply_seconds = 0.0;

        for (std::size_t iblock = 1; iblock < info.blocks.size(); ++iblock) {
            const se::huygens::Block& block = info.blocks[iblock];
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block, source_stride,
                    target_x_stride, target_z_stride);
            const se::huygens::OneWayLayerTables tables =
                se::huygens::read_one_way_layer_tables(table_prefix, block.id);
            se::huygens::validate_one_way_layer_tables(tables, geometry, block);

            const float maximum_tau = *std::max_element(
                tables.traveltime.values.begin(), tables.traveltime.values.end());
            const se::huygens::FrequencyKirchhoffFilter filter(
                frequency, filter_dt, filter_length,
                maximum_tau, filter_lookup_subsamples);
            const std::vector<float> weights =
                se::huygens::trapezoidal_weights(model, geometry.source_ix);

            se::huygens::OneWayKernelData kernel;
            kernel.rows = static_cast<int>(geometry.targets.size());
            kernel.sources = static_cast<int>(geometry.source_ix.size());
            kernel.source_z = model.z(block.source_iz);
            kernel.quadrature_weights = &weights;
            kernel.tables = &tables;
            kernel.filter = &filter;

            std::cout << "Direct block " << block.id
                      << ": matrix " << kernel.rows << " x "
                      << kernel.sources << ", rhs=" << rhs_count << '\n';

            const auto started = Clock::now();
            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                const std::vector<se::huygens::Complex> input =
                    se::huygens::gather_boundary_values(
                        model, block, geometry.source_ix,
                        wavefields[static_cast<std::size_t>(rhs)]);
                const std::vector<se::huygens::Complex> target_values =
                    se::huygens::direct_one_way_apply(kernel, input);
                se::huygens::inject_target_values(
                    model, geometry.targets, target_values,
                    wavefields[static_cast<std::size_t>(rhs)]);
            }
            const double seconds = std::chrono::duration<double>(
                Clock::now() - started).count();
            total_apply_seconds += seconds;

            const int column = static_cast<int>(iblock) - 1;
            timing_values[static_cast<std::size_t>(column) * metrics + 0] =
                static_cast<float>(seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 1] =
                static_cast<float>(kernel.sources);
            timing_values[static_cast<std::size_t>(column) * metrics + 2] =
                static_cast<float>(kernel.rows);
            timing_values[static_cast<std::size_t>(column) * metrics + 3] =
                static_cast<float>(rhs_count);
        }

        const auto& output_wavefield =
            wavefields[static_cast<std::size_t>(output_rhs)];
        se::huygens::write_wavefield_component_rsf(
            output_real, model, output_wavefield, frequency,
            "direct_one_way_kirchhoff", false);
        se::huygens::write_wavefield_component_rsf(
            output_imag, model, output_wavefield, frequency,
            "direct_one_way_kirchhoff", true);
        se::huygens::write_timing_rsf(
            timing_file, timing_values, metrics, propagation_blocks,
            "direct_one_way_kirchhoff",
            {"direct_apply_seconds", "source_points", "target_points", "rhs_count"},
            {{"total_direct_apply_seconds", static_cast<float>(total_apply_seconds)},
             {"frequency_hz", frequency},
             {"rhs_count", static_cast<float>(rhs_count)},
             {"one_way_spatial_phase_sign", -1.0f},
             {"output_source_ix", static_cast<float>(source_positions[output_rhs])}});

        std::cout << "Direct one-way Kirchhoff sweep completed.\n"
                  << "Total integration time for " << rhs_count << " RHS: "
                  << total_apply_seconds << " s\n"
                  << "Output RHS/source: " << output_rhs << '/'
                  << source_positions[output_rhs] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "wave_frequency_direct: " << error.what() << '\n';
        return 1;
    }
}
