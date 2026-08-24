#include "huygens_cli.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/huygens_sweep.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr float kPi = 3.14159265358979323846f;

se::huygens::Table3D make_table_shape(
    const se::huygens::Model2D& model,
    const se::huygens::LayerGeometry& geometry)
{
    se::huygens::Table3D table;
    table.nz_target = static_cast<int>(geometry.target_iz.size());
    table.nx_target = static_cast<int>(geometry.target_ix.size());
    table.nsource = static_cast<int>(geometry.source_ix.size());
    table.dz_target = model.dz *
        (geometry.target_iz.size() > 1
             ? geometry.target_iz[1] - geometry.target_iz[0]
             : 1);
    table.dx_target = model.dx *
        (geometry.target_ix.size() > 1
             ? geometry.target_ix[1] - geometry.target_ix[0]
             : 1);
    table.dx_source = model.dx *
        (geometry.source_ix.size() > 1
             ? geometry.source_ix[1] - geometry.source_ix[0]
             : 1);
    table.oz_target = model.z(geometry.target_iz.front());
    table.ox_target = model.x(geometry.target_ix.front());
    table.ox_source = model.x(geometry.source_ix.front());
    table.values.resize(static_cast<std::size_t>(table.rows()) * table.nsource);
    return table;
}

} // namespace

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        huygens_cli::initialize(argc, argv);

        const std::string velocity = huygens_cli::required_string("velocity");
        const std::string block_file = huygens_cli::required_string("block_file");
        const std::string output_prefix = huygens_cli::optional_string(
            "output_prefix", "huygens_tt/travel");
        const std::string timing_file = huygens_cli::optional_string(
            "timing", output_prefix + "_timing.rsf");
        const int source_stride = huygens_cli::optional_int("source_stride", 1);
        const int target_x_stride = huygens_cli::optional_int("target_x_stride", 1);
        const int target_z_stride = huygens_cli::optional_int("target_z_stride", 1);
        const int requested_threads = huygens_cli::optional_int("threads", 0);
        const float max_table_mb = huygens_cli::optional_float("max_table_mb", 8192.0f);

        if (source_stride < 1 || target_x_stride < 1 || target_z_stride < 1) {
            throw std::invalid_argument("all strides must be positive");
        }
        if (!(max_table_mb > 0.0f)) {
            throw std::invalid_argument("max_table_mb must be positive");
        }
        if (requested_threads < 0) {
            throw std::invalid_argument("threads must be non-negative");
        }

        int available_threads = 1;
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        if (requested_threads > 0) {
            omp_set_num_threads(requested_threads);
        }
        available_threads = requested_threads > 0
            ? requested_threads
            : omp_get_max_threads();
#else
        if (requested_threads > 1) {
            throw std::runtime_error(
                "threads>1 was requested, but travel_time_solver was built without OpenMP");
        }
#endif

        std::cout << "FMM source-level OpenMP workers available: "
                  << available_threads
                  << " (requested=" << requested_threads << ")\n";

        const se::huygens::Model2D model = se::huygens::read_velocity_model(velocity);
        const se::huygens::BlockInfo info = se::huygens::read_block_info(block_file);
        se::huygens::validate_block_info(info, &model);

        const int propagation_blocks = static_cast<int>(info.blocks.size()) - 1;
        const int metrics = 4;
        std::vector<float> timing_values(
            static_cast<std::size_t>(metrics) * propagation_blocks, 0.0f);
        double total_fmm_seconds = 0.0;

        for (std::size_t iblock = 1; iblock < info.blocks.size(); ++iblock) {
            const se::huygens::Block& block = info.blocks[iblock];
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block, source_stride, target_x_stride, target_z_stride);

            se::huygens::LayerTables tables;
            tables.traveltime = make_table_shape(model, geometry);
            tables.amplitude = make_table_shape(model, geometry);
            tables.normal_traveltime_derivative = make_table_shape(model, geometry);

            const double table_mb =
                3.0 * static_cast<double>(tables.traveltime.values.size()) *
                sizeof(float) / 1.0e6;
            if (table_mb > max_table_mb) {
                throw std::runtime_error(
                    "block " + std::to_string(block.id) + " needs " +
                    std::to_string(table_mb) +
                    " MB for tau/amp/dtaun, exceeding max_table_mb");
            }

            std::cout << "Block " << block.id
                      << ": sources=" << geometry.source_ix.size()
                      << ", targets=" << geometry.targets.size()
                      << ", table memory=" << table_mb << " MB\n";

            const auto started = Clock::now();
            const int source_count =
                static_cast<int>(geometry.source_ix.size());
            const int progress_step = std::max(1, source_count / 20);
            const int block_threads = std::max(
                1, std::min(available_threads, source_count));

            std::atomic<int> completed_sources{0};
            std::atomic<bool> failed{false};
            std::mutex failure_mutex;
            std::string failure_message;

            std::cout << "  OpenMP workers for this block: "
                      << block_threads << '\n';

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(block_threads)
#endif
            for (int isource = 0; isource < source_count; ++isource) {
                if (failed.load(std::memory_order_relaxed)) {
                    continue;
                }

                try {
                    const int source_ix =
                        geometry.source_ix[static_cast<std::size_t>(isource)];
                    const int source_iz = block.source_iz;

                    // Each source is independent.  The three FMM solves remain
                    // sequential inside one worker so that the outer source
                    // loop owns all OpenMP parallelism and nested teams are
                    // avoided.
                    const std::vector<float> tau_center =
                        se::huygens::solve_fmm(model, source_ix, source_iz);
                    const std::vector<float> tau_minus =
                        se::huygens::solve_fmm(model, source_ix, source_iz - 1);
                    const std::vector<float> tau_plus =
                        se::huygens::solve_fmm(model, source_ix, source_iz + 1);

                    for (std::size_t ix_local = 0;
                         ix_local < geometry.target_ix.size(); ++ix_local) {
                        const int ix = geometry.target_ix[ix_local];
                        for (std::size_t iz_local = 0;
                             iz_local < geometry.target_iz.size(); ++iz_local) {
                            const int iz = geometry.target_iz[iz_local];
                            const std::size_t grid = model.index(ix, iz);
                            const std::size_t out = tables.traveltime.index(
                                isource,
                                static_cast<int>(ix_local),
                                static_cast<int>(iz_local));

                            const float tau =
                                std::max(tau_center[grid], 1.0e-8f);
                            const float dtaudz_source =
                                (tau_plus[grid] - tau_minus[grid]) /
                                (2.0f * model.dz);

                            tables.traveltime.values[out] = tau;
                            // Frequency-independent 2-D geometric amplitude:
                            // G = A/sqrt(omega) *
                            //     exp(i(omega*tau + pi/4)).
                            tables.amplitude.values[out] =
                                1.0f / std::sqrt(8.0f * kPi * tau);
                            // The outward normal of the lower half-space is -z.
                            tables.normal_traveltime_derivative.values[out] =
                                -dtaudz_source;
                        }
                    }

                    const int done =
                        completed_sources.fetch_add(
                            1, std::memory_order_relaxed) + 1;
                    if (done % progress_step == 0 || done == source_count) {
#ifdef _OPENMP
#pragma omp critical(huygens_fmm_progress)
#endif
                        {
                            std::cout << "  FMM sources completed "
                                      << done << '/' << source_count << '\n';
                        }
                    }
                } catch (const std::exception& error) {
                    bool expected = false;
                    if (failed.compare_exchange_strong(expected, true)) {
                        std::lock_guard<std::mutex> lock(failure_mutex);
                        failure_message =
                            "block " + std::to_string(block.id) +
                            ", source " + std::to_string(isource) +
                            ": " + error.what();
                    }
                } catch (...) {
                    bool expected = false;
                    if (failed.compare_exchange_strong(expected, true)) {
                        std::lock_guard<std::mutex> lock(failure_mutex);
                        failure_message =
                            "block " + std::to_string(block.id) +
                            ", source " + std::to_string(isource) +
                            ": unknown FMM failure";
                    }
                }
            }

            if (failed.load()) {
                throw std::runtime_error(
                    failure_message.empty()
                        ? "parallel FMM source loop failed"
                        : failure_message);
            }

            const double seconds = std::chrono::duration<double>(
                Clock::now() - started).count();
            total_fmm_seconds += seconds;

            se::huygens::write_table_rsf(
                se::huygens::layer_table_filename(
                    output_prefix, block.id, "tau"),
                model, block, tables.traveltime, "traveltime",
                source_stride, target_x_stride, target_z_stride);
            se::huygens::write_table_rsf(
                se::huygens::layer_table_filename(
                    output_prefix, block.id, "amp"),
                model, block, tables.amplitude, "simplified_geometric_amplitude",
                source_stride, target_x_stride, target_z_stride);
            se::huygens::write_table_rsf(
                se::huygens::layer_table_filename(
                    output_prefix, block.id, "dtaun"),
                model, block, tables.normal_traveltime_derivative,
                "source_normal_traveltime_derivative",
                source_stride, target_x_stride, target_z_stride);

            const int column = static_cast<int>(iblock) - 1;
            timing_values[static_cast<std::size_t>(column) * metrics + 0] =
                static_cast<float>(seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 1] =
                static_cast<float>(table_mb);
            timing_values[static_cast<std::size_t>(column) * metrics + 2] =
                static_cast<float>(geometry.source_ix.size());
            timing_values[static_cast<std::size_t>(column) * metrics + 3] =
                static_cast<float>(geometry.targets.size());
        }

        se::huygens::write_timing_rsf(
            timing_file,
            timing_values,
            metrics,
            propagation_blocks,
            "travel_time_solver",
            {"fmm_seconds", "three_table_megabytes",
             "source_points", "target_points"},
            {{"total_fmm_seconds", static_cast<float>(total_fmm_seconds)},
             {"openmp_workers", static_cast<float>(available_threads)}});

        std::cout << "Travel-time preprocessing completed.\n"
                  << "Table prefix: " << output_prefix << '\n'
                  << "Timing RSF: " << timing_file << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "travel_time_solver: " << error.what() << '\n';
        return 1;
    }
}
