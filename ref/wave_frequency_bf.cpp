#include "huygens_cli.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/huygens_sweep.hpp>
#include <SERECKIRCH/include/se_butterfly.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr float kPi = 3.14159265358979323846f;

std::vector<double> make_boundary_coordinates(
    const se::huygens::Model2D& model,
    const se::huygens::Block& block,
    const std::vector<int>& source_ix)
{
    std::vector<double> coordinates(2 * source_ix.size());
    const double z = model.z(block.source_iz);
    for (std::size_t i = 0; i < source_ix.size(); ++i) {
        coordinates[2 * i] = model.x(source_ix[i]);
        coordinates[2 * i + 1] = z;
    }
    return coordinates;
}

double complex_l2_norm(const std::vector<se::huygens::Complex>& values)
{
    long double squared_norm = 0.0;
    for (const se::huygens::Complex& value : values) {
        squared_norm += std::norm(value);
    }
    return std::sqrt(static_cast<double>(squared_norm));
}

void accumulate_statistics(se::butterfly::Statistics& total,
                           const se::butterfly::Statistics& item)
{
    total.setup_seconds += item.setup_seconds;
    total.row_tree_seconds += item.row_tree_seconds;
    total.column_tree_seconds += item.column_tree_seconds;
    total.structure_seconds += item.structure_seconds;
    total.compression_seconds += item.compression_seconds;
    total.build_seconds += item.build_seconds;
    total.apply_pack_seconds += item.apply_pack_seconds;
    total.apply_multiply_seconds += item.apply_multiply_seconds;
    total.apply_unpack_seconds += item.apply_unpack_seconds;
    total.apply_seconds += item.apply_seconds;
    total.library_fill_seconds += item.library_fill_seconds;
    total.library_entry_seconds += item.library_entry_seconds;
    total.library_entry_traverse_seconds += item.library_entry_traverse_seconds;
    total.library_entry_butterfly_seconds +=
        item.library_entry_butterfly_seconds;
    total.library_entry_communication_seconds +=
        item.library_entry_communication_seconds;
    total.library_multiply_seconds += item.library_multiply_seconds;
    total.compressed_megabytes += item.compressed_megabytes;
    total.peak_megabytes = std::max(
        total.peak_megabytes, item.peak_megabytes);
    total.maximum_rank = std::max(
        total.maximum_rank, item.maximum_rank);
    total.sampled_entries += item.sampled_entries;
    total.applied_vectors += item.applied_vectors;
}

double dense_matrix_megabytes(int rows, int columns)
{
    return static_cast<double>(rows) * static_cast<double>(columns) *
           sizeof(se::huygens::Complex) / (1024.0 * 1024.0);
}

void print_matrix_report(const std::string& label,
                         const se::butterfly::Statistics& stats,
                         int rows,
                         int columns)
{
    const double dense_mb = dense_matrix_megabytes(rows, columns);
    const double compression_ratio = stats.compressed_megabytes > 0.0
        ? dense_mb / stats.compressed_megabytes
        : 0.0;
    const double dense_entries =
        static_cast<double>(rows) * static_cast<double>(columns);
    const double sampled_percent = dense_entries > 0.0
        ? 100.0 * static_cast<double>(stats.sampled_entries) / dense_entries
        : 0.0;

    const std::ios::fmtflags old_flags = std::cout.flags();
    const std::streamsize old_precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3)
              << "  [" << label << "]\n"
              << "    setup=" << stats.setup_seconds
              << " s, tree: row=" << stats.row_tree_seconds
              << " s, column=" << stats.column_tree_seconds
              << " s, structure=" << stats.structure_seconds << " s\n"
              << "    compression=" << stats.compression_seconds
              << " s, total build=" << stats.build_seconds << " s\n"
              << "    apply: pack=" << stats.apply_pack_seconds
              << " s, multiply=" << stats.apply_multiply_seconds
              << " s, unpack=" << stats.apply_unpack_seconds
              << " s, total=" << stats.apply_seconds << " s\n"
              << "    memory: compressed=" << stats.compressed_megabytes
              << " MiB, peak=" << stats.peak_megabytes
              << " MiB, dense=" << dense_mb
              << " MiB, dense/compressed=" << compression_ratio << "x\n"
              << "    rank_max=" << stats.maximum_rank
              << ", sampled=" << stats.sampled_entries
              << " (" << sampled_percent << "% of dense entries)\n"
              << "    BPACK internal: fill=" << stats.library_fill_seconds
              << " s, entry=" << stats.library_entry_seconds
              << " s, entry_BF=" << stats.library_entry_butterfly_seconds
              << " s, multiply=" << stats.library_multiply_seconds << " s\n";
    if (compression_ratio <= 1.0) {
        std::cout << "    WARNING: compressed storage is not smaller than dense storage.\n";
    }
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

void print_runtime_configuration(int requested_threads, int verbosity)
{
    std::cout << "\n========== Butterfly runtime configuration ==========\n";
#if defined(NDEBUG)
    std::cout << "Build mode              : Release-like (NDEBUG)\n";
#else
    std::cout << "Build mode              : Debug/non-Release WARNING\n";
#endif
#if defined(KIRCH_NATIVE_OPTIMIZATION)
    std::cout << "Native CPU tuning       : enabled\n";
#else
    std::cout << "Native CPU tuning       : disabled\n";
#endif
#ifdef _OPENMP
    std::cout << "OpenMP                  : enabled, max_threads="
              << omp_get_max_threads()
              << ", requested=" << requested_threads << '\n';
#else
    std::cout << "OpenMP                  : disabled\n";
#endif
    std::cout << "ButterflyPACK raw output: "
              << (verbosity < 0 ? "hidden" : "enabled")
              << " (verbosity=" << verbosity << ")\n"
              << "=====================================================\n";
}
}

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        huygens_cli::initialize(argc, argv);

#ifndef KIRCH_HAS_BUTTERFLYPACK_FLOAT
        throw std::runtime_error(
            "wave_frequency_bf requires KIRCH_BPACK_ENABLE_FLOAT=ON");
#else
        const std::string velocity = huygens_cli::required_string("velocity");
        const std::string block_file = huygens_cli::required_string("block_file");
        const std::string table_prefix = huygens_cli::required_string("table_prefix");
        const std::string output_real = huygens_cli::optional_string(
            "output_real", "wave_frequency_bf_real.rsf");
        const std::string output_imag = huygens_cli::optional_string(
            "output_imag", "wave_frequency_bf_imag.rsf");
        const std::string timing_file = huygens_cli::optional_string(
            "timing", "wave_frequency_bf_timing.rsf");

        const float frequency = huygens_cli::optional_float("frequency", 25.0f);
        const float source_amplitude = huygens_cli::optional_float(
            "source_amplitude", 1.0f);
        const float source_radius = huygens_cli::optional_float("source_radius", 0.0f);
        const float tolerance = huygens_cli::optional_float("tol", 1.0e-4f);
        const int leaf_size = huygens_cli::optional_int("leaf", 64);
        const int verbosity = huygens_cli::optional_int("verbosity", 0);
        const int source_ix_parameter = huygens_cli::optional_int("source_ix", -1);
        const int source_iz = huygens_cli::optional_int("source_iz", 0);
        const int source_stride = huygens_cli::optional_int("source_stride", 1);
        const int target_x_stride = huygens_cli::optional_int("target_x_stride", 1);
        const int target_z_stride = huygens_cli::optional_int("target_z_stride", 1);
        const int threads = huygens_cli::optional_int("threads", 0);

        if (!(frequency > 0.0f) || !(tolerance > 0.0f) ||
            source_stride < 1 || leaf_size < 4) {
            throw std::invalid_argument(
                "frequency, tol and source_stride must be positive; leaf must be >=4");
        }
        if (target_x_stride != 1 || target_z_stride != 1) {
            throw std::invalid_argument(
                "minimal recursive prototype requires target_x_stride=1 and target_z_stride=1");
        }
        huygens_cli::set_openmp_threads(threads);
        print_runtime_configuration(threads, verbosity);

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
        const int metrics = 22;
        std::vector<float> timing_values(
            static_cast<std::size_t>(metrics) * propagation_blocks, 0.0f);
        se::butterfly::Statistics total_statistics;
        double total_dense_megabytes = 0.0;
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

            std::vector<se::huygens::Complex> value_input(
                geometry.source_ix.size());
            std::vector<se::huygens::Complex> normal_input(
                geometry.source_ix.size());
            for (std::size_t source = 0; source < geometry.source_ix.size(); ++source) {
                value_input[source] = boundary_input[2 * source];
                normal_input[source] = boundary_input[2 * source + 1];
            }

            se::huygens::KernelData kernel;
            kernel.rows = static_cast<int>(geometry.targets.size());
            kernel.sources = static_cast<int>(geometry.source_ix.size());
            kernel.omega = omega;
            kernel.quadrature_weights = &weights;
            kernel.tables = &tables;

            const std::vector<double> row_coordinates =
                se::huygens::make_row_coordinates(model, geometry.targets);
            const std::vector<double> column_coordinates =
                make_boundary_coordinates(model, block, geometry.source_ix);

            se::butterfly::Options options;
            options.tolerance = tolerance;
            options.leaf_size = leaf_size;
            options.verbosity = verbosity;
            options.coordinate_dimension = 2;

            std::cout << "Butterfly block " << block.id
                      << ": two matrices " << kernel.rows << " x "
                      << kernel.sources << '\n';

            std::vector<se::huygens::Complex> target_values;
            se::butterfly::Statistics block_statistics;

            // Keep the two Cauchy channels in separate ButterflyPACK matrices.
            // Giving U and dU/dn almost identical coordinates in one matrix can
            // make geometric sorting treat the two physical operators as one
            // duplicated point set and noticeably reduce compression accuracy.
            {
                se::butterfly::Matrix<float> value_butterfly(
                    kernel.rows, kernel.sources,
                    row_coordinates, column_coordinates,
                    [&kernel](int row, int source) {
                        return kernel.entry(row, 2 * source);
                    },
                    options);
                target_values = value_butterfly.apply(value_input);
                const se::butterfly::Statistics stats =
                    value_butterfly.statistics();
                accumulate_statistics(block_statistics, stats);
                print_matrix_report("U value channel", stats,
                                    kernel.rows, kernel.sources);
            }

            {
                se::butterfly::Matrix<float> normal_butterfly(
                    kernel.rows, kernel.sources,
                    row_coordinates, column_coordinates,
                    [&kernel](int row, int source) {
                        return kernel.entry(row, 2 * source + 1);
                    },
                    options);
                const std::vector<se::huygens::Complex> normal_values =
                    normal_butterfly.apply(normal_input);
                for (std::size_t row = 0; row < target_values.size(); ++row) {
                    target_values[row] += normal_values[row];
                }
                const se::butterfly::Statistics stats =
                    normal_butterfly.statistics();
                accumulate_statistics(block_statistics, stats);
                print_matrix_report("normal derivative channel", stats,
                                    kernel.rows, kernel.sources);
            }

            const double input_norm = complex_l2_norm(boundary_input);
            const double output_norm = complex_l2_norm(target_values);
            if (!std::isfinite(output_norm) ||
                (input_norm > 0.0 && output_norm == 0.0)) {
                throw std::runtime_error(
                    "ButterflyPACK returned an invalid or zero propagation block");
            }
            const double block_dense_megabytes =
                2.0 * dense_matrix_megabytes(kernel.rows, kernel.sources);
            const double block_compression_ratio =
                block_statistics.compressed_megabytes > 0.0
                ? block_dense_megabytes /
                  block_statistics.compressed_megabytes
                : 0.0;
            std::cout << "  [block " << block.id << " summary]\n"
                      << "    build=" << block_statistics.build_seconds
                      << " s (trees="
                      << block_statistics.row_tree_seconds +
                         block_statistics.column_tree_seconds
                      << " s, structure="
                      << block_statistics.structure_seconds
                      << " s, compression="
                      << block_statistics.compression_seconds << " s)\n"
                      << "    apply=" << block_statistics.apply_seconds
                      << " s, build/apply="
                      << block_statistics.build_seconds /
                         std::max(block_statistics.apply_seconds, 1.0e-12)
                      << "x\n"
                      << "    memory: compressed_sum="
                      << block_statistics.compressed_megabytes
                      << " MiB, live_peak="
                      << block_statistics.peak_megabytes
                      << " MiB, dense_equivalent="
                      << block_dense_megabytes
                      << " MiB, dense/compressed="
                      << block_compression_ratio << "x\n"
                      << "    input_l2=" << input_norm
                      << ", output_l2=" << output_norm << "\n\n";

            se::huygens::inject_target_values(
                model, geometry.targets, target_values, wavefield);

            accumulate_statistics(total_statistics, block_statistics);
            total_dense_megabytes += block_dense_megabytes;

            const int column = static_cast<int>(iblock) - 1;
            timing_values[static_cast<std::size_t>(column) * metrics + 0] =
                static_cast<float>(block_statistics.build_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 1] =
                static_cast<float>(block_statistics.apply_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 2] =
                static_cast<float>(block_statistics.compressed_megabytes);
            timing_values[static_cast<std::size_t>(column) * metrics + 3] =
                static_cast<float>(block_statistics.peak_megabytes);
            timing_values[static_cast<std::size_t>(column) * metrics + 4] =
                static_cast<float>(block_statistics.maximum_rank);
            timing_values[static_cast<std::size_t>(column) * metrics + 5] =
                static_cast<float>(kernel.sources);
            timing_values[static_cast<std::size_t>(column) * metrics + 6] =
                static_cast<float>(kernel.rows);
            timing_values[static_cast<std::size_t>(column) * metrics + 7] =
                static_cast<float>(block_statistics.setup_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 8] =
                static_cast<float>(block_statistics.row_tree_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 9] =
                static_cast<float>(block_statistics.column_tree_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 10] =
                static_cast<float>(block_statistics.structure_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 11] =
                static_cast<float>(block_statistics.compression_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 12] =
                static_cast<float>(block_statistics.apply_pack_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 13] =
                static_cast<float>(block_statistics.apply_multiply_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 14] =
                static_cast<float>(block_statistics.apply_unpack_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 15] =
                static_cast<float>(
                    block_statistics.sampled_entries / 1.0e6);
            timing_values[static_cast<std::size_t>(column) * metrics + 16] =
                static_cast<float>(block_dense_megabytes);
            timing_values[static_cast<std::size_t>(column) * metrics + 17] =
                static_cast<float>(block_compression_ratio);
            timing_values[static_cast<std::size_t>(column) * metrics + 18] =
                static_cast<float>(block_statistics.library_fill_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 19] =
                static_cast<float>(block_statistics.library_entry_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 20] =
                static_cast<float>(
                    block_statistics.library_entry_butterfly_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 21] =
                static_cast<float>(
                    block_statistics.library_multiply_seconds);
        }

        se::huygens::write_wavefield_component_rsf(
            output_real, model, wavefield, frequency,
            "butterfly_huygens_sweep", false);
        se::huygens::write_wavefield_component_rsf(
            output_imag, model, wavefield, frequency,
            "butterfly_huygens_sweep", true);
        se::huygens::write_timing_rsf(
            timing_file, timing_values, metrics, propagation_blocks,
            "butterfly_huygens_sweep",
            {"build_seconds", "apply_seconds", "compressed_megabytes",
             "peak_megabytes", "maximum_rank", "source_points",
             "target_points", "setup_seconds", "row_tree_seconds",
             "column_tree_seconds", "structure_seconds",
             "compression_seconds", "apply_pack_seconds",
             "apply_multiply_seconds", "apply_unpack_seconds",
             "sampled_entries_million", "dense_megabytes",
             "compression_ratio", "library_fill_seconds",
             "library_entry_seconds", "library_entry_bf_seconds",
             "library_multiply_seconds"},
            {{"total_build_seconds",
              static_cast<float>(total_statistics.build_seconds)},
             {"total_tree_seconds",
              static_cast<float>(total_statistics.row_tree_seconds +
                                 total_statistics.column_tree_seconds)},
             {"total_structure_seconds",
              static_cast<float>(total_statistics.structure_seconds)},
             {"total_compression_seconds",
              static_cast<float>(total_statistics.compression_seconds)},
             {"total_apply_seconds",
              static_cast<float>(total_statistics.apply_seconds)},
             {"total_build_plus_apply_seconds",
              static_cast<float>(total_statistics.build_seconds +
                                 total_statistics.apply_seconds)},
             {"total_compressed_megabytes",
              static_cast<float>(total_statistics.compressed_megabytes)},
             {"maximum_live_peak_megabytes",
              static_cast<float>(total_statistics.peak_megabytes)},
             {"dense_equivalent_megabytes",
              static_cast<float>(total_dense_megabytes)},
             {"overall_compression_ratio",
              static_cast<float>(
                  total_dense_megabytes /
                  std::max(total_statistics.compressed_megabytes, 1.0e-12))},
             {"frequency_hz", frequency},
             {"tolerance", tolerance}});

        std::cout << "========== Butterfly total summary ==========\n"
                  << "Build total       : "
                  << total_statistics.build_seconds << " s\n"
                  << "  row+column trees: "
                  << total_statistics.row_tree_seconds +
                     total_statistics.column_tree_seconds << " s\n"
                  << "  structures      : "
                  << total_statistics.structure_seconds << " s\n"
                  << "  compression     : "
                  << total_statistics.compression_seconds << " s\n"
                  << "Apply total       : "
                  << total_statistics.apply_seconds << " s\n"
                  << "Build/apply ratio : "
                  << total_statistics.build_seconds /
                     std::max(total_statistics.apply_seconds, 1.0e-12)
                  << "x\n"
                  << "Compressed sum    : "
                  << total_statistics.compressed_megabytes << " MiB\n"
                  << "Maximum live peak : "
                  << total_statistics.peak_megabytes << " MiB\n"
                  << "Dense equivalent  : "
                  << total_dense_megabytes << " MiB\n"
                  << "Dense/compressed  : "
                  << total_dense_megabytes /
                     std::max(total_statistics.compressed_megabytes, 1.0e-12)
                  << "x\n"
                  << "=============================================\n"
                  << "Real: " << output_real << '\n'
                  << "Imag: " << output_imag << '\n'
                  << "Timing: " << timing_file << '\n';
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << "wave_frequency_bf: " << error.what() << '\n';
        return 1;
    }
}
