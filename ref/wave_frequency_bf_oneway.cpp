#include "huygens_cli.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/bf1d.h>
#include <SERECKIRCH/include/huygens_sweep.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct FactorDeleter {
    void operator()(BFStrictSegmentedFactor* factor) const
    {
        bf1d_strict_segmented_destroy(factor);
    }
};
using FactorPointer = std::unique_ptr<BFStrictSegmentedFactor, FactorDeleter>;

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

using FFTStorage = std::vector<float>;

void copy_to_fftw(const std::vector<se::huygens::Complex>& input,
                  FFTStorage& output)
{
    output.resize(2 * input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[2 * i] = input[i].real();
        output[2 * i + 1] = input[i].imag();
    }
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
            "output_real", "wave_frequency_bf_real.rsf");
        const std::string output_imag = huygens_cli::optional_string(
            "output_imag", "wave_frequency_bf_imag.rsf");
        const std::string timing_file = huygens_cli::optional_string(
            "timing", "wave_frequency_bf_timing.rsf");

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
        const int p = huygens_cli::optional_int("bf_p", 12);
        const int leaf = huygens_cli::optional_int("bf_n_leaf", 16);
        const int panel_levels = huygens_cli::optional_int("bf_panel_levels", 1);
        const float amp_eps = huygens_cli::optional_float("bf_amp_eps", 1.0e-20f);
        const float phase_tol = huygens_cli::optional_float("bf_phase_tol", 1.0f);
        const int threads = huygens_cli::optional_int("threads", 0);

        if (!(frequency > 0.0f) || !(filter_dt > 0.0f) ||
            !(filter_length > 0.0f) || filter_lookup_subsamples < 1 ||
            rhs_count < 1 || p < 2 || leaf < 4 || p >= leaf ||
            panel_levels < 0 || !(phase_tol > 0.0f) || amp_eps < 0.0f) {
            throw std::invalid_argument("invalid phase-aware butterfly parameters");
        }
        if (source_stride != 1 || target_x_stride != 1 || target_z_stride != 1) {
            throw std::invalid_argument(
                "phase-aware bf1d path currently requires all spatial strides to equal one");
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
        const int metrics = 7;
        std::vector<float> timing_values(
            static_cast<std::size_t>(metrics) * propagation_blocks, 0.0f);
        double total_build_seconds = 0.0;
        double total_apply_seconds = 0.0;
        const float omega =
            2.0f * 3.14159265358979323846f * frequency;

        for (std::size_t iblock = 1; iblock < info.blocks.size(); ++iblock) {
            const se::huygens::Block& block = info.blocks[iblock];
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block, source_stride,
                    target_x_stride, target_z_stride);
            const se::huygens::OneWayLayerTables tables =
                se::huygens::read_one_way_layer_tables(table_prefix, block.id);
            se::huygens::validate_one_way_layer_tables(tables, geometry, block);

            if (static_cast<int>(geometry.source_ix.size()) != model.nx ||
                static_cast<int>(geometry.target_ix.size()) != model.nx) {
                throw std::runtime_error(
                    "bf1d requires one datum source and one target per lateral grid point");
            }

            const float maximum_tau = *std::max_element(
                tables.traveltime.values.begin(), tables.traveltime.values.end());
            const se::huygens::FrequencyKirchhoffFilter filter(
                frequency, filter_dt, filter_length,
                maximum_tau, filter_lookup_subsamples);
            const std::vector<float> weights =
                se::huygens::trapezoidal_weights(model, geometry.source_ix);

            se::huygens::OneWayKernelData kernel;
            kernel.rows = static_cast<int>(geometry.targets.size());
            kernel.sources = model.nx;
            kernel.source_z = model.z(block.source_iz);
            kernel.quadrature_weights = &weights;
            kernel.tables = &tables;
            kernel.filter = &filter;

            const int target_depths =
                static_cast<int>(geometry.target_iz.size());
            std::vector<FactorPointer> factors(
                static_cast<std::size_t>(target_depths));

            std::cout << "Phase-aware butterfly block " << block.id
                      << ": " << target_depths << " matrices "
                      << model.nx << " x " << model.nx
                      << ", rhs=" << rhs_count << '\n';

            const auto build_started = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int iz_local = 0; iz_local < target_depths; ++iz_local) {
                std::vector<float> tau_storage(
                    static_cast<std::size_t>(model.nx) * model.nx);
                std::vector<float*> tau_rows(static_cast<std::size_t>(model.nx));
                FFTStorage amplitude(
                    2 * static_cast<std::size_t>(model.nx) * model.nx);
                for (int ix_target = 0; ix_target < model.nx; ++ix_target) {
                    tau_rows[static_cast<std::size_t>(ix_target)] =
                        tau_storage.data() +
                        static_cast<std::size_t>(ix_target) * model.nx;
                    const int row = ix_target * target_depths + iz_local;
                    for (int source = 0; source < model.nx; ++source) {
                        const std::size_t matrix_index =
                            static_cast<std::size_t>(ix_target) * model.nx + source;
                        tau_storage[matrix_index] = kernel.traveltime(row, source);
                        const se::huygens::Complex value =
                            kernel.phase_removed_amplitude(row, source);
                        amplitude[2 * matrix_index] = value.real();
                        amplitude[2 * matrix_index + 1] = value.imag();
                    }
                }
                factors[static_cast<std::size_t>(iz_local)].reset(
                    bf1d_strict_segmented_create_phase_amp(
                        model.nx, tau_rows.data(), reinterpret_cast<const fftwf_complex*>(amplitude.data()), omega,
                        p, leaf, panel_levels, amp_eps, phase_tol));
                if (!factors[static_cast<std::size_t>(iz_local)]) {
                    throw std::runtime_error("bf1d factor construction returned null");
                }
            }
            const double build_seconds = std::chrono::duration<double>(
                Clock::now() - build_started).count();
            total_build_seconds += build_seconds;

            std::vector<std::vector<se::huygens::Complex>> boundary_inputs(
                static_cast<std::size_t>(rhs_count));
            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                boundary_inputs[static_cast<std::size_t>(rhs)] =
                    se::huygens::gather_boundary_values(
                        model, block, geometry.source_ix,
                        wavefields[static_cast<std::size_t>(rhs)]);
            }
            std::vector<FFTStorage> packed_inputs(
                static_cast<std::size_t>(rhs_count));
            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                copy_to_fftw(boundary_inputs[static_cast<std::size_t>(rhs)],
                             packed_inputs[static_cast<std::size_t>(rhs)]);
            }
            std::vector<std::vector<se::huygens::Complex>> target_values(
                static_cast<std::size_t>(rhs_count),
                std::vector<se::huygens::Complex>(geometry.targets.size()));

            const auto apply_started = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int iz_local = 0; iz_local < target_depths; ++iz_local) {
                FFTStorage output(
                    2 * static_cast<std::size_t>(model.nx));
                for (int rhs = 0; rhs < rhs_count; ++rhs) {
                    const auto& input =
                        packed_inputs[static_cast<std::size_t>(rhs)];
                    bf1d_strict_segmented_apply(
                        factors[static_cast<std::size_t>(iz_local)].get(),
                        reinterpret_cast<const fftwf_complex*>(input.data()),
                        reinterpret_cast<fftwf_complex*>(output.data()));
                    for (int ix_target = 0; ix_target < model.nx; ++ix_target) {
                        const std::size_t row =
                            static_cast<std::size_t>(ix_target) * target_depths +
                            iz_local;
                        target_values[static_cast<std::size_t>(rhs)][row] =
                            se::huygens::Complex(
                                output[2 * static_cast<std::size_t>(ix_target)],
                                output[2 * static_cast<std::size_t>(ix_target) + 1]);
                    }
                }
            }
            const double apply_seconds = std::chrono::duration<double>(
                Clock::now() - apply_started).count();
            total_apply_seconds += apply_seconds;

            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                se::huygens::inject_target_values(
                    model, geometry.targets,
                    target_values[static_cast<std::size_t>(rhs)],
                    wavefields[static_cast<std::size_t>(rhs)]);
            }

            const double dense_megabytes_per_depth =
                static_cast<double>(model.nx) * model.nx *
                (sizeof(float) + sizeof(fftwf_complex)) / 1.0e6;
            std::cout << "  build=" << build_seconds
                      << " s, apply=" << apply_seconds
                      << " s, apply/RHS=" << apply_seconds / rhs_count
                      << " s, dense workspace/depth="
                      << dense_megabytes_per_depth << " MB\n";

            const int column = static_cast<int>(iblock) - 1;
            timing_values[static_cast<std::size_t>(column) * metrics + 0] =
                static_cast<float>(build_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 1] =
                static_cast<float>(apply_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 2] =
                static_cast<float>(apply_seconds / rhs_count);
            timing_values[static_cast<std::size_t>(column) * metrics + 3] =
                static_cast<float>(model.nx);
            timing_values[static_cast<std::size_t>(column) * metrics + 4] =
                static_cast<float>(kernel.rows);
            timing_values[static_cast<std::size_t>(column) * metrics + 5] =
                static_cast<float>(rhs_count);
            timing_values[static_cast<std::size_t>(column) * metrics + 6] =
                static_cast<float>(dense_megabytes_per_depth);
        }

        const auto& output_wavefield =
            wavefields[static_cast<std::size_t>(output_rhs)];
        se::huygens::write_wavefield_component_rsf(
            output_real, model, output_wavefield, frequency,
            "phase_aware_butterfly_one_way_kirchhoff", false);
        se::huygens::write_wavefield_component_rsf(
            output_imag, model, output_wavefield, frequency,
            "phase_aware_butterfly_one_way_kirchhoff", true);
        se::huygens::write_timing_rsf(
            timing_file, timing_values, metrics, propagation_blocks,
            "phase_aware_butterfly_one_way_kirchhoff",
            {"build_seconds", "apply_seconds", "apply_seconds_per_rhs",
             "source_points", "target_points", "rhs_count",
             "dense_workspace_megabytes_per_depth"},
            {{"total_build_seconds", static_cast<float>(total_build_seconds)},
             {"total_apply_seconds", static_cast<float>(total_apply_seconds)},
             {"total_build_plus_apply_seconds",
              static_cast<float>(total_build_seconds + total_apply_seconds)},
             {"frequency_hz", frequency},
             {"rhs_count", static_cast<float>(rhs_count)},
             {"bf_p", static_cast<float>(p)},
             {"bf_n_leaf", static_cast<float>(leaf)},
             {"bf_phase_tol", phase_tol},
             {"one_way_spatial_phase_sign", -1.0f},
             {"output_source_ix", static_cast<float>(source_positions[output_rhs])}});

        std::cout << "Phase-aware butterfly one-way sweep completed.\n"
                  << "Total build time: " << total_build_seconds << " s\n"
                  << "Total apply time for " << rhs_count << " RHS: "
                  << total_apply_seconds << " s\n"
                  << "Output RHS/source: " << output_rhs << '/'
                  << source_positions[output_rhs] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "wave_frequency_bf: " << error.what() << '\n';
        return 1;
    }
}
