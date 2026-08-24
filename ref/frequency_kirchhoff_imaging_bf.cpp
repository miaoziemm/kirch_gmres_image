#include "frequency_kirchhoff_imaging_common.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/bf1d.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fki = frequency_kirchhoff_imaging;

namespace {

struct FactorDeleter {
    void operator()(BFStrictSegmentedFactor* factor) const
    {
        bf1d_strict_segmented_destroy(factor);
    }
};
using FactorPointer =
    std::unique_ptr<BFStrictSegmentedFactor, FactorDeleter>;
using PackedComplex = std::vector<float>;

void pack_complex(const std::vector<fki::Complex>& input,
                  PackedComplex& output)
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
        const fki::CommonOptions options =
            fki::parse_common_options("frequency_kirchhoff_imaging_bf");

        const int p = huygens_cli::optional_int("bf_p", 12);
        const int leaf = huygens_cli::optional_int("bf_n_leaf", 16);
        const int panel_levels =
            huygens_cli::optional_int("bf_panel_levels", 1);
        const float amp_eps =
            huygens_cli::optional_float("bf_amp_eps", 1.0e-20f);
        const float phase_tol =
            huygens_cli::optional_float("bf_phase_tol", 0.5f);
        const int requested_build_threads =
            huygens_cli::optional_int("bf_build_threads", 0);
        const float max_build_workspace_mb =
            huygens_cli::optional_float("bf_build_workspace_mb", 512.0f);
        const int max_factor_count =
            huygens_cli::optional_int("bf_max_factor_count", 200000);

        if (p < 2 || leaf < 4 || p >= leaf || panel_levels < 0 ||
            amp_eps < 0.0f || !(phase_tol > 0.0f) ||
            requested_build_threads < 0 ||
            !(max_build_workspace_mb > 0.0f) || max_factor_count < 1) {
            throw std::invalid_argument("invalid phase-aware BF parameters");
        }
        huygens_cli::set_openmp_threads(options.threads);

        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(options.velocity);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(options.block_file);
        se::huygens::validate_block_info(info, &model);

        fki::SeismicData data(options.seismic_data, options.cmp != 0);
        const fki::FrequencyAxis axis =
            fki::make_frequency_axis(data, options);
        const std::vector<int> shots =
            fki::selected_shots(data, options);
        fki::validate_boundary_state_memory(
            options, shots.size(), axis.frequencies.size(), model.nx);
        const std::vector<float> model_weights =
            fki::full_boundary_weights(model);
        const std::vector<fki::Complex> source_spectrum =
            fki::ricker_spectrum(
                data, axis, options.fdom, options.source_time,
                options.source_amplitude);

        fki::print_configuration(
            options, data, axis, shots, model,
            "phase-aware butterfly, frequency bank per block");
        std::cout
            << "BF parameters        : p=" << p
            << ", leaf=" << leaf
            << ", panel_levels=" << panel_levels
            << ", phase_tol=" << phase_tol << '\n';

        double fft_seconds = 0.0;
        int skipped_receivers = 0;
        std::vector<fki::ShotBoundaryState> states;
        states.reserve(shots.size());
        for (const int shot : shots) {
            states.push_back(fki::initialize_shot_state(
                data, shot, model, options, axis,
                source_spectrum, model_weights,
                fft_seconds, skipped_receivers));
        }

        const std::size_t grid_size =
            static_cast<std::size_t>(model.nx) * model.nz;
        std::vector<float> image(grid_size, 0.0f);
        std::vector<float> illumination(grid_size, 0.0f);

        const int metrics = 8;
        const int block_count = static_cast<int>(info.blocks.size());
        std::vector<float> timing_values(
            static_cast<std::size_t>(metrics) * block_count, 0.0f);
        double total_filter_seconds = 0.0;
        double total_build_seconds = 0.0;
        double total_apply_seconds = 0.0;

        for (std::size_t iblock = 0;
             iblock < info.blocks.size(); ++iblock) {
            const se::huygens::Block block =
                fki::imaging_propagation_block(info, iblock);
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block, options.source_stride,
                    options.target_x_stride, options.target_z_stride);
            const se::huygens::OneWayLayerTables tables =
                se::huygens::read_one_way_layer_tables(
                    options.table_prefix, block.id);
            se::huygens::validate_one_way_layer_tables(
                tables, geometry, block);

            if (static_cast<int>(geometry.source_ix.size()) != model.nx ||
                static_cast<int>(geometry.target_ix.size()) != model.nx) {
                throw std::runtime_error(
                    "phase-aware bf1d imaging requires every lateral grid point");
            }

            const int target_depth_count =
                static_cast<int>(geometry.target_iz.size());
            const std::size_t factor_count =
                axis.frequencies.size() *
                static_cast<std::size_t>(target_depth_count);
            if (factor_count > static_cast<std::size_t>(max_factor_count)) {
                throw std::runtime_error(
                    "requested all-frequency BF bank contains " +
                    std::to_string(factor_count) +
                    " factors, exceeding bf_max_factor_count; narrow the "
                    "frequency range/increase frequency_stride or raise the limit");
            }

            const float maximum_tau = *std::max_element(
                tables.traveltime.values.begin(),
                tables.traveltime.values.end());
            const float filter_dt = options.filter_dt > 0.0f
                ? options.filter_dt
                : data.dt();

            const auto filter_started = fki::Clock::now();
            std::vector<std::unique_ptr<se::huygens::FrequencyKirchhoffFilter>>
                filters(axis.frequencies.size());
            for (std::size_t ifrequency = 0;
                 ifrequency < axis.frequencies.size(); ++ifrequency) {
                filters[ifrequency] = std::make_unique<
                    se::huygens::FrequencyKirchhoffFilter>(
                        axis.frequencies[ifrequency], filter_dt,
                        options.filter_length, maximum_tau,
                        options.filter_lookup_subsamples);
            }
            const double filter_seconds = std::chrono::duration<double>(
                fki::Clock::now() - filter_started).count();
            total_filter_seconds += filter_seconds;

            // Transpose the source-major RSF table once.  Every frequency
            // reuses the same phase (traveltime) matrices.
            const std::size_t matrix_elements =
                static_cast<std::size_t>(model.nx) * model.nx;
            std::vector<float> tau_by_depth(
                static_cast<std::size_t>(target_depth_count) * matrix_elements);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int iz_local = 0;
                 iz_local < target_depth_count; ++iz_local) {
                float* matrix = tau_by_depth.data() +
                    static_cast<std::size_t>(iz_local) * matrix_elements;
                for (int ix_target = 0; ix_target < model.nx; ++ix_target) {
                    for (int source = 0; source < model.nx; ++source) {
                        matrix[static_cast<std::size_t>(ix_target) * model.nx +
                               source] =
                            std::max(
                                tables.traveltime.values[
                                    tables.traveltime.index(
                                        source, ix_target, iz_local)],
                                1.0e-8f);
                    }
                }
            }

            const double workspace_per_task_mb =
                static_cast<double>(matrix_elements) *
                (sizeof(float) + sizeof(fftwf_complex) +
                 sizeof(float) + sizeof(fftwf_complex)) / 1.0e6;
            int build_threads = 1;
#ifdef _OPENMP
            const int available_threads = options.threads > 0
                ? options.threads
                : omp_get_max_threads();
            const int workspace_limited = std::max(
                1, static_cast<int>(
                       max_build_workspace_mb /
                       std::max(workspace_per_task_mb, 1.0e-6)));
            build_threads = requested_build_threads > 0
                ? requested_build_threads
                : std::min(available_threads, workspace_limited);
            build_threads = std::max(
                1, std::min(build_threads, static_cast<int>(factor_count)));
#else
            if (requested_build_threads > 1) {
                throw std::runtime_error(
                    "bf_build_threads>1 requested without OpenMP");
            }
#endif

            std::vector<FactorPointer> factors(factor_count);
            std::atomic<bool> build_failed{false};
            std::mutex build_error_mutex;
            std::string build_error;

            std::cout
                << "BF block " << block.id
                << ": prebuilding all " << axis.frequencies.size()
                << " frequencies x " << target_depth_count
                << " depths = " << factor_count
                << " factors before any apply; build_threads="
                << build_threads << '\n';

            const auto build_started = fki::Clock::now();
#ifdef _OPENMP
#pragma omp parallel num_threads(build_threads)
#endif
            {
                // Reuse construction scratch per OpenMP worker.  Creating the
                // dense amplitude and row-pointer arrays for every factor was
                // a noticeable allocator overhead for large frequency banks.
                std::vector<float*> tau_rows(
                    static_cast<std::size_t>(model.nx));
                std::vector<float> amplitude(2 * matrix_elements);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                for (long long factor_index = 0;
                     factor_index < static_cast<long long>(factor_count);
                     ++factor_index) {
                    if (build_failed.load(std::memory_order_relaxed)) continue;
                    try {
                        const std::size_t index =
                            static_cast<std::size_t>(factor_index);
                        const std::size_t ifrequency =
                            index / static_cast<std::size_t>(target_depth_count);
                        const int iz_local = static_cast<int>(
                            index % static_cast<std::size_t>(target_depth_count));
                        float* tau_matrix = tau_by_depth.data() +
                            static_cast<std::size_t>(iz_local) * matrix_elements;
                        for (int ix = 0; ix < model.nx; ++ix) {
                            tau_rows[static_cast<std::size_t>(ix)] =
                                tau_matrix +
                                static_cast<std::size_t>(ix) * model.nx;
                        }

                        se::huygens::OneWayKernelData kernel;
                        kernel.rows = static_cast<int>(geometry.targets.size());
                        kernel.sources = model.nx;
                        kernel.source_z = model.z(block.source_iz);
                        kernel.quadrature_weights = &model_weights;
                        kernel.tables = &tables;
                        kernel.filter = filters[ifrequency].get();
                        for (int ix_target = 0;
                             ix_target < model.nx; ++ix_target) {
                            const int row =
                                ix_target * target_depth_count + iz_local;
                            for (int source = 0; source < model.nx; ++source) {
                                const std::size_t matrix_index =
                                    static_cast<std::size_t>(ix_target) *
                                        model.nx + source;
                                const fki::Complex value =
                                    kernel.phase_removed_amplitude(row, source);
                                amplitude[2 * matrix_index] = value.real();
                                amplitude[2 * matrix_index + 1] = value.imag();
                            }
                        }
                        const float omega =
                            2.0f * fki::kPi * axis.frequencies[ifrequency];
                        factors[index].reset(
                            bf1d_strict_segmented_create_phase_amp(
                                model.nx, tau_rows.data(),
                                reinterpret_cast<const fftwf_complex*>(
                                    amplitude.data()),
                                omega, p, leaf, panel_levels,
                                amp_eps, phase_tol));
                        if (!factors[index]) {
                            throw std::runtime_error(
                                "bf1d factor construction returned null");
                        }
                    } catch (const std::exception& error) {
                        bool expected = false;
                        if (build_failed.compare_exchange_strong(expected, true)) {
                            std::lock_guard<std::mutex> lock(build_error_mutex);
                            build_error = error.what();
                        }
                    } catch (...) {
                        bool expected = false;
                        if (build_failed.compare_exchange_strong(expected, true)) {
                            std::lock_guard<std::mutex> lock(build_error_mutex);
                            build_error =
                                "unknown BF factor construction error";
                        }
                    }
                }
            }
            if (build_failed.load()) {
                throw std::runtime_error(
                    "all-frequency BF bank construction failed: " +
                    build_error);
            }
            const double build_seconds = std::chrono::duration<double>(
                fki::Clock::now() - build_started).count();
            total_build_seconds += build_seconds;

            const std::size_t target_values_per_frequency =
                static_cast<std::size_t>(target_depth_count) * model.nx;
            const auto apply_started = fki::Clock::now();

            for (fki::ShotBoundaryState& state : states) {
                PackedComplex packed_source;
                PackedComplex packed_receiver;
                pack_complex(state.source, packed_source);
                pack_complex(state.receiver_conjugate, packed_receiver);

                std::vector<fki::Complex> source_output(
                    axis.frequencies.size() * target_values_per_frequency);
                std::vector<fki::Complex> receiver_output(
                    source_output.size());

#ifdef _OPENMP
#pragma omp parallel
#endif
                {
                    std::vector<float> source_line(
                        2 * static_cast<std::size_t>(model.nx));
                    std::vector<float> receiver_line(
                        2 * static_cast<std::size_t>(model.nx));
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                    for (long long factor_index = 0;
                         factor_index < static_cast<long long>(factor_count);
                         ++factor_index) {
                        const std::size_t index =
                            static_cast<std::size_t>(factor_index);
                        const std::size_t ifrequency =
                            index / static_cast<std::size_t>(target_depth_count);
                        const int iz_local = static_cast<int>(
                            index % static_cast<std::size_t>(target_depth_count));
                        const std::size_t input_offset =
                            2 * ifrequency * static_cast<std::size_t>(model.nx);
                        bf1d_strict_segmented_apply(
                            factors[index].get(),
                            reinterpret_cast<const fftwf_complex*>(
                                packed_source.data() + input_offset),
                            reinterpret_cast<fftwf_complex*>(
                                source_line.data()));
                        bf1d_strict_segmented_apply(
                            factors[index].get(),
                            reinterpret_cast<const fftwf_complex*>(
                                packed_receiver.data() + input_offset),
                            reinterpret_cast<fftwf_complex*>(
                                receiver_line.data()));
                        for (int ix = 0; ix < model.nx; ++ix) {
                            const std::size_t output_index =
                                (ifrequency *
                                     static_cast<std::size_t>(target_depth_count) +
                                 static_cast<std::size_t>(iz_local)) *
                                    static_cast<std::size_t>(model.nx) + ix;
                            source_output[output_index] = fki::Complex(
                                source_line[2 * static_cast<std::size_t>(ix)],
                                source_line[2 * static_cast<std::size_t>(ix) + 1]);
                            receiver_output[output_index] = fki::Complex(
                                receiver_line[2 * static_cast<std::size_t>(ix)],
                                receiver_line[2 * static_cast<std::size_t>(ix) + 1]);
                        }
                    }
                }

                fki::accumulate_image(
                    model, axis, geometry.target_iz,
                    source_output, receiver_output,
                    image, illumination);
                if (iblock + 1 < info.blocks.size()) {
                    fki::extract_next_boundary(
                        model, axis, geometry.target_iz,
                        info.blocks[iblock + 1].source_iz,
                        source_output, receiver_output, state);
                }
            }

            const double apply_seconds = std::chrono::duration<double>(
                fki::Clock::now() - apply_started).count();
            total_apply_seconds += apply_seconds;

            const int column = static_cast<int>(iblock);
            timing_values[static_cast<std::size_t>(column) * metrics + 0] =
                static_cast<float>(filter_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 1] =
                static_cast<float>(build_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 2] =
                static_cast<float>(apply_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 3] =
                static_cast<float>(factor_count);
            timing_values[static_cast<std::size_t>(column) * metrics + 4] =
                static_cast<float>(states.size());
            timing_values[static_cast<std::size_t>(column) * metrics + 5] =
                static_cast<float>(axis.frequencies.size());
            timing_values[static_cast<std::size_t>(column) * metrics + 6] =
                static_cast<float>(target_depth_count);
            timing_values[static_cast<std::size_t>(column) * metrics + 7] =
                static_cast<float>(build_threads);

            std::cout
                << "  bank build=" << build_seconds
                << " s, apply/correlation=" << apply_seconds
                << " s, apply/shot="
                << apply_seconds / std::max<std::size_t>(states.size(), 1)
                << " s\n";
        }

        fki::finish_image(options, model, image, illumination);
        se::huygens::write_image_rsf(
            options.image, model, image,
            "frequency_kirchhoff_imaging_phase_aware_bf",
            {{"frequency_min_hz", axis.frequencies.front()},
             {"frequency_max_hz", axis.frequencies.back()},
             {"frequency_count", static_cast<float>(axis.frequencies.size())},
             {"shot_count", static_cast<float>(states.size())},
             {"cmp_geometry", static_cast<float>(options.cmp)},
             {"cross_correlation_conjugate", 1.0f},
             {"bf_p", static_cast<float>(p)},
             {"bf_n_leaf", static_cast<float>(leaf)},
             {"bf_phase_tol", phase_tol}});
        if (!options.illumination.empty()) {
            se::huygens::write_image_rsf(
                options.illumination, model, illumination,
                "frequency_kirchhoff_source_illumination",
                {{"frequency_count",
                  static_cast<float>(axis.frequencies.size())},
                 {"shot_count", static_cast<float>(states.size())}});
        }
        se::huygens::write_timing_rsf(
            options.timing, timing_values, metrics, block_count,
            "frequency_kirchhoff_imaging_phase_aware_bf",
            {"filter_setup_seconds", "all_frequency_bank_build_seconds",
             "apply_and_image_seconds", "factor_count", "shot_count",
             "frequency_count", "target_depth_count", "build_threads"},
            {{"fft_seconds", static_cast<float>(fft_seconds)},
             {"total_filter_setup_seconds",
              static_cast<float>(total_filter_seconds)},
             {"total_factor_build_seconds",
              static_cast<float>(total_build_seconds)},
             {"total_apply_and_image_seconds",
              static_cast<float>(total_apply_seconds)},
             {"skipped_receiver_coordinates",
              static_cast<float>(skipped_receivers)}});

        std::cout
            << "Phase-aware BF frequency-domain Kirchhoff imaging completed.\n"
            << "FFT/data preparation: " << fft_seconds << " s\n"
            << "Filter setup:         " << total_filter_seconds << " s\n"
            << "All-frequency banks:  " << total_build_seconds << " s\n"
            << "Apply + image:        " << total_apply_seconds << " s\n"
            << "Image:                " << options.image << '\n'
            << "Timing:               " << options.timing << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "frequency_kirchhoff_imaging_bf: "
                  << error.what() << '\n';
        return 1;
    }
}
