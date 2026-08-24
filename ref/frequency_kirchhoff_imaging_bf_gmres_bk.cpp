#include "frequency_kirchhoff_imaging_common.hpp"
#include "matlab_gmres.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/bf1d.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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

void pack_frequency_batch(const std::vector<fki::Complex>& input,
                          std::size_t frequency_begin,
                          std::size_t frequency_count,
                          int nx,
                          PackedComplex& output)
{
    const std::size_t point_count =
        frequency_count * static_cast<std::size_t>(nx);
    output.resize(2 * point_count);
    const std::size_t input_begin =
        frequency_begin * static_cast<std::size_t>(nx);
    if (input_begin + point_count > input.size()) {
        throw std::out_of_range("frequency batch exceeds boundary state");
    }

    for (std::size_t i = 0; i < point_count; ++i) {
        output[2 * i] = input[input_begin + i].real();
        output[2 * i + 1] = input[input_begin + i].imag();
    }
}

void accumulate_image_batch(
    const se::huygens::Model2D& model,
    const fki::FrequencyAxis& axis,
    std::size_t frequency_begin,
    std::size_t frequency_count,
    const std::vector<int>& target_depths,
    int depth_begin,
    int depth_count,
    const std::vector<fki::Complex>& source_output,
    const std::vector<fki::Complex>& receiver_conjugate_output,
    std::vector<float>& image,
    std::vector<float>& illumination)
{
    const std::size_t per_frequency =
        static_cast<std::size_t>(depth_count) * model.nx;
    const std::size_t expected = frequency_count * per_frequency;
    if (source_output.size() != expected ||
        receiver_conjugate_output.size() != expected) {
        throw std::invalid_argument(
            "batched propagated wavefield size is invalid");
    }

    for (std::size_t ifrequency_local = 0;
         ifrequency_local < frequency_count;
         ++ifrequency_local) {
        const std::size_t ifrequency =
            frequency_begin + ifrequency_local;
        const float weight = axis.imaging_weights[ifrequency];

        for (int iz_batch = 0; iz_batch < depth_count; ++iz_batch) {
            const int iz_local = depth_begin + iz_batch;
            const int iz =
                target_depths[static_cast<std::size_t>(iz_local)];

            for (int ix = 0; ix < model.nx; ++ix) {
                const std::size_t local =
                    (ifrequency_local *
                         static_cast<std::size_t>(depth_count) +
                     static_cast<std::size_t>(iz_batch)) *
                        static_cast<std::size_t>(model.nx) +
                    static_cast<std::size_t>(ix);
                const fki::Complex source = source_output[local];
                const fki::Complex receiver_conjugate =
                    receiver_conjugate_output[local];
                const std::size_t grid = model.index(ix, iz);

                image[grid] +=
                    weight * (source * receiver_conjugate).real();
                illumination[grid] += weight * std::norm(source);
            }
        }
    }
}

void extract_next_boundary_batch(
    const se::huygens::Model2D& model,
    std::size_t frequency_begin,
    std::size_t frequency_count,
    const std::vector<int>& target_depths,
    int depth_begin,
    int depth_count,
    int next_depth,
    const std::vector<fki::Complex>& source_output,
    const std::vector<fki::Complex>& receiver_output,
    fki::ShotBoundaryState& state)
{
    const int next_local = fki::find_local_depth(
        target_depths, next_depth);
    if (next_local < depth_begin ||
        next_local >= depth_begin + depth_count) {
        return;
    }

    const int iz_batch = next_local - depth_begin;
    for (std::size_t ifrequency_local = 0;
         ifrequency_local < frequency_count;
         ++ifrequency_local) {
        const std::size_t ifrequency =
            frequency_begin + ifrequency_local;

        for (int ix = 0; ix < model.nx; ++ix) {
            const std::size_t output_index =
                (ifrequency_local *
                     static_cast<std::size_t>(depth_count) +
                 static_cast<std::size_t>(iz_batch)) *
                    static_cast<std::size_t>(model.nx) +
                static_cast<std::size_t>(ix);
            const std::size_t boundary_index =
                ifrequency * static_cast<std::size_t>(model.nx) +
                static_cast<std::size_t>(ix);

            state.source[boundary_index] =
                source_output[output_index];
            state.receiver_conjugate[boundary_index] =
                receiver_output[output_index];
        }
    }
}

void correct_complete_block(
    const se::huygens::Model2D& model, const std::vector<int>& target_iz,
    float frequency, const matlab_gmres::Parameters& parameters,
    int shot, int block, std::vector<fki::Complex>& values,
    const char* field_name, std::ofstream& metrics_stream)
{
    const int nz = static_cast<int>(target_iz.size());
    const int nx = model.nx;
    if (static_cast<int>(values.size()) != nz * nx) {
        throw std::invalid_argument("GMRES correction requires one complete frequency block");
    }
    std::vector<float> velocity(static_cast<std::size_t>(nz) * nx);
    matlab_gmres::Vector initial(nz * nx);
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            const int mi = matlab_gmres::index(iz, ix, nz);
            velocity[static_cast<std::size_t>(mi)] =
                model.velocity[model.index(ix, target_iz[static_cast<std::size_t>(iz)])];
            const fki::Complex value = values[static_cast<std::size_t>(iz) * nx + ix];
            initial[mi] = std::complex<double>(value.real(), value.imag());
        }
    }
    std::vector<matlab_gmres::IterationMetric> metrics;
    const auto corrected = matlab_gmres::correct(
        velocity, initial, nz, nx, model.dz, model.dx, frequency,
        parameters, &metrics);
    const int blend_rows = std::min(parameters.absorbing_rows, nz / 2);
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            const int mi = matlab_gmres::index(iz, ix, nz);
            double weight = 1.0;
            if (blend_rows > 0 && iz < blend_rows) {
                weight = 0.5 - 0.5 * std::cos(
                    3.14159265358979323846 * static_cast<double>(iz) / blend_rows);
            } else if (blend_rows > 0 && iz >= nz - blend_rows) {
                weight = 0.5 - 0.5 * std::cos(
                    3.14159265358979323846 * static_cast<double>(nz - 1 - iz) / blend_rows);
            }
            const auto value = weight * corrected[mi] + (1.0 - weight) * initial[mi];
            values[static_cast<std::size_t>(iz) * nx + ix] = fki::Complex(
                static_cast<float>(value.real()), static_cast<float>(value.imag()));
        }
    }
    for (const auto& m : metrics) {
        metrics_stream << shot << ',' << block << ',' << frequency << ','
            << field_name << ',' << m.outer << ',' << m.gmres_steps << ','
            << m.residual_before << ',' << m.residual_after_gmres << ','
            << m.gmres_correction_norm << ','
            << static_cast<int>(m.gmres_accepted) << '\n';
    }
}

void phase_shift_first_block(const se::huygens::Model2D& model,
                             const se::huygens::Block& block,
                             const std::vector<int>& target_iz,
                             float frequency,
                             const PackedComplex& packed,
                             std::vector<fki::Complex>& output)
{
    const int nx = model.nx;
    std::vector<fki::Complex> spectrum(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
        spectrum[static_cast<std::size_t>(ix)] = fki::Complex(
            packed[2 * static_cast<std::size_t>(ix)],
            packed[2 * static_cast<std::size_t>(ix) + 1]);
    }
    fftwf_plan forward = fftwf_plan_dft_1d(
        nx, reinterpret_cast<fftwf_complex*>(spectrum.data()),
        reinterpret_cast<fftwf_complex*>(spectrum.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    if (!forward) throw std::runtime_error("first-block phase-shift FFT plan failed");
    fftwf_execute(forward);
    fftwf_destroy_plan(forward);

    double velocity_sum = 0.0;
    std::size_t velocity_count = 0;
    for (const int iz : target_iz) {
        for (int ix = 0; ix < nx; ++ix) {
            velocity_sum += model.velocity[model.index(ix, iz)];
            ++velocity_count;
        }
    }
    const double v0 = velocity_sum / std::max<std::size_t>(velocity_count, 1);
    const double omega = 2.0 * 3.14159265358979323846 * frequency;
    const double k0 = omega / v0;
    output.resize(static_cast<std::size_t>(target_iz.size()) * nx);
    std::vector<fki::Complex> line(static_cast<std::size_t>(nx));
    for (std::size_t iz_local = 0; iz_local < target_iz.size(); ++iz_local) {
        const double distance = model.z(target_iz[iz_local]) - model.z(block.source_iz);
        for (int ik = 0; ik < nx; ++ik) {
            const int signed_k = ik <= nx / 2 ? ik : ik - nx;
            const double kx = 2.0 * 3.14159265358979323846 * signed_k /
                              (static_cast<double>(nx) * model.dx);
            const double kz2 = k0 * k0 - kx * kx;
            const fki::Complex propagator = kz2 >= 0.0
                ? std::exp(fki::Complex(0.0f, static_cast<float>(-std::sqrt(kz2) * distance)))
                : fki::Complex(static_cast<float>(std::exp(-std::sqrt(-kz2) * std::abs(distance))), 0.0f);
            line[static_cast<std::size_t>(ik)] = spectrum[static_cast<std::size_t>(ik)] * propagator;
        }
        fftwf_plan inverse = fftwf_plan_dft_1d(
            nx, reinterpret_cast<fftwf_complex*>(line.data()),
            reinterpret_cast<fftwf_complex*>(line.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!inverse) throw std::runtime_error("first-block inverse phase-shift FFT plan failed");
        fftwf_execute(inverse);
        fftwf_destroy_plan(inverse);
        for (int ix = 0; ix < nx; ++ix) {
            output[iz_local * static_cast<std::size_t>(nx) + ix] =
                line[static_cast<std::size_t>(ix)] / static_cast<float>(nx);
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        huygens_cli::initialize(argc, argv);
        const fki::CommonOptions options =
            fki::parse_common_options(
                "frequency_kirchhoff_imaging_bf_gmres");

        const int p = huygens_cli::optional_int("bf_p", 12);
        const int leaf =
            huygens_cli::optional_int("bf_n_leaf", 16);
        const int panel_levels =
            huygens_cli::optional_int("bf_panel_levels", 1);
        const float amp_eps =
            huygens_cli::optional_float("bf_amp_eps", 1.0e-20f);
        const float phase_tol =
            huygens_cli::optional_float("bf_phase_tol", 0.5f);
        const int requested_build_threads =
            huygens_cli::optional_int("bf_build_threads", 0);
        const float max_build_workspace_mb =
            huygens_cli::optional_float(
                "bf_build_workspace_mb", 512.0f);
        const int max_factor_count =
            huygens_cli::optional_int(
                "bf_max_factor_count", 200000);
        const int requested_frequency_batch =
            huygens_cli::optional_int(
                "bf_frequency_batch", 4);
        const int requested_depth_batch =
            huygens_cli::optional_int(
                "bf_depth_batch", 1);
        matlab_gmres::Parameters correction;
        correction.outer_iterations = huygens_cli::optional_int("gmres_outer", 3);
        correction.gmres_restart = huygens_cli::optional_int("gmres_restart", 30);
        correction.gmres_cycles = huygens_cli::optional_int("gmres_cycles", 1);
        correction.absorbing_rows = huygens_cli::optional_int("gmres_nabs", 6);
        correction.gmres_tolerance = huygens_cli::optional_float("gmres_tolerance", 1.0e-12f);
        correction.damp_max = huygens_cli::optional_float("gmres_damp_max", 2.0f);
        const std::string correction_csv = huygens_cli::optional_string(
            "correction_csv", "gmres_metrics.csv");

        if (p < 2 || leaf < 4 || p >= leaf ||
            panel_levels < 0 || amp_eps < 0.0f ||
            !(phase_tol > 0.0f) ||
            requested_build_threads < 0 ||
            !(max_build_workspace_mb > 0.0f) ||
            max_factor_count < 1 ||
            requested_frequency_batch < 1 ||
            requested_depth_batch < 1) {
            throw std::invalid_argument(
                "invalid phase-aware BF parameters");
        }
        if (correction.outer_iterations != 3 || correction.gmres_restart < 1 ||
            correction.gmres_cycles < 1 || correction.absorbing_rows < 0 ||
            !(correction.gmres_tolerance > 0.0) || !(correction.damp_max >= 0.0)) {
            throw std::invalid_argument(
                "invalid MATLAB GMRES parameters; gmres_outer must equal 3");
        }

        huygens_cli::set_openmp_threads(options.threads);

        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(
                options.velocity);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(
                options.block_file);
        se::huygens::validate_block_info(info, &model);

        fki::SeismicData data(
            options.seismic_data, options.cmp != 0);
        const fki::FrequencyAxis axis =
            fki::make_frequency_axis(data, options);
        const std::vector<int> shots =
            fki::selected_shots(data, options);

        fki::validate_boundary_state_memory(
            options, shots.size(),
            axis.frequencies.size(), model.nx);

        const std::vector<float> model_weights =
            fki::full_boundary_weights(model);
        const std::vector<fki::Complex> source_spectrum =
            fki::ricker_spectrum(
                data, axis, options.fdom,
                options.source_time,
                options.source_amplitude);

        fki::print_configuration(
            options, data, axis, shots, model,
            "phase-aware butterfly, low-memory tiled factors");

        std::cout
            << "BF parameters        : p=" << p
            << ", leaf=" << leaf
            << ", panel_levels=" << panel_levels
            << ", phase_tol=" << phase_tol << '\n'
            << "BF resident tile     : frequency_batch="
            << requested_frequency_batch
            << ", depth_batch="
            << requested_depth_batch << '\n';

        double fft_seconds = 0.0;
        int skipped_receivers = 0;
        std::vector<fki::ShotBoundaryState> states;
        states.reserve(shots.size());

        for (const int shot : shots) {
            states.push_back(
                fki::initialize_shot_state(
                    data, shot, model, options, axis,
                    source_spectrum, model_weights,
                    fft_seconds, skipped_receivers));
        }

        const std::size_t grid_size =
            static_cast<std::size_t>(model.nx) *
            model.nz;
        std::vector<float> image(grid_size, 0.0f);
        std::vector<float> illumination(
            grid_size, 0.0f);

        const int metrics = 10;
        const int block_count =
            static_cast<int>(info.blocks.size());
        std::vector<float> timing_values(
            static_cast<std::size_t>(metrics) *
                block_count,
            0.0f);
        std::ofstream correction_stream(correction_csv);
        if (!correction_stream) {
            throw std::runtime_error("cannot open correction_csv: " + correction_csv);
        }
        correction_stream << std::setprecision(17)
            << "shot,block,frequency,field,outer,gmres_steps,residual_before,"
               "residual_after_gmres,gmres_correction_norm,gmres_accepted\n";

        double total_filter_seconds = 0.0;
        double total_build_seconds = 0.0;
        double total_apply_seconds = 0.0;

        for (std::size_t iblock = 0;
             iblock < info.blocks.size();
             ++iblock) {
            const se::huygens::Block block =
                fki::imaging_propagation_block(
                    info, iblock);
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block,
                    options.source_stride,
                    options.target_x_stride,
                    options.target_z_stride);
            const se::huygens::OneWayLayerTables tables =
                se::huygens::read_one_way_layer_tables(
                    options.table_prefix, block.id);
            se::huygens::validate_one_way_layer_tables(
                tables, geometry, block);

            if (static_cast<int>(
                    geometry.source_ix.size()) != model.nx ||
                static_cast<int>(
                    geometry.target_ix.size()) != model.nx) {
                throw std::runtime_error(
                    "phase-aware bf1d imaging requires "
                    "every lateral grid point");
            }

            const int target_depth_count =
                static_cast<int>(
                    geometry.target_iz.size());
            // The MATLAB correction acts on a complete local block.  Retain
            // exactly one frequency and all block depths in each tile.
            const int frequency_batch = 1;
            const int depth_batch = target_depth_count;
            const int peak_resident_factors =
                frequency_batch * depth_batch;
            const std::size_t total_factor_count =
                axis.frequencies.size() *
                static_cast<std::size_t>(
                    target_depth_count);

            if (peak_resident_factors >
                max_factor_count) {
                throw std::runtime_error(
                    "resident BF tile contains " +
                    std::to_string(
                        peak_resident_factors) +
                    " factors, exceeding "
                    "bf_max_factor_count");
            }

            const float maximum_tau =
                *std::max_element(
                    tables.traveltime.values.begin(),
                    tables.traveltime.values.end());
            const float filter_dt =
                options.filter_dt > 0.0f
                    ? options.filter_dt
                    : data.dt();

            const std::size_t matrix_elements =
                static_cast<std::size_t>(model.nx) *
                model.nx;
            const double scratch_per_thread_mb =
                static_cast<double>(
                    matrix_elements) *
                (sizeof(float) +
                 2.0 * sizeof(float)) /
                1.0e6;
            const double tau_tile_mb =
                static_cast<double>(
                    depth_batch) *
                matrix_elements *
                sizeof(float) /
                1.0e6;

            int build_threads = 1;
#ifdef _OPENMP
            const int available_threads =
                options.threads > 0
                    ? options.threads
                    : omp_get_max_threads();
            const int workspace_limited =
                std::max(
                    1,
                    static_cast<int>(
                        max_build_workspace_mb /
                        std::max(
                            scratch_per_thread_mb,
                            1.0e-6)));
            const int requested_or_available =
                requested_build_threads > 0
                    ? requested_build_threads
                    : available_threads;

            build_threads = std::max(
                1,
                std::min({
                    requested_or_available,
                    workspace_limited,
                    peak_resident_factors}));
#else
            if (requested_build_threads > 1) {
                throw std::runtime_error(
                    "bf_build_threads>1 requested "
                    "without OpenMP");
            }
#endif

            std::cout
                << "BF block " << block.id
                << ": total factors="
                << total_factor_count
                << ", resident factors<="
                << peak_resident_factors
                << ", tau tile=" << tau_tile_mb
                << " MB, tau cache<="
                << (static_cast<double>(target_depth_count) *
                    matrix_elements * sizeof(float) / 1.0e6)
                << " MB, scratch/thread="
                << scratch_per_thread_mb
                << " MB, build_threads="
                << build_threads << '\n';

            double block_filter_seconds = 0.0;
            double block_build_seconds = 0.0;
            double block_apply_seconds = 0.0;

            // Priority 1: traveltime matrices are frequency independent.
            // Convert the source-major RSF table into target-major dense matrices
            // once per propagation block, then reuse them in every frequency tile.
            const int depth_tile_count =
                (target_depth_count + depth_batch - 1) / depth_batch;
            std::vector<std::vector<float>> tau_cache(
                static_cast<std::size_t>(depth_tile_count));
            const auto tau_cache_started = fki::Clock::now();

            for (int depth_begin = 0;
                 depth_begin < target_depth_count;
                 depth_begin += depth_batch) {
                const int depth_count = std::min(
                    depth_batch, target_depth_count - depth_begin);
                const int depth_tile = depth_begin / depth_batch;
                std::vector<float>& tau_by_depth =
                    tau_cache[static_cast<std::size_t>(depth_tile)];
                tau_by_depth.resize(
                    static_cast<std::size_t>(depth_count) *
                    matrix_elements);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int iz_batch = 0;
                     iz_batch < depth_count;
                     ++iz_batch) {
                    const int iz_local = depth_begin + iz_batch;
                    float* tau_matrix =
                        tau_by_depth.data() +
                        static_cast<std::size_t>(iz_batch) *
                            matrix_elements;
                    for (int ix_target = 0;
                         ix_target < model.nx;
                         ++ix_target) {
                        for (int source = 0;
                             source < model.nx;
                             ++source) {
                            tau_matrix[
                                static_cast<std::size_t>(ix_target) *
                                    model.nx +
                                source] =
                                std::max(
                                    tables.traveltime.values[
                                        tables.traveltime.index(
                                            source,
                                            ix_target,
                                            iz_local)],
                                    1.0e-8f);
                        }
                    }
                }
            }
            block_build_seconds +=
                std::chrono::duration<double>(
                    fki::Clock::now() - tau_cache_started)
                    .count();

            for (std::size_t frequency_begin = 0;
                 frequency_begin <
                    axis.frequencies.size();
                 frequency_begin +=
                    static_cast<std::size_t>(
                        frequency_batch)) {
                const std::size_t frequency_count =
                    std::min(
                        static_cast<std::size_t>(
                            frequency_batch),
                        axis.frequencies.size() -
                            frequency_begin);

                const auto filter_started =
                    fki::Clock::now();
                std::vector<std::unique_ptr<
                    se::huygens::
                        FrequencyKirchhoffFilter>>
                    filters(frequency_count);

                for (std::size_t ifrequency_local = 0;
                     ifrequency_local <
                        frequency_count;
                     ++ifrequency_local) {
                    const std::size_t ifrequency =
                        frequency_begin +
                        ifrequency_local;
                    filters[ifrequency_local] =
                        std::make_unique<
                            se::huygens::
                                FrequencyKirchhoffFilter>(
                            axis.frequencies[
                                ifrequency],
                            filter_dt,
                            options.filter_length,
                            maximum_tau,
                            options.
                                filter_lookup_subsamples);
                }

                const double filter_seconds =
                    std::chrono::duration<double>(
                        fki::Clock::now() -
                        filter_started)
                        .count();
                block_filter_seconds +=
                    filter_seconds;

                struct PackedState {
                    PackedComplex source;
                    PackedComplex receiver;
                };
                std::vector<PackedState> packed_states(
                    states.size());

                for (std::size_t ishot = 0;
                     ishot < states.size();
                     ++ishot) {
                    pack_frequency_batch(
                        states[ishot].source,
                        frequency_begin,
                        frequency_count,
                        model.nx,
                        packed_states[ishot].source);
                    pack_frequency_batch(
                        states[ishot].
                            receiver_conjugate,
                        frequency_begin,
                        frequency_count,
                        model.nx,
                        packed_states[ishot].
                            receiver);
                }

                for (int depth_begin = 0;
                     depth_begin <
                        target_depth_count;
                     depth_begin += depth_batch) {
                    const int depth_count =
                        std::min(
                            depth_batch,
                            target_depth_count -
                                depth_begin);
                    const std::size_t factor_count =
                        frequency_count *
                        static_cast<std::size_t>(
                            depth_count);

                    const int depth_tile =
                        depth_begin / depth_batch;
                    std::vector<float>& tau_by_depth =
                        tau_cache[
                            static_cast<std::size_t>(depth_tile)];

                    std::vector<FactorPointer>
                        factors(factor_count);
                    std::atomic<bool>
                        build_failed{false};
                    std::mutex build_error_mutex;
                    std::string build_error;

                    const auto build_started =
                        fki::Clock::now();

#ifdef _OPENMP
#pragma omp parallel num_threads(build_threads)
#endif
                    {
                        std::vector<float*> tau_rows(
                            static_cast<std::size_t>(
                                model.nx));
                        std::vector<float> amplitude(
                            2 * matrix_elements);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                        for (long long factor_index = 0;
                             factor_index <
                                static_cast<long long>(
                                    factor_count);
                             ++factor_index) {
                            if (build_failed.load(
                                    std::memory_order_relaxed)) {
                                continue;
                            }

                            try {
                                const std::size_t index =
                                    static_cast<
                                        std::size_t>(
                                        factor_index);
                                const std::size_t
                                    ifrequency_local =
                                        index /
                                        static_cast<
                                            std::size_t>(
                                            depth_count);
                                const int iz_batch =
                                    static_cast<int>(
                                        index %
                                        static_cast<
                                            std::size_t>(
                                            depth_count));
                                const int iz_local =
                                    depth_begin +
                                    iz_batch;
                                float* tau_matrix =
                                    tau_by_depth.data() +
                                    static_cast<
                                        std::size_t>(
                                        iz_batch) *
                                        matrix_elements;

                                for (int ix = 0;
                                     ix < model.nx;
                                     ++ix) {
                                    tau_rows[
                                        static_cast<
                                            std::size_t>(
                                            ix)] =
                                        tau_matrix +
                                        static_cast<
                                            std::size_t>(
                                            ix) *
                                            model.nx;
                                }

                                se::huygens::
                                    OneWayKernelData
                                        kernel;
                                kernel.rows =
                                    static_cast<int>(
                                        geometry.targets.
                                            size());
                                kernel.sources =
                                    model.nx;
                                kernel.source_z =
                                    model.z(
                                        block.source_iz);
                                kernel.
                                    quadrature_weights =
                                        &model_weights;
                                kernel.tables =
                                    &tables;
                                kernel.filter =
                                    filters[
                                        ifrequency_local]
                                        .get();

                                for (int ix_target = 0;
                                     ix_target <
                                        model.nx;
                                     ++ix_target) {
                                    const int row =
                                        ix_target *
                                            target_depth_count +
                                        iz_local;

                                    for (int source = 0;
                                         source <
                                            model.nx;
                                         ++source) {
                                        const std::size_t
                                            matrix_index =
                                                static_cast<
                                                    std::size_t>(
                                                    ix_target) *
                                                    model.nx +
                                                source;
                                        const fki::Complex
                                            value =
                                                kernel.
                                                    phase_removed_amplitude(
                                                        row,
                                                        source);
                                        amplitude[
                                            2 *
                                            matrix_index] =
                                            value.real();
                                        amplitude[
                                            2 *
                                                matrix_index +
                                            1] =
                                            value.imag();
                                    }
                                }

                                const std::size_t
                                    ifrequency =
                                        frequency_begin +
                                        ifrequency_local;
                                const float omega =
                                    2.0f * fki::kPi *
                                    axis.frequencies[
                                        ifrequency];

                                factors[index].reset(
                                    bf1d_strict_segmented_create_phase_amp(
                                        model.nx,
                                        tau_rows.data(),
                                        reinterpret_cast<
                                            const fftwf_complex*>(
                                            amplitude.data()),
                                        omega,
                                        p,
                                        leaf,
                                        panel_levels,
                                        amp_eps,
                                        phase_tol));

                                if (!factors[index]) {
                                    throw std::
                                        runtime_error(
                                            "bf1d factor construction returned null");
                                }
                            } catch (
                                const std::exception&
                                    error) {
                                bool expected = false;
                                if (build_failed.
                                        compare_exchange_strong(
                                            expected,
                                            true)) {
                                    std::lock_guard<
                                        std::mutex>
                                        lock(
                                            build_error_mutex);
                                    build_error =
                                        error.what();
                                }
                            } catch (...) {
                                bool expected = false;
                                if (build_failed.
                                        compare_exchange_strong(
                                            expected,
                                            true)) {
                                    std::lock_guard<
                                        std::mutex>
                                        lock(
                                            build_error_mutex);
                                    build_error =
                                        "unknown BF factor construction error";
                                }
                            }
                        }
                    }

                    if (build_failed.load()) {
                        throw std::runtime_error(
                            "low-memory BF tile "
                            "construction failed: " +
                            build_error);
                    }

                    block_build_seconds +=
                        std::chrono::duration<double>(
                            fki::Clock::now() -
                            build_started)
                            .count();

                    const auto apply_started =
                        fki::Clock::now();

                    for (std::size_t ishot = 0;
                         ishot < states.size();
                         ++ishot) {
                        std::vector<fki::Complex>
                            source_output(
                                frequency_count *
                                static_cast<
                                    std::size_t>(
                                    depth_count) *
                                model.nx);
                        std::vector<fki::Complex>
                            receiver_output(
                                source_output.size());

#ifdef _OPENMP
#pragma omp parallel
#endif
                        {
                            std::vector<float>
                                source_line(
                                    2 *
                                    static_cast<
                                        std::size_t>(
                                        model.nx));
                            std::vector<float>
                                receiver_line(
                                    2 *
                                    static_cast<
                                        std::size_t>(
                                        model.nx));

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                            for (
                                long long factor_index = 0;
                                factor_index <
                                    static_cast<
                                        long long>(
                                        factor_count);
                                ++factor_index) {
                                const std::size_t index =
                                    static_cast<
                                        std::size_t>(
                                        factor_index);
                                const std::size_t
                                    ifrequency_local =
                                        index /
                                        static_cast<
                                            std::size_t>(
                                            depth_count);
                                const int iz_batch =
                                    static_cast<int>(
                                        index %
                                        static_cast<
                                            std::size_t>(
                                            depth_count));
                                const std::size_t
                                    input_offset =
                                        2 *
                                        ifrequency_local *
                                        static_cast<
                                            std::size_t>(
                                            model.nx);

                                bf1d_strict_segmented_apply(
                                    factors[index].get(),
                                    reinterpret_cast<
                                        const fftwf_complex*>(
                                        packed_states[
                                            ishot]
                                            .source
                                            .data() +
                                        input_offset),
                                    reinterpret_cast<
                                        fftwf_complex*>(
                                        source_line.
                                            data()));

                                bf1d_strict_segmented_apply(
                                    factors[index].get(),
                                    reinterpret_cast<
                                        const fftwf_complex*>(
                                        packed_states[
                                            ishot]
                                            .receiver
                                            .data() +
                                        input_offset),
                                    reinterpret_cast<
                                        fftwf_complex*>(
                                        receiver_line.
                                            data()));

                                for (int ix = 0;
                                     ix < model.nx;
                                     ++ix) {
                                    const std::size_t
                                        output_index =
                                            (ifrequency_local *
                                                 static_cast<
                                                     std::size_t>(
                                                     depth_count) +
                                             static_cast<
                                                 std::size_t>(
                                                 iz_batch)) *
                                                static_cast<
                                                    std::size_t>(
                                                    model.nx) +
                                            static_cast<
                                                std::size_t>(
                                                ix);

                                    source_output[
                                        output_index] =
                                        fki::Complex(
                                            source_line[
                                                2 *
                                                static_cast<
                                                    std::size_t>(
                                                    ix)],
                                            source_line[
                                                2 *
                                                    static_cast<
                                                        std::size_t>(
                                                        ix) +
                                                1]);

                                    receiver_output[
                                        output_index] =
                                        fki::Complex(
                                            receiver_line[
                                                2 *
                                                static_cast<
                                                    std::size_t>(
                                                    ix)],
                                            receiver_line[
                                                2 *
                                                    static_cast<
                                                        std::size_t>(
                                                        ix) +
                                                1]);
                                }
                            }
                        }

                        if (iblock == 0) {
                            phase_shift_first_block(
                                model, block, geometry.target_iz,
                                axis.frequencies[frequency_begin],
                                packed_states[ishot].source, source_output);
                            phase_shift_first_block(
                                model, block, geometry.target_iz,
                                axis.frequencies[frequency_begin],
                                packed_states[ishot].receiver, receiver_output);
                        } else {
                            if (frequency_count != 1 || depth_begin != 0 ||
                                depth_count != target_depth_count) {
                                throw std::runtime_error(
                                    "MATLAB correction tile must contain one complete block");
                            }
                            const float frequency = axis.frequencies[frequency_begin];
                            correct_complete_block(
                                model, geometry.target_iz, frequency, correction,
                                shots[ishot], block.id, source_output, "source",
                                correction_stream);
                            correct_complete_block(
                                model, geometry.target_iz, frequency, correction,
                                shots[ishot], block.id, receiver_output, "receiver",
                                correction_stream);
                        }

                        accumulate_image_batch(
                            model,
                            axis,
                            frequency_begin,
                            frequency_count,
                            geometry.target_iz,
                            depth_begin,
                            depth_count,
                            source_output,
                            receiver_output,
                            image,
                            illumination);

                        if (iblock + 1 <
                            info.blocks.size()) {
                            extract_next_boundary_batch(
                                model,
                                frequency_begin,
                                frequency_count,
                                geometry.target_iz,
                                depth_begin,
                                depth_count,
                                info.blocks[
                                    iblock + 1]
                                    .source_iz,
                                source_output,
                                receiver_output,
                                states[ishot]);
                        }
                    }

                    block_apply_seconds +=
                        std::chrono::duration<double>(
                            fki::Clock::now() -
                            apply_started)
                            .count();

                    // Factors are destroyed here; the frequency-independent
                    // traveltime cache remains until this block completes.
                }
            }

            total_filter_seconds +=
                block_filter_seconds;
            total_build_seconds +=
                block_build_seconds;
            total_apply_seconds +=
                block_apply_seconds;

            const int column =
                static_cast<int>(iblock);
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                0] =
                static_cast<float>(
                    block_filter_seconds);
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                1] =
                static_cast<float>(
                    block_build_seconds);
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                2] =
                static_cast<float>(
                    block_apply_seconds);
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                3] =
                static_cast<float>(
                    total_factor_count);
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                4] =
                static_cast<float>(
                    peak_resident_factors);
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                5] =
                static_cast<float>(
                    states.size());
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                6] =
                static_cast<float>(
                    axis.frequencies.size());
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                7] =
                static_cast<float>(
                    target_depth_count);
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                8] =
                static_cast<float>(
                    frequency_batch);
            timing_values[
                static_cast<std::size_t>(column) *
                    metrics +
                9] =
                static_cast<float>(
                    depth_batch);

            std::cout
                << "  build="
                << block_build_seconds
                << " s, apply/correlation="
                << block_apply_seconds
                << " s, peak resident factors="
                << peak_resident_factors
                << '\n';
        }

        fki::finish_image(
            options, model, image, illumination);

        se::huygens::write_image_rsf(
            options.image,
            model,
            image,
            "frequency_kirchhoff_imaging_bf_gmres",
            {{"frequency_min_hz",
              axis.frequencies.front()},
             {"frequency_max_hz",
              axis.frequencies.back()},
             {"frequency_count",
              static_cast<float>(
                  axis.frequencies.size())},
             {"shot_count",
              static_cast<float>(
                  states.size())},
             {"cmp_geometry",
              static_cast<float>(
                  options.cmp)},
             {"cross_correlation_conjugate",
              1.0f},
             {"bf_p",
              static_cast<float>(p)},
             {"bf_n_leaf",
              static_cast<float>(leaf)},
             {"bf_phase_tol",
              phase_tol},
             {"bf_frequency_batch",
              static_cast<float>(
                  requested_frequency_batch)},
             {"bf_depth_batch",
              static_cast<float>(
                  requested_depth_batch)}});

        if (!options.illumination.empty()) {
            se::huygens::write_image_rsf(
                options.illumination,
                model,
                illumination,
                "frequency_kirchhoff_source_illumination",
                {{"frequency_count",
                  static_cast<float>(
                      axis.frequencies.size())},
                 {"shot_count",
                  static_cast<float>(
                      states.size())}});
        }

        se::huygens::write_timing_rsf(
            options.timing,
            timing_values,
            metrics,
            block_count,
            "frequency_kirchhoff_imaging_bf_gmres",
            {"filter_setup_seconds",
             "factor_build_seconds",
             "apply_and_image_seconds",
             "total_factor_count",
             "peak_resident_factor_count",
             "shot_count",
             "frequency_count",
             "target_depth_count",
             "frequency_batch",
             "depth_batch"},
            {{"fft_seconds",
              static_cast<float>(
                  fft_seconds)},
             {"total_filter_setup_seconds",
              static_cast<float>(
                  total_filter_seconds)},
             {"total_factor_build_seconds",
              static_cast<float>(
                  total_build_seconds)},
             {"total_apply_and_image_seconds",
              static_cast<float>(
                  total_apply_seconds)},
             {"skipped_receiver_coordinates",
              static_cast<float>(
                  skipped_receivers)}});

        std::cout
            << "Block BF + restarted GMRES imaging completed.\n"
            << "FFT/data preparation: "
            << fft_seconds << " s\n"
            << "Filter setup:         "
            << total_filter_seconds << " s\n"
            << "BF factor build:      "
            << total_build_seconds << " s\n"
            << "Apply + image:        "
            << total_apply_seconds << " s\n"
            << "Image:                "
            << options.image << '\n'
            << "Timing:               "
            << options.timing << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "frequency_kirchhoff_imaging_bf: "
            << error.what() << '\n';
        return 1;
    }
}
