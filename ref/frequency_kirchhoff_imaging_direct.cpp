#include "frequency_kirchhoff_imaging_common.hpp"
#include "program_help.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fki = frequency_kirchhoff_imaging;

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        huygens_cli::initialize(argc, argv);
        const fki::CommonOptions options =
            fki::parse_common_options("frequency_kirchhoff_imaging_direct");
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
            options, data, axis, shots, model, "direct one-way integral");

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

        const int metrics = 6;
        const int block_count = static_cast<int>(info.blocks.size());
        std::vector<float> timing_values(
            static_cast<std::size_t>(metrics) * block_count, 0.0f);
        double total_filter_seconds = 0.0;
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
                    "direct frequency imaging requires every lateral grid point");
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
            std::vector<se::huygens::OneWayKernelData> kernels(
                axis.frequencies.size());
            for (std::size_t ifrequency = 0;
                 ifrequency < axis.frequencies.size(); ++ifrequency) {
                filters[ifrequency] = std::make_unique<
                    se::huygens::FrequencyKirchhoffFilter>(
                        axis.frequencies[ifrequency], filter_dt,
                        options.filter_length, maximum_tau,
                        options.filter_lookup_subsamples);
                kernels[ifrequency].rows =
                    static_cast<int>(geometry.targets.size());
                kernels[ifrequency].sources = model.nx;
                kernels[ifrequency].source_z = model.z(block.source_iz);
                kernels[ifrequency].quadrature_weights = &model_weights;
                kernels[ifrequency].tables = &tables;
                kernels[ifrequency].filter = filters[ifrequency].get();
            }
            const double filter_seconds = std::chrono::duration<double>(
                fki::Clock::now() - filter_started).count();
            total_filter_seconds += filter_seconds;

            const std::size_t target_values_per_frequency =
                geometry.target_iz.size() * static_cast<std::size_t>(model.nx);
            const auto apply_started = fki::Clock::now();

            for (fki::ShotBoundaryState& state : states) {
                std::vector<fki::Complex> source_output(
                    axis.frequencies.size() * target_values_per_frequency);
                std::vector<fki::Complex> receiver_output(
                    source_output.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (long long ifrequency = 0;
                     ifrequency < static_cast<long long>(axis.frequencies.size());
                     ++ifrequency) {
                    const std::size_t frequency =
                        static_cast<std::size_t>(ifrequency);
                    const std::size_t input_offset =
                        frequency * static_cast<std::size_t>(model.nx);
                    const std::size_t output_offset =
                        frequency * target_values_per_frequency;
                    std::vector<fki::Complex> source_rows(
                        target_values_per_frequency);
                    std::vector<fki::Complex> receiver_rows(
                        target_values_per_frequency);

                    // OneWayKernelData rows use x-major storage with local
                    // depth as the fastest index:
                    //     row = ix * depth_count + iz_local.
                    // The imaging and boundary routines use depth-major
                    // storage:
                    //     packed = iz_local * nx + ix.
                    // Convert explicitly before accumulation and before
                    // extracting the next propagation datum.
                    fki::direct_apply_pair(
                        kernels[frequency],
                        state.source.data() + input_offset,
                        state.receiver_conjugate.data() + input_offset,
                        source_rows.data(),
                        receiver_rows.data());

                    const std::size_t depth_count =
                        geometry.target_iz.size();
                    for (int ix = 0; ix < model.nx; ++ix) {
                        for (std::size_t iz_local = 0;
                             iz_local < depth_count; ++iz_local) {
                            const std::size_t row =
                                static_cast<std::size_t>(ix) * depth_count +
                                iz_local;
                            const std::size_t packed =
                                iz_local * static_cast<std::size_t>(model.nx) +
                                static_cast<std::size_t>(ix);
                            source_output[output_offset + packed] =
                                source_rows[row];
                            receiver_output[output_offset + packed] =
                                receiver_rows[row];
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
                static_cast<float>(apply_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 2] =
                static_cast<float>(states.size());
            timing_values[static_cast<std::size_t>(column) * metrics + 3] =
                static_cast<float>(axis.frequencies.size());
            timing_values[static_cast<std::size_t>(column) * metrics + 4] =
                static_cast<float>(geometry.target_iz.size());
            timing_values[static_cast<std::size_t>(column) * metrics + 5] =
                static_cast<float>(geometry.targets.size());

            std::cout
                << "Direct imaging block " << block.id
                << ": filter setup=" << filter_seconds
                << " s, propagation/correlation=" << apply_seconds
                << " s, frequencies=" << axis.frequencies.size()
                << ", shots=" << states.size() << '\n';
        }

        fki::finish_image(options, model, image, illumination);
        se::huygens::write_image_rsf(
            options.image, model, image,
            "frequency_kirchhoff_imaging_direct",
            {{"frequency_min_hz", axis.frequencies.front()},
             {"frequency_max_hz", axis.frequencies.back()},
             {"frequency_count", static_cast<float>(axis.frequencies.size())},
             {"shot_count", static_cast<float>(states.size())},
             {"cmp_geometry", static_cast<float>(options.cmp)},
             {"cross_correlation_conjugate", 1.0f}});
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
            "frequency_kirchhoff_imaging_direct",
            {"filter_setup_seconds", "apply_and_image_seconds",
             "shot_count", "frequency_count", "target_depth_count",
             "target_point_count"},
            {{"fft_seconds", static_cast<float>(fft_seconds)},
             {"total_filter_setup_seconds",
              static_cast<float>(total_filter_seconds)},
             {"total_apply_and_image_seconds",
              static_cast<float>(total_apply_seconds)},
             {"skipped_receiver_coordinates",
              static_cast<float>(skipped_receivers)}});

        std::cout
            << "Direct frequency-domain Kirchhoff imaging completed.\n"
            << "FFT/data preparation: " << fft_seconds << " s\n"
            << "Filter setup:         " << total_filter_seconds << " s\n"
            << "Propagation + image:  " << total_apply_seconds << " s\n"
            << "Image:                " << options.image << '\n'
            << "Timing:               " << options.timing << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "frequency_kirchhoff_imaging_direct: "
                  << error.what() << '\n';
        return 1;
    }
}
