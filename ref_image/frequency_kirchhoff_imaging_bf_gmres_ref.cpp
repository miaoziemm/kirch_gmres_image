#include "frequency_kirchhoff_imaging_common.hpp"
#include "global_preconditioned_gmres.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/efmm.h>
#include <SERECKIRCH/include/se_butterfly.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fki = frequency_kirchhoff_imaging;
namespace gpg = global_preconditioned_gmres;

namespace {

constexpr const char* kProgramName =
    "frequency_kirchhoff_imaging_bf_global_gmres";
constexpr double kRayPhase = -0.75 * gpg::kPi;
constexpr double kOverlapAmplitudeFloor = 1.0e-3;
constexpr double kOverlapScaleCap = 2.0;

std::vector<fki::Complex> frequency_boundary(
    const std::vector<fki::Complex>& field, std::size_t frequency, int nx)
{
    const std::size_t first = frequency * static_cast<std::size_t>(nx);
    return {field.begin() + first, field.begin() + first + nx};
}

std::vector<fki::Complex> depth_major(
    const std::vector<fki::Complex>& x_major, int nx, int nz)
{
    std::vector<fki::Complex> result(static_cast<std::size_t>(nx) * nz);
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            result[static_cast<std::size_t>(iz) * nx + ix] =
                x_major[static_cast<std::size_t>(ix) * nz + iz];
        }
    }
    return result;
}

se::huygens::Block receiver_propagation_block(
    const se::huygens::BlockInfo& blocks, std::size_t block_index)
{
    se::huygens::Block block =
        fki::imaging_propagation_block(blocks, block_index);

    // The block file stores contiguous official output intervals.  For every
    // recursive block, recompute the rows immediately below its datum so that
    // the two neighboring Kirchhoff fields share physical samples to align.
    if (block_index > 0 && blocks.overlap_rows > 0) {
        const int shared_start = block.source_iz + 1;
        if (shared_start < block.target_start_iz)
            block.target_start_iz = shared_start;
    }
    return block;
}

std::vector<int> shared_depths(const std::vector<int>& first,
                               const std::vector<int>& second)
{
    std::vector<int> result;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < first.size() && j < second.size()) {
        if (first[i] < second[j]) {
            ++i;
        } else if (second[j] < first[i]) {
            ++j;
        } else {
            result.push_back(first[i]);
            ++i;
            ++j;
        }
    }
    return result;
}

int local_depth(const std::vector<int>& depths, int iz)
{
    const auto found = std::lower_bound(depths.begin(), depths.end(), iz);
    return found == depths.end() || *found != iz
        ? -1
        : static_cast<int>(std::distance(depths.begin(), found));
}

struct ReceiverOverlap {
    int nx = 0;
    std::size_t frequency_count = 0;
    std::vector<int> depths;
    std::vector<fki::Complex> field;

    bool empty() const { return depths.empty(); }

    std::size_t index(std::size_t shot, std::size_t frequency,
                      std::size_t depth, int ix) const
    {
        return (((shot * frequency_count + frequency) * depths.size() + depth) *
                    static_cast<std::size_t>(nx) + ix);
    }
};

ReceiverOverlap make_receiver_overlap(const std::vector<int>& depths, int nx,
                                      std::size_t shot_count,
                                      std::size_t frequency_count)
{
    ReceiverOverlap overlap;
    overlap.nx = nx;
    overlap.frequency_count = frequency_count;
    overlap.depths = depths;
    overlap.field.assign(shot_count * frequency_count * depths.size() *
                         static_cast<std::size_t>(nx),
                         fki::Complex(0.0f, 0.0f));
    return overlap;
}

void blend_receiver_overlap(const ReceiverOverlap& previous,
                            std::size_t shot, std::size_t frequency,
                            const std::vector<int>& target_iz,
                            std::vector<fki::Complex>& current)
{
    if (previous.empty()) return;

    double maximum_amplitude = 0.0;
    for (std::size_t idepth = 0; idepth < previous.depths.size(); ++idepth) {
        const int current_iz = local_depth(target_iz, previous.depths[idepth]);
        if (current_iz < 0) continue;
        for (int ix = 0; ix < previous.nx; ++ix) {
            const fki::Complex old_value = previous.field[
                previous.index(shot, frequency, idepth, ix)];
            const fki::Complex new_value = current[
                static_cast<std::size_t>(current_iz) * previous.nx + ix];
            maximum_amplitude = std::max(
                maximum_amplitude,
                std::max(static_cast<double>(std::abs(old_value)),
                         static_cast<double>(std::abs(new_value))));
        }
    }

    const double amplitude_floor =
        kOverlapAmplitudeFloor * maximum_amplitude;
    std::complex<double> numerator(0.0, 0.0);
    double denominator = 0.0;
    for (std::size_t idepth = 0; idepth < previous.depths.size(); ++idepth) {
        const int current_iz = local_depth(target_iz, previous.depths[idepth]);
        if (current_iz < 0) continue;
        for (int ix = 0; ix < previous.nx; ++ix) {
            const fki::Complex old_value = previous.field[
                previous.index(shot, frequency, idepth, ix)];
            const fki::Complex new_value = current[
                static_cast<std::size_t>(current_iz) * previous.nx + ix];
            const std::complex<double> old_field(
                old_value.real(), old_value.imag());
            const std::complex<double> new_field(
                new_value.real(), new_value.imag());
            if (std::abs(old_field) < amplitude_floor ||
                std::abs(new_field) < amplitude_floor) {
                continue;
            }
            numerator += std::conj(new_field) * old_field;
            denominator += std::norm(new_field);
        }
    }

    std::complex<double> scale(1.0, 0.0);
    if (denominator > std::numeric_limits<double>::min()) {
        scale = numerator / denominator;
        if (!std::isfinite(scale.real()) || !std::isfinite(scale.imag()) ||
            std::abs(scale) < std::numeric_limits<double>::min()) {
            scale = std::complex<double>(1.0, 0.0);
        }
    }

    const double raw_scale_magnitude = std::abs(scale);
    if (raw_scale_magnitude > kOverlapScaleCap) {
        scale *= kOverlapScaleCap / raw_scale_magnitude;
    } else if (raw_scale_magnitude < 1.0 / kOverlapScaleCap) {
        scale *= 1.0 / (kOverlapScaleCap * raw_scale_magnitude);
    }

    const double scale_magnitude = std::abs(scale);
    const double scale_phase = std::arg(scale);
    for (std::size_t idepth = 0; idepth < previous.depths.size(); ++idepth) {
        const int current_iz = local_depth(target_iz, previous.depths[idepth]);
        if (current_iz < 0) continue;

        const double new_weight = previous.depths.size() == 1
            ? 0.5
            : static_cast<double>(idepth) /
                  static_cast<double>(previous.depths.size() - 1);
        const double old_weight = 1.0 - new_weight;
        const std::complex<double> local_scale = std::polar(
            std::pow(scale_magnitude, old_weight), scale_phase * old_weight);

        for (int ix = 0; ix < previous.nx; ++ix) {
            const fki::Complex old_value = previous.field[
                previous.index(shot, frequency, idepth, ix)];
            const std::size_t current_index =
                static_cast<std::size_t>(current_iz) * previous.nx + ix;
            const fki::Complex new_value = current[current_index];
            const std::complex<double> fused =
                old_weight * std::complex<double>(
                    old_value.real(), old_value.imag()) +
                new_weight * local_scale * std::complex<double>(
                    new_value.real(), new_value.imag());
            current[current_index] = fki::Complex(
                static_cast<float>(fused.real()),
                static_cast<float>(fused.imag()));
        }
    }
}

void store_receiver_overlap(const std::vector<int>& target_iz,
                            const std::vector<fki::Complex>& field,
                            std::size_t shot, std::size_t frequency,
                            ReceiverOverlap& overlap)
{
    for (std::size_t idepth = 0; idepth < overlap.depths.size(); ++idepth) {
        const int current_iz = local_depth(target_iz, overlap.depths[idepth]);
        if (current_iz < 0) continue;
        for (int ix = 0; ix < overlap.nx; ++ix) {
            overlap.field[overlap.index(shot, frequency, idepth, ix)] =
                field[static_cast<std::size_t>(current_iz) * overlap.nx + ix];
        }
    }
}

std::vector<fki::Complex> phase_shift(
    const se::huygens::Model2D& model, const se::huygens::Block& block,
    const std::vector<int>& target_iz, float frequency,
    const std::vector<fki::Complex>& boundary)
{
    const int nx = model.nx;
    std::vector<fki::Complex> spectrum = boundary;
    fftwf_plan forward = fftwf_plan_dft_1d(
        nx, reinterpret_cast<fftwf_complex*>(spectrum.data()),
        reinterpret_cast<fftwf_complex*>(spectrum.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(forward);
    fftwf_destroy_plan(forward);

    double mean_velocity = 0.0;
    for (const int iz : target_iz) {
        for (int ix = 0; ix < nx; ++ix)
            mean_velocity += model.velocity[model.index(ix, iz)];
    }
    mean_velocity /= static_cast<double>(target_iz.size() * nx);

    const double k0 = 2.0 * gpg::kPi * frequency / mean_velocity;
    std::vector<fki::Complex> result(
        static_cast<std::size_t>(target_iz.size()) * nx);
    std::vector<fki::Complex> line(static_cast<std::size_t>(nx));
    fftwf_plan inverse = fftwf_plan_dft_1d(
        nx, reinterpret_cast<fftwf_complex*>(line.data()),
        reinterpret_cast<fftwf_complex*>(line.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);

    for (std::size_t iz = 0; iz < target_iz.size(); ++iz) {
        const double distance = model.z(target_iz[iz]) - model.z(block.source_iz);
        for (int ik = 0; ik < nx; ++ik) {
            const int signed_k = ik <= nx / 2 ? ik : ik - nx;
            const double kx = 2.0 * gpg::kPi * signed_k / (nx * model.dx);
            const double kz2 = k0 * k0 - kx * kx;
            const fki::Complex propagator = kz2 >= 0.0
                ? std::exp(fki::Complex(0.0f,
                    static_cast<float>(-std::sqrt(kz2) * distance)))
                : fki::Complex(static_cast<float>(
                    std::exp(-std::sqrt(-kz2) * std::abs(distance))), 0.0f);
            line[ik] = spectrum[ik] * propagator;
        }

        fftwf_execute(inverse);
        for (int ix = 0; ix < nx; ++ix)
            result[iz * static_cast<std::size_t>(nx) + ix] =
                line[ix] / static_cast<float>(nx);
    }

    fftwf_destroy_plan(inverse);
    return result;
}

void save_receiver_boundary(const se::huygens::Model2D& model,
                            std::size_t frequency,
                            const std::vector<int>& target_iz, int next_iz,
                            const std::vector<fki::Complex>& field,
                            fki::ShotBoundaryState& state)
{
    const int local_iz = fki::find_local_depth(target_iz, next_iz);
    if (local_iz < 0) return;

    for (int ix = 0; ix < model.nx; ++ix) {
        state.receiver_conjugate[frequency *
            static_cast<std::size_t>(model.nx) + ix] =
            field[static_cast<std::size_t>(local_iz) * model.nx + ix];
    }
}

std::size_t receiver_index(std::size_t shot, std::size_t frequency,
                           std::size_t grid, std::size_t frequency_count,
                           std::size_t grid_size)
{
    return (shot * frequency_count + frequency) * grid_size + grid;
}

void save_receiver_block(const se::huygens::Model2D& model,
                         const std::vector<int>& target_iz,
                         const std::vector<fki::Complex>& field,
                         std::size_t shot, std::size_t frequency,
                         std::size_t frequency_count,
                         std::vector<fki::Complex>& receiver_kirchhoff_field)
{
    const std::size_t grid_size =
        static_cast<std::size_t>(model.nx) * model.nz;
    for (std::size_t iz = 0; iz < target_iz.size(); ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            receiver_kirchhoff_field[receiver_index(
                shot, frequency, model.index(ix, target_iz[iz]),
                frequency_count, grid_size)] =
                field[iz * static_cast<std::size_t>(model.nx) + ix];
        }
    }
}

int nearest_index(float coordinate, float origin, float step, int count)
{
    return std::clamp(
        static_cast<int>(std::lround((coordinate - origin) / step)),
        0, count - 1);
}

enum class EfmmLayout { zfast, xfast };

std::size_t efmm_index(EfmmLayout layout, int iz, int ix, int nz, int nx)
{
    return layout == EfmmLayout::zfast
        ? static_cast<std::size_t>(ix) * nz + iz
        : static_cast<std::size_t>(iz) * nx + ix;
}

std::vector<float> efmm_velocity(const se::huygens::Model2D& model,
                                 EfmmLayout layout)
{
    std::vector<float> velocity(
        static_cast<std::size_t>(model.nx) * model.nz);
    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            velocity[efmm_index(layout, iz, ix, model.nz, model.nx)] =
                model.velocity[model.index(ix, iz)];
        }
    }
    return velocity;
}

EfmmLayout detect_efmm_layout(const std::vector<float>& time,
                              int nz, int nx, int source_iz, int source_ix)
{
    const float minimum = *std::min_element(time.begin(), time.end());
    const float tolerance = std::max(1.0e-7f, 1.0e-4f * std::abs(minimum));
    const bool zfast = std::abs(time[efmm_index(
        EfmmLayout::zfast, source_iz, source_ix, nz, nx)] - minimum) <= tolerance;
    const bool xfast = std::abs(time[efmm_index(
        EfmmLayout::xfast, source_iz, source_ix, nz, nx)] - minimum) <= tolerance;
    if (zfast == xfast) {
        throw std::runtime_error("cannot determine eFMM travel-time layout");
    }
    return zfast ? EfmmLayout::zfast : EfmmLayout::xfast;
}

std::vector<float> source_traveltime(
    const se::huygens::Model2D& model, int source_ix, int source_iz)
{
    const auto solve = [&](EfmmLayout layout) {
        efmm_t solver{};
        if (efmm_init(&solver, model.nx, model.nz, model.dx, model.dz,
                      source_ix * model.dx, source_iz * model.dz) != 0) {
            throw std::runtime_error("efmm_init failed");
        }
        try {
            std::vector<float> velocity = efmm_velocity(model, layout);
            if (efmm_set_vel(&solver, velocity.data()) != 0 ||
                efmm_solver(&solver) != 0) {
                throw std::runtime_error("eFMM solve failed");
            }
            std::vector<float> time(
                solver.tt, solver.tt + static_cast<std::size_t>(model.nx) * model.nz);
            efmm_free(&solver);
            return time;
        } catch (...) {
            efmm_free(&solver);
            throw;
        }
    };

    std::vector<float> raw = solve(EfmmLayout::zfast);
    EfmmLayout layout = detect_efmm_layout(
        raw, model.nz, model.nx, source_iz, source_ix);
    if (layout != EfmmLayout::zfast) {
        raw = solve(layout);
        layout = detect_efmm_layout(
            raw, model.nz, model.nx, source_iz, source_ix);
    }

    std::vector<float> time(
        static_cast<std::size_t>(model.nx) * model.nz);
    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            time[model.index(ix, iz)] =
                raw[efmm_index(layout, iz, ix, model.nz, model.nx)];
        }
    }
    return time;
}

gpg::Vector point_ray(const se::huygens::Model2D& model,
                      const std::vector<float>& time, float frequency,
                      gpg::Complex source, int source_ix, int source_iz)
{
    const double omega = 2.0 * gpg::kPi * frequency;
    const double tmin = std::max(model.dx, model.dz) /
        model.velocity[model.index(source_ix, source_iz)];
    const gpg::Complex discrete_source = source *
        (static_cast<double>(model.dx) * model.dz);
    gpg::Vector field(static_cast<Eigen::Index>(model.nx) * model.nz);

    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            const double travel_time =
                std::max(0.0, static_cast<double>(time[model.index(ix, iz)]));
            const double effective_time = std::max(travel_time, tmin);
            const double amplitude =
                1.0 / std::sqrt(8.0 * gpg::kPi * omega * effective_time);
            field[gpg::index(iz, ix, model.nx)] = discrete_source *
                std::polar(amplitude, omega * travel_time + kRayPhase);
        }
    }
    return field;
}

std::vector<float> pad_velocity(const se::huygens::Model2D& model)
{
    const int nz = model.nz + 2 * gpg::kPml;
    const int nx = model.nx + 2 * gpg::kPml;
    std::vector<float> padded(static_cast<std::size_t>(nz) * nx);

    for (int iz = 0; iz < nz; ++iz) {
        const int physical_iz = std::clamp(iz - gpg::kPml, 0, model.nz - 1);
        for (int ix = 0; ix < nx; ++ix) {
            const int physical_ix = std::clamp(ix - gpg::kPml, 0, model.nx - 1);
            padded[gpg::index(iz, ix, nx)] =
                model.velocity[model.index(physical_ix, physical_iz)];
        }
    }
    return padded;
}

gpg::Vector pad_field(const gpg::Vector& field, int nz, int nx)
{
    const int padded_nz = nz + 2 * gpg::kPml;
    const int padded_nx = nx + 2 * gpg::kPml;
    gpg::Vector padded(static_cast<Eigen::Index>(padded_nz) * padded_nx);

    for (int iz = 0; iz < padded_nz; ++iz) {
        const int physical_iz = std::clamp(iz - gpg::kPml, 0, nz - 1);
        for (int ix = 0; ix < padded_nx; ++ix) {
            const int physical_ix = std::clamp(ix - gpg::kPml, 0, nx - 1);
            padded[gpg::index(iz, ix, padded_nx)] =
                field[gpg::index(physical_iz, physical_ix, nx)];
        }
    }
    return padded;
}

gpg::Vector crop_field(const gpg::Vector& padded, int nz, int nx)
{
    const int padded_nx = nx + 2 * gpg::kPml;
    gpg::Vector field(static_cast<Eigen::Index>(nz) * nx);

    for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            field[gpg::index(iz, ix, nx)] =
                padded[gpg::index(iz + gpg::kPml, ix + gpg::kPml, padded_nx)];
        }
    }
    return field;
}

gpg::Vector lower_mask(const gpg::Vector& field, int nz, int nx,
                       int first_iz)
{
    gpg::Vector masked = gpg::Vector::Zero(field.size());
    for (int iz = first_iz; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            masked[gpg::index(iz, ix, nx)] = field[gpg::index(iz, ix, nx)];
        }
    }
    return masked;
}

struct FieldCorrection {
    gpg::Vector field;
    gpg::Vector update;
    int steps = 0;
    double residual_before = 0.0;
    double residual_after = 0.0;
};

gpg::Vector receiver_kirchhoff_vector(
    const se::huygens::Model2D& model,
    const std::vector<fki::Complex>& receiver_kirchhoff_field,
    std::size_t offset)
{
    gpg::Vector field(static_cast<Eigen::Index>(model.nx) * model.nz);
    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            const fki::Complex value =
                receiver_kirchhoff_field[offset + model.index(ix, iz)];
            field[gpg::index(iz, ix, model.nx)] =
                gpg::Complex(value.real(), value.imag());
        }
    }
    return field;
}

FieldCorrection correct_global_field(const gpg::Sparse& matrix,
                                     const gpg::Vector& input_field,
                                     int nz, int nx, int correction_iz,
                                     int restart, int outer)
{
    const int padded_nz = nz + 2 * gpg::kPml;
    const int padded_nx = nx + 2 * gpg::kPml;
    const int padded_correction_iz = correction_iz + gpg::kPml;
    const gpg::Vector padded_field = pad_field(input_field, nz, nx);
    const gpg::Vector initial = lower_mask(
        padded_field, padded_nz, padded_nx, padded_correction_iz);
    const gpg::Vector a_field = matrix * padded_field;
    const gpg::Vector rhs = matrix * initial - lower_mask(
        a_field, padded_nz, padded_nx, padded_correction_iz);
    const gpg::Result result = gpg::solve(
        matrix, rhs, initial, restart, outer);
    const gpg::Vector solved = crop_field(result.field, nz, nx);

    gpg::Vector corrected = input_field;
    for (int iz = correction_iz; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            corrected[gpg::index(iz, ix, nx)] =
                solved[gpg::index(iz, ix, nx)];
        }
    }
    return {corrected, corrected - input_field, result.steps,
            result.residual_before, result.residual_after};
}

void write_field(const std::filesystem::path& directory,
                 const se::huygens::Model2D& model, const gpg::Vector& field,
                 int shot, std::size_t frequency_index, float frequency,
                 const char* name)
{
    const std::size_t size = static_cast<std::size_t>(model.nx) * model.nz;
    std::vector<float> real(size), imag(size);
    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            const gpg::Complex value = field[gpg::index(iz, ix, model.nx)];
            const std::size_t grid = model.index(ix, iz);
            real[grid] = static_cast<float>(value.real());
            imag[grid] = static_cast<float>(value.imag());
        }
    }

    std::ostringstream stem;
    stem << name << "_shot" << shot
         << "_ifreq" << frequency_index + 1
         << "_f" << std::fixed << std::setprecision(6) << frequency << "Hz";
    se::huygens::write_image_rsf(
        (directory / (stem.str() + "_real.rsf")).string(), model, real,
        std::string(name) + "_real");
    se::huygens::write_image_rsf(
        (directory / (stem.str() + "_imag.rsf")).string(), model, imag,
        std::string(name) + "_imag");
}

void write_traveltime(const std::filesystem::path& directory,
                      const se::huygens::Model2D& model,
                      const std::vector<float>& time, int shot)
{
    se::huygens::write_image_rsf(
        (directory / ("source_traveltime_shot" + std::to_string(shot) +
                      ".rsf")).string(),
        model, time, "point_source_traveltime");
}

} // namespace

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        huygens_cli::initialize(argc, argv);
        const fki::CommonOptions options = fki::parse_common_options(kProgramName);
        huygens_cli::set_openmp_threads(options.threads);

        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(options.velocity);
        const se::huygens::BlockInfo blocks =
            se::huygens::read_block_info(options.block_file);
        se::huygens::validate_block_info(blocks, &model);

        const int source_iz = nearest_index(
            huygens_cli::optional_float("global_source_z", model.oz),
            model.oz, model.dz, model.nz);
        const int correction_iz = nearest_index(
            huygens_cli::optional_float(
                "global_source_correction_z0",
                model.oz + std::min(model.nz - 1, source_iz + 30) * model.dz),
            model.oz, model.dz, model.nz);
        const int gmres_restart = huygens_cli::optional_int(
            "gmres_restart", gpg::kDefaultRestart);
        const int gmres_outer = huygens_cli::optional_int(
            "gmres_outer", gpg::kDefaultOuter);
        if (gmres_restart < 1 || gmres_outer < 1) {
            throw std::invalid_argument(
                "gmres_restart and gmres_outer must both be positive");
        }
        const std::string field_output =
            huygens_cli::optional_string("source_wavefield_output_dir", "");
        const std::string image_before_correction =
            huygens_cli::optional_string("image_before_correction", "");
        const std::string illumination_before_correction =
            huygens_cli::optional_string(
                "illumination_before_correction", "");

        fki::SeismicData data(options.seismic_data, options.cmp != 0);
        const fki::FrequencyAxis axis = fki::make_frequency_axis(data, options);
        const std::vector<int> shots = fki::selected_shots(data, options);
        const std::vector<float> weights = fki::full_boundary_weights(model);
        const std::vector<fki::Complex> source_spectrum = fki::ricker_spectrum(
            data, axis, options.fdom, options.source_time, options.source_amplitude);

        std::cout << "Global source/receiver GMRES: restart=" << gmres_restart
                  << ", outer=" << gmres_outer
                  << ", max steps=" << gmres_restart * gmres_outer
                  << ", z0=" << model.z(correction_iz) << '\n';

        double fft_seconds = 0.0;
        int skipped_receivers = 0;
        std::vector<fki::ShotBoundaryState> states;
        states.reserve(shots.size());
        for (const int shot : shots) {
            states.push_back(fki::initialize_shot_state(
                data, shot, model, options, axis, source_spectrum, weights,
                fft_seconds, skipped_receivers));
        }

        const std::size_t grid_size =
            static_cast<std::size_t>(model.nx) * model.nz;
        const std::size_t frequency_count = axis.frequencies.size();
        // This array holds the complete receiver Kirchhoff continuation for
        // every shot, frequency, and physical depth before GMRES is invoked.
        std::vector<fki::Complex> receiver_kirchhoff_field(
            states.size() * frequency_count * grid_size,
            fki::Complex(0.0f, 0.0f));
        // Keep the two imaging conditions separate for a direct comparison:
        // ray/Kirchhoff before GMRES and source/receiver after GMRES.
        std::vector<float> image_before_correction_field(grid_size, 0.0f);
        std::vector<float> illumination_before_correction_field(
            grid_size, 0.0f);
        std::vector<float> image_after_correction(grid_size, 0.0f);
        std::vector<float> illumination_after_correction(grid_size, 0.0f);
        std::vector<float> timing(4 * blocks.blocks.size(), 0.0f);

        const se::butterfly::Options butterfly_options = [] {
            se::butterfly::Options options;
            options.tolerance = 1.0e-4f;
            options.leaf_size = 64;
            options.verbosity = -1;
            options.coordinate_dimension = 2;
            options.lr_level = 100;
            options.sample_parameter = 1.5f;
            options.forward_n15_flag = 0;
            options.nearest_neighbors = 0;
            options.compression_pattern = 1;
            options.less_adapt = 1;
            options.rank_detection_factor = 0.3f;
            options.reuse_geometry = 1;
            return options;
        }();

        double total_filter_seconds = 0.0;
        double total_build_seconds = 0.0;
        double total_apply_seconds = 0.0;

        std::vector<std::vector<int>> receiver_block_depths(
            blocks.blocks.size());
        for (std::size_t iblock = 0; iblock < blocks.blocks.size(); ++iblock) {
            const se::huygens::Block block =
                receiver_propagation_block(blocks, iblock);
            receiver_block_depths[iblock] =
                se::huygens::make_layer_geometry(
                    model, block, options.source_stride,
                    options.target_x_stride, options.target_z_stride).target_iz;
        }
        ReceiverOverlap previous_receiver_overlap;

        // Original recursive receiver continuation: the surface near block
        // uses its homogeneous one-way phase shift; every following block is
        // the Kirchhoff integral evaluated by a Butterfly matrix.  Neighboring
        // blocks recompute their shared rows, then align and blend them before
        // accepting either field into the complete receiver wavefield.
        for (std::size_t iblock = 0; iblock < blocks.blocks.size(); ++iblock) {
            const se::huygens::Block block =
                receiver_propagation_block(blocks, iblock);
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block, options.source_stride,
                    options.target_x_stride, options.target_z_stride);
            const se::huygens::OneWayLayerTables tables =
                se::huygens::read_one_way_layer_tables(options.table_prefix, block.id);

            const bool first_block = iblock == 0;
            const float maximum_time = first_block ? 0.0f :
                *std::max_element(tables.traveltime.values.begin(),
                                  tables.traveltime.values.end());
            const float filter_dt = options.filter_dt > 0.0f
                ? options.filter_dt : data.dt();

            std::vector<double> rows;
            std::vector<double> columns;
            if (!first_block) {
                rows.reserve(2 * geometry.targets.size());
                for (const auto& point : geometry.targets) {
                    rows.push_back(model.ox + point.ix * model.dx);
                    rows.push_back(model.oz + point.iz * model.dz);
                }
                columns.reserve(2 * geometry.source_ix.size());
                for (const int ix : geometry.source_ix) {
                    columns.push_back(model.ox + ix * model.dx);
                    columns.push_back(model.z(block.source_iz));
                }
            }

            double filter_seconds = 0.0;
            double build_seconds = 0.0;
            double apply_seconds = 0.0;

            ReceiverOverlap next_receiver_overlap;
            if (iblock + 1 < blocks.blocks.size()) {
                next_receiver_overlap = make_receiver_overlap(
                    shared_depths(receiver_block_depths[iblock],
                                  receiver_block_depths[iblock + 1]),
                    model.nx, states.size(), frequency_count);
            }

            for (std::size_t ifrequency = 0;
                 ifrequency < frequency_count; ++ifrequency) {
                const float frequency = axis.frequencies[ifrequency];
                std::vector<std::vector<fki::Complex>> propagated;

                if (first_block) {
                    propagated.reserve(states.size());
                    for (const auto& state : states) {
                        propagated.push_back(phase_shift(
                            model, block, geometry.target_iz, frequency,
                            frequency_boundary(
                                state.receiver_conjugate, ifrequency, model.nx)));
                    }
                } else {
                    const auto filter_start = fki::Clock::now();
                    const se::huygens::FrequencyKirchhoffFilter filter(
                        frequency, filter_dt, options.filter_length,
                        maximum_time, options.filter_lookup_subsamples);
                    filter_seconds += std::chrono::duration<double>(
                        fki::Clock::now() - filter_start).count();

                    se::huygens::OneWayKernelData kernel;
                    kernel.rows = static_cast<int>(geometry.targets.size());
                    kernel.sources = static_cast<int>(geometry.source_ix.size());
                    kernel.source_z = model.z(block.source_iz);
                    kernel.quadrature_weights = &weights;
                    kernel.tables = &tables;
                    kernel.filter = &filter;

                    std::vector<std::vector<fki::Complex>> input;
                    input.reserve(states.size());
                    for (const auto& state : states) {
                        input.push_back(frequency_boundary(
                            state.receiver_conjugate, ifrequency, model.nx));
                    }

                    const auto build_start = fki::Clock::now();
                    se::butterfly::Matrix<float> butterfly(
                        kernel.rows, kernel.sources, rows, columns,
                        [&kernel](int row, int source) {
                            return kernel.entry(row, source);
                        },
                        butterfly_options);
                    build_seconds += std::chrono::duration<double>(
                        fki::Clock::now() - build_start).count();

                    const auto apply_start = fki::Clock::now();
                    const std::vector<std::vector<fki::Complex>> output =
                        butterfly.apply_many(input);
                    apply_seconds += std::chrono::duration<double>(
                        fki::Clock::now() - apply_start).count();

                    propagated.reserve(output.size());
                    for (const auto& field : output) {
                        propagated.push_back(
                            depth_major(field, model.nx,
                                        static_cast<int>(geometry.target_iz.size())));
                    }
                }

                for (std::size_t ishot = 0; ishot < states.size(); ++ishot) {
                    blend_receiver_overlap(
                        previous_receiver_overlap, ishot, ifrequency,
                        geometry.target_iz, propagated[ishot]);
                    save_receiver_block(
                        model, geometry.target_iz, propagated[ishot], ishot,
                        ifrequency, frequency_count, receiver_kirchhoff_field);
                    store_receiver_overlap(
                        geometry.target_iz, propagated[ishot], ishot,
                        ifrequency, next_receiver_overlap);
                    if (iblock + 1 < blocks.blocks.size()) {
                        save_receiver_boundary(
                            model, ifrequency, geometry.target_iz,
                            blocks.blocks[iblock + 1].source_iz,
                            propagated[ishot], states[ishot]);
                    }
                }
            }

            timing[4 * iblock] = static_cast<float>(filter_seconds);
            timing[4 * iblock + 1] = static_cast<float>(build_seconds);
            timing[4 * iblock + 2] = static_cast<float>(apply_seconds);
            timing[4 * iblock + 3] =
                static_cast<float>(filter_seconds + build_seconds + apply_seconds);
            total_filter_seconds += filter_seconds;
            total_build_seconds += build_seconds;
            total_apply_seconds += apply_seconds;
            previous_receiver_overlap = std::move(next_receiver_overlap);
        }

        std::filesystem::path output_directory;
        if (!field_output.empty()) {
            output_directory = field_output;
            std::filesystem::create_directories(output_directory);
        }

        std::vector<int> source_ix(states.size());
        std::vector<std::vector<float>> traveltime(states.size());
        for (std::size_t ishot = 0; ishot < states.size(); ++ishot) {
            source_ix[ishot] = nearest_index(
                states[ishot].shot_x, model.ox, model.dx, model.nx);
            traveltime[ishot] =
                source_traveltime(model, source_ix[ishot], source_iz);
            if (!field_output.empty()) {
                write_traveltime(
                    output_directory, model, traveltime[ishot], shots[ishot]);
            }
        }

        const std::vector<float> velocity = pad_velocity(model);
        const int padded_nz = model.nz + 2 * gpg::kPml;
        const int padded_nx = model.nx + 2 * gpg::kPml;

        for (std::size_t ifrequency = 0;
             ifrequency < frequency_count; ++ifrequency) {
            const float frequency = axis.frequencies[ifrequency];
            const gpg::Sparse source_matrix = gpg::build_helmholtz(
                velocity, padded_nz, padded_nx, model.dz, model.dx, frequency);
            const gpg::Sparse receiver_matrix = gpg::build_helmholtz(
                velocity, padded_nz, padded_nx, model.dz, model.dx, frequency,
                true);

            for (std::size_t ishot = 0; ishot < states.size(); ++ishot) {
                const gpg::Complex source = std::conj(gpg::Complex(
                    source_spectrum[ifrequency].real(),
                    source_spectrum[ifrequency].imag()));
                const gpg::Vector ray = point_ray(
                    model, traveltime[ishot], frequency, source,
                    source_ix[ishot], source_iz);
                const FieldCorrection source_corrected = correct_global_field(
                    source_matrix, ray, model.nz, model.nx, correction_iz,
                    gmres_restart, gmres_outer);

                const std::size_t offset = receiver_index(
                    ishot, ifrequency, 0, frequency_count, grid_size);
                // GMRES starts only after the complete recursive Kirchhoff /
                // Butterfly receiver field has been assembled above.
                const gpg::Vector receiver_kirchhoff =
                    receiver_kirchhoff_vector(
                        model, receiver_kirchhoff_field, offset);
                const FieldCorrection receiver_corrected = correct_global_field(
                    receiver_matrix, receiver_kirchhoff, model.nz, model.nx,
                    correction_iz, gmres_restart, gmres_outer);

                std::cout << "shot=" << shots[ishot]
                          << " f=" << frequency
                          << " source=" << source_corrected.steps
                          << " (" << source_corrected.residual_before
                          << " -> " << source_corrected.residual_after << ')'
                          << " receiver=" << receiver_corrected.steps
                          << " (" << receiver_corrected.residual_before
                          << " -> " << receiver_corrected.residual_after << ')'
                          << '\n';

                if (!field_output.empty()) {
                    write_field(output_directory, model, ray, shots[ishot],
                                ifrequency, frequency, "source_ray");
                    write_field(output_directory, model, source_corrected.field,
                                shots[ishot], ifrequency, frequency,
                                "source_gmres");
                    write_field(output_directory, model, source_corrected.update,
                                shots[ishot], ifrequency, frequency,
                                "source_update");
                    write_field(output_directory, model, receiver_kirchhoff,
                                shots[ishot], ifrequency, frequency,
                                "receiver_kirchhoff");
                    write_field(output_directory, model, receiver_corrected.field,
                                shots[ishot], ifrequency, frequency,
                                "receiver_gmres");
                    write_field(output_directory, model, receiver_corrected.update,
                                shots[ishot], ifrequency, frequency,
                                "receiver_update");
                }

                const float weight = axis.imaging_weights[ifrequency];
                for (int iz = 0; iz < model.nz; ++iz) {
                    for (int ix = 0; ix < model.nx; ++ix) {
                        const int local = gpg::index(iz, ix, model.nx);
                        const std::size_t grid = model.index(ix, iz);
                        const gpg::Complex source_before = ray[local];
                        const gpg::Complex receiver_before =
                            receiver_kirchhoff[local];
                        image_before_correction_field[grid] +=
                            weight * static_cast<float>(
                                (source_before * receiver_before).real());
                        illumination_before_correction_field[grid] +=
                            weight * static_cast<float>(std::norm(source_before));

                        const gpg::Complex source_after =
                            source_corrected.field[local];
                        const gpg::Complex receiver_after =
                            receiver_corrected.field[local];
                        image_after_correction[grid] += weight * static_cast<float>(
                            (source_after * receiver_after).real());
                        illumination_after_correction[grid] +=
                            weight * static_cast<float>(std::norm(source_after));
                    }
                }
            }
        }

        fki::finish_image(
            options, model, image_before_correction_field,
            illumination_before_correction_field);
        fki::finish_image(
            options, model, image_after_correction,
            illumination_after_correction);
        if (!image_before_correction.empty()) {
            se::huygens::write_image_rsf(
                image_before_correction, model, image_before_correction_field,
                "source_ray_receiver_kirchhoff_cross_correlation",
                {{"frequency_min_hz", axis.frequencies.front()},
                 {"frequency_max_hz", axis.frequencies.back()},
                 {"frequency_count", static_cast<float>(frequency_count)},
                 {"shot_count", static_cast<float>(states.size())}});
        }
        if (!illumination_before_correction.empty()) {
            se::huygens::write_image_rsf(
                illumination_before_correction, model,
                illumination_before_correction_field,
                "source_ray_illumination");
        }
        se::huygens::write_image_rsf(
            options.image, model, image_after_correction,
            "source_receiver_gmres_cross_correlation",
            {{"frequency_min_hz", axis.frequencies.front()},
             {"frequency_max_hz", axis.frequencies.back()},
             {"frequency_count", static_cast<float>(frequency_count)},
             {"shot_count", static_cast<float>(states.size())}});
        if (!options.illumination.empty()) {
            se::huygens::write_image_rsf(
                options.illumination, model, illumination_after_correction,
                "point_source_gmres_illumination");
        }

        se::huygens::write_timing_rsf(
            options.timing, timing, 4, static_cast<int>(blocks.blocks.size()),
            kProgramName,
            {"filter_seconds", "bpack_build_seconds",
             "bpack_apply_seconds", "total_seconds"},
            {{"fft_seconds", static_cast<float>(fft_seconds)},
             {"total_filter_seconds", static_cast<float>(total_filter_seconds)},
             {"total_bpack_build_seconds", static_cast<float>(total_build_seconds)},
             {"total_bpack_apply_seconds", static_cast<float>(total_apply_seconds)},
             {"skipped_receiver_coordinates",
              static_cast<float>(skipped_receivers)}});

        if (!image_before_correction.empty()) {
            std::cout << "Image before correction: "
                      << image_before_correction << '\n';
        }
        std::cout << "Image after correction: " << options.image << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << kProgramName << ": " << error.what() << '\n';
        return 1;
    }
}
