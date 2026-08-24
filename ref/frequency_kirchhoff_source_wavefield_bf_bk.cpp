#include "frequency_kirchhoff_imaging_common.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/se_butterfly.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fki = frequency_kirchhoff_imaging;

namespace {

constexpr const char* kProgramName =
    "frequency_kirchhoff_source_wavefield_bf";
constexpr double kPi = 3.14159265358979323846;

struct Options {
    std::string velocity;
    std::string block_file;
    std::string table_prefix;
    std::string output_real;
    std::string output_imag;
    std::string seam_csv;

    // These defaults reproduce the source side of
    // model/run_mar_single_shot_bf_gmres_terminal_logs_threads_unified.sh.
    float requested_frequency = 10.0f;
    float source_x = 5.376f;
    int nt = 6000;
    float dt = 0.0007f;
    float t0 = 0.0f;
    int nfft = 8192;
    float fdom = 20.0f;
    float source_time = 0.075f;
    float source_amplitude = 1.0f;

    float filter_dt = 0.0f;
    float filter_length = 0.025f;
    int filter_lookup_subsamples = 64;

    int threads = 0;

    float bpack_tolerance = 1.0e-4f;
    int bpack_leaf = 64;
    int bpack_verbosity = -1;
    int bpack_progress = 1;
    int bpack_lr_level = 100;
    float bpack_sample_parameter = 1.5f;
    int bpack_forward_n15 = 0;
    int bpack_knn = 0;
    int bpack_pat_comp = 1;
    int bpack_less_adapt = 1;
    float bpack_rdetect_factor = 0.3f;
    int bpack_reuse_tree = 1;

    int seam_enable = 1;
    float seam_amp_rel_floor = 1.0e-3f;
    float seam_scale_cap = 2.0f;
};

Options parse_options()
{
    Options o;
    o.velocity = huygens_cli::required_string("velocity");
    o.block_file = huygens_cli::required_string("block_file");
    o.table_prefix = huygens_cli::required_string("table_prefix");
    o.output_real = huygens_cli::optional_string(
        "output_real", "source_wavefield_real.rsf");
    o.output_imag = huygens_cli::optional_string(
        "output_imag", "source_wavefield_imag.rsf");
    o.seam_csv = huygens_cli::optional_string(
        "seam_csv", "source_wavefield_seam.csv");

    o.requested_frequency =
        huygens_cli::optional_float("frequency", 10.0f);
    o.source_x = huygens_cli::optional_float("source_x", 5.376f);
    o.nt = huygens_cli::optional_int("nt", 6000);
    o.dt = huygens_cli::optional_float("dt", 0.0007f);
    o.t0 = huygens_cli::optional_float("t0", 0.0f);
    o.nfft = huygens_cli::optional_int("nfft", 8192);
    o.fdom = huygens_cli::optional_float("fdom", 20.0f);
    o.source_time = huygens_cli::optional_float("source_time", 0.075f);
    o.source_amplitude =
        huygens_cli::optional_float("source_amplitude", 1.0f);

    o.filter_dt = huygens_cli::optional_float("filter_dt", 0.0f);
    o.filter_length =
        huygens_cli::optional_float("filter_length", 0.025f);
    o.filter_lookup_subsamples =
        huygens_cli::optional_int("filter_lookup_subsamples", 64);

    o.threads = huygens_cli::optional_int("threads", 0);

    o.bpack_tolerance =
        huygens_cli::optional_float("bpack_tol", 1.0e-4f);
    o.bpack_leaf = huygens_cli::optional_int("bpack_leaf", 64);
    o.bpack_verbosity =
        huygens_cli::optional_int("bpack_verbosity", -1);
    o.bpack_progress =
        huygens_cli::optional_int("bpack_progress", 1);
    o.bpack_lr_level =
        huygens_cli::optional_int("bpack_lrlevel", 100);
    o.bpack_sample_parameter =
        huygens_cli::optional_float("bpack_sample_para", 1.5f);
    o.bpack_forward_n15 =
        huygens_cli::optional_int("bpack_forward_n15", 0);
    o.bpack_knn = huygens_cli::optional_int("bpack_knn", 0);
    o.bpack_pat_comp =
        huygens_cli::optional_int("bpack_pat_comp", 1);
    o.bpack_less_adapt =
        huygens_cli::optional_int("bpack_less_adapt", 1);
    o.bpack_rdetect_factor =
        huygens_cli::optional_float("bpack_rdetect_factor", 0.3f);
    o.bpack_reuse_tree =
        huygens_cli::optional_int("bpack_reuse_tree", 1);

    o.seam_enable = huygens_cli::optional_int("seam_enable", 1);
    o.seam_amp_rel_floor =
        huygens_cli::optional_float("seam_amp_rel_floor", 1.0e-3f);
    o.seam_scale_cap =
        huygens_cli::optional_float("seam_scale_cap", 2.0f);

    if (!(o.requested_frequency > 0.0f) ||
        !std::isfinite(o.requested_frequency) ||
        !std::isfinite(o.source_x) ||
        o.nt < 2 || !(o.dt > 0.0f) || !std::isfinite(o.dt) ||
        !std::isfinite(o.t0) || o.nfft < o.nt ||
        !(o.fdom > 0.0f) || !std::isfinite(o.fdom) ||
        !std::isfinite(o.source_time) ||
        !(o.source_amplitude != 0.0f) ||
        !std::isfinite(o.source_amplitude)) {
        throw std::invalid_argument("invalid source/frequency sampling parameters");
    }
    if (o.requested_frequency > 0.5f / o.dt) {
        throw std::invalid_argument("requested frequency exceeds Nyquist");
    }
    if (o.filter_dt < 0.0f || !(o.filter_length > 0.0f) ||
        o.filter_lookup_subsamples < 1) {
        throw std::invalid_argument("invalid one-way filter parameters");
    }
    if (o.threads < 0 ||
        !(o.bpack_tolerance > 0.0f) ||
        !std::isfinite(o.bpack_tolerance) ||
        o.bpack_leaf < 4 ||
        (o.bpack_progress != 0 && o.bpack_progress != 1) ||
        o.bpack_lr_level < 0 ||
        !(o.bpack_sample_parameter > 0.0f) ||
        !std::isfinite(o.bpack_sample_parameter) ||
        o.bpack_forward_n15 < 0 || o.bpack_forward_n15 > 2 ||
        o.bpack_knn < 0 ||
        o.bpack_pat_comp < 1 || o.bpack_pat_comp > 3 ||
        (o.bpack_less_adapt != 0 && o.bpack_less_adapt != 1) ||
        !(o.bpack_rdetect_factor > 0.0f) ||
        !std::isfinite(o.bpack_rdetect_factor) ||
        (o.bpack_reuse_tree != 0 && o.bpack_reuse_tree != 1)) {
        throw std::invalid_argument("invalid ButterflyPACK parameters");
    }
    if ((o.seam_enable != 0 && o.seam_enable != 1) ||
        o.seam_amp_rel_floor < 0.0f ||
        !(o.seam_scale_cap >= 1.0f)) {
        throw std::invalid_argument("invalid Scheme-A seam parameters");
    }
    return o;
}

struct SourceSpectrumSample {
    int bin = 0;
    float frequency = 0.0f;
    fki::Complex value = fki::Complex(0.0f, 0.0f);
};

// This is the single-bin equivalent of fki::ricker_spectrum() used by the
// production imaging program. The requested frequency is snapped to the
// nearest FFT bin because frequency_kirchhoff_imaging_bf_gmres also operates
// only on discrete FFT bins.
SourceSpectrumSample production_ricker_sample(const Options& o)
{
    const float df =
        1.0f / (static_cast<float>(o.nfft) * o.dt);
    int bin = static_cast<int>(
        std::llround(static_cast<double>(o.requested_frequency / df)));
    bin = std::max(1, std::min(bin, o.nfft / 2));
    const float frequency = static_cast<float>(bin) * df;

    float source_time = o.source_time;
    if (!(source_time >= o.t0)) {
        source_time = o.t0 + 1.5f / o.fdom;
    }

    std::vector<float> samples(
        static_cast<std::size_t>(o.nfft), 0.0f);
    const float p2 =
        static_cast<float>(kPi * kPi) * o.fdom * o.fdom;
    for (int it = 0; it < o.nt; ++it) {
        const float t = o.t0 + static_cast<float>(it) * o.dt;
        const float delay = t - source_time;
        const float q = p2 * delay * delay;
        samples[static_cast<std::size_t>(it)] =
            o.source_amplitude *
            (1.0f - 2.0f * q) * std::exp(-q);
    }

    const std::size_t spectrum_size =
        static_cast<std::size_t>(o.nfft / 2 + 1);
    std::vector<float> spectrum_storage(
        2 * spectrum_size, 0.0f);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(
        o.nfft, samples.data(),
        reinterpret_cast<fftwf_complex*>(spectrum_storage.data()),
        FFTW_ESTIMATE);
    if (plan == nullptr) {
        throw std::runtime_error(
            "FFTW failed to create the source Ricker FFT plan");
    }
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    const std::size_t offset =
        2 * static_cast<std::size_t>(bin);
    fki::Complex value(
        spectrum_storage[offset],
        spectrum_storage[offset + 1]);
    const float omega =
        2.0f * static_cast<float>(kPi) * frequency;
    value *= o.dt *
        std::exp(fki::Complex(0.0f, -omega * o.t0));

    SourceSpectrumSample result;
    result.bin = bin;
    result.frequency = frequency;
    result.value = value;
    return result;
}

se::huygens::Block scheme_a_propagation_block(
    const se::huygens::BlockInfo& info,
    std::size_t block_index,
    bool enable_scheme_a)
{
    se::huygens::Block block =
        fki::imaging_propagation_block(info, block_index);
    if (enable_scheme_a &&
        block_index > 0 &&
        info.overlap_rows > 0) {
        const int overlap_target_start =
            block.source_iz + 1;
        if (overlap_target_start >= block.target_start_iz) {
            throw std::runtime_error(
                "Scheme A expected source_iz+1 above official target_start_iz");
        }
        block.target_start_iz = overlap_target_start;
    }
    return block;
}

int exact_depth_index(
    const std::vector<int>& depths, int iz)
{
    const auto it =
        std::lower_bound(depths.begin(), depths.end(), iz);
    if (it == depths.end() || *it != iz) return -1;
    return static_cast<int>(
        std::distance(depths.begin(), it));
}

std::vector<int> intersect_depths(
    const std::vector<int>& a,
    const std::vector<int>& b)
{
    std::vector<int> result;
    result.reserve(std::min(a.size(), b.size()));
    std::set_intersection(
        a.begin(), a.end(), b.begin(), b.end(),
        std::back_inserter(result));
    return result;
}

struct OverlapCache {
    int nx = 0;
    std::vector<int> depths;
    std::vector<fki::Complex> source;

    bool empty() const noexcept
    {
        return depths.empty();
    }

    std::size_t index(
        std::size_t idepth, int ix) const
    {
        return idepth *
                   static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(ix);
    }
};

OverlapCache make_overlap_cache(
    const std::vector<int>& depths, int nx)
{
    OverlapCache cache;
    cache.nx = nx;
    cache.depths = depths;
    cache.source.assign(
        depths.size() * static_cast<std::size_t>(nx),
        fki::Complex(0.0f, 0.0f));
    return cache;
}

std::complex<double> estimate_complex_alignment(
    const OverlapCache& previous,
    const std::vector<fki::Complex>& current,
    const std::vector<int>& current_depths,
    double amplitude_relative_floor,
    double scale_cap,
    std::size_t* accepted_samples)
{
    if (accepted_samples) *accepted_samples = 0;
    if (previous.empty()) return {1.0, 0.0};

    double maximum_amplitude = 0.0;
    for (std::size_t idepth = 0;
         idepth < previous.depths.size();
         ++idepth) {
        const int current_local =
            exact_depth_index(
                current_depths,
                previous.depths[idepth]);
        if (current_local < 0) continue;
        for (int ix = 0; ix < previous.nx; ++ix) {
            const fki::Complex ref =
                previous.source[
                    previous.index(idepth, ix)];
            const fki::Complex cur =
                current[
                    static_cast<std::size_t>(
                        current_local) *
                        previous.nx +
                    ix];
            maximum_amplitude = std::max(
                maximum_amplitude,
                std::max(
                    static_cast<double>(std::abs(ref)),
                    static_cast<double>(std::abs(cur))));
        }
    }

    const double threshold =
        amplitude_relative_floor *
        maximum_amplitude;
    std::complex<double> numerator(0.0, 0.0);
    double denominator = 0.0;
    std::size_t used = 0;

    for (std::size_t idepth = 0;
         idepth < previous.depths.size();
         ++idepth) {
        const int current_local =
            exact_depth_index(
                current_depths,
                previous.depths[idepth]);
        if (current_local < 0) continue;
        for (int ix = 0; ix < previous.nx; ++ix) {
            const fki::Complex ref_f =
                previous.source[
                    previous.index(idepth, ix)];
            const fki::Complex cur_f =
                current[
                    static_cast<std::size_t>(
                        current_local) *
                        previous.nx +
                    ix];
            const std::complex<double> ref(
                ref_f.real(), ref_f.imag());
            const std::complex<double> cur(
                cur_f.real(), cur_f.imag());
            if (std::abs(ref) < threshold ||
                std::abs(cur) < threshold) {
                continue;
            }
            numerator += std::conj(cur) * ref;
            denominator += std::norm(cur);
            ++used;
        }
    }

    if (accepted_samples) *accepted_samples = used;
    if (used == 0 ||
        !(denominator >
          std::numeric_limits<double>::min())) {
        return {1.0, 0.0};
    }

    std::complex<double> alpha =
        numerator / denominator;
    const double magnitude = std::abs(alpha);
    if (!(magnitude > 0.0) ||
        !std::isfinite(magnitude)) {
        return {1.0, 0.0};
    }

    if (scale_cap > 1.0) {
        const double minimum_scale =
            1.0 / scale_cap;
        const double clipped =
            std::min(
                scale_cap,
                std::max(
                    minimum_scale, magnitude));
        alpha *= clipped / magnitude;
    }
    return alpha;
}

std::complex<double> relaxed_complex_scale(
    const std::complex<double>& alpha,
    double amount)
{
    const double magnitude =
        std::max(std::abs(alpha), 1.0e-30);
    return std::polar(
        std::pow(magnitude, amount),
        std::arg(alpha) * amount);
}

std::complex<double> blend_overlap(
    const OverlapCache& previous,
    const std::vector<int>& current_depths,
    std::vector<fki::Complex>& current,
    double amplitude_relative_floor,
    double scale_cap,
    std::size_t* accepted_samples)
{
    const std::complex<double> alpha =
        estimate_complex_alignment(
            previous, current, current_depths,
            amplitude_relative_floor,
            scale_cap, accepted_samples);

    for (std::size_t idepth = 0;
         idepth < previous.depths.size();
         ++idepth) {
        const int current_local =
            exact_depth_index(
                current_depths,
                previous.depths[idepth]);
        if (current_local < 0) continue;

        const double s =
            previous.depths.size() == 1
                ? 0.5
                : static_cast<double>(idepth) /
                      static_cast<double>(
                          previous.depths.size() - 1);
        const double previous_weight =
            0.5 * (1.0 + std::cos(kPi * s));
        const double current_weight =
            1.0 - previous_weight;
        const std::complex<double> local_scale =
            relaxed_complex_scale(
                alpha, previous_weight);

        for (int ix = 0;
             ix < previous.nx;
             ++ix) {
            const std::size_t current_index =
                static_cast<std::size_t>(
                    current_local) *
                    previous.nx +
                ix;
            const fki::Complex ref_f =
                previous.source[
                    previous.index(
                        idepth, ix)];
            const fki::Complex cur_f =
                current[current_index];
            const std::complex<double> ref(
                ref_f.real(), ref_f.imag());
            const std::complex<double> cur(
                cur_f.real(), cur_f.imag());
            const std::complex<double> fused =
                previous_weight * ref +
                current_weight *
                    local_scale * cur;
            current[current_index] =
                fki::Complex(
                    static_cast<float>(
                        fused.real()),
                    static_cast<float>(
                        fused.imag()));
        }
    }
    return alpha;
}

void store_overlap(
    const std::vector<int>& current_depths,
    const std::vector<fki::Complex>& current,
    OverlapCache& next)
{
    if (next.empty()) return;
    for (std::size_t idepth = 0;
         idepth < next.depths.size();
         ++idepth) {
        const int local =
            exact_depth_index(
                current_depths,
                next.depths[idepth]);
        if (local < 0) {
            throw std::runtime_error(
                "next overlap depth is absent from current source field");
        }
        for (int ix = 0; ix < next.nx; ++ix) {
            next.source[
                next.index(idepth, ix)] =
                current[
                    static_cast<std::size_t>(
                        local) *
                        next.nx +
                    ix];
        }
    }
}

void phase_shift_first_block(
    const se::huygens::Model2D& model,
    const se::huygens::Block& block,
    const std::vector<int>& target_iz,
    float frequency,
    const std::vector<fki::Complex>& boundary,
    std::vector<fki::Complex>& output)
{
    // Same first-block implementation as
    // frequency_kirchhoff_imaging_bf_gmres.cpp.
    const int nx = model.nx;
    if (boundary.size() !=
        static_cast<std::size_t>(nx)) {
        throw std::invalid_argument(
            "first-block boundary size is invalid");
    }

    std::vector<fki::Complex> spectrum = boundary;
    fftwf_plan forward = fftwf_plan_dft_1d(
        nx,
        reinterpret_cast<fftwf_complex*>(
            spectrum.data()),
        reinterpret_cast<fftwf_complex*>(
            spectrum.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);
    if (!forward) {
        throw std::runtime_error(
            "first-block phase-shift FFT plan failed");
    }
    fftwf_execute(forward);
    fftwf_destroy_plan(forward);

    double velocity_sum = 0.0;
    std::size_t velocity_count = 0;
    for (const int iz : target_iz) {
        for (int ix = 0; ix < nx; ++ix) {
            velocity_sum +=
                model.velocity[
                    model.index(ix, iz)];
            ++velocity_count;
        }
    }
    const double v0 =
        velocity_sum /
        std::max<std::size_t>(
            velocity_count, 1);
    const double omega =
        2.0 * kPi * frequency;
    const double k0 = omega / v0;

    output.resize(
        target_iz.size() *
        static_cast<std::size_t>(nx));
    std::vector<fki::Complex> line(
        static_cast<std::size_t>(nx));
    fftwf_plan inverse =
        fftwf_plan_dft_1d(
            nx,
            reinterpret_cast<fftwf_complex*>(
                line.data()),
            reinterpret_cast<fftwf_complex*>(
                line.data()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
    if (!inverse) {
        throw std::runtime_error(
            "first-block inverse phase-shift FFT plan failed");
    }

    for (std::size_t iz_local = 0;
         iz_local < target_iz.size();
         ++iz_local) {
        const double distance =
            model.z(target_iz[iz_local]) -
            model.z(block.source_iz);
        for (int ik = 0; ik < nx; ++ik) {
            const int signed_k =
                ik <= nx / 2 ? ik : ik - nx;
            const double kx =
                2.0 * kPi *
                static_cast<double>(signed_k) /
                (static_cast<double>(nx) *
                 model.dx);
            const double kz2 =
                k0 * k0 - kx * kx;
            const fki::Complex propagator =
                kz2 >= 0.0
                    ? std::exp(
                          fki::Complex(
                              0.0f,
                              static_cast<float>(
                                  -std::sqrt(kz2) *
                                  distance)))
                    : fki::Complex(
                          static_cast<float>(
                              std::exp(
                                  -std::sqrt(-kz2) *
                                  std::abs(distance))),
                          0.0f);
            line[static_cast<std::size_t>(ik)] =
                spectrum[
                    static_cast<std::size_t>(ik)] *
                propagator;
        }
        fftwf_execute(inverse);
        for (int ix = 0; ix < nx; ++ix) {
            output[
                iz_local *
                    static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix)] =
                line[
                    static_cast<std::size_t>(ix)] /
                static_cast<float>(nx);
        }
    }
    fftwf_destroy_plan(inverse);
}

std::vector<double> full_block_row_coordinates(
    const se::huygens::Model2D& model,
    const se::huygens::LayerGeometry& geometry)
{
    std::vector<double> coordinates(
        2 * geometry.targets.size());
    for (std::size_t row = 0;
         row < geometry.targets.size();
         ++row) {
        const auto& point =
            geometry.targets[row];
        coordinates[2 * row] =
            static_cast<double>(model.ox) +
            static_cast<double>(point.ix) *
                model.dx;
        coordinates[2 * row + 1] =
            static_cast<double>(model.oz) +
            static_cast<double>(point.iz) *
                model.dz;
    }
    return coordinates;
}

std::vector<double> datum_column_coordinates(
    const se::huygens::Model2D& model,
    const se::huygens::Block& block,
    const se::huygens::LayerGeometry& geometry)
{
    std::vector<double> coordinates(
        2 * geometry.source_ix.size());
    const double datum_z =
        static_cast<double>(model.oz) +
        static_cast<double>(
            block.source_iz) *
            model.dz;
    for (std::size_t source = 0;
         source < geometry.source_ix.size();
         ++source) {
        coordinates[2 * source] =
            static_cast<double>(model.ox) +
            static_cast<double>(
                geometry.source_ix[source]) *
                model.dx;
        coordinates[2 * source + 1] =
            datum_z;
    }
    return coordinates;
}

void full_block_xmajor_to_depthmajor(
    const se::huygens::Model2D& model,
    const se::huygens::LayerGeometry& geometry,
    const std::vector<fki::Complex>& rows,
    std::vector<fki::Complex>& depth_major)
{
    const int depth_count =
        static_cast<int>(
            geometry.target_iz.size());
    const std::size_t expected =
        static_cast<std::size_t>(model.nx) *
        static_cast<std::size_t>(depth_count);
    if (rows.size() != expected ||
        geometry.targets.size() != expected) {
        throw std::runtime_error(
            "full-block ButterflyPACK output has invalid dimensions");
    }

    depth_major.assign(
        expected, fki::Complex(0.0f, 0.0f));

    // make_layer_geometry stores target rows x-major:
    // row = ix * depth_count + iz_local.
    for (int ix = 0; ix < model.nx; ++ix) {
        for (int iz_local = 0;
             iz_local < depth_count;
             ++iz_local) {
            const std::size_t row =
                static_cast<std::size_t>(ix) *
                    depth_count +
                static_cast<std::size_t>(
                    iz_local);
            const auto& point =
                geometry.targets[row];
            if (point.ix != ix ||
                point.iz !=
                    geometry.target_iz[
                        static_cast<std::size_t>(
                            iz_local)]) {
                throw std::runtime_error(
                    "unexpected target ordering in full-block ButterflyPACK geometry");
            }
            const std::size_t destination =
                static_cast<std::size_t>(
                    iz_local) *
                    model.nx +
                ix;
            depth_major[destination] =
                rows[row];
        }
    }
}

void copy_block_to_global(
    const se::huygens::Model2D& model,
    const std::vector<int>& target_depths,
    const std::vector<fki::Complex>& block_values,
    std::vector<fki::Complex>& global,
    std::vector<int>& depth_coverage)
{
    const std::size_t expected =
        target_depths.size() *
        static_cast<std::size_t>(model.nx);
    if (block_values.size() != expected) {
        throw std::invalid_argument(
            "block source wavefield size is invalid");
    }
    if (depth_coverage.size() !=
        static_cast<std::size_t>(model.nz)) {
        throw std::invalid_argument(
            "depth coverage array does not match model nz");
    }
    for (std::size_t iz_local = 0;
         iz_local < target_depths.size();
         ++iz_local) {
        const int iz =
            target_depths[iz_local];
        if (iz < 0 || iz >= model.nz) {
            throw std::out_of_range(
                "block target depth lies outside the velocity model");
        }
        ++depth_coverage[
            static_cast<std::size_t>(iz)];
        for (int ix = 0;
             ix < model.nx;
             ++ix) {
            global[model.index(ix, iz)] =
                block_values[
                    iz_local *
                        static_cast<std::size_t>(
                            model.nx) +
                    ix];
        }
    }
}

std::vector<fki::Complex> extract_boundary(
    const se::huygens::Model2D& model,
    const std::vector<int>& target_depths,
    int next_depth,
    const std::vector<fki::Complex>& block_values)
{
    const int local =
        fki::find_local_depth(
            target_depths, next_depth);
    if (local < 0) {
        throw std::runtime_error(
            "next propagation datum is absent from current source block");
    }

    std::vector<fki::Complex> boundary(
        static_cast<std::size_t>(
            model.nx));
    for (int ix = 0;
         ix < model.nx;
         ++ix) {
        boundary[
            static_cast<std::size_t>(ix)] =
            block_values[
                static_cast<std::size_t>(
                    local) *
                    model.nx +
                ix];
    }
    return boundary;
}

void write_raw_component(
    const std::string& filename,
    const std::vector<fki::Complex>& wavefield,
    bool imaginary)
{
    std::ofstream stream(
        filename, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open output binary: " +
            filename);
    }
    std::vector<float> values(
        wavefield.size());
    for (std::size_t i = 0;
         i < wavefield.size();
         ++i) {
        values[i] =
            imaginary
                ? wavefield[i].imag()
                : wavefield[i].real();
    }
    stream.write(
        reinterpret_cast<const char*>(
            values.data()),
        static_cast<std::streamsize>(
            values.size() *
            sizeof(float)));
    if (!stream) {
        throw std::runtime_error(
            "failed while writing output binary: " +
            filename);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(
            argc, argv)) {
        return 0;
    }

    try {
        huygens_cli::initialize(argc, argv);
        const Options options =
            parse_options();
        huygens_cli::set_openmp_threads(
            options.threads);

#ifndef KIRCH_HAS_BUTTERFLYPACK_FLOAT
        throw std::runtime_error(
            "frequency_kirchhoff_source_wavefield_bf requires "
            "KIRCH_ENABLE_BUTTERFLYPACK=ON and "
            "KIRCH_BPACK_ENABLE_FLOAT=ON");
#else
        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(
                options.velocity);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(
                options.block_file);
        se::huygens::validate_block_info(
            info, &model);

        if (options.seam_enable != 0 &&
            info.overlap_rows <= 0) {
            throw std::invalid_argument(
                "seam_enable=1 requires overlap_rows > 0");
        }

        if (options.source_x <
                model.x(0) - 1.0e-5f ||
            options.source_x >
                model.x(model.nx - 1) +
                    1.0e-5f) {
            throw std::invalid_argument(
                "source_x lies outside velocity model");
        }

        const SourceSpectrumSample source_sample =
            production_ricker_sample(options);
        const float frequency =
            source_sample.frequency;
        const float effective_filter_dt =
            options.filter_dt > 0.0f
                ? options.filter_dt
                : options.dt;

        const std::vector<float> model_weights =
            fki::full_boundary_weights(model);

        std::vector<fki::Complex> boundary(
            static_cast<std::size_t>(
                model.nx),
            fki::Complex(0.0f, 0.0f));
        if (!fki::deposit_integral_sample(
                boundary, 0,
                model, model_weights,
                options.source_x,
                source_sample.value)) {
            throw std::runtime_error(
                "failed to place source on surface datum");
        }

        std::vector<fki::Complex> wavefield(
            static_cast<std::size_t>(
                model.nx) *
                model.nz,
            fki::Complex(0.0f, 0.0f));
        for (int ix = 0;
             ix < model.nx;
             ++ix) {
            wavefield[
                model.index(ix, 0)] =
                boundary[
                    static_cast<std::size_t>(
                        ix)];
        }

        // Track every global depth row that is actually assembled. Row zero
        // is the deposited surface source; all deeper rows must be supplied
        // by one or more propagation blocks.
        std::vector<int> depth_coverage(
            static_cast<std::size_t>(model.nz), 0);
        depth_coverage[0] = 1;

        std::vector<std::vector<int>>
            block_target_depths(
                info.blocks.size());
        for (std::size_t ib = 0;
             ib < info.blocks.size();
             ++ib) {
            const se::huygens::Block block =
                scheme_a_propagation_block(
                    info, ib,
                    options.seam_enable != 0);
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block,
                    1, 1, 1);
            block_target_depths[ib] =
                geometry.target_iz;
        }

        if (options.seam_enable != 0) {
            for (std::size_t ib = 0;
                 ib + 1 <
                     block_target_depths.size();
                 ++ib) {
                const auto common =
                    intersect_depths(
                        block_target_depths[ib],
                        block_target_depths[
                            ib + 1]);
                if (common.size() !=
                    static_cast<std::size_t>(
                        info.overlap_rows)) {
                    throw std::runtime_error(
                        "Scheme-A overlap does not match block geometry");
                }
            }
        }

        std::ofstream seam_stream(
            options.seam_csv);
        if (!seam_stream) {
            throw std::runtime_error(
                "cannot open seam_csv: " +
                options.seam_csv);
        }
        seam_stream
            << std::setprecision(17)
            << "block,frequency,overlap_rows,samples,"
               "alpha_real,alpha_imag,alpha_abs,alpha_phase_rad\n";

        std::cout
            << "Single-frequency SOURCE wavefield "
               "extracted from production imaging path\n"
            << "requested frequency : "
            << options.requested_frequency
            << " Hz\n"
            << "FFT bin             : "
            << source_sample.bin << '\n'
            << "actual frequency    : "
            << frequency << " Hz\n"
            << "source x            : "
            << options.source_x << '\n'
            << "source spectrum     : "
            << source_sample.value << '\n'
            << "source sampling     : nt="
            << options.nt
            << ", dt=" << options.dt
            << ", t0=" << options.t0
            << ", nfft=" << options.nfft
            << ", fdom=" << options.fdom
            << ", source_time="
            << options.source_time << '\n'
            << "model               : nx="
            << model.nx
            << ", nz=" << model.nz
            << ", dx=" << model.dx
            << ", dz=" << model.dz << '\n'
            << "blocks              : "
            << info.blocks.size()
            << ", overlap_rows="
            << info.overlap_rows << '\n'
            << "first block         : production phase-shift launch\n"
            << "later blocks        : ONE full-block ButterflyPACK "
               "operator per block\n"
            << "receiver field      : not allocated\n"
            << "GMRES               : not allocated\n"
            << "image/illumination  : not allocated\n";

        OverlapCache previous_overlap;
        double total_filter_seconds = 0.0;
        double total_build_seconds = 0.0;
        double total_apply_seconds = 0.0;

        for (std::size_t iblock = 0;
             iblock < info.blocks.size();
             ++iblock) {
            const se::huygens::Block block =
                scheme_a_propagation_block(
                    info, iblock,
                    options.seam_enable != 0);
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block, 1, 1, 1);
            const se::huygens::OneWayLayerTables tables =
                se::huygens::read_one_way_layer_tables(
                    options.table_prefix,
                    block.id);
            try {
                se::huygens::validate_one_way_layer_tables(
                    tables, geometry, block);
            } catch (const std::exception& error) {
                if (options.seam_enable != 0 &&
                    iblock > 0 &&
                    info.overlap_rows > 0) {
                    throw std::runtime_error(
                        "Scheme A requires traveltime tables "
                        "with include_propagation_overlap=1. "
                        "Original validation error: " +
                        std::string(error.what()));
                }
                throw;
            }

            if (static_cast<int>(
                    geometry.source_ix.size()) !=
                    model.nx ||
                static_cast<int>(
                    geometry.target_ix.size()) !=
                    model.nx) {
                throw std::runtime_error(
                    "source-only ButterflyPACK path requires "
                    "all lateral grid points");
            }

            const int target_depth_count =
                static_cast<int>(
                    geometry.target_iz.size());
            const std::size_t row_count =
                geometry.targets.size();
            if (row_count !=
                static_cast<std::size_t>(
                    model.nx) *
                    target_depth_count) {
                throw std::runtime_error(
                    "target geometry is not one complete 2-D block");
            }

            OverlapCache next_overlap;
            if (options.seam_enable != 0 &&
                iblock + 1 <
                    info.blocks.size()) {
                next_overlap =
                    make_overlap_cache(
                        intersect_depths(
                            block_target_depths[
                                iblock],
                            block_target_depths[
                                iblock + 1]),
                        model.nx);
            }

            std::vector<fki::Complex>
                block_output;

            std::cout
                << "\nBlock " << block.id
                << ": source_iz="
                << block.source_iz
                << ", target=["
                << geometry.target_iz.front()
                << ','
                << geometry.target_iz.back()
                << ']'
                << ", matrix="
                << row_count
                << 'x'
                << geometry.source_ix.size()
                << '\n';

            if (iblock == 0) {
                const auto started =
                    fki::Clock::now();
                phase_shift_first_block(
                    model, block,
                    geometry.target_iz,
                    frequency,
                    boundary,
                    block_output);
                const double elapsed =
                    std::chrono::duration<double>(
                        fki::Clock::now() -
                        started).count();
                total_apply_seconds += elapsed;
                std::cout
                    << "  phase-shift apply="
                    << elapsed << " s\n";
            } else {
                const auto filter_started =
                    fki::Clock::now();
                const float maximum_tau =
                    *std::max_element(
                        tables.traveltime.values.begin(),
                        tables.traveltime.values.end());
                const se::huygens::FrequencyKirchhoffFilter filter(
                    frequency,
                    effective_filter_dt,
                    options.filter_length,
                    maximum_tau,
                    options.filter_lookup_subsamples);
                const double filter_seconds =
                    std::chrono::duration<double>(
                        fki::Clock::now() -
                        filter_started).count();
                total_filter_seconds +=
                    filter_seconds;

                se::huygens::OneWayKernelData kernel;
                kernel.rows =
                    static_cast<int>(
                        row_count);
                kernel.sources =
                    static_cast<int>(
                        geometry.source_ix.size());
                kernel.source_z =
                    model.z(block.source_iz);
                kernel.quadrature_weights =
                    &model_weights;
                kernel.tables = &tables;
                kernel.filter = &filter;

                const std::vector<double>
                    row_coordinates =
                        full_block_row_coordinates(
                            model, geometry);
                const std::vector<double>
                    column_coordinates =
                        datum_column_coordinates(
                            model, block,
                            geometry);

                se::butterfly::Options
                    bpack_options;
                bpack_options.tolerance =
                    options.bpack_tolerance;
                bpack_options.leaf_size =
                    options.bpack_leaf;
                bpack_options.verbosity =
                    options.bpack_verbosity;
                bpack_options.coordinate_dimension =
                    2;
                bpack_options.lr_level =
                    options.bpack_lr_level;
                bpack_options.sample_parameter =
                    options.bpack_sample_parameter;
                bpack_options.forward_n15_flag =
                    options.bpack_forward_n15;
                bpack_options.nearest_neighbors =
                    options.bpack_knn;
                bpack_options.compression_pattern =
                    options.bpack_pat_comp;
                bpack_options.less_adapt =
                    options.bpack_less_adapt;
                bpack_options.rank_detection_factor =
                    options.bpack_rdetect_factor;
                bpack_options.reuse_geometry =
                    options.bpack_reuse_tree;

                if (options.bpack_progress != 0) {
                    std::cout
                        << "  [BPACK build START]"
                        << " matrix="
                        << row_count
                        << 'x'
                        << geometry.source_ix.size()
                        << '\n';
                }

                const auto build_started =
                    fki::Clock::now();
                auto butterfly =
                    std::make_unique<
                        se::butterfly::Matrix<float>>(
                        static_cast<int>(
                            row_count),
                        static_cast<int>(
                            geometry.source_ix.size()),
                        row_coordinates,
                        column_coordinates,
                        [&kernel](
                            int row,
                            int source) {
                            return kernel.entry(
                                row, source);
                        },
                        bpack_options);
                const double build_seconds =
                    std::chrono::duration<double>(
                        fki::Clock::now() -
                        build_started).count();
                total_build_seconds +=
                    build_seconds;

                const auto apply_started =
                    fki::Clock::now();
                const std::vector<fki::Complex>
                    xmajor_output =
                        butterfly->apply(
                            boundary);
                const double apply_seconds =
                    std::chrono::duration<double>(
                        fki::Clock::now() -
                        apply_started).count();
                total_apply_seconds +=
                    apply_seconds;

                full_block_xmajor_to_depthmajor(
                    model, geometry,
                    xmajor_output,
                    block_output);

                const auto stats =
                    butterfly->statistics();

                if (options.bpack_progress != 0) {
                    std::cout
                        << "  [BPACK DONE]"
                        << " build="
                        << build_seconds
                        << " s"
                        << ", apply="
                        << apply_seconds
                        << " s"
                        << ", compressed="
                        << stats.compressed_megabytes
                        << " MiB"
                        << ", peak="
                        << stats.peak_megabytes
                        << " MiB"
                        << ", rank_max="
                        << stats.maximum_rank
                        << '\n';
                }

                std::cout
                    << "  filter="
                    << filter_seconds
                    << " s\n";
            }

            if (options.seam_enable != 0 &&
                iblock > 0 &&
                !previous_overlap.empty()) {
                std::size_t samples = 0;
                const std::complex<double> alpha =
                    blend_overlap(
                        previous_overlap,
                        geometry.target_iz,
                        block_output,
                        options.seam_amp_rel_floor,
                        options.seam_scale_cap,
                        &samples);
                seam_stream
                    << block.id << ','
                    << frequency << ','
                    << previous_overlap.depths.size()
                    << ','
                    << samples << ','
                    << alpha.real() << ','
                    << alpha.imag() << ','
                    << std::abs(alpha) << ','
                    << std::arg(alpha)
                    << '\n';
                std::cout
                    << "  seam alpha="
                    << alpha
                    << ", samples="
                    << samples << '\n';
            }

            if (options.seam_enable != 0 &&
                !next_overlap.empty()) {
                store_overlap(
                    geometry.target_iz,
                    block_output,
                    next_overlap);
            }

            copy_block_to_global(
                model,
                geometry.target_iz,
                block_output,
                wavefield,
                depth_coverage);

            if (iblock + 1 <
                info.blocks.size()) {
                boundary =
                    extract_boundary(
                        model,
                        geometry.target_iz,
                        info.blocks[
                            iblock + 1].source_iz,
                        block_output);
            }

            previous_overlap =
                options.seam_enable != 0
                    ? std::move(next_overlap)
                    : OverlapCache{};
        }

        int minimum_depth_coverage =
            std::numeric_limits<int>::max();
        int maximum_depth_coverage = 0;
        for (int iz = 0; iz < model.nz; ++iz) {
            const int count =
                depth_coverage[
                    static_cast<std::size_t>(iz)];
            if (count < 1) {
                throw std::runtime_error(
                    "global source-wavefield assembly left depth iz=" +
                    std::to_string(iz) + " unwritten");
            }
            minimum_depth_coverage =
                std::min(minimum_depth_coverage, count);
            maximum_depth_coverage =
                std::max(maximum_depth_coverage, count);
        }

        // The repository RSF writer converts the internal x-fast field
        // (ix + iz*nx) back to standard RSF order with depth as axis 1:
        // n1=model.nz and n2=model.nx.
        se::huygens::write_wavefield_component_rsf(
            options.output_real,
            model, wavefield, frequency,
            kProgramName, false);
        se::huygens::write_wavefield_component_rsf(
            options.output_imag,
            model, wavefield, frequency,
            kProgramName, true);

        std::cout
            << "\nSource wavefield completed.\n"
            << "Actual frequency    : "
            << frequency << " Hz\n"
            << "Global grid         : n1=nz="
            << model.nz << ", n2=nx="
            << model.nx << '\n'
            << "Depth coverage      : min="
            << minimum_depth_coverage
            << ", max="
            << maximum_depth_coverage
            << " (max>1 is expected in Scheme-A overlaps)\n"
            << "Filter setup total  : "
            << total_filter_seconds
            << " s\n"
            << "BPACK build total   : "
            << total_build_seconds
            << " s\n"
            << "Propagation total   : "
            << total_apply_seconds
            << " s\n"
            << "Real RSF            : "
            << options.output_real
            << '\n'
            << "Imag RSF            : "
            << options.output_imag
            << '\n'
            << "RSF axes            : n1=depth (nz), n2=x (nx)\n"
            << "RSF payload         : native_float, .rsf@\n";
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr
            << kProgramName
            << ": "
            << error.what()
            << '\n';
        return 1;
    }
}
