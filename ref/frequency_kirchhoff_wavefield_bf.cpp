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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fki = frequency_kirchhoff_imaging;

namespace {

constexpr const char* kProgramName = "frequency_kirchhoff_wavefield_bf";
constexpr double kPi = 3.14159265358979323846;

struct Options {
    std::string velocity;
    std::string block_file;
    std::string table_prefix;
    std::string output_real;
    std::string output_imag;
    std::string seam_csv;

    float frequency = 10.0f;
    int source_ix = -1;
    int source_iz = 0;
    float source_amplitude = 1.0f;
    float source_phase_deg = 0.0f;

    float filter_dt = 0.001f;
    float filter_length = 0.025f;
    int filter_lookup_subsamples = 64;

    int source_stride = 1;
    int target_x_stride = 1;
    int target_z_stride = 1;
    int threads = 0;

    float bpack_tolerance = 1.0e-4f;
    int bpack_leaf = 64;
    int bpack_verbosity = -1;
    int bpack_depth_chunk = 20;
    int bpack_progress = 1;
    int bpack_progress_every = 1;
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
        "output_real", "frequency_kirchhoff_wavefield_bf_real.bin");
    o.output_imag = huygens_cli::optional_string(
        "output_imag", "frequency_kirchhoff_wavefield_bf_imag.bin");
    o.seam_csv = huygens_cli::optional_string(
        "seam_csv", "frequency_kirchhoff_wavefield_bf_seam.csv");

    o.frequency = huygens_cli::optional_float("frequency", 10.0f);
    o.source_ix = huygens_cli::optional_int("source_ix", -1);
    o.source_iz = huygens_cli::optional_int("source_iz", 0);
    o.source_amplitude = huygens_cli::optional_float("source_amplitude", 1.0f);
    o.source_phase_deg = huygens_cli::optional_float("source_phase_deg", 0.0f);

    o.filter_dt = huygens_cli::optional_float("filter_dt", 0.001f);
    o.filter_length = huygens_cli::optional_float("filter_length", 0.025f);
    o.filter_lookup_subsamples =
        huygens_cli::optional_int("filter_lookup_subsamples", 64);

    o.source_stride = huygens_cli::optional_int("source_stride", 1);
    o.target_x_stride = huygens_cli::optional_int("target_x_stride", 1);
    o.target_z_stride = huygens_cli::optional_int("target_z_stride", 1);
    o.threads = huygens_cli::optional_int("threads", 0);

    o.bpack_tolerance = huygens_cli::optional_float("bpack_tol", 1.0e-4f);
    o.bpack_leaf = huygens_cli::optional_int("bpack_leaf", 64);
    o.bpack_verbosity = huygens_cli::optional_int("bpack_verbosity", -1);
    o.bpack_depth_chunk = huygens_cli::optional_int("bpack_depth_chunk", 20);
    o.bpack_progress = huygens_cli::optional_int("bpack_progress", 1);
    o.bpack_progress_every =
        huygens_cli::optional_int("bpack_progress_every", 1);
    o.bpack_lr_level = huygens_cli::optional_int("bpack_lrlevel", 100);
    o.bpack_sample_parameter =
        huygens_cli::optional_float("bpack_sample_para", 1.5f);
    o.bpack_forward_n15 =
        huygens_cli::optional_int("bpack_forward_n15", 0);
    o.bpack_knn = huygens_cli::optional_int("bpack_knn", 0);
    o.bpack_pat_comp = huygens_cli::optional_int("bpack_pat_comp", 1);
    o.bpack_less_adapt = huygens_cli::optional_int("bpack_less_adapt", 1);
    o.bpack_rdetect_factor =
        huygens_cli::optional_float("bpack_rdetect_factor", 0.3f);
    o.bpack_reuse_tree = huygens_cli::optional_int("bpack_reuse_tree", 1);

    o.seam_enable = huygens_cli::optional_int("seam_enable", 1);
    o.seam_amp_rel_floor =
        huygens_cli::optional_float("seam_amp_rel_floor", 1.0e-3f);
    o.seam_scale_cap =
        huygens_cli::optional_float("seam_scale_cap", 2.0f);

    if (!(o.frequency > 0.0f) || !std::isfinite(o.frequency) ||
        !(o.source_amplitude != 0.0f) || !std::isfinite(o.source_amplitude) ||
        !std::isfinite(o.source_phase_deg) ||
        !(o.filter_dt > 0.0f) || !(o.filter_length > 0.0f) ||
        o.filter_lookup_subsamples < 1) {
        throw std::invalid_argument("invalid frequency/source/filter parameters");
    }
    if (o.source_iz != 0) {
        throw std::invalid_argument(
            "source_iz must be 0; this program follows the imaging program's "
            "surface-datum first-block launch");
    }
    if (o.source_stride != 1 || o.target_x_stride != 1 ||
        o.target_z_stride != 1) {
        throw std::invalid_argument(
            "frequency_kirchhoff_wavefield_bf requires all spatial strides to equal one");
    }
    if (o.threads < 0 ||
        !(o.bpack_tolerance > 0.0f) || !std::isfinite(o.bpack_tolerance) ||
        o.bpack_leaf < 4 || o.bpack_depth_chunk < 1 ||
        (o.bpack_progress != 0 && o.bpack_progress != 1) ||
        o.bpack_progress_every < 1 || o.bpack_lr_level < 0 ||
        !(o.bpack_sample_parameter > 0.0f) ||
        !std::isfinite(o.bpack_sample_parameter) ||
        o.bpack_forward_n15 < 0 || o.bpack_forward_n15 > 2 ||
        o.bpack_knn < 0 || o.bpack_pat_comp < 1 || o.bpack_pat_comp > 3 ||
        (o.bpack_less_adapt != 0 && o.bpack_less_adapt != 1) ||
        !(o.bpack_rdetect_factor > 0.0f) ||
        !std::isfinite(o.bpack_rdetect_factor) ||
        (o.bpack_reuse_tree != 0 && o.bpack_reuse_tree != 1)) {
        throw std::invalid_argument("invalid rectangular ButterflyPACK parameters");
    }
    if ((o.seam_enable != 0 && o.seam_enable != 1) ||
        o.seam_amp_rel_floor < 0.0f ||
        !(o.seam_scale_cap >= 1.0f)) {
        throw std::invalid_argument("invalid overlap/seam parameters");
    }
    return o;
}

se::huygens::Block propagation_block(const se::huygens::BlockInfo& info,
                                     std::size_t block_index,
                                     bool seam_enable)
{
    se::huygens::Block block =
        fki::imaging_propagation_block(info, block_index);
    if (seam_enable && block_index > 0 && info.overlap_rows > 0) {
        const int overlap_target_start = block.source_iz + 1;
        if (overlap_target_start >= block.target_start_iz) {
            throw std::runtime_error(
                "overlap geometry is inconsistent: source_iz+1 must lie above "
                "the official target_start_iz");
        }
        block.target_start_iz = overlap_target_start;
    }
    return block;
}

int exact_depth_index(const std::vector<int>& depths, int iz)
{
    const auto it = std::lower_bound(depths.begin(), depths.end(), iz);
    if (it == depths.end() || *it != iz) return -1;
    return static_cast<int>(std::distance(depths.begin(), it));
}

std::vector<int> intersect_depths(const std::vector<int>& a,
                                  const std::vector<int>& b)
{
    std::vector<int> result;
    result.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(result));
    return result;
}

struct OverlapCache {
    int nx = 0;
    std::vector<int> depths;
    std::vector<fki::Complex> values;

    bool empty() const noexcept { return depths.empty(); }

    std::size_t index(std::size_t idepth, int ix) const
    {
        return idepth * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(ix);
    }
};

OverlapCache make_overlap_cache(const std::vector<int>& depths, int nx)
{
    OverlapCache cache;
    cache.nx = nx;
    cache.depths = depths;
    cache.values.assign(
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
    for (std::size_t idepth = 0; idepth < previous.depths.size(); ++idepth) {
        const int current_local =
            exact_depth_index(current_depths, previous.depths[idepth]);
        if (current_local < 0) continue;
        for (int ix = 0; ix < previous.nx; ++ix) {
            const fki::Complex ref_f =
                previous.values[previous.index(idepth, ix)];
            const fki::Complex cur_f =
                current[static_cast<std::size_t>(current_local) *
                            static_cast<std::size_t>(previous.nx) +
                        static_cast<std::size_t>(ix)];
            maximum_amplitude = std::max(
                maximum_amplitude,
                std::max(static_cast<double>(std::abs(ref_f)),
                         static_cast<double>(std::abs(cur_f))));
        }
    }

    const double threshold = amplitude_relative_floor * maximum_amplitude;
    std::complex<double> numerator(0.0, 0.0);
    double denominator = 0.0;
    std::size_t used = 0;

    for (std::size_t idepth = 0; idepth < previous.depths.size(); ++idepth) {
        const int current_local =
            exact_depth_index(current_depths, previous.depths[idepth]);
        if (current_local < 0) continue;
        for (int ix = 0; ix < previous.nx; ++ix) {
            const fki::Complex ref_f =
                previous.values[previous.index(idepth, ix)];
            const fki::Complex cur_f =
                current[static_cast<std::size_t>(current_local) *
                            static_cast<std::size_t>(previous.nx) +
                        static_cast<std::size_t>(ix)];
            const std::complex<double> ref(ref_f.real(), ref_f.imag());
            const std::complex<double> cur(cur_f.real(), cur_f.imag());
            if (std::abs(ref) < threshold || std::abs(cur) < threshold) continue;
            numerator += std::conj(cur) * ref;
            denominator += std::norm(cur);
            ++used;
        }
    }

    if (accepted_samples) *accepted_samples = used;
    if (used == 0 || !(denominator > std::numeric_limits<double>::min())) {
        return {1.0, 0.0};
    }

    std::complex<double> alpha = numerator / denominator;
    const double magnitude = std::abs(alpha);
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
        return {1.0, 0.0};
    }

    if (scale_cap > 1.0) {
        const double minimum_scale = 1.0 / scale_cap;
        const double clipped =
            std::min(scale_cap, std::max(minimum_scale, magnitude));
        alpha *= clipped / magnitude;
    }
    return alpha;
}

std::complex<double> relaxed_complex_scale(
    const std::complex<double>& alpha, double amount)
{
    const double magnitude = std::max(std::abs(alpha), 1.0e-30);
    return std::polar(std::pow(magnitude, amount), std::arg(alpha) * amount);
}

std::complex<double> blend_overlap(
    const OverlapCache& previous,
    const std::vector<int>& current_depths,
    std::vector<fki::Complex>& current,
    double amplitude_relative_floor,
    double scale_cap,
    std::size_t* accepted_samples)
{
    const std::complex<double> alpha = estimate_complex_alignment(
        previous, current, current_depths,
        amplitude_relative_floor, scale_cap, accepted_samples);

    for (std::size_t idepth = 0; idepth < previous.depths.size(); ++idepth) {
        const int current_local =
            exact_depth_index(current_depths, previous.depths[idepth]);
        if (current_local < 0) continue;

        const double s = previous.depths.size() == 1
                             ? 0.5
                             : static_cast<double>(idepth) /
                                   static_cast<double>(previous.depths.size() - 1);
        const double previous_weight =
            0.5 * (1.0 + std::cos(kPi * s));
        const double current_weight = 1.0 - previous_weight;
        const std::complex<double> local_scale =
            relaxed_complex_scale(alpha, previous_weight);

        for (int ix = 0; ix < previous.nx; ++ix) {
            const std::size_t current_index =
                static_cast<std::size_t>(current_local) *
                    static_cast<std::size_t>(previous.nx) +
                static_cast<std::size_t>(ix);
            const fki::Complex ref_f =
                previous.values[previous.index(idepth, ix)];
            const fki::Complex cur_f = current[current_index];
            const std::complex<double> ref(ref_f.real(), ref_f.imag());
            const std::complex<double> cur(cur_f.real(), cur_f.imag());
            const std::complex<double> fused =
                previous_weight * ref +
                current_weight * local_scale * cur;
            current[current_index] = fki::Complex(
                static_cast<float>(fused.real()),
                static_cast<float>(fused.imag()));
        }
    }
    return alpha;
}

void store_overlap(const std::vector<int>& current_depths,
                   const std::vector<fki::Complex>& current,
                   OverlapCache& next)
{
    if (next.empty()) return;
    for (std::size_t idepth = 0; idepth < next.depths.size(); ++idepth) {
        const int local = exact_depth_index(current_depths, next.depths[idepth]);
        if (local < 0) {
            throw std::runtime_error(
                "next overlap depth is absent from current block output");
        }
        for (int ix = 0; ix < next.nx; ++ix) {
            next.values[next.index(idepth, ix)] =
                current[static_cast<std::size_t>(local) *
                            static_cast<std::size_t>(next.nx) +
                        static_cast<std::size_t>(ix)];
        }
    }
}

struct BpackDepthChunk {
    int depth_begin = 0;
    int depth_count = 0;
    std::vector<int> full_rows;
    std::vector<std::size_t> depth_major_indices;
    std::vector<double> row_coordinates;
};

std::vector<BpackDepthChunk> make_bpack_depth_chunks(
    const se::huygens::Model2D& model,
    const se::huygens::LayerGeometry& geometry,
    int requested_depth_chunk)
{
    if (requested_depth_chunk < 1) {
        throw std::invalid_argument("bpack_depth_chunk must be positive");
    }
    const int target_depth_count =
        static_cast<int>(geometry.target_iz.size());
    if (target_depth_count < 1 ||
        geometry.targets.size() !=
            static_cast<std::size_t>(target_depth_count) *
                static_cast<std::size_t>(model.nx)) {
        throw std::runtime_error(
            "ButterflyPACK chunking requires a complete x-by-depth target grid");
    }

    const int chunk_depths =
        std::min(requested_depth_chunk, target_depth_count);
    std::vector<BpackDepthChunk> chunks;
    chunks.reserve(static_cast<std::size_t>(
        (target_depth_count + chunk_depths - 1) / chunk_depths));

    for (int depth_begin = 0; depth_begin < target_depth_count;
         depth_begin += chunk_depths) {
        BpackDepthChunk chunk;
        chunk.depth_begin = depth_begin;
        chunk.depth_count =
            std::min(chunk_depths, target_depth_count - depth_begin);
        const std::size_t rows =
            static_cast<std::size_t>(chunk.depth_count) *
            static_cast<std::size_t>(model.nx);
        chunk.full_rows.reserve(rows);
        chunk.depth_major_indices.reserve(rows);
        chunk.row_coordinates.reserve(2 * rows);

        for (int ix = 0; ix < model.nx; ++ix) {
            for (int iz_chunk = 0; iz_chunk < chunk.depth_count; ++iz_chunk) {
                const int iz_local = depth_begin + iz_chunk;
                const std::size_t full_row =
                    static_cast<std::size_t>(ix) *
                        static_cast<std::size_t>(target_depth_count) +
                    static_cast<std::size_t>(iz_local);
                if (full_row >= geometry.targets.size()) {
                    throw std::runtime_error(
                        "ButterflyPACK chunk row exceeds target geometry");
                }
                const se::huygens::GridPoint& point =
                    geometry.targets[full_row];
                if (point.ix != ix ||
                    point.iz !=
                        geometry.target_iz[static_cast<std::size_t>(iz_local)]) {
                    throw std::runtime_error(
                        "ButterflyPACK target ordering is incompatible with chunking");
                }
                chunk.full_rows.push_back(static_cast<int>(full_row));
                chunk.depth_major_indices.push_back(
                    static_cast<std::size_t>(iz_local) *
                        static_cast<std::size_t>(model.nx) +
                    static_cast<std::size_t>(ix));
                chunk.row_coordinates.push_back(
                    static_cast<double>(model.ox) +
                    static_cast<double>(point.ix) * model.dx);
                chunk.row_coordinates.push_back(
                    static_cast<double>(model.oz) +
                    static_cast<double>(point.iz) * model.dz);
            }
        }
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

void scatter_chunk_output(const std::vector<fki::Complex>& chunk_rows,
                          const BpackDepthChunk& chunk,
                          std::vector<fki::Complex>& full_depth_major)
{
    if (chunk_rows.size() != chunk.depth_major_indices.size()) {
        throw std::invalid_argument(
            "ButterflyPACK output size does not match chunk mapping");
    }
    for (std::size_t row = 0; row < chunk_rows.size(); ++row) {
        const std::size_t destination = chunk.depth_major_indices[row];
        if (destination >= full_depth_major.size()) {
            throw std::out_of_range(
                "ButterflyPACK chunk destination exceeds block output");
        }
        full_depth_major[destination] = chunk_rows[row];
    }
}

void phase_shift_first_block(const se::huygens::Model2D& model,
                             const se::huygens::Block& block,
                             const std::vector<int>& target_iz,
                             float frequency,
                             const std::vector<fki::Complex>& boundary,
                             std::vector<fki::Complex>& output)
{
    const int nx = model.nx;
    if (boundary.size() != static_cast<std::size_t>(nx)) {
        throw std::invalid_argument("first-block boundary size is invalid");
    }

    std::vector<fki::Complex> spectrum = boundary;
    fftwf_plan forward = fftwf_plan_dft_1d(
        nx, reinterpret_cast<fftwf_complex*>(spectrum.data()),
        reinterpret_cast<fftwf_complex*>(spectrum.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);
    if (!forward) {
        throw std::runtime_error("first-block phase-shift FFT plan failed");
    }
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
    const double v0 =
        velocity_sum / std::max<std::size_t>(velocity_count, 1);
    const double omega = 2.0 * kPi * static_cast<double>(frequency);
    const double k0 = omega / v0;

    output.resize(target_iz.size() * static_cast<std::size_t>(nx));
    std::vector<fki::Complex> line(static_cast<std::size_t>(nx));
    fftwf_plan inverse = fftwf_plan_dft_1d(
        nx, reinterpret_cast<fftwf_complex*>(line.data()),
        reinterpret_cast<fftwf_complex*>(line.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);
    if (!inverse) {
        throw std::runtime_error(
            "first-block inverse phase-shift FFT plan failed");
    }

    for (std::size_t iz_local = 0; iz_local < target_iz.size(); ++iz_local) {
        const double distance =
            model.z(target_iz[iz_local]) - model.z(block.source_iz);
        for (int ik = 0; ik < nx; ++ik) {
            const int signed_k = ik <= nx / 2 ? ik : ik - nx;
            const double kx =
                2.0 * kPi * static_cast<double>(signed_k) /
                (static_cast<double>(nx) * model.dx);
            const double kz2 = k0 * k0 - kx * kx;
            const fki::Complex propagator =
                kz2 >= 0.0
                    ? std::exp(fki::Complex(
                          0.0f,
                          static_cast<float>(-std::sqrt(kz2) * distance)))
                    : fki::Complex(
                          static_cast<float>(
                              std::exp(-std::sqrt(-kz2) *
                                       std::abs(distance))),
                          0.0f);
            line[static_cast<std::size_t>(ik)] =
                spectrum[static_cast<std::size_t>(ik)] * propagator;
        }
        fftwf_execute(inverse);
        for (int ix = 0; ix < nx; ++ix) {
            output[iz_local * static_cast<std::size_t>(nx) +
                   static_cast<std::size_t>(ix)] =
                line[static_cast<std::size_t>(ix)] /
                static_cast<float>(nx);
        }
    }
    fftwf_destroy_plan(inverse);
}

void copy_block_to_global(const se::huygens::Model2D& model,
                          const std::vector<int>& target_depths,
                          const std::vector<fki::Complex>& block_values,
                          std::vector<fki::Complex>& global)
{
    const std::size_t expected =
        target_depths.size() * static_cast<std::size_t>(model.nx);
    if (block_values.size() != expected) {
        throw std::invalid_argument("block wavefield size is invalid");
    }
    for (std::size_t iz_local = 0; iz_local < target_depths.size(); ++iz_local) {
        const int iz = target_depths[iz_local];
        for (int ix = 0; ix < model.nx; ++ix) {
            global[model.index(ix, iz)] =
                block_values[iz_local * static_cast<std::size_t>(model.nx) +
                             static_cast<std::size_t>(ix)];
        }
    }
}

std::vector<fki::Complex> extract_boundary(
    const se::huygens::Model2D& model,
    const std::vector<int>& target_depths,
    int next_depth,
    const std::vector<fki::Complex>& block_values)
{
    const int local = fki::find_local_depth(target_depths, next_depth);
    if (local < 0) {
        throw std::runtime_error(
            "next propagation datum is absent from current block output");
    }
    std::vector<fki::Complex> boundary(
        static_cast<std::size_t>(model.nx));
    for (int ix = 0; ix < model.nx; ++ix) {
        boundary[static_cast<std::size_t>(ix)] =
            block_values[static_cast<std::size_t>(local) *
                             static_cast<std::size_t>(model.nx) +
                         static_cast<std::size_t>(ix)];
    }
    return boundary;
}

void write_raw_component(const std::string& filename,
                         const std::vector<fki::Complex>& wavefield,
                         bool imaginary)
{
    std::ofstream stream(filename, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open output binary: " + filename);
    }
    std::vector<float> values(wavefield.size());
    for (std::size_t i = 0; i < wavefield.size(); ++i) {
        values[i] = imaginary ? wavefield[i].imag() : wavefield[i].real();
    }
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(
                     values.size() * sizeof(float)));
    if (!stream) {
        throw std::runtime_error("failed while writing output binary: " + filename);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        huygens_cli::initialize(argc, argv);
        const Options options = parse_options();
        huygens_cli::set_openmp_threads(options.threads);

#ifndef KIRCH_HAS_BUTTERFLYPACK_FLOAT
        throw std::runtime_error(
            "frequency_kirchhoff_wavefield_bf requires "
            "KIRCH_ENABLE_BUTTERFLYPACK=ON and KIRCH_BPACK_ENABLE_FLOAT=ON");
#else
        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(options.velocity);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(options.block_file);
        se::huygens::validate_block_info(info, &model);

        if (options.seam_enable != 0 && info.overlap_rows <= 0) {
            throw std::invalid_argument(
                "seam_enable=1 requires block_info overlap_rows > 0");
        }

        const int source_ix =
            options.source_ix >= 0 ? options.source_ix : model.nx / 2;
        if (source_ix < 0 || source_ix >= model.nx) {
            throw std::invalid_argument("source_ix lies outside the velocity model");
        }

        const std::vector<float> boundary_weights =
            fki::full_boundary_weights(model);
        std::vector<fki::Complex> boundary(
            static_cast<std::size_t>(model.nx),
            fki::Complex(0.0f, 0.0f));

        const double phase_rad =
            static_cast<double>(options.source_phase_deg) * kPi / 180.0;
        const fki::Complex source_value(
            static_cast<float>(
                static_cast<double>(options.source_amplitude) *
                std::cos(phase_rad)),
            static_cast<float>(
                static_cast<double>(options.source_amplitude) *
                std::sin(phase_rad)));
        const float source_x = model.x(source_ix);
        if (!fki::deposit_integral_sample(
                boundary, 0, model, boundary_weights,
                source_x, source_value)) {
            throw std::runtime_error(
                "failed to place the monochromatic source on the surface datum");
        }

        std::vector<fki::Complex> wavefield(
            static_cast<std::size_t>(model.nx) *
                static_cast<std::size_t>(model.nz),
            fki::Complex(0.0f, 0.0f));
        for (int ix = 0; ix < model.nx; ++ix) {
            wavefield[model.index(ix, options.source_iz)] =
                boundary[static_cast<std::size_t>(ix)];
        }

        std::vector<std::vector<int>> block_target_depths(info.blocks.size());
        for (std::size_t ib = 0; ib < info.blocks.size(); ++ib) {
            const se::huygens::Block preview =
                propagation_block(info, ib, options.seam_enable != 0);
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, preview,
                    options.source_stride,
                    options.target_x_stride,
                    options.target_z_stride);
            block_target_depths[ib] = geometry.target_iz;
        }

        if (options.seam_enable != 0) {
            for (std::size_t ib = 0; ib + 1 < block_target_depths.size(); ++ib) {
                const std::vector<int> common =
                    intersect_depths(block_target_depths[ib],
                                     block_target_depths[ib + 1]);
                if (common.size() !=
                    static_cast<std::size_t>(info.overlap_rows)) {
                    throw std::runtime_error(
                        "computational overlap between blocks " +
                        std::to_string(ib) + " and " +
                        std::to_string(ib + 1) + " is " +
                        std::to_string(common.size()) +
                        " rows; expected " +
                        std::to_string(info.overlap_rows));
                }
            }
        }

        std::ofstream seam_stream(options.seam_csv);
        if (!seam_stream) {
            throw std::runtime_error(
                "cannot open seam_csv: " + options.seam_csv);
        }
        seam_stream << std::setprecision(17)
                    << "block,frequency,overlap_rows,samples,"
                       "alpha_real,alpha_imag,alpha_abs,alpha_phase_rad\n";

        std::cout
            << "Single-frequency source wavefield\n"
            << "  frequency      : " << options.frequency << " Hz\n"
            << "  source_ix      : " << source_ix << '\n'
            << "  source_x       : " << source_x << '\n'
            << "  source_iz      : " << options.source_iz << '\n'
            << "  source value   : " << source_value << '\n'
            << "  model          : nx=" << model.nx
            << ", nz=" << model.nz
            << ", dx=" << model.dx
            << ", dz=" << model.dz << '\n'
            << "  blocks         : " << info.blocks.size()
            << ", overlap_rows=" << info.overlap_rows << '\n'
            << "  first block    : homogeneous phase-shift launch\n"
            << "  later blocks   : rectangular ButterflyPACK 1D datum -> 2D block\n"
            << "  GMRES          : disabled/not compiled into this program\n"
            << "  receiver field : not allocated\n"
            << "  image          : not allocated\n"
            << "  geometry cache : disabled (single frequency has no cross-frequency reuse)\n";

        OverlapCache previous_overlap;
        double total_filter_seconds = 0.0;
        double total_build_seconds = 0.0;
        double total_apply_seconds = 0.0;

        for (std::size_t iblock = 0; iblock < info.blocks.size(); ++iblock) {
            const se::huygens::Block block =
                propagation_block(info, iblock, options.seam_enable != 0);
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block,
                    options.source_stride,
                    options.target_x_stride,
                    options.target_z_stride);
            const se::huygens::OneWayLayerTables tables =
                se::huygens::read_one_way_layer_tables(
                    options.table_prefix, block.id);
            try {
                se::huygens::validate_one_way_layer_tables(
                    tables, geometry, block);
            } catch (const std::exception& error) {
                if (options.seam_enable != 0 && iblock > 0 &&
                    info.overlap_rows > 0) {
                    throw std::runtime_error(
                        "overlap propagation requires traveltime tables generated "
                        "with include_propagation_overlap=1. Original error: " +
                        std::string(error.what()));
                }
                throw;
            }

            if (static_cast<int>(geometry.source_ix.size()) != model.nx ||
                static_cast<int>(geometry.target_ix.size()) != model.nx) {
                throw std::runtime_error(
                    "rectangular ButterflyPACK wavefield propagation requires "
                    "every lateral grid point");
            }
            if (geometry.target_iz != block_target_depths[iblock]) {
                throw std::runtime_error(
                    "precomputed block target depths changed unexpectedly");
            }

            const int target_depth_count =
                static_cast<int>(geometry.target_iz.size());
            std::vector<fki::Complex> block_output(
                static_cast<std::size_t>(target_depth_count) *
                    static_cast<std::size_t>(model.nx),
                fki::Complex(0.0f, 0.0f));

            OverlapCache next_overlap;
            if (options.seam_enable != 0 &&
                iblock + 1 < info.blocks.size()) {
                next_overlap = make_overlap_cache(
                    intersect_depths(block_target_depths[iblock],
                                     block_target_depths[iblock + 1]),
                    model.nx);
            }

            std::cout
                << "\nBlock " << block.id
                << ": source_iz=" << block.source_iz
                << ", computational target=["
                << geometry.target_iz.front() << ','
                << geometry.target_iz.back() << ']'
                << ", official target=["
                << info.blocks[iblock].target_start_iz << ','
                << info.blocks[iblock].target_end_iz << ']'
                << ", overlap_prev=" << previous_overlap.depths.size()
                << ", overlap_next=" << next_overlap.depths.size()
                << '\n';

            if (iblock == 0) {
                const auto started = fki::Clock::now();
                phase_shift_first_block(
                    model, block, geometry.target_iz,
                    options.frequency, boundary, block_output);
                const double elapsed = std::chrono::duration<double>(
                    fki::Clock::now() - started).count();
                total_apply_seconds += elapsed;
                std::cout << "  phase-shift apply=" << elapsed << " s\n";
            } else {
                const auto filter_started = fki::Clock::now();
                const float maximum_tau = *std::max_element(
                    tables.traveltime.values.begin(),
                    tables.traveltime.values.end());
                const se::huygens::FrequencyKirchhoffFilter filter(
                    options.frequency,
                    options.filter_dt,
                    options.filter_length,
                    maximum_tau,
                    options.filter_lookup_subsamples);
                const double filter_seconds = std::chrono::duration<double>(
                    fki::Clock::now() - filter_started).count();
                total_filter_seconds += filter_seconds;

                se::huygens::OneWayKernelData kernel;
                kernel.rows = static_cast<int>(geometry.targets.size());
                kernel.sources = static_cast<int>(geometry.source_ix.size());
                kernel.source_z = model.z(block.source_iz);
                kernel.quadrature_weights = &boundary_weights;
                kernel.tables = &tables;
                kernel.filter = &filter;

                std::vector<double> column_coordinates(
                    2 * geometry.source_ix.size());
                const double datum_z =
                    static_cast<double>(model.oz) +
                    static_cast<double>(block.source_iz) * model.dz;
                for (std::size_t isource = 0;
                     isource < geometry.source_ix.size(); ++isource) {
                    column_coordinates[2 * isource] =
                        static_cast<double>(model.ox) +
                        static_cast<double>(geometry.source_ix[isource]) *
                            model.dx;
                    column_coordinates[2 * isource + 1] = datum_z;
                }

                const std::vector<BpackDepthChunk> chunks =
                    make_bpack_depth_chunks(
                        model, geometry, options.bpack_depth_chunk);

                se::butterfly::Options bpack_options;
                bpack_options.tolerance = options.bpack_tolerance;
                bpack_options.leaf_size = options.bpack_leaf;
                bpack_options.verbosity = options.bpack_verbosity;
                bpack_options.coordinate_dimension = 2;
                bpack_options.lr_level = options.bpack_lr_level;
                bpack_options.sample_parameter =
                    options.bpack_sample_parameter;
                bpack_options.forward_n15_flag =
                    options.bpack_forward_n15;
                bpack_options.nearest_neighbors = options.bpack_knn;
                bpack_options.compression_pattern = options.bpack_pat_comp;
                bpack_options.less_adapt = options.bpack_less_adapt;
                bpack_options.rank_detection_factor =
                    options.bpack_rdetect_factor;
                bpack_options.reuse_geometry =
                    options.bpack_reuse_tree;

                double block_build_seconds = 0.0;
                double block_apply_seconds = 0.0;

                for (std::size_t ichunk = 0; ichunk < chunks.size(); ++ichunk) {
                    const BpackDepthChunk& chunk = chunks[ichunk];
                    const bool print_progress =
                        options.bpack_progress != 0 &&
                        (ichunk == 0 ||
                         ichunk + 1 == chunks.size() ||
                         (ichunk %
                              static_cast<std::size_t>(
                                  options.bpack_progress_every) ==
                          0));

                    if (print_progress) {
                        const int first_depth =
                            geometry.target_iz[
                                static_cast<std::size_t>(chunk.depth_begin)];
                        const int last_depth =
                            geometry.target_iz[
                                static_cast<std::size_t>(
                                    chunk.depth_begin +
                                    chunk.depth_count - 1)];
                        std::cout
                            << "  [BPACK build START] chunk="
                            << (ichunk + 1) << '/' << chunks.size()
                            << ", depths=[" << first_depth
                            << ',' << last_depth << ']'
                            << ", matrix=" << chunk.full_rows.size()
                            << 'x' << geometry.source_ix.size() << '\n';
                    }

                    const auto build_started = fki::Clock::now();
                    auto butterfly =
                        std::make_unique<se::butterfly::Matrix<float>>(
                            static_cast<int>(chunk.full_rows.size()),
                            static_cast<int>(geometry.source_ix.size()),
                            chunk.row_coordinates,
                            column_coordinates,
                            [&kernel, &chunk](int row, int source) {
                                if (row < 0 ||
                                    static_cast<std::size_t>(row) >=
                                        chunk.full_rows.size()) {
                                    throw std::out_of_range(
                                        "ButterflyPACK callback row is invalid");
                                }
                                return kernel.entry(
                                    chunk.full_rows[
                                        static_cast<std::size_t>(row)],
                                    source);
                            },
                            bpack_options);
                    const double build_seconds =
                        std::chrono::duration<double>(
                            fki::Clock::now() - build_started).count();
                    block_build_seconds += build_seconds;

                    const auto apply_started = fki::Clock::now();
                    const std::vector<fki::Complex> output =
                        butterfly->apply(boundary);
                    const double apply_seconds =
                        std::chrono::duration<double>(
                            fki::Clock::now() - apply_started).count();
                    block_apply_seconds += apply_seconds;

                    scatter_chunk_output(
                        output, chunk, block_output);

                    if (print_progress) {
                        const se::butterfly::Statistics stats =
                            butterfly->statistics();
                        std::cout
                            << "  [BPACK DONE] chunk="
                            << (ichunk + 1) << '/' << chunks.size()
                            << ", build=" << build_seconds << " s"
                            << ", apply=" << apply_seconds << " s"
                            << ", compressed="
                            << stats.compressed_megabytes << " MiB"
                            << ", peak=" << stats.peak_megabytes << " MiB"
                            << ", rank_max=" << stats.maximum_rank << '\n';
                    }
                }

                total_build_seconds += block_build_seconds;
                total_apply_seconds += block_apply_seconds;
                std::cout
                    << "  filter=" << filter_seconds << " s"
                    << ", BPACK build=" << block_build_seconds << " s"
                    << ", BPACK apply=" << block_apply_seconds << " s\n";
            }

            if (options.seam_enable != 0 &&
                iblock > 0 && !previous_overlap.empty()) {
                std::size_t samples = 0;
                const std::complex<double> alpha = blend_overlap(
                    previous_overlap,
                    geometry.target_iz,
                    block_output,
                    options.seam_amp_rel_floor,
                    options.seam_scale_cap,
                    &samples);
                seam_stream
                    << block.id << ',' << options.frequency << ','
                    << previous_overlap.depths.size() << ','
                    << samples << ','
                    << alpha.real() << ','
                    << alpha.imag() << ','
                    << std::abs(alpha) << ','
                    << std::arg(alpha) << '\n';
                std::cout
                    << "  seam alpha=" << alpha
                    << ", samples=" << samples << '\n';
            }

            if (options.seam_enable != 0 && !next_overlap.empty()) {
                store_overlap(
                    geometry.target_iz, block_output, next_overlap);
            }

            copy_block_to_global(
                model, geometry.target_iz, block_output, wavefield);

            if (iblock + 1 < info.blocks.size()) {
                boundary = extract_boundary(
                    model,
                    geometry.target_iz,
                    info.blocks[iblock + 1].source_iz,
                    block_output);
            }

            previous_overlap = options.seam_enable != 0
                ? std::move(next_overlap)
                : OverlapCache{};
        }

        write_raw_component(options.output_real, wavefield, false);
        write_raw_component(options.output_imag, wavefield, true);

        std::cout
            << "\nSingle-frequency ButterflyPACK wavefield completed.\n"
            << "Filter setup total : " << total_filter_seconds << " s\n"
            << "BPACK build total  : " << total_build_seconds << " s\n"
            << "Propagation total  : " << total_apply_seconds << " s\n"
            << "Real binary        : " << options.output_real << '\n'
            << "Imag binary        : " << options.output_imag << '\n'
            << "Binary layout      : float32, x-fast; index=ix+iz*nx\n"
            << "MATLAB read        : "
               "A=reshape(fread(fopen(file,'rb'),nx*nz,'float32=>single'),"
               "[nx,nz]).';\n";
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << kProgramName << ": " << error.what() << '\n';
        return 1;
    }
}
