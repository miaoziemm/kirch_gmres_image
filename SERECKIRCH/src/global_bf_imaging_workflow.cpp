#include <SERECKIRCH/include/global_bf_imaging_workflow.hpp>
#include <SERECKIRCH/include/frequency_kirchhoff_imaging_common.hpp>
#include <global_preconditioned_gmres.hpp>
#include <SERECKIRCH/include/efmm.h>

#include <SERECKIRCH/include/se_butterfly.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <cstdint>
#include <memory>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fki = frequency_kirchhoff_imaging;

namespace {

constexpr const char* kProgramName = "frequency_kirchhoff_imaging_bf_global_gmres";
constexpr const char* kMethodDescription =
    "reference point-ray source + recursive ButterflyPACK receiver + unpreconditioned restarted GMRES";

using PackedComplex = std::vector<float>;

/*
 * Scheme-A computational geometry.
 *
 * BlockInfo intentionally keeps the official output intervals contiguous.
 * For block k>0, however, source_iz lies overlap_rows samples above the
 * previous official target end.  To obtain a real wavefield overlap, Scheme A
 * recomputes the rows immediately below that propagation datum:
 *
 *     source_iz + 1 ... target_end_iz
 *
 * The extra rows source_iz+1 ... previous_target_end are used only for
 * inter-block alignment/blending and partition-of-unity imaging.  The block
 * file itself is not changed.
 */
se::huygens::Block scheme_a_propagation_block(
    const se::huygens::BlockInfo& info,
    std::size_t block_index,
    bool enable_scheme_a)
{
    se::huygens::Block block =
        fki::imaging_propagation_block(info, block_index);

    if (enable_scheme_a && block_index > 0 && info.overlap_rows > 0) {
        const int overlap_target_start = block.source_iz + 1;
        if (overlap_target_start >= block.target_start_iz) {
            throw std::runtime_error(
                "Scheme A expected source_iz+1 to lie above the official "
                "target_start_iz for a positive propagation overlap");
        }
        block.target_start_iz = overlap_target_start;
    }
    return block;
}

/* -------------------------------------------------------------------------
 * Scheme A: overlap-consistent block coupling
 *
 * The original program propagated each block recursively, but each block
 * contributed its complete target-depth interval directly to the image.  A
 * small phase/amplitude mismatch between neighboring local wavefields could
 * therefore appear as a horizontal seam.  The helpers below add three pieces:
 *
 *   1) cache the actual wavefield in the geometric overlap of two blocks;
 *   2) estimate a complex least-squares scale before blending the two fields;
 *   3) use a partition of unity for both wavefield blending and imaging.
 *
 * The complex scale is local to the overlap.  It is NOT applied to the whole
 * current block: its magnitude and phase are continuously relaxed back to one
 * while the previous-block contribution is tapered to zero.  Consequently the
 * bottom of the overlap is exactly the unmodified current-block wavefield and
 * no second seam is introduced there.
 * ------------------------------------------------------------------------- */

struct OverlapCache {
    int nx = 0;
    std::size_t shot_count = 0;
    std::size_t frequency_count = 0;
    std::vector<int> depths;
    std::vector<fki::Complex> source;
    std::vector<fki::Complex> receiver_conjugate;

    bool empty() const { return depths.empty(); }

    std::size_t index(std::size_t shot,
                      std::size_t frequency,
                      std::size_t depth,
                      int ix) const
    {
        return ((((shot * frequency_count) + frequency) * depths.size() + depth) *
                    static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix));
    }
};

std::vector<int> intersect_depths(const std::vector<int>& a,
                                  const std::vector<int>& b)
{
    std::vector<int> result;
    result.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(result));
    return result;
}

int exact_depth_index(const std::vector<int>& depths, int iz)
{
    const auto it = std::lower_bound(depths.begin(), depths.end(), iz);
    if (it == depths.end() || *it != iz) return -1;
    return static_cast<int>(std::distance(depths.begin(), it));
}

OverlapCache make_overlap_cache(const std::vector<int>& depths,
                                int nx,
                                std::size_t shot_count,
                                std::size_t frequency_count)
{
    OverlapCache cache;
    cache.nx = nx;
    cache.shot_count = shot_count;
    cache.frequency_count = frequency_count;
    cache.depths = depths;
    if (depths.empty()) return cache;

    const std::size_t count = shot_count * frequency_count * depths.size() *
                              static_cast<std::size_t>(nx);
    cache.source.resize(count);
    cache.receiver_conjugate.resize(count);
    return cache;
}

double overlap_cache_mb(const OverlapCache& cache)
{
    return static_cast<double>(cache.source.size() +
                               cache.receiver_conjugate.size()) *
           sizeof(fki::Complex) / 1048576.0;
}

std::vector<std::vector<float>> build_image_partition_weights(
    const std::vector<std::vector<int>>& block_depths,
    int model_nz)
{
    std::vector<std::vector<float>> weights(block_depths.size());
    for (std::size_t iblock = 0; iblock < block_depths.size(); ++iblock) {
        weights[iblock].assign(block_depths[iblock].size(), 1.0f);
    }

    // Apply complementary raised-cosine tapers on every adjacent overlap.
    // If a block overlaps both neighbors, the left and right tapers multiply,
    // leaving the central non-overlap part at unit weight.
    for (std::size_t iblock = 0; iblock + 1 < block_depths.size(); ++iblock) {
        const std::vector<int> common = intersect_depths(
            block_depths[iblock], block_depths[iblock + 1]);
        if (common.empty()) continue;

        for (std::size_t j = 0; j < common.size(); ++j) {
            const double s = common.size() == 1
                                 ? 0.5
                                 : static_cast<double>(j) /
                                       static_cast<double>(common.size() - 1);
            const double previous_weight =
                0.5 * (1.0 + std::cos(3.14159265358979323846 * s));
            const double current_weight = 1.0 - previous_weight;
            const int iprev = exact_depth_index(block_depths[iblock], common[j]);
            const int icurr = exact_depth_index(block_depths[iblock + 1], common[j]);
            if (iprev >= 0) {
                weights[iblock][static_cast<std::size_t>(iprev)] *=
                    static_cast<float>(previous_weight);
            }
            if (icurr >= 0) {
                weights[iblock + 1][static_cast<std::size_t>(icurr)] *=
                    static_cast<float>(current_weight);
            }
        }
    }

    // Normalize by global depth.  For the usual two-block overlap this is
    // already exactly one, but normalization also makes the code robust to an
    // unusual three-way overlap or target-z subsampling.
    std::vector<double> sums(static_cast<std::size_t>(model_nz), 0.0);
    for (std::size_t iblock = 0; iblock < block_depths.size(); ++iblock) {
        for (std::size_t i = 0; i < block_depths[iblock].size(); ++i) {
            const int iz = block_depths[iblock][i];
            if (iz >= 0 && iz < model_nz) {
                sums[static_cast<std::size_t>(iz)] += weights[iblock][i];
            }
        }
    }
    for (std::size_t iblock = 0; iblock < block_depths.size(); ++iblock) {
        for (std::size_t i = 0; i < block_depths[iblock].size(); ++i) {
            const int iz = block_depths[iblock][i];
            if (iz >= 0 && iz < model_nz) {
                const double sum = sums[static_cast<std::size_t>(iz)];
                if (sum > 0.0) {
                    weights[iblock][i] =
                        static_cast<float>(weights[iblock][i] / sum);
                }
            }
        }
    }
    return weights;
}

std::complex<double> estimate_complex_alignment(
    const OverlapCache& previous,
    const std::vector<fki::Complex>& current,
    const std::vector<int>& current_depths,
    std::size_t shot,
    std::size_t frequency,
    bool source_field,
    double amplitude_relative_floor,
    double scale_cap,
    std::size_t* accepted_samples)
{
    if (accepted_samples) *accepted_samples = 0;
    if (previous.empty()) return std::complex<double>(1.0, 0.0);

    double maximum_amplitude = 0.0;
    for (std::size_t idepth = 0; idepth < previous.depths.size(); ++idepth) {
        const int current_local =
            exact_depth_index(current_depths, previous.depths[idepth]);
        if (current_local < 0) continue;
        for (int ix = 0; ix < previous.nx; ++ix) {
            const fki::Complex ref = source_field
                ? previous.source[previous.index(shot, frequency, idepth, ix)]
                : previous.receiver_conjugate[
                      previous.index(shot, frequency, idepth, ix)];
            const fki::Complex cur = current[
                static_cast<std::size_t>(current_local) * previous.nx + ix];
            maximum_amplitude = std::max(
                maximum_amplitude,
                std::max(static_cast<double>(std::abs(ref)),
                         static_cast<double>(std::abs(cur))));
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
            const fki::Complex ref_f = source_field
                ? previous.source[previous.index(shot, frequency, idepth, ix)]
                : previous.receiver_conjugate[
                      previous.index(shot, frequency, idepth, ix)];
            const fki::Complex cur_f = current[
                static_cast<std::size_t>(current_local) * previous.nx + ix];
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
        return std::complex<double>(1.0, 0.0);
    }

    std::complex<double> alpha = numerator / denominator;
    double magnitude = std::abs(alpha);
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
        return std::complex<double>(1.0, 0.0);
    }

    if (scale_cap > 1.0) {
        const double minimum_scale = 1.0 / scale_cap;
        const double clipped = std::min(scale_cap, std::max(minimum_scale, magnitude));
        alpha *= clipped / magnitude;
    }
    return alpha;
}

std::complex<double> relaxed_complex_scale(const std::complex<double>& alpha,
                                           double amount)
{
    // amount=1 -> alpha; amount=0 -> one.  Polar interpolation avoids the
    // zero crossing that linear complex interpolation can produce near a
    // phase difference of pi.
    const double magnitude = std::max(std::abs(alpha), 1.0e-30);
    const double phase = std::arg(alpha);
    return std::polar(std::pow(magnitude, amount), phase * amount);
}

void blend_one_overlap_field(
    const OverlapCache& previous,
    std::size_t shot,
    std::size_t frequency,
    const std::vector<int>& current_depths,
    std::vector<fki::Complex>& current,
    bool source_field,
    double amplitude_relative_floor,
    double scale_cap,
    std::complex<double>* estimated_scale,
    std::size_t* accepted_samples)
{
    if (previous.empty()) {
        if (estimated_scale) *estimated_scale = std::complex<double>(1.0, 0.0);
        if (accepted_samples) *accepted_samples = 0;
        return;
    }

    std::size_t used = 0;
    const std::complex<double> alpha = estimate_complex_alignment(
        previous, current, current_depths, shot, frequency, source_field,
        amplitude_relative_floor, scale_cap, &used);
    if (estimated_scale) *estimated_scale = alpha;
    if (accepted_samples) *accepted_samples = used;

    for (std::size_t idepth = 0; idepth < previous.depths.size(); ++idepth) {
        const int current_local =
            exact_depth_index(current_depths, previous.depths[idepth]);
        if (current_local < 0) continue;

        const double s = previous.depths.size() == 1
                             ? 0.5
                             : static_cast<double>(idepth) /
                                   static_cast<double>(previous.depths.size() - 1);
        const double previous_weight =
            0.5 * (1.0 + std::cos(3.14159265358979323846 * s));
        const double current_weight = 1.0 - previous_weight;
        const std::complex<double> local_scale =
            relaxed_complex_scale(alpha, previous_weight);

        for (int ix = 0; ix < previous.nx; ++ix) {
            const std::size_t current_index =
                static_cast<std::size_t>(current_local) * previous.nx + ix;
            const fki::Complex ref_f = source_field
                ? previous.source[previous.index(shot, frequency, idepth, ix)]
                : previous.receiver_conjugate[
                      previous.index(shot, frequency, idepth, ix)];
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
}

void store_next_overlap(
    const std::vector<int>& current_depths,
    const std::vector<fki::Complex>& source,
    const std::vector<fki::Complex>& receiver_conjugate,
    std::size_t shot,
    std::size_t frequency,
    OverlapCache& next)
{
    if (next.empty()) return;
    for (std::size_t idepth = 0; idepth < next.depths.size(); ++idepth) {
        const int local = exact_depth_index(current_depths, next.depths[idepth]);
        if (local < 0) {
            throw std::runtime_error(
                "next overlap depth is absent from current block target depths");
        }
        for (int ix = 0; ix < next.nx; ++ix) {
            const std::size_t local_index =
                static_cast<std::size_t>(local) * next.nx + ix;
            const std::size_t cache_index = next.index(shot, frequency, idepth, ix);
            next.source[cache_index] = source[local_index];
            next.receiver_conjugate[cache_index] = receiver_conjugate[local_index];
        }
    }
}

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
            static_cast<std::size_t>(target_depth_count) * model.nx) {
        throw std::runtime_error(
            "ButterflyPACK depth chunking requires a complete x-by-depth target grid");
    }

    const int chunk_depths = std::min(requested_depth_chunk, target_depth_count);
    std::vector<BpackDepthChunk> chunks;
    chunks.reserve(static_cast<std::size_t>(
        (target_depth_count + chunk_depths - 1) / chunk_depths));

    for (int depth_begin = 0; depth_begin < target_depth_count;
         depth_begin += chunk_depths) {
        BpackDepthChunk chunk;
        chunk.depth_begin = depth_begin;
        chunk.depth_count = std::min(
            chunk_depths, target_depth_count - depth_begin);
        const std::size_t rows =
            static_cast<std::size_t>(chunk.depth_count) * model.nx;
        chunk.full_rows.reserve(rows);
        chunk.depth_major_indices.reserve(rows);
        chunk.row_coordinates.reserve(2 * rows);

        for (int ix = 0; ix < model.nx; ++ix) {
            for (int iz_chunk = 0; iz_chunk < chunk.depth_count; ++iz_chunk) {
                const int iz_local = depth_begin + iz_chunk;
                const std::size_t full_row =
                    static_cast<std::size_t>(ix) * target_depth_count +
                    static_cast<std::size_t>(iz_local);
                if (full_row >= geometry.targets.size()) {
                    throw std::runtime_error(
                        "ButterflyPACK chunk row exceeds target geometry");
                }
                const se::huygens::GridPoint& point = geometry.targets[full_row];
                if (point.ix != ix ||
                    point.iz != geometry.target_iz[static_cast<std::size_t>(iz_local)]) {
                    throw std::runtime_error(
                        "ButterflyPACK target ordering is incompatible with chunking");
                }
                chunk.full_rows.push_back(static_cast<int>(full_row));
                chunk.depth_major_indices.push_back(
                    static_cast<std::size_t>(iz_local) * model.nx + ix);
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

void scatter_bpack_chunk_output(
    const std::vector<fki::Complex>& chunk_rows,
    const BpackDepthChunk& chunk,
    std::vector<fki::Complex>& full_depth_major)
{
    if (chunk_rows.size() != chunk.depth_major_indices.size()) {
        throw std::invalid_argument(
            "ButterflyPACK chunk output size does not match chunk mapping");
    }
    for (std::size_t row = 0; row < chunk_rows.size(); ++row) {
        const std::size_t destination = chunk.depth_major_indices[row];
        if (destination >= full_depth_major.size()) {
            throw std::out_of_range(
                "ButterflyPACK chunk destination exceeds full block output");
        }
        full_depth_major[destination] = chunk_rows[row];
    }
}

std::vector<fki::Complex> boundary_frequency(
    const std::vector<fki::Complex>& state,
    std::size_t frequency,
    int nx)
{
    const std::size_t begin = frequency * static_cast<std::size_t>(nx);
    const std::size_t end = begin + static_cast<std::size_t>(nx);
    if (end > state.size()) {
        throw std::out_of_range(
            "ButterflyPACK boundary frequency exceeds stored state");
    }
    return std::vector<fki::Complex>(state.begin() + begin, state.begin() + end);
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


void phase_shift_first_block(const se::huygens::Model2D& model,
                             const se::huygens::Block& block,
                             const std::vector<int>& target_iz,
                             float frequency,
                             double mean_velocity,
                             const std::vector<fki::Complex>& boundary,
                             std::size_t frequency_index,
                             std::vector<fki::Complex>& output)
{
    const int nx = model.nx;
    const std::size_t boundary_offset =
        frequency_index * static_cast<std::size_t>(nx);
    if (boundary_offset + static_cast<std::size_t>(nx) > boundary.size()) {
        throw std::out_of_range(
            "phase-shift frequency exceeds boundary state");
    }
    std::vector<fki::Complex> spectrum(
        boundary.begin() + static_cast<std::ptrdiff_t>(boundary_offset),
        boundary.begin() + static_cast<std::ptrdiff_t>(
            boundary_offset + static_cast<std::size_t>(nx)));
    fftwf_plan forward = fftwf_plan_dft_1d(
        nx, reinterpret_cast<fftwf_complex*>(spectrum.data()),
        reinterpret_cast<fftwf_complex*>(spectrum.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    if (!forward) throw std::runtime_error("first-block phase-shift FFT plan failed");
    fftwf_execute(forward);
    fftwf_destroy_plan(forward);

    const double omega = 2.0 * 3.14159265358979323846 * frequency;
    const double k0 = omega / mean_velocity;
    output.resize(static_cast<std::size_t>(target_iz.size()) * nx);
    std::vector<fki::Complex> line(static_cast<std::size_t>(nx));
    // FFTW plans describe the transform layout, not the values.  Reuse one
    // inverse plan for every depth instead of planning the identical transform
    // once per output row.
    fftwf_plan inverse = fftwf_plan_dft_1d(
        nx, reinterpret_cast<fftwf_complex*>(line.data()),
        reinterpret_cast<fftwf_complex*>(line.data()), FFTW_BACKWARD,
        FFTW_ESTIMATE);
    if (!inverse) {
        throw std::runtime_error("first-block inverse phase-shift FFT plan failed");
    }
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
        fftwf_execute(inverse);
        for (int ix = 0; ix < nx; ++ix) {
            output[iz_local * static_cast<std::size_t>(nx) + ix] =
                line[static_cast<std::size_t>(ix)] / static_cast<float>(nx);
        }
    }
    fftwf_destroy_plan(inverse);
}

namespace gpg = global_preconditioned_gmres;

std::vector<float> solve_global_source_traveltime(
    const se::huygens::Model2D& model, float source_x, float source_z)
{
    const float relative_x = source_x - model.ox;
    if (relative_x < -1.0e-4f ||
        relative_x > (model.nx - 1) * model.dx + 1.0e-4f) {
        throw std::invalid_argument(
            "source x coordinate lies outside velocity model");
    }

    efmm_t solver{};
    if (efmm_init(&solver, model.nx, model.nz, model.dx, model.dz,
                  relative_x, source_z - model.oz) != 0) {
        throw std::runtime_error("efmm_init failed for global source field");
    }
    try {
        if (efmm_set_vel(
                &solver, const_cast<float*>(model.velocity.data())) != 0) {
            throw std::runtime_error(
                "efmm_set_vel failed for global source field");
        }
        const int status = efmm_solver(&solver);
        if (status != 0) {
            throw std::runtime_error(
                "efmm_solver failed with status " + std::to_string(status));
        }
        const std::size_t count =
            static_cast<std::size_t>(model.nx) * model.nz;
        std::vector<float> traveltime(solver.tt, solver.tt + count);
        efmm_free(&solver);
        return traveltime;
    } catch (...) {
        efmm_free(&solver);
        throw;
    }
}

int nearest_global_source_ix(
    const se::huygens::Model2D& model, float source_x)
{
    const double coordinate =
        (static_cast<double>(source_x) - model.ox) / model.dx;
    return std::clamp(
        static_cast<int>(std::llround(coordinate)), 0, model.nx - 1);
}

gpg::Vector build_global_source_aexp(
    const se::huygens::Model2D& model,
    const std::vector<float>& traveltime,
    float frequency,
    fki::Complex source_spectrum,
    int source_ix,
    int source_iz)
{
    const std::size_t count =
        static_cast<std::size_t>(model.nx) * model.nz;
    if (traveltime.size() != count || !(frequency > 0.0f)) {
        throw std::invalid_argument("invalid global Aexp parameters");
    }

    const double omega = 2.0 * gpg::kPi * frequency;
    const double minimum_time = std::max(model.dx, model.dz) /
        model.velocity[model.index(source_ix, source_iz)];
    const gpg::Complex spectrum = std::conj(gpg::Complex(
        static_cast<double>(source_spectrum.real()),
        static_cast<double>(source_spectrum.imag()))) *
        (static_cast<double>(model.dx) * model.dz);

    gpg::Vector field(static_cast<Eigen::Index>(count));
    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            const std::size_t grid = model.index(ix, iz);
            const double travel_time = std::max(
                0.0, static_cast<double>(traveltime[grid]));
            const double effective_time = std::max(travel_time, minimum_time);
            const double amplitude = 1.0 /
                std::sqrt(8.0 * gpg::kPi * omega * effective_time);
            const double phase = omega * travel_time + 0.25 * gpg::kPi;
            field[static_cast<Eigen::Index>(grid)] =
                spectrum * std::polar(amplitude, phase);
        }
    }
    return field;
}

gpg::Vector pad_global_field(const gpg::Vector& field, int nz, int nx)
{
    const int pnz = nz + 2 * gpg::kPml;
    const int pnx = nx + 2 * gpg::kPml;
    gpg::Vector padded(static_cast<Eigen::Index>(pnz) * pnx);
    for (int iz = 0; iz < pnz; ++iz) {
        const int piz = std::clamp(iz - gpg::kPml, 0, nz - 1);
        for (int ix = 0; ix < pnx; ++ix) {
            const int pix = std::clamp(ix - gpg::kPml, 0, nx - 1);
            padded[gpg::index(iz, ix, pnx)] = field[gpg::index(piz, pix, nx)];
        }
    }
    return padded;
}

gpg::Vector lower_global_mask(const gpg::Vector& field, int nz, int nx,
                              int first_iz)
{
    gpg::Vector masked = gpg::Vector::Zero(field.size());
    for (int iz = first_iz; iz < nz; ++iz)
        for (int ix = 0; ix < nx; ++ix)
            masked[gpg::index(iz, ix, nx)] = field[gpg::index(iz, ix, nx)];
    return masked;
}

gpg::Vector reference_correct_field(const gpg::Sparse& matrix,
                                    const gpg::Vector& input, int nz, int nx,
                                    int correction_iz, int restart, int outer,
                                    gpg::Result* metric)
{
    const int pnz = nz + 2 * gpg::kPml;
    const int pnx = nx + 2 * gpg::kPml;
    const int first = correction_iz + gpg::kPml;
    const gpg::Vector padded = pad_global_field(input, nz, nx);
    const gpg::Vector initial = lower_global_mask(padded, pnz, pnx, first);
    const gpg::Vector rhs = matrix * initial -
        lower_global_mask(matrix * padded, pnz, pnx, first);
    gpg::Result result = gpg::solve(matrix, rhs, initial, restart, outer);
    gpg::Vector corrected = input;
    for (int iz = correction_iz; iz < nz; ++iz)
        for (int ix = 0; ix < nx; ++ix)
            corrected[gpg::index(iz, ix, nx)] =
                result.field[gpg::index(iz + gpg::kPml,
                                        ix + gpg::kPml, pnx)];
    if (metric != nullptr) *metric = std::move(result);
    return corrected;
}

class ReceiverBlockStore {
public:
    ReceiverBlockStore(std::filesystem::path directory, std::string mode,
                       double estimated_megabytes, double memory_limit_megabytes,
                       std::size_t shots, std::size_t frequencies,
                       const se::huygens::Model2D& model)
        : directory_(std::move(directory)), shots_(shots), frequencies_(frequencies),
          nx_(model.nx), grid_size_(static_cast<std::size_t>(model.nx) * model.nz)
    {
        if (mode != "auto" && mode != "memory" && mode != "disk")
            throw std::invalid_argument("receiver_store must be auto, memory, or disk");
        memory_backed_ = mode == "memory" ||
            (mode == "auto" && estimated_megabytes <= memory_limit_megabytes);
        if (mode == "memory" && estimated_megabytes > memory_limit_megabytes)
            throw std::runtime_error("receiver_store=memory requires " +
                std::to_string(estimated_megabytes) +
                " MiB, exceeding global_storage_max_mb=" +
                std::to_string(memory_limit_megabytes));
        if (!memory_backed_) std::filesystem::create_directories(directory_);
    }

    ~ReceiverBlockStore()
    {
        std::error_code error;
        if (!memory_backed_) {
            for (auto& block : blocks_) {
                block.stream.close();
                std::filesystem::remove(block.path, error);
                error.clear();
            }
            // Remove the configured directory only when it is empty; never delete
            // unrelated user files if a shared scratch parent was supplied.
            std::filesystem::remove(directory_, error);
        }
    }

    void begin_block(const std::vector<int>& depths,
                     const std::vector<float>& weights)
    {
        if (depths.size() != weights.size())
            throw std::invalid_argument("receiver-store depth/weight size mismatch");
        Block block;
        block.depths = depths;
        block.weights = weights;
        block.values_per_record = depths.size() * static_cast<std::size_t>(nx_);
        if (memory_backed_) {
            block.memory.assign(shots_ * frequencies_ * block.values_per_record,
                                fki::Complex(0.0f, 0.0f));
        } else {
            block.path = directory_ / ("receiver_block_" +
                std::to_string(blocks_.size()) + ".bin");
            block.stream.open(block.path, std::ios::binary | std::ios::in |
                                          std::ios::out | std::ios::trunc);
            if (!block.stream)
                throw std::runtime_error("cannot create receiver scratch file: " + block.path.string());
        }
        blocks_.push_back(std::move(block));
    }

    void write(std::size_t shot, std::size_t frequency,
               const std::vector<fki::Complex>& field)
    {
        Block& block = blocks_.back();
        if (field.size() != block.values_per_record)
            throw std::invalid_argument("receiver-store field size mismatch");
        const std::uint64_t record = frequency * shots_ + shot;
        const std::uint64_t value_offset = record * block.values_per_record;
        if (memory_backed_) {
            std::copy(field.begin(), field.end(), block.memory.begin() +
                      static_cast<std::ptrdiff_t>(value_offset));
        } else {
            const std::uint64_t byte_offset = value_offset * sizeof(fki::Complex);
            block.stream.clear();
            block.stream.seekp(static_cast<std::streamoff>(byte_offset));
            block.stream.write(reinterpret_cast<const char*>(field.data()),
                               static_cast<std::streamsize>(field.size() * sizeof(fki::Complex)));
            if (!block.stream) throw std::runtime_error("receiver scratch write failed");
            bytes_written_ += field.size() * sizeof(fki::Complex);
        }
    }

    void finalize_writes()
    {
        if (writes_finalized_) return;
        if (!memory_backed_) {
            for (Block& block : blocks_) {
                block.stream.flush();
                if (!block.stream)
                    throw std::runtime_error("receiver scratch flush failed");
            }
        }
        writes_finalized_ = true;
    }

    gpg::Vector load(std::size_t shot, std::size_t frequency,
                     const se::huygens::Model2D& model)
    {
        if (!writes_finalized_)
            throw std::logic_error("receiver store must be finalized before reading");
        gpg::Vector output = gpg::Vector::Zero(static_cast<Eigen::Index>(grid_size_));
        std::vector<fki::Complex> read_buffer;
        for (Block& block : blocks_) {
            read_buffer.resize(block.values_per_record);
            const std::uint64_t record = frequency * shots_ + shot;
            const std::uint64_t value_offset = record * block.values_per_record;
            if (memory_backed_) {
                std::copy_n(block.memory.begin() + static_cast<std::ptrdiff_t>(value_offset),
                            block.values_per_record, read_buffer.begin());
            } else {
                // std::fstream has shared seek state.  Keep only disk access
                // serialized; GMRES and memory-backed loads remain parallel.
                std::lock_guard<std::mutex> lock(read_mutex_);
                const std::uint64_t byte_offset = value_offset * sizeof(fki::Complex);
                block.stream.clear();
                block.stream.seekg(static_cast<std::streamoff>(byte_offset));
                block.stream.read(reinterpret_cast<char*>(read_buffer.data()),
                                  static_cast<std::streamsize>(read_buffer.size() * sizeof(fki::Complex)));
                if (!block.stream) throw std::runtime_error("receiver scratch read failed");
                bytes_read_ += read_buffer.size() * sizeof(fki::Complex);
            }
            for (std::size_t ld = 0; ld < block.depths.size(); ++ld) {
                const int iz = block.depths[ld];
                const double weight = block.weights[ld];
                for (int ix = 0; ix < nx_; ++ix) {
                    const fki::Complex value = read_buffer[ld * static_cast<std::size_t>(nx_) + ix];
                    output[static_cast<Eigen::Index>(model.index(ix, iz))] +=
                        weight * gpg::Complex(value.real(), value.imag());
                }
            }
        }
        return output;
    }

    double written_megabytes() const { return bytes_written_ / 1048576.0; }
    double read_megabytes() const { return bytes_read_ / 1048576.0; }
    bool memory_backed() const { return memory_backed_; }

private:
    struct Block {
        std::vector<int> depths;
        std::vector<float> weights;
        std::size_t values_per_record = 0;
        std::filesystem::path path;
        std::fstream stream;
        std::vector<fki::Complex> memory;
    };
    std::filesystem::path directory_;
    std::size_t shots_ = 0, frequencies_ = 0;
    int nx_ = 0;
    std::size_t grid_size_ = 0;
    std::vector<Block> blocks_;
    bool memory_backed_ = false;
    bool writes_finalized_ = false;
    std::mutex read_mutex_;
    std::uint64_t bytes_written_ = 0, bytes_read_ = 0;
};

} // namespace

namespace kirch::imaging {

struct GlobalBfImagingWorkflow::Impl {
    std::size_t shots = 0;
    std::size_t frequencies = 0;
    int frequency_threads = 1;
    std::function<void()> calculate_receivers;
    std::function<void(std::size_t)> begin_frequency;
    std::function<void(std::size_t, std::size_t)> calculate_source;
    std::function<void(std::size_t, std::size_t)> correct;
    std::function<void(std::size_t, std::size_t)> correlate;
    std::function<void(std::size_t)> end_frequency;
    std::function<void()> finish;
};

GlobalBfImagingWorkflow::GlobalBfImagingWorkflow(
    std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GlobalBfImagingWorkflow::GlobalBfImagingWorkflow(
    GlobalBfImagingWorkflow&&) noexcept = default;
GlobalBfImagingWorkflow& GlobalBfImagingWorkflow::operator=(
    GlobalBfImagingWorkflow&&) noexcept = default;
GlobalBfImagingWorkflow::~GlobalBfImagingWorkflow() = default;
std::size_t GlobalBfImagingWorkflow::shot_count() const { return impl_->shots; }
std::size_t GlobalBfImagingWorkflow::frequency_count() const { return impl_->frequencies; }
int GlobalBfImagingWorkflow::frequency_parallelism() const { return impl_->frequency_threads; }
void GlobalBfImagingWorkflow::calculate_receiver_wavefields() { impl_->calculate_receivers(); }
void GlobalBfImagingWorkflow::begin_frequency(std::size_t i) { impl_->begin_frequency(i); }
void GlobalBfImagingWorkflow::calculate_source_wavefield(std::size_t s, std::size_t f) { impl_->calculate_source(s, f); }
void GlobalBfImagingWorkflow::iteratively_correct_wavefields(std::size_t s, std::size_t f) { impl_->correct(s, f); }
void GlobalBfImagingWorkflow::cross_correlate_image(std::size_t s, std::size_t f) { impl_->correlate(s, f); }
void GlobalBfImagingWorkflow::end_frequency(std::size_t i) { impl_->end_frequency(i); }
void GlobalBfImagingWorkflow::process_frequency(std::size_t frequency)
{
    begin_frequency(frequency);
    for (std::size_t shot = 0; shot < shot_count(); ++shot) {
        calculate_source_wavefield(shot, frequency);
        iteratively_correct_wavefields(shot, frequency);
        cross_correlate_image(shot, frequency);
    }
    end_frequency(frequency);
}
void GlobalBfImagingWorkflow::finish() { impl_->finish(); }

int with_global_bf_imaging_workflow(
    int argc, char** argv,
    const std::function<void(GlobalBfImagingWorkflow&)>& orchestration)
{
    try {
        huygens_cli::initialize(argc, argv);
        const fki::CommonOptions options =
            // Keep the legacy program's default image/timing filenames so the
            // external input/output contract is unchanged.  Supplying image=
            // and timing= explicitly continues to work exactly as before.
            fki::parse_common_options("frequency_kirchhoff_imaging_bf_gmres");

        const float bpack_tolerance =
            huygens_cli::optional_float("bpack_tol", 1.0e-4f);
        const int bpack_leaf =
            huygens_cli::optional_int("bpack_leaf", 64);
        const int bpack_verbosity =
            huygens_cli::optional_int("bpack_verbosity", -1);
        const int bpack_depth_chunk =
            huygens_cli::optional_int("bpack_depth_chunk", 20);
        const int bpack_progress =
            huygens_cli::optional_int("bpack_progress", 1);
        const int bpack_progress_every =
            huygens_cli::optional_int("bpack_progress_every", 1);
        const int bpack_lr_level =
            huygens_cli::optional_int("bpack_lrlevel", 100);
        const float bpack_sample_parameter =
            huygens_cli::optional_float("bpack_sample_para", 1.5f);
        const int bpack_forward_n15 =
            huygens_cli::optional_int("bpack_forward_n15", 0);
        const int bpack_knn =
            huygens_cli::optional_int("bpack_knn", 10);
        const int bpack_pat_comp =
            huygens_cli::optional_int("bpack_pat_comp", 1);
        const int bpack_less_adapt =
            huygens_cli::optional_int("bpack_less_adapt", 1);
        const float bpack_rdetect_factor =
            huygens_cli::optional_float("bpack_rdetect_factor", 0.3f);
        const int bpack_reuse_tree =
            huygens_cli::optional_int("bpack_reuse_tree", 1);
        const float bpack_geometry_cache_mb =
            huygens_cli::optional_float("bpack_geometry_cache_mb", 1024.0f);
        const int gmres_enable =
            huygens_cli::optional_int("gmres_enable", 1);
        const int gmres_frequency_threads = huygens_cli::optional_int(
            "gmres_frequency_threads", 0);
        const std::string correction_csv = huygens_cli::optional_string(
            "correction_csv", "global_gmres_metrics.csv");

        const int gmres_outer = huygens_cli::optional_int(
            "global_gmres_iterations", gpg::kDefaultOuter);
        const int gmres_restart = huygens_cli::optional_int(
            "gmres_restart", gpg::kDefaultRestart);
        const float global_storage_max_mb = huygens_cli::optional_float(
            "global_storage_max_mb", 8192.0f);
        const std::string receiver_store = huygens_cli::optional_string(
            "receiver_store", "auto");
        const std::string receiver_store_dir = huygens_cli::optional_string(
            "receiver_store_dir", ".kirch_receiver_scratch");
        const std::string per_shot_image_dir = huygens_cli::optional_string(
            "per_shot_image_dir", "");
        const float per_shot_image_max_mb = huygens_cli::optional_float(
            "per_shot_image_max_mb", 4096.0f);
        const float global_source_z = huygens_cli::optional_float(
            "global_source_z", 0.0f);
        const float global_source_correction_z0 = huygens_cli::optional_float(
            "global_source_correction_z0", 0.105f);

        // Scheme-A seam controls.  seam_enable=0 reproduces the original
        // wavefield path (apart from the partition weights, which are also
        // disabled in that case).
        const int seam_enable = huygens_cli::optional_int("seam_enable", 1);
        const float seam_amp_rel_floor = huygens_cli::optional_float(
            "seam_amp_rel_floor", 1.0e-3f);
        const float seam_scale_cap = huygens_cli::optional_float(
            "seam_scale_cap", 2.0f);
        const float seam_cache_max_mb = huygens_cli::optional_float(
            "seam_cache_max_mb", 4096.0f);
        const std::string seam_csv = huygens_cli::optional_string(
            "seam_csv", "seam_alignment_metrics.csv");

        if (!(bpack_tolerance > 0.0f) || !std::isfinite(bpack_tolerance) ||
            bpack_leaf < 4 || bpack_depth_chunk < 1 ||
            (bpack_progress != 0 && bpack_progress != 1) ||
            bpack_progress_every < 1 || bpack_lr_level < 0 ||
            !(bpack_sample_parameter > 0.0f) ||
            !std::isfinite(bpack_sample_parameter) ||
            bpack_forward_n15 < 0 || bpack_forward_n15 > 2 ||
            bpack_knn < 0 ||
            bpack_pat_comp < 1 || bpack_pat_comp > 3 ||
            (bpack_less_adapt != 0 && bpack_less_adapt != 1) ||
            !(bpack_rdetect_factor > 0.0f) ||
            !std::isfinite(bpack_rdetect_factor) ||
            (bpack_reuse_tree != 0 && bpack_reuse_tree != 1) ||
            bpack_geometry_cache_mb < 0.0f ||
            !std::isfinite(bpack_geometry_cache_mb)) {
            throw std::invalid_argument(
                "invalid rectangular ButterflyPACK parameters");
        }
        if (gmres_enable != 0 && gmres_enable != 1) {
            throw std::invalid_argument("gmres_enable must be 0 or 1");
        }
        if (gmres_frequency_threads < 0)
            throw std::invalid_argument("gmres_frequency_threads must be non-negative");
        if (gmres_outer < 1 || gmres_restart < 1) {
            throw std::invalid_argument(
                "global_gmres_iterations and gmres_restart must be positive");
        }
        if ((seam_enable != 0 && seam_enable != 1) ||
            seam_amp_rel_floor < 0.0f ||
            !(seam_scale_cap >= 1.0f) ||
            !(seam_cache_max_mb > 0.0f)) {
            throw std::invalid_argument("invalid Scheme-A seam parameters");
        }

        huygens_cli::set_openmp_threads(options.threads);

        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(
                options.velocity);
        const int global_source_iz = std::clamp(
            static_cast<int>(std::llround((global_source_z - model.oz) / model.dz)),
            0, model.nz - 1);
        const int global_correction_iz = std::clamp(
            static_cast<int>(std::llround(
                (global_source_correction_z0 - model.oz) / model.dz)),
            global_source_iz, model.nz - 1);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(
                options.block_file);
        se::huygens::validate_block_info(info, &model);
        if (seam_enable != 0 && info.overlap_rows <= 0) {
            throw std::invalid_argument(
                "seam_enable=1 requires block_info overlap_rows > 0");
        }

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
            kMethodDescription);

        std::cout
            << "Global GMRES         : "
            << (gmres_enable != 0 ? "enabled" : "disabled") << '\n'
            << "Global iterations    : " << gmres_outer << '\n'
            << "Frequency threads    : "
            << (gmres_frequency_threads > 0
                    ? gmres_frequency_threads
                    : std::min(options.threads > 0 ? options.threads : 4, 4)) << '\n'
            << "GMRES restart        : " << gmres_restart << '\n'
            << "Source/correction z  : " << model.z(global_source_iz) << "/"
            << model.z(global_correction_iz) << '\n'
            << "GMRES preconditioner : none (reference restarted GMRES)\n";

        std::cout
            << "Butterfly backend    : rectangular ButterflyPACK (one operator/frequency/block)\n"
            << "BPACK parameters     : tol=" << bpack_tolerance
            << ", leaf=" << bpack_leaf
            << ", depth_chunk=" << bpack_depth_chunk
            << ", verbosity=" << bpack_verbosity << '\n'
            << "BPACK representation : LRlevel=" << bpack_lr_level
            << ", sample_para=" << bpack_sample_parameter
            << ", forwardN15flag=" << bpack_forward_n15
            << ", knn=" << bpack_knn
            << (bpack_lr_level >= 1
                    ? " (BUTTERFLY mode)\n"
                    : " (LOW-RANK mode)\n")
            << "BPACK construction   : pat_comp=" << bpack_pat_comp
            << ", less_adapt=" << bpack_less_adapt
            << ", rdetect_factor=" << bpack_rdetect_factor
            << ", tree_reuse=" << bpack_reuse_tree
            << ", geometry_cache_cap=" << bpack_geometry_cache_mb << " MB\n"
            << "BPACK progress       : enable=" << bpack_progress
            << ", every=" << bpack_progress_every << " operator(s)\n"
            << "BPACK RHS strategy   : apply_many(receiver only for all selected shots)\n"
            << "Scheme-A seam        : enable=" << seam_enable
            << ", amp_floor=" << seam_amp_rel_floor
            << ", scale_cap=" << seam_scale_cap
            << ", cache_cap=" << seam_cache_max_mb << " MB\n";

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
        const bool write_per_shot_images = !per_shot_image_dir.empty();
        const double per_shot_required_mb = write_per_shot_images
            ? 2.0 * states.size() * grid_size * sizeof(float) / 1048576.0 : 0.0;
        if (per_shot_required_mb > per_shot_image_max_mb) {
            throw std::runtime_error("per-shot before/after images require " +
                std::to_string(per_shot_required_mb) +
                " MiB, exceeding per_shot_image_max_mb=" +
                std::to_string(per_shot_image_max_mb));
        }
        std::vector<std::vector<float>> per_shot_before, per_shot_after;
        if (write_per_shot_images) {
            std::filesystem::create_directories(per_shot_image_dir);
            per_shot_before.assign(states.size(), std::vector<float>(grid_size, 0.0f));
            per_shot_after.assign(states.size(), std::vector<float>(grid_size, 0.0f));
            std::cout << "Per-shot images       : " << per_shot_image_dir
                      << " (before + after correction, " << per_shot_required_mb
                      << " MiB accumulation buffers)\n";
        }

        // In the global-correction executable the source branch is NOT
        // propagated by Kirchhoff.  Keep the shared ButterflyPACK apply-many
        // layout intact by making the legacy source datum identically zero;
        // the actual source initial field is built later from FMM traveltime.
        for (auto& state : states) {
            std::fill(state.source.begin(), state.source.end(),
                      fki::Complex(0.0f, 0.0f));
        }

        // Disk tiles retain the true overlap rows for every propagation block.
        // Include those rows in the auto-backend estimate so memory mode never
        // exceeds its configured limit merely because blocks overlap.
        const std::size_t receiver_storage_depths = static_cast<std::size_t>(model.nz) +
            static_cast<std::size_t>(std::max(info.overlap_rows, 0)) *
            (info.blocks.empty() ? 0 : info.blocks.size() - 1);
        const double former_global_receiver_mb =
            static_cast<double>(states.size()) * axis.frequencies.size() *
            model.nx * receiver_storage_depths * sizeof(fki::Complex) / 1048576.0;
        ReceiverBlockStore receiver_store_backend(
            receiver_store_dir, receiver_store, former_global_receiver_mb,
            global_storage_max_mb, states.size(), axis.frequencies.size(), model);
        std::cout << "Receiver storage      : "
                  << (receiver_store_backend.memory_backed() ? "memory" : "disk-backed block tiles")
                  << (receiver_store_backend.memory_backed() ? "" : " at " + receiver_store_dir)
                  << " (estimated full field " << former_global_receiver_mb << " MiB)\n";

        const int metrics = 12;
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

        std::ofstream seam_stream(seam_csv);
        if (!seam_stream) {
            throw std::runtime_error("cannot open seam_csv: " + seam_csv);
        }
        seam_stream << std::setprecision(17)
            << "shot,block,frequency,field,overlap_rows,samples,"
               "alpha_real,alpha_imag,alpha_abs,alpha_phase_rad\n";

        // Precompute each propagation block's target depths before any field
        // calculation.  This lets the imaging weights form an exact partition
        // of unity across the true geometric overlaps rather than assuming an
        // overlap from block-file row counts alone.
        std::vector<std::vector<int>> block_target_depths(info.blocks.size());
        for (std::size_t ib = 0; ib < info.blocks.size(); ++ib) {
            const se::huygens::Block preview_block =
                scheme_a_propagation_block(
                    info, ib, seam_enable != 0);
            const se::huygens::LayerGeometry preview_geometry =
                se::huygens::make_layer_geometry(
                    model, preview_block, options.source_stride,
                    options.target_x_stride, options.target_z_stride);
            block_target_depths[ib] = preview_geometry.target_iz;
        }
        std::vector<std::vector<float>> image_partition_weights(
            block_target_depths.size());
        if (seam_enable != 0) {
            image_partition_weights = build_image_partition_weights(
                block_target_depths, model.nz);
            for (std::size_t ib = 0; ib + 1 < block_target_depths.size(); ++ib) {
                const std::vector<int> common = intersect_depths(
                    block_target_depths[ib], block_target_depths[ib + 1]);
                const std::size_t expected =
                    static_cast<std::size_t>(info.overlap_rows);
                if (common.size() != expected) {
                    throw std::runtime_error(
                        "Scheme A computational geometry produced " +
                        std::to_string(common.size()) +
                        " shared target rows between propagation blocks " +
                        std::to_string(ib) + " and " +
                        std::to_string(ib + 1) + "; expected " +
                        std::to_string(expected));
                }
            }
        } else {
            for (std::size_t ib = 0; ib < block_target_depths.size(); ++ib) {
                image_partition_weights[ib].assign(
                    block_target_depths[ib].size(), 1.0f);
            }
        }

        OverlapCache previous_overlap;

        double total_filter_seconds = 0.0;
        double total_build_seconds = 0.0;
        double total_apply_seconds = 0.0;
        double total_bpack_library_build_seconds = 0.0;
        double total_propagation_seconds = 0.0;

        auto calculate_receiver_wavefields = [&]() {
        for (std::size_t iblock = 0;
             iblock < info.blocks.size();
             ++iblock) {
            const se::huygens::Block block =
                scheme_a_propagation_block(
                    info, iblock, seam_enable != 0);
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(
                    model, block,
                    options.source_stride,
                    options.target_x_stride,
                    options.target_z_stride);
            const bool uses_butterfly = iblock != 0;
            se::huygens::OneWayLayerTables tables;
            if (uses_butterfly) {
                tables = se::huygens::read_one_way_layer_tables(
                    options.table_prefix, block.id);
                try {
                    se::huygens::validate_one_way_layer_tables(
                        tables, geometry, block);
                } catch (const std::exception& error) {
                    if (seam_enable != 0 && info.overlap_rows > 0) {
                        throw std::runtime_error(
                            "Scheme A requires traveltime tables that include "
                            "the propagation-overlap rows. Regenerate them "
                            "with travel_time_solver "
                            "include_propagation_overlap=1. Original validation "
                            "error: " + std::string(error.what()));
                    }
                    throw;
                }
            }

            if (static_cast<int>(
                    geometry.source_ix.size()) != model.nx ||
                static_cast<int>(
                    geometry.target_ix.size()) != model.nx) {
                throw std::runtime_error(
                    "rectangular ButterflyPACK imaging requires "
                    "every lateral grid point");
            }

            const int target_depth_count =
                static_cast<int>(
                    geometry.target_iz.size());
            double first_block_mean_velocity = 0.0;
            if (!uses_butterfly) {
                std::size_t velocity_count = 0;
                for (const int iz : geometry.target_iz) {
                    for (int ix = 0; ix < model.nx; ++ix) {
                        first_block_mean_velocity +=
                            model.velocity[model.index(ix, iz)];
                        ++velocity_count;
                    }
                }
                first_block_mean_velocity /=
                    std::max<std::size_t>(velocity_count, 1);
            }

            if (geometry.target_iz != block_target_depths[iblock]) {
                throw std::runtime_error(
                    "precomputed block target depths changed unexpectedly");
            }

            receiver_store_backend.begin_block(
                geometry.target_iz, image_partition_weights[iblock]);

            OverlapCache next_overlap;
            if (seam_enable != 0) {
                if (iblock > 0) {
                    const std::vector<int> expected_previous = intersect_depths(
                        block_target_depths[iblock - 1],
                        block_target_depths[iblock]);
                    if (previous_overlap.depths != expected_previous) {
                        throw std::runtime_error(
                            "previous overlap cache does not match current block geometry");
                    }
                }
                if (iblock + 1 < info.blocks.size()) {
                    const std::vector<int> next_depths = intersect_depths(
                        block_target_depths[iblock],
                        block_target_depths[iblock + 1]);
                    next_overlap = make_overlap_cache(
                        next_depths, model.nx, states.size(),
                        axis.frequencies.size());
                    const double cache_mb = overlap_cache_mb(next_overlap);
                    if (cache_mb > seam_cache_max_mb) {
                        throw std::runtime_error(
                            "Scheme-A next-overlap cache requires " +
                            std::to_string(cache_mb) +
                            " MB, exceeding seam_cache_max_mb=" +
                            std::to_string(seam_cache_max_mb));
                    }
                }
            }

            std::cout << "  computational target=["
                      << geometry.target_iz.front() << ','
                      << geometry.target_iz.back() << ']'
                      << ", official target=["
                      << info.blocks[iblock].target_start_iz << ','
                      << info.blocks[iblock].target_end_iz << ']'
                      << ", overlap with previous="
                      << previous_overlap.depths.size()
                      << " rows, overlap with next="
                      << next_overlap.depths.size()
                      << " rows";
            if (!next_overlap.empty()) {
                std::cout << ", next overlap cache="
                          << overlap_cache_mb(next_overlap) << " MB";
            }
            std::cout << '\n';

            // The first block keeps the existing homogeneous phase-shift launch.
            // Later blocks use rectangular ButterflyPACK, but a very tall
            // nx*nz_block by nx operator is expensive to compress. Split the
            // complete block into shallow 2-D depth strips. Each strip still
            // maps the same 1-D propagation datum to multiple target depths.
            const std::size_t bpack_columns = geometry.source_ix.size();
            if (bpack_columns >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    "rectangular ButterflyPACK column count exceeds 32-bit API limits");
            }

            const float maximum_tau = uses_butterfly
                ? *std::max_element(tables.traveltime.values.begin(),
                                    tables.traveltime.values.end())
                : 0.0f;
            const float filter_dt = options.filter_dt > 0.0f
                ? options.filter_dt
                : data.dt();

            std::vector<double> column_coordinates;
            std::vector<BpackDepthChunk> bpack_chunks;
            if (uses_butterfly) {
                // OneWayKernelData has exactly one [x,z] column coordinate per
                // datum sample. The Cauchy helper is intentionally not used.
                column_coordinates.resize(2 * geometry.source_ix.size());
                const double datum_z =
                    static_cast<double>(model.oz) +
                    static_cast<double>(block.source_iz) * model.dz;
                for (std::size_t isource = 0;
                     isource < geometry.source_ix.size(); ++isource) {
                    column_coordinates[2 * isource] =
                        static_cast<double>(model.ox) +
                        static_cast<double>(geometry.source_ix[isource]) * model.dx;
                    column_coordinates[2 * isource + 1] = datum_z;
                }
                bpack_chunks = make_bpack_depth_chunks(
                    model, geometry, bpack_depth_chunk);
            }

            const std::size_t bpack_chunks_per_frequency = bpack_chunks.size();
            const std::size_t bpack_operator_count = uses_butterfly
                ? axis.frequencies.size() * bpack_chunks_per_frequency
                : 0;
            const std::size_t bpack_max_rows = uses_butterfly
                ? static_cast<std::size_t>(model.nx) *
                      std::min(bpack_depth_chunk, target_depth_count)
                : static_cast<std::size_t>(target_depth_count) * model.nx;

            std::cout
                << "BPACK block " << block.id
                << ": operators=" << bpack_operator_count
                << ", chunks/frequency=" << bpack_chunks_per_frequency
                << ", max matrix=" << bpack_max_rows << 'x' << bpack_columns
                << ", depth_chunk="
                << (uses_butterfly
                        ? std::min(bpack_depth_chunk, target_depth_count)
                        : target_depth_count)
                << ", RHS/operator="
                << (uses_butterfly ? states.size() : 0)
                << ", receiver tile=1freq x " << target_depth_count << "depth"
                << (uses_butterfly
                        ? ""
                        : ", phase-shift block (BPACK build skipped)")
                << std::endl;

            double block_filter_seconds = 0.0;
            double block_build_seconds = 0.0;
            double block_bpack_apply_seconds = 0.0;
            double block_post_seconds = 0.0;
            double block_apply_seconds = 0.0;
            double block_library_build_seconds = 0.0;
            double block_compressed_mb = 0.0;
            double block_peak_mb = 0.0;
            double block_max_rank = 0.0;
            std::size_t block_sampled_entries = 0;
            std::vector<float> bpack_geometry_cache;
            std::size_t bpack_geometry_rows = 0;
            double bpack_geometry_cache_seconds = 0.0;

            if (uses_butterfly && bpack_geometry_cache_mb > 0.0f) {
                bpack_geometry_rows = geometry.targets.size();
                if (bpack_columns != 0 &&
                    bpack_geometry_rows >
                        std::numeric_limits<std::size_t>::max() / bpack_columns) {
                    throw std::overflow_error(
                        "ButterflyPACK geometry cache size overflows size_t");
                }
                const std::size_t cache_entries =
                    bpack_geometry_rows * bpack_columns;
                const long double required_mb =
                    static_cast<long double>(cache_entries) * sizeof(float) /
                    (1024.0L * 1024.0L);
                if (required_mb <=
                    static_cast<long double>(bpack_geometry_cache_mb)) {
                    bpack_geometry_cache.resize(cache_entries);
                    se::huygens::OneWayKernelData geometry_kernel;
                    geometry_kernel.rows =
                        static_cast<int>(geometry.targets.size());
                    geometry_kernel.sources =
                        static_cast<int>(bpack_columns);
                    geometry_kernel.source_z = model.z(block.source_iz);
                    geometry_kernel.quadrature_weights = &model_weights;
                    geometry_kernel.tables = &tables;

                    const auto cache_started = fki::Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                    for (int source = 0;
                         source < geometry_kernel.sources; ++source) {
                        const std::size_t source_offset =
                            static_cast<std::size_t>(source) *
                            bpack_geometry_rows;
                        for (int row = 0; row < geometry_kernel.rows; ++row) {
                            const float tau =
                                geometry_kernel.traveltime(row, source);
                            bpack_geometry_cache[
                                source_offset +
                                static_cast<std::size_t>(row)] =
                                geometry_kernel.geometry_factor(
                                    row, source, tau);
                        }
                    }
                    bpack_geometry_cache_seconds =
                        std::chrono::duration<double>(
                            fki::Clock::now() - cache_started).count();
                    std::cout
                        << "  BPACK geometry cache: enabled, "
                        << static_cast<double>(required_mb) << " MiB"
                        << ", build=" << bpack_geometry_cache_seconds << " s"
                        << ", reused across " << axis.frequencies.size()
                        << " frequencies\n";
                } else {
                    std::cout
                        << "  BPACK geometry cache: disabled, requires "
                        << static_cast<double>(required_mb) << " MiB > cap "
                        << bpack_geometry_cache_mb << " MiB\n";
                }
            }

            for (std::size_t ifrequency = 0;
                 ifrequency < axis.frequencies.size(); ++ifrequency) {
                const float frequency = axis.frequencies[ifrequency];

                std::vector<std::vector<fki::Complex>> propagated_depth_major;
                if (uses_butterfly) {
                    propagated_depth_major.assign(
                        states.size(),
                        std::vector<fki::Complex>(
                            static_cast<std::size_t>(target_depth_count) * model.nx,
                            fki::Complex(0.0f, 0.0f)));

                    const auto filter_started = fki::Clock::now();
                    const se::huygens::FrequencyKirchhoffFilter filter(
                        frequency, filter_dt, options.filter_length,
                        maximum_tau, options.filter_lookup_subsamples);
                    block_filter_seconds += std::chrono::duration<double>(
                        fki::Clock::now() - filter_started).count();

                    se::huygens::OneWayKernelData kernel;
                    kernel.rows = static_cast<int>(geometry.targets.size());
                    kernel.sources = static_cast<int>(bpack_columns);
                    kernel.source_z = model.z(block.source_iz);
                    kernel.quadrature_weights = &model_weights;
                    kernel.tables = &tables;
                    kernel.filter = &filter;

                    se::butterfly::Options bpack_options;
                    bpack_options.tolerance = bpack_tolerance;
                    bpack_options.leaf_size = bpack_leaf;
                    bpack_options.verbosity = bpack_verbosity;
                    bpack_options.coordinate_dimension = 2;
                    bpack_options.lr_level = bpack_lr_level;
                    bpack_options.sample_parameter = bpack_sample_parameter;
                    bpack_options.forward_n15_flag = bpack_forward_n15;
                    bpack_options.nearest_neighbors = bpack_knn;
                    bpack_options.compression_pattern = bpack_pat_comp;
                    bpack_options.less_adapt = bpack_less_adapt;
                    bpack_options.rank_detection_factor = bpack_rdetect_factor;
                    bpack_options.reuse_geometry = bpack_reuse_tree;

                    std::vector<std::vector<fki::Complex>> rhs_inputs;
                    rhs_inputs.reserve(states.size());
                    for (std::size_t ishot = 0; ishot < states.size(); ++ishot) {
                        rhs_inputs.push_back(boundary_frequency(
                            states[ishot].receiver_conjugate,
                            ifrequency, model.nx));
                    }

#ifndef KIRCH_HAS_BUTTERFLYPACK_FLOAT
                    throw std::runtime_error(
                        "rectangular frequency_kirchhoff_imaging_bf_gmres requires "
                        "KIRCH_ENABLE_BUTTERFLYPACK=ON and KIRCH_BPACK_ENABLE_FLOAT=ON");
#else
                    for (std::size_t ichunk = 0;
                         ichunk < bpack_chunks.size(); ++ichunk) {
                        const BpackDepthChunk& chunk = bpack_chunks[ichunk];
                        const std::size_t operator_index =
                            ifrequency * bpack_chunks.size() + ichunk + 1;
                        const std::size_t chunk_rows = chunk.full_rows.size();
                        const int first_depth = geometry.target_iz[
                            static_cast<std::size_t>(chunk.depth_begin)];
                        const int last_depth = geometry.target_iz[
                            static_cast<std::size_t>(
                                chunk.depth_begin + chunk.depth_count - 1)];
                        const bool print_progress = bpack_progress != 0 &&
                            (operator_index == 1 ||
                             operator_index == bpack_operator_count ||
                             ((operator_index - 1) %
                                  static_cast<std::size_t>(bpack_progress_every) == 0));

                        if (print_progress) {
                            std::cout
                                << "    [BPACK build START] block=" << block.id
                                << " op=" << operator_index << '/'
                                << bpack_operator_count
                                << " freq=" << (ifrequency + 1) << '/'
                                << axis.frequencies.size()
                                << " f=" << frequency << " Hz"
                                << " chunk=" << (ichunk + 1) << '/'
                                << bpack_chunks.size()
                                << " depths=[" << first_depth << ',' << last_depth << ']'
                                << " matrix=" << chunk_rows << 'x' << bpack_columns
                                << " cum_build=" << block_build_seconds << " s"
                                << " cum_apply=" << block_bpack_apply_seconds << " s"
                                << std::endl;
                        }

                        const auto build_started = fki::Clock::now();
                        auto butterfly = std::make_unique<
                            se::butterfly::Matrix<float>>(
                                static_cast<int>(chunk_rows),
                                static_cast<int>(bpack_columns),
                                chunk.row_coordinates,
                                column_coordinates,
                                [&kernel, &chunk,
                                 &bpack_geometry_cache,
                                 bpack_geometry_rows,
                                 &filter](int row, int source) {
                                    if (row < 0 ||
                                        static_cast<std::size_t>(row) >=
                                            chunk.full_rows.size()) {
                                        throw std::out_of_range(
                                            "ButterflyPACK chunk callback row is invalid");
                                    }
                                    const int full_row =
                                        chunk.full_rows[
                                            static_cast<std::size_t>(row)];
                                    if (!bpack_geometry_cache.empty()) {
                                        const float tau =
                                            kernel.traveltime(full_row, source);
                                        const std::size_t cache_index =
                                            static_cast<std::size_t>(source) *
                                                bpack_geometry_rows +
                                            static_cast<std::size_t>(full_row);
                                        return bpack_geometry_cache[cache_index] *
                                               filter.response(tau);
                                    }
                                    return kernel.entry(full_row, source);
                                },
                                bpack_options);
                        const double build_wall = std::chrono::duration<double>(
                            fki::Clock::now() - build_started).count();
                        block_build_seconds += build_wall;
                        if (print_progress) {
                            std::cout
                                << "    [BPACK build DONE ] block=" << block.id
                                << " op=" << operator_index << '/'
                                << bpack_operator_count
                                << " build=" << build_wall << " s"
                                << " built=" << operator_index << '/'
                                << bpack_operator_count
                                << " cum_build=" << block_build_seconds << " s"
                                << std::endl;
                        }

                        const auto bpack_apply_started = fki::Clock::now();
                        const std::vector<std::vector<fki::Complex>> chunk_rows_out =
                            butterfly->apply_many(rhs_inputs);
                        const double apply_wall = std::chrono::duration<double>(
                            fki::Clock::now() - bpack_apply_started).count();
                        block_bpack_apply_seconds += apply_wall;

                        const se::butterfly::Statistics bpack_stats =
                            butterfly->statistics();
                        butterfly.reset();

                        block_library_build_seconds += bpack_stats.build_seconds;
                        block_compressed_mb += bpack_stats.compressed_megabytes;
                        block_peak_mb = std::max(
                            block_peak_mb, bpack_stats.peak_megabytes);
                        block_max_rank = std::max(
                            block_max_rank, bpack_stats.maximum_rank);
                        block_sampled_entries += bpack_stats.sampled_entries;

                        if (chunk_rows_out.size() != states.size()) {
                            throw std::runtime_error(
                                "ButterflyPACK apply_many returned an unexpected RHS count");
                        }
                        for (std::size_t irhs = 0;
                             irhs < chunk_rows_out.size(); ++irhs) {
                            scatter_bpack_chunk_output(
                                chunk_rows_out[irhs], chunk,
                                propagated_depth_major[irhs]);
                        }

                        if (print_progress) {
                            std::cout
                                << "    [BPACK apply DONE] block=" << block.id
                                << " op=" << operator_index << '/'
                                << bpack_operator_count
                                << " apply=" << apply_wall << " s"
                                << " applied=" << operator_index << '/'
                                << bpack_operator_count
                                << " cum_build=" << block_build_seconds << " s"
                                << " cum_apply=" << block_bpack_apply_seconds << " s"
                                << " sampled=" << bpack_stats.sampled_entries
                                << " compressed=" << bpack_stats.compressed_megabytes
                                << " MiB"
                                << " rank_max=" << bpack_stats.maximum_rank
                                << " setup=" << bpack_stats.setup_seconds << " s"
                                << " tree="
                                << (bpack_stats.row_tree_seconds +
                                    bpack_stats.column_tree_seconds)
                                << " s"
                                << " structure=" << bpack_stats.structure_seconds
                                << " s"
                                << " compression=" << bpack_stats.compression_seconds
                                << " s"
                                << " entry=" << bpack_stats.library_entry_seconds
                                << " s"
                                << " entry_bf="
                                << bpack_stats.library_entry_butterfly_seconds
                                << " s"
                                << " tree_cache="
                                << (bpack_stats.geometry_reused != 0
                                        ? "hit"
                                        : "miss")
                                << std::endl;
                        }
                    }
#endif
                }

                const auto post_started = fki::Clock::now();
                for (std::size_t ishot = 0; ishot < states.size(); ++ishot) {
                    // Global imaging constructs the source field later from
                    // eFMM traveltime.  Recursive Kirchhoff propagation therefore
                    // carries only the receiver field; the zero source buffer is
                    // retained solely for the shared overlap/boundary container.
                    std::vector<fki::Complex> source_output(
                        static_cast<std::size_t>(target_depth_count) * model.nx,
                        fki::Complex(0.0f, 0.0f));
                    // The global workflow never images this local receiver tile
                    // before overlap alignment: it stores the aligned field and
                    // performs imaging only after the complete field is assembled.
                    // Build the propagation buffer directly and avoid a full-tile copy.
                    std::vector<fki::Complex> receiver_for_propagation;
                    if (!uses_butterfly) {
                        phase_shift_first_block(
                            model, block, geometry.target_iz, frequency,
                            first_block_mean_velocity,
                            states[ishot].receiver_conjugate, ifrequency,
                            receiver_for_propagation);
                    } else {
                        receiver_for_propagation =
                            std::move(propagated_depth_major[ishot]);
                    }
                    std::vector<fki::Complex> source_for_propagation =
                        source_output;

                    if (seam_enable != 0 && iblock > 0 &&
                        !previous_overlap.empty()) {
                        std::complex<double> source_alpha(1.0, 0.0);
                        std::complex<double> receiver_alpha(1.0, 0.0);
                        std::size_t source_samples = 0;
                        std::size_t receiver_samples = 0;

                        blend_one_overlap_field(
                            previous_overlap, ishot, ifrequency,
                            geometry.target_iz,
                            source_for_propagation, true,
                            seam_amp_rel_floor, seam_scale_cap,
                            &source_alpha, &source_samples);
                        blend_one_overlap_field(
                            previous_overlap, ishot, ifrequency,
                            geometry.target_iz,
                            receiver_for_propagation, false,
                            seam_amp_rel_floor, seam_scale_cap,
                            &receiver_alpha, &receiver_samples);

                        seam_stream
                            << shots[ishot] << ',' << block.id << ','
                            << frequency << ",source,"
                            << previous_overlap.depths.size() << ','
                            << source_samples << ','
                            << source_alpha.real() << ','
                            << source_alpha.imag() << ','
                            << std::abs(source_alpha) << ','
                            << std::arg(source_alpha) << '\n'
                            << shots[ishot] << ',' << block.id << ','
                            << frequency << ",receiver,"
                            << previous_overlap.depths.size() << ','
                            << receiver_samples << ','
                            << receiver_alpha.real() << ','
                            << receiver_alpha.imag() << ','
                            << std::abs(receiver_alpha) << ','
                            << std::arg(receiver_alpha) << '\n';
                    }

                    /*
                     * Cache and transmit only the overlap-consistent
                     * propagation copy.  This keeps the recursive datum
                     * continuous without altering the raw field used by the
                     * imaging condition.
                     */
                    if (seam_enable != 0 && !next_overlap.empty()) {
                        store_next_overlap(
                            geometry.target_iz,
                            source_for_propagation,
                            receiver_for_propagation,
                            ishot,
                            ifrequency,
                            next_overlap);
                    }

                    // Global mode postpones the imaging condition until the
                    // complete receiver Kirchhoff field has been assembled and
                    // globally corrected.  Use the overlap-consistent
                    // propagation copy and the same partition-of-unity weights
                    // to build one full receiver field per shot/frequency.
                    receiver_store_backend.write(
                        ishot, ifrequency, receiver_for_propagation);

                    if (iblock + 1 < info.blocks.size()) {
                        extract_next_boundary_batch(
                            model,
                            ifrequency,
                            1,
                            geometry.target_iz,
                            0,
                            target_depth_count,
                            info.blocks[iblock + 1].source_iz,
                            source_for_propagation,
                            receiver_for_propagation,
                            states[ishot]);
                    }
                }
                const double post_wall = std::chrono::duration<double>(
                    fki::Clock::now() - post_started).count();
                block_post_seconds += post_wall;
                block_apply_seconds =
                    block_bpack_apply_seconds + block_post_seconds;
                if (bpack_progress != 0 && uses_butterfly &&
                    (((ifrequency + 1) %
                          static_cast<std::size_t>(bpack_progress_every)) == 0 ||
                     ifrequency + 1 == axis.frequencies.size())) {
                    std::cout
                        << "    [frequency done] block=" << block.id
                        << " freq=" << (ifrequency + 1) << '/'
                        << axis.frequencies.size()
                        << " f=" << frequency << " Hz"
                        << " BPACK built="
                        << ((ifrequency + 1) * bpack_chunks_per_frequency)
                        << '/' << bpack_operator_count
                        << " cum_build=" << block_build_seconds << " s"
                        << " cum_bpack_apply=" << block_bpack_apply_seconds << " s"
                        << " cum_GMRES/image=" << block_post_seconds << " s"
                        << std::endl;
                }
            }

            // The next block needs only the immediately preceding overlap.
            // Releasing the old cache here keeps the additional memory bounded
            // by at most two neighboring overlap caches while a block runs.
            if (seam_enable != 0) {
                previous_overlap = std::move(next_overlap);
            } else {
                previous_overlap = OverlapCache{};
            }

            total_filter_seconds +=
                block_filter_seconds;
            total_build_seconds +=
                block_build_seconds;
            total_apply_seconds +=
                block_apply_seconds;
            total_bpack_library_build_seconds +=
                block_library_build_seconds;
            total_propagation_seconds += block_bpack_apply_seconds;

            const int column = static_cast<int>(iblock);
            timing_values[static_cast<std::size_t>(column) * metrics + 0] =
                static_cast<float>(block_filter_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 1] =
                static_cast<float>(block_build_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 2] =
                static_cast<float>(block_apply_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 3] =
                static_cast<float>(block_library_build_seconds);
            timing_values[static_cast<std::size_t>(column) * metrics + 4] =
                static_cast<float>(bpack_operator_count);
            timing_values[static_cast<std::size_t>(column) * metrics + 5] =
                static_cast<float>(bpack_max_rows);
            timing_values[static_cast<std::size_t>(column) * metrics + 6] =
                static_cast<float>(bpack_columns);
            timing_values[static_cast<std::size_t>(column) * metrics + 7] =
                static_cast<float>(block_sampled_entries);
            timing_values[static_cast<std::size_t>(column) * metrics + 8] =
                static_cast<float>(block_compressed_mb);
            timing_values[static_cast<std::size_t>(column) * metrics + 9] =
                static_cast<float>(block_peak_mb);
            timing_values[static_cast<std::size_t>(column) * metrics + 10] =
                static_cast<float>(block_max_rank);
            timing_values[static_cast<std::size_t>(column) * metrics + 11] =
                static_cast<float>(target_depth_count);

            std::cout
                << "  build=" << block_build_seconds
                << " s (BPACK internal=" << block_library_build_seconds << " s)"
                << ", BPACK apply=" << block_bpack_apply_seconds << " s"
                << ", GMRES/image=" << block_post_seconds << " s"
                << ", apply/GMRES/correlation=" << block_apply_seconds << " s"
                << ", sampled=" << block_sampled_entries
                << ", compressed_sum=" << block_compressed_mb << " MiB"
                << ", peak=" << block_peak_mb << " MiB"
                << ", rank_max=" << block_max_rank
                << '\n';
        }
        receiver_store_backend.finalize_writes();
        };

        // ------------------------------------------------------------------
        // Global correction stage.  Kirchhoff/ButterflyPACK has now supplied
        // one complete receiver-conjugate initial field per shot/frequency.
        // The source initial field is generated independently from eFMM
        // traveltime and is never taken from the Kirchhoff source branch.
        // ------------------------------------------------------------------
        fki::Clock::time_point global_started;
        bool global_started_set = false;
        std::vector<std::vector<float>> source_traveltimes(states.size());
        for (std::size_t ishot = 0; ishot < states.size(); ++ishot) {
            source_traveltimes[ishot] = solve_global_source_traveltime(
                model, states[ishot].shot_x, model.z(global_source_iz));
        }

        const int padded_nz = model.nz + 2 * gpg::kPml;
        const int padded_nx = model.nx + 2 * gpg::kPml;
        std::vector<float> padded_velocity(
            static_cast<std::size_t>(padded_nz) * padded_nx);
        for (int iz = 0; iz < padded_nz; ++iz) {
            const int piz = std::clamp(iz - gpg::kPml, 0, model.nz - 1);
            for (int ix = 0; ix < padded_nx; ++ix) {
                const int pix = std::clamp(ix - gpg::kPml, 0, model.nx - 1);
                padded_velocity[static_cast<std::size_t>(gpg::index(iz, ix, padded_nx))] =
                    model.velocity[model.index(pix, piz)];
            }
        }

        double global_factor_seconds = 0.0;
        double global_solve_seconds = 0.0;
        struct FrequencyState {
            gpg::Sparse source_matrix;
            gpg::Sparse receiver_matrix;
            gpg::Vector source_field;
            gpg::Vector receiver_field;
            std::size_t shot = 0;
        };
        std::vector<FrequencyState> frequency_states(axis.frequencies.size());
        std::mutex metrics_mutex;

        auto begin_frequency = [&](std::size_t ifrequency) {
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (!global_started_set) {
                    global_started = fki::Clock::now();
                    global_started_set = true;
                }
            }
            if (ifrequency >= axis.frequencies.size())
                throw std::out_of_range("frequency index is out of range");
            FrequencyState& active = frequency_states[ifrequency];
            if (gmres_enable != 0) {
                const auto factor_started = fki::Clock::now();
                active.source_matrix = gpg::build_helmholtz(
                    padded_velocity, padded_nz, padded_nx, model.dz, model.dx,
                    axis.frequencies[ifrequency]);
                active.receiver_matrix = gpg::build_helmholtz(
                    padded_velocity, padded_nz, padded_nx, model.dz, model.dx,
                    axis.frequencies[ifrequency], true);
                if (ifrequency == 0) {
                    std::cout << "GMRES padded grid     : " << padded_nz << 'x'
                              << padded_nx << " (PML=" << gpg::kPml << ")\n";
                }
                const double elapsed = std::chrono::duration<double>(
                    fki::Clock::now() - factor_started).count();
                std::lock_guard<std::mutex> lock(metrics_mutex);
                global_factor_seconds += elapsed;
            }
        };

        auto calculate_source_wavefield = [&](std::size_t ishot,
                                               std::size_t ifrequency) {
            if (ishot >= states.size() || ifrequency >= frequency_states.size())
                throw std::out_of_range("shot/frequency stage is out of order");
            FrequencyState& active = frequency_states[ifrequency];
            active.shot = ishot;
            const int source_ix = nearest_global_source_ix(
                model, states[ishot].shot_x);
            active.source_field = build_global_source_aexp(
                model, source_traveltimes[ishot], axis.frequencies[ifrequency],
                source_spectrum[ifrequency], source_ix, global_source_iz);
            active.receiver_field = receiver_store_backend.load(
                ishot, ifrequency, model);
            if (write_per_shot_images) {
                const float weight = axis.imaging_weights[ifrequency];
                for (std::size_t grid = 0; grid < grid_size; ++grid) {
                    const gpg::Complex us = active.source_field[
                        static_cast<Eigen::Index>(grid)];
                    const gpg::Complex ur = active.receiver_field[
                        static_cast<Eigen::Index>(grid)];
                    const float value = weight * static_cast<float>(
                        (us * std::conj(ur)).real());
#pragma omp atomic update
                    per_shot_before[ishot][grid] += value;
                }
            }
        };

        auto iteratively_correct = [&](std::size_t ishot,
                                       std::size_t ifrequency) {
            FrequencyState& active = frequency_states[ifrequency];
            if (ishot != active.shot)
                throw std::logic_error("wavefield correction called out of order");
            if (gmres_enable == 0) return;
            gpg::Result source_metric;
            gpg::Result receiver_metric;
            const gpg::Vector source_before = active.source_field;
            const gpg::Vector receiver_before = active.receiver_field;
            const auto solve_started = fki::Clock::now();
            active.source_field = reference_correct_field(
                active.source_matrix, active.source_field, model.nz, model.nx,
                global_correction_iz, gmres_restart, gmres_outer, &source_metric);
            active.receiver_field = reference_correct_field(
                active.receiver_matrix, active.receiver_field, model.nz, model.nx,
                global_correction_iz, gmres_restart, gmres_outer, &receiver_metric);
            std::lock_guard<std::mutex> lock(metrics_mutex);
            global_solve_seconds += std::chrono::duration<double>(
                fki::Clock::now() - solve_started).count();
            correction_stream
                << shots[ishot] << ",-1," << axis.frequencies[ifrequency]
                << ",source,1," << source_metric.steps << ','
                << source_metric.residual_before << ','
                << source_metric.residual_after << ','
                << (active.source_field - source_before).norm() << ",1\n"
                << shots[ishot] << ",-1," << axis.frequencies[ifrequency]
                << ",receiver,1," << receiver_metric.steps << ','
                << receiver_metric.residual_before << ','
                << receiver_metric.residual_after << ','
                << (active.receiver_field - receiver_before).norm() << ",1\n";
        };

        auto cross_correlate = [&](std::size_t ishot,
                                   std::size_t ifrequency) {
            FrequencyState& active = frequency_states[ifrequency];
            if (ishot != active.shot)
                throw std::logic_error("cross correlation called out of order");
            const float weight = axis.imaging_weights[ifrequency];
            const gpg::Vector& physical_source = active.source_field;
            const gpg::Vector& physical_receiver = active.receiver_field;
            for (std::size_t grid = 0; grid < grid_size; ++grid) {
                const gpg::Complex us = physical_source[
                    static_cast<Eigen::Index>(grid)];
                const gpg::Complex ur = physical_receiver[
                    static_cast<Eigen::Index>(grid)];
                const float after_value = weight * static_cast<float>(
                    (us * std::conj(ur)).real());
#pragma omp atomic update
                image[grid] += after_value;
                const float illumination_value = weight * static_cast<float>(std::norm(us));
#pragma omp atomic update
                illumination[grid] += illumination_value;
                if (write_per_shot_images)
#pragma omp atomic update
                    per_shot_after[ishot][grid] += after_value;
            }
        };

        auto end_frequency = [&](std::size_t ifrequency) {
            if (bpack_progress != 0 &&
                (ifrequency == 0 || ifrequency + 1 == axis.frequencies.size() ||
                 (ifrequency + 1) % static_cast<std::size_t>(
                     std::max(bpack_progress_every, 1)) == 0)) {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                std::cout << "  [global correction] frequency="
                          << (ifrequency + 1) << '/' << axis.frequencies.size()
                          << " f=" << axis.frequencies[ifrequency]
                          << " Hz, iterations="
                          << (gmres_enable != 0 ? gmres_outer : 0)
                          << '\n';
            }
        };

        auto implementation = std::make_unique<GlobalBfImagingWorkflow::Impl>();
        implementation->shots = states.size();
        implementation->frequencies = axis.frequencies.size();
        implementation->frequency_threads = gmres_frequency_threads > 0
            ? gmres_frequency_threads
            : std::min(options.threads > 0 ? options.threads : 4, 4);
        implementation->calculate_receivers = calculate_receiver_wavefields;
        implementation->begin_frequency = begin_frequency;
        implementation->calculate_source = calculate_source_wavefield;
        implementation->correct = iteratively_correct;
        implementation->correlate = cross_correlate;
        implementation->end_frequency = end_frequency;
        implementation->finish = [] {};
        GlobalBfImagingWorkflow workflow(std::move(implementation));
        orchestration(workflow);

        std::cout
            << "Global correction     : "
            << (gmres_enable != 0 ? "enabled" : "disabled") << '\n'
            << "GMRES preconditioner : none\n"
            << "Global factor time   : " << global_factor_seconds << " s\n"
            << "Global solve time    : " << global_solve_seconds << " s\n"
            << "Global stage wall    : "
            << std::chrono::duration<double>(
                   fki::Clock::now() - global_started).count()
            << " s\n";

        fki::finish_image(
            options, model, image, illumination);

        if (write_per_shot_images) {
            for (std::size_t ishot = 0; ishot < states.size(); ++ishot) {
                const std::string prefix = (std::filesystem::path(per_shot_image_dir) /
                    ("shot_" + std::to_string(shots[ishot]))).string();
                se::huygens::write_image_rsf(prefix + "_before_gmres.rsf", model,
                    per_shot_before[ishot], "per_shot_before_global_gmres");
                se::huygens::write_image_rsf(prefix + "_after_gmres.rsf", model,
                    per_shot_after[ishot], "per_shot_after_global_gmres");
            }
        }
        std::cout << "Receiver scratch I/O : wrote " << receiver_store_backend.written_megabytes()
                  << " MiB, read " << receiver_store_backend.read_megabytes() << " MiB\n";

        se::huygens::write_image_rsf(
            options.image,
            model,
            image,
            kProgramName,
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
             {"bpack_tolerance",
              bpack_tolerance},
             {"bpack_leaf",
              static_cast<float>(bpack_leaf)},
             {"bpack_depth_chunk",
              static_cast<float>(bpack_depth_chunk)},
             {"seam_enable",
              static_cast<float>(seam_enable)},
             {"seam_amp_rel_floor",
              seam_amp_rel_floor},
             {"seam_scale_cap",
              seam_scale_cap}});

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
            kProgramName,
            {"filter_setup_seconds",
             "bpack_wall_build_seconds",
             "apply_gmres_image_seconds",
             "bpack_library_build_seconds",
             "bpack_operator_count",
             "bpack_matrix_rows",
             "bpack_matrix_columns",
             "bpack_sampled_entries",
             "bpack_compressed_mb_sum",
             "bpack_peak_mb",
             "bpack_max_rank",
             "target_depth_count"},
            {{"fft_seconds",
              static_cast<float>(
                  fft_seconds)},
             {"total_filter_setup_seconds",
              static_cast<float>(
                  total_filter_seconds)},
             {"total_bpack_build_seconds",
              static_cast<float>(
                  total_build_seconds)},
             {"total_bpack_library_build_seconds",
              static_cast<float>(
                  total_bpack_library_build_seconds)},
             {"bpack_tolerance",
              bpack_tolerance},
             {"bpack_leaf",
              static_cast<float>(bpack_leaf)},
             {"bpack_depth_chunk",
              static_cast<float>(bpack_depth_chunk)},
             {"total_apply_and_image_seconds",
              static_cast<float>(
                  total_apply_seconds)},
             {"skipped_receiver_coordinates",
              static_cast<float>(
                  skipped_receivers)},
             {"global_factor_seconds", static_cast<float>(global_factor_seconds)},
             {"global_solve_seconds", static_cast<float>(global_solve_seconds)},
             {"receiver_scratch_written_mb",
              static_cast<float>(receiver_store_backend.written_megabytes())},
             {"receiver_scratch_read_mb",
              static_cast<float>(receiver_store_backend.read_megabytes())}});

        std::cout
            << "Rectangular ButterflyPACK + restarted GMRES imaging completed.\n"
            << "FFT/data preparation: "
            << fft_seconds << " s\n"
            << "Filter setup:         "
            << total_filter_seconds << " s\n"
            << "BPACK operator build: "
            << total_build_seconds << " s"
            << " (internal=" << total_bpack_library_build_seconds << " s)\n"
            << "Apply + image:        "
            << total_apply_seconds << " s\n"
            << "Image:                "
            << options.image << '\n'
            << "Timing:               "
            << options.timing << '\n'
            << "Seam metrics:         "
            << seam_csv << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << kProgramName << ": "
            << error.what() << '\n';
        return 1;
    }
}

} // namespace kirch::imaging
