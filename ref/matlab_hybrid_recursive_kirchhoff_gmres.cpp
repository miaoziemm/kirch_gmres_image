// Four-block recursive FMM/Kirchhoff initial field followed by a global
// Kirchhoff-field-phase coarse correction and GMRES experiment.
//
// Important implementation facts:
//   * The original translation is included, not edited. Its main is renamed so
//     that its tested Helmholtz, GMRES, reference, and I/O routines remain
//     shared. This file provides its own coarse-space builder.
//   * Traveltimes are computed by the repository efmm solver for every datum
//     source. No straight-ray or Euclidean traveltime substitute is used.
//   * The simplified one-way Kirchhoff kernel is the same phase/filter form as
//     OneWayKernelData in SERECKIRCH/src/huygens_sweep.cpp.
//   * Matrix-vector products use the repository's actual multi-level
//     bf1d_strict_segmented_* butterfly factorization. They are checked against
//     direct dense summation at three depths per propagated block.
//   * The first block is the locally valid primary-source field, as in the
//     Luo-Qian-Burridge Huygens sweep. Blocks 2-4 are recursive integrations.
//   * Each recursive target includes the shared rows in the preceding block;
//     those rows determine a complex least-squares fit and a linear cross-fade
//     before the propagated field is accepted in the new block.
//   * The global coarse-space carrier is U_K/(|U_K|+epsilon), with full-domain
//     support chi=1 and epsilon derived from the RMS amplitude of U_K. Main-source
//     FMM traveltime, exp(i*omega*T), and the reference solution are not inputs
//     to the coarse-space construction.
//   * nx/nz and sx/sz describe the physical model.  A separate nabs-cell
//     absorbing layer is added on every side, and every output in fields/ is
//     cropped back to the physical grid.

#define main matlab_hybrid_rayphase_gmres_original_main
#include "matlab_hybrid_rayphase_gmres.cpp"
#undef main

#include <SERECKIRCH/include/bf1d.h>
#include <SERECKIRCH/include/efmm.h>

#include <array>
#include <atomic>
#include <mutex>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using ComplexF = std::complex<float>;

struct RecursiveSettings {
    int block_count = 4;
    int overlap_rows = 20;
    int threads = 0;
    int bf_p = 12;
    int bf_leaf = 16;
    int bf_panel_levels = 1;
    float bf_amp_eps = 1.0e-20f;
    float bf_phase_tol = 0.5f;
    float filter_dt = 0.001f;
    float filter_length = 0.025f;
    int filter_lookup_subsamples = 32;
    bool validate_direct = true;
    bool save_fmm_samples = true;
    bool fixed_outer_iterations = false;
    double coarse_phase_epsilon_rms_fraction = 1.0e-3;
    std::string reference_real;
    std::string reference_imag;
};

struct FmmModel {
    int nx = 0;
    int nz = 0;
    float dx = 1.0f;
    float dz = 1.0f;
    std::vector<float> velocity; // x-fast: ix + iz*nx

    std::size_t index(int ix, int iz) const
    {
        return static_cast<std::size_t>(ix) +
               static_cast<std::size_t>(iz) * nx;
    }
};

struct ExplicitAbsorbingGrid {
    int physical_nx = 0;
    int physical_nz = 0;
    int physical_x0 = 0;
    int physical_z0 = 0;
    int padding_left = 0;
    int padding_right = 0;
    int padding_top = 0;
    int padding_bottom = 0;
};

Model add_explicit_absorbing_padding(const Model& physical, int nabs)
{
    if (nabs < 0) {
        throw std::invalid_argument("nabs must be non-negative");
    }
    Model computational;
    computational.nx = physical.nx + 2 * nabs;
    computational.nz = physical.nz + 2 * nabs;
    computational.full_nx = physical.full_nx;
    computational.full_nz = physical.full_nz;
    computational.crop_x0 = physical.crop_x0;
    computational.crop_z0 = physical.crop_z0;
    computational.source_x = physical.source_x + nabs;
    computational.source_z = physical.source_z + nabs;
    computational.velocity.resize(
        static_cast<std::size_t>(computational.nx) * computational.nz);

    // Constant edge extrapolation changes only the artificial padding.  Every
    // sample in the physical model is copied without modification.
    for (int ix = 0; ix < computational.nx; ++ix) {
        const int physical_x = std::clamp(ix - nabs, 0, physical.nx - 1);
        for (int iz = 0; iz < computational.nz; ++iz) {
            const int physical_z = std::clamp(iz - nabs, 0, physical.nz - 1);
            computational.velocity[computational.index(iz, ix)] =
                physical.velocity[physical.index(physical_z, physical_x)];
        }
    }
    return computational;
}

ExplicitAbsorbingGrid make_absorbing_grid(const Model& physical,
                                          const Model& computational,
                                          int nabs)
{
    ExplicitAbsorbingGrid grid;
    grid.physical_nx = physical.nx;
    grid.physical_nz = physical.nz;
    grid.physical_x0 = nabs;
    grid.physical_z0 = nabs;
    grid.padding_left = nabs;
    grid.padding_right = nabs;
    grid.padding_top = nabs;
    grid.padding_bottom = nabs;
    if (computational.nx != physical.nx + 2 * nabs ||
        computational.nz != physical.nz + 2 * nabs ||
        computational.source_x != physical.source_x + nabs ||
        computational.source_z != physical.source_z + nabs) {
        throw std::runtime_error("inconsistent explicit absorbing-grid mapping");
    }
    return grid;
}

Vector crop_to_physical(const Vector& computational_field,
                        const Model& physical,
                        const Model& computational,
                        const ExplicitAbsorbingGrid& grid)
{
    if (computational_field.size() != computational.nx * computational.nz) {
        throw std::invalid_argument("computational complex field has wrong size");
    }
    Vector output(physical.nx * physical.nz);
    for (int ix = 0; ix < physical.nx; ++ix) {
        for (int iz = 0; iz < physical.nz; ++iz) {
            output[physical.index(iz, ix)] = computational_field[
                computational.index(
                    iz + grid.physical_z0, ix + grid.physical_x0)];
        }
    }
    return output;
}

std::vector<double> crop_to_physical(
    const std::vector<double>& computational_field,
    const Model& physical,
    const Model& computational,
    const ExplicitAbsorbingGrid& grid)
{
    if (computational_field.size() !=
        static_cast<std::size_t>(computational.nx) * computational.nz) {
        throw std::invalid_argument("computational real field has wrong size");
    }
    std::vector<double> output(
        static_cast<std::size_t>(physical.nx) * physical.nz);
    for (int ix = 0; ix < physical.nx; ++ix) {
        for (int iz = 0; iz < physical.nz; ++iz) {
            output[static_cast<std::size_t>(physical.index(iz, ix))] =
                computational_field[static_cast<std::size_t>(
                    computational.index(
                        iz + grid.physical_z0, ix + grid.physical_x0))];
        }
    }
    return output;
}

void write_physical_complex_field(
    const fs::path& prefix,
    const Vector& computational_field,
    const Model& physical,
    const Model& computational,
    const ExplicitAbsorbingGrid& grid)
{
    write_real_imag_field(
        prefix,
        crop_to_physical(
            computational_field, physical, computational, grid));
}

void write_physical_real_field(
    const fs::path& filename,
    const std::vector<double>& computational_field,
    const Model& physical,
    const Model& computational,
    const ExplicitAbsorbingGrid& grid)
{
    write_float_field(
        filename,
        crop_to_physical(
            computational_field, physical, computational, grid));
}

struct AbsorbingAudit {
    double source_eta = 0.0;
    double maximum_physical_eta = 0.0;
    std::size_t damped_padding_samples = 0;
    std::size_t expected_padding_samples = 0;
};

AbsorbingAudit audit_explicit_absorbing_grid(
    const Model& physical,
    const Model& computational,
    const ExplicitAbsorbingGrid& grid,
    const HelmholtzSystem& system)
{
    if (system.eta.size() !=
        static_cast<std::size_t>(computational.nx) * computational.nz) {
        throw std::runtime_error("absorbing eta field has wrong size");
    }
    AbsorbingAudit audit;
    audit.expected_padding_samples =
        static_cast<std::size_t>(computational.nx) * computational.nz -
        static_cast<std::size_t>(physical.nx) * physical.nz;
    audit.source_eta = system.eta[static_cast<std::size_t>(
        computational.index(
            computational.source_z, computational.source_x))];

    for (int ix = 0; ix < computational.nx; ++ix) {
        for (int iz = 0; iz < computational.nz; ++iz) {
            const bool in_physical =
                ix >= grid.physical_x0 &&
                ix < grid.physical_x0 + physical.nx &&
                iz >= grid.physical_z0 &&
                iz < grid.physical_z0 + physical.nz;
            const double eta = system.eta[static_cast<std::size_t>(
                computational.index(iz, ix))];
            if (in_physical) {
                audit.maximum_physical_eta =
                    std::max(audit.maximum_physical_eta, std::abs(eta));
            } else if (eta > 0.0) {
                ++audit.damped_padding_samples;
            }
        }
    }
    if (audit.source_eta != 0.0 || audit.maximum_physical_eta != 0.0) {
        throw std::runtime_error(
            "absorbing damping leaked into the physical model or source");
    }
    if (audit.damped_padding_samples != audit.expected_padding_samples) {
        throw std::runtime_error(
            "absorbing damping does not cover exactly the explicit padding");
    }
    return audit;
}

struct RecursiveBlock {
    int id = 0;
    int output_start_iz = 0;
    int output_end_iz = 0;
    int datum_iz = 0;

    int integration_start_iz() const
    {
        return id == 0 ? output_start_iz : datum_iz + 1;
    }

    int integration_depth_count() const
    {
        return output_end_iz - integration_start_iz() + 1;
    }
};

struct BlockMetric {
    RecursiveBlock block;
    int fmm_sources = 0;
    double table_megabytes = 0.0;
    double fmm_seconds = 0.0;
    double butterfly_build_seconds = 0.0;
    double butterfly_apply_seconds = 0.0;
    double direct_check_seconds = 0.0;
    double direct_relative_error = 0.0;
    ComplexF overlap_scale = ComplexF(1.0f, 0.0f);
    double overlap_mismatch_before = 0.0;
    double overlap_mismatch_after = 0.0;
    float minimum_traveltime = 0.0f;
    float maximum_traveltime = 0.0f;
};

struct RecursiveResult {
    std::vector<ComplexF> one_way_field;
    std::vector<std::vector<ComplexF>> block_snapshots;
    std::vector<double> primary_traveltime;
    std::vector<std::vector<double>> datum_middle_source_traveltimes;
    std::vector<BlockMetric> block_metrics;
    double primary_fmm_seconds = 0.0;
    double total_seconds = 0.0;
};

void print_recursive_help()
{
    std::cout
        << "Four-block recursive FMM + bf1d Kirchhoff-field coarse correction + GMRES\n\n"
        << "Required:\n"
        << "  model=FILE        smoothed raw float32 model in MATLAB [nz,nx] order\n\n"
        << "Physical-grid defaults:\n"
        << "  nx=1024 nz=298 dx=10.5 dz=10.5 frequency=15 sx=513 sz=1\n"
        << "  sx/sz are 1-based physical-model coordinates.\n\n"
        << "Explicit absorbing padding:\n"
        << "  nabs=25 damp_max=2\n"
        << "  computational grid=(nx+2*nabs) x (nz+2*nabs)\n"
        << "  the source is shifted by nabs; damping is zero in the physical grid\n"
        << "  and nonzero only in the constant-velocity edge padding.\n\n"
        << "Iteration defaults:\n"
        << "  q_coarse=2 gmres_steps_per_outer=3 target_relres=0.05\n\n"
        << "Recursive Kirchhoff defaults:\n"
        << "  block_count=4 overlap_rows=20 threads=0\n"
        << "  bf_p=12 bf_leaf=16 bf_panel_levels=1 bf_phase_tol=0.5\n"
        << "  filter_dt=0.001 filter_length=0.025 filter_lookup_subsamples=32\n"
        << "  validate_direct=1 save_fmm_samples=1\n"
        << "  fixed_outer_iterations=0  set to 1 to run exactly outer_max iterations\n\n"
        << "Kirchhoff-field coarse carrier:\n"
        << "  coarse_phase_epsilon_rms_fraction=1e-3\n"
        << "  carrier=U_K/(abs(U_K)+epsilon), full-domain chi=1\n"
        << "  No primary FMM traveltime or exp(i*omega*T) is used here.\n\n"
        << "Reference:\n"
        << "  reference_real=FILE reference_imag=FILE  split float32 reference\n"
        << "  read references must use the padded computational dimensions\n"
        << "  or use reference_mode=direct|read|none from the original program\n";
}

RecursiveSettings parse_recursive_settings(const Arguments& arguments)
{
    RecursiveSettings settings;
    settings.block_count = arguments.get_int("block_count", settings.block_count);
    settings.overlap_rows = arguments.get_int("overlap_rows", settings.overlap_rows);
    settings.threads = arguments.get_int("threads", settings.threads);
    settings.bf_p = arguments.get_int("bf_p", settings.bf_p);
    settings.bf_leaf = arguments.get_int("bf_leaf", settings.bf_leaf);
    settings.bf_panel_levels = arguments.get_int(
        "bf_panel_levels", settings.bf_panel_levels);
    settings.bf_amp_eps = static_cast<float>(arguments.get_double(
        "bf_amp_eps", settings.bf_amp_eps));
    settings.bf_phase_tol = static_cast<float>(arguments.get_double(
        "bf_phase_tol", settings.bf_phase_tol));
    settings.filter_dt = static_cast<float>(arguments.get_double(
        "filter_dt", settings.filter_dt));
    settings.filter_length = static_cast<float>(arguments.get_double(
        "filter_length", settings.filter_length));
    settings.filter_lookup_subsamples = arguments.get_int(
        "filter_lookup_subsamples", settings.filter_lookup_subsamples);
    settings.validate_direct = arguments.get_bool(
        "validate_direct", settings.validate_direct);
    settings.save_fmm_samples = arguments.get_bool(
        "save_fmm_samples", settings.save_fmm_samples);
    settings.fixed_outer_iterations = arguments.get_bool(
        "fixed_outer_iterations", settings.fixed_outer_iterations);
    settings.coarse_phase_epsilon_rms_fraction = arguments.get_double(
        "coarse_phase_epsilon_rms_fraction",
        arguments.get_double(
            "coarse_amp_floor_rms_fraction",
            settings.coarse_phase_epsilon_rms_fraction));
    settings.reference_real = arguments.get("reference_real", "");
    settings.reference_imag = arguments.get("reference_imag", "");

    if (settings.block_count != 4) {
        throw std::invalid_argument(
            "block_count must be exactly 4 for this controlled experiment");
    }
    if (settings.overlap_rows < 1 || settings.threads < 0 ||
        settings.bf_p < 2 || settings.bf_leaf < 4 ||
        settings.bf_p >= settings.bf_leaf || settings.bf_panel_levels < 0 ||
        !(settings.bf_phase_tol > 0.0f) || settings.bf_amp_eps < 0.0f ||
        !(settings.filter_dt > 0.0f) || !(settings.filter_length > 0.0f) ||
        settings.filter_lookup_subsamples < 1 ||
        !(settings.coarse_phase_epsilon_rms_fraction > 0.0) ||
        !std::isfinite(settings.coarse_phase_epsilon_rms_fraction)) {
        throw std::invalid_argument("invalid recursive FMM/butterfly parameters");
    }
    if (settings.reference_real.empty() != settings.reference_imag.empty()) {
        throw std::invalid_argument(
            "reference_real and reference_imag must be supplied together");
    }
    return settings;
}

FmmModel make_fmm_model(const Model& model, const Parameters& p)
{
    FmmModel output;
    output.nx = model.nx;
    output.nz = model.nz;
    output.dx = static_cast<float>(p.dx);
    output.dz = static_cast<float>(p.dz);
    output.velocity.resize(static_cast<std::size_t>(model.nx) * model.nz);
    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            output.velocity[output.index(ix, iz)] = static_cast<float>(
                model.velocity[model.index(iz, ix)]);
        }
    }
    return output;
}

std::vector<RecursiveBlock> make_four_blocks(const FmmModel& model,
                                             int source_z,
                                             int overlap_rows,
                                             int physical_z0,
                                             int physical_nz)
{
    if (model.nz < 16 || source_z < 0 || source_z >= model.nz ||
        physical_z0 < 0 || physical_nz < 16 ||
        physical_z0 + physical_nz > model.nz) {
        throw std::invalid_argument("invalid grid/source for four-block split");
    }
    std::vector<RecursiveBlock> blocks(4);
    for (int id = 0; id < 4; ++id) {
        RecursiveBlock block;
        block.id = id;
        const int physical_start = physical_z0 + static_cast<int>(
            (static_cast<long long>(id) * physical_nz) / 4);
        const int physical_end = physical_z0 + static_cast<int>(
            (static_cast<long long>(id + 1) * physical_nz) / 4) - 1;
        // Keep the three recursive datums tied to physical-model quarters.
        // Only the first/last blocks absorb the artificial top/bottom padding.
        block.output_start_iz = id == 0 ? 0 : physical_start;
        block.output_end_iz = id == 3 ? model.nz - 1 : physical_end;
        block.datum_iz = id == 0
            ? source_z
            : block.output_start_iz - overlap_rows;
        if (id > 0 && (block.datum_iz < 1 ||
                       block.datum_iz >= block.output_start_iz ||
                       block.datum_iz > blocks[static_cast<std::size_t>(id - 1)].output_end_iz)) {
            throw std::invalid_argument(
                "overlap_rows does not place the datum inside the preceding block");
        }
        blocks[static_cast<std::size_t>(id)] = block;
    }
    if (source_z > blocks.front().output_end_iz) {
        throw std::invalid_argument("source lies below the first block");
    }
    if (blocks[1].datum_iz <= source_z) {
        throw std::invalid_argument(
            "the first recursive datum must be below the physical source; "
            "reduce overlap_rows or enlarge the vertical crop");
    }
    return blocks;
}

std::vector<float> solve_repository_fmm(const FmmModel& model,
                                        int source_x,
                                        int source_z)
{
    if (source_x < 0 || source_x >= model.nx ||
        source_z < 0 || source_z >= model.nz) {
        throw std::invalid_argument("FMM source is outside the active model");
    }
    efmm_t solver{};
    if (efmm_init(&solver, model.nx, model.nz, model.dx, model.dz,
                  source_x * model.dx, source_z * model.dz) != 0) {
        throw std::runtime_error("efmm_init failed");
    }
    try {
        if (efmm_set_vel(&solver, const_cast<float*>(model.velocity.data())) != 0) {
            throw std::runtime_error("efmm_set_vel failed");
        }
        const int status = efmm_solver(&solver);
        if (status != 0) {
            throw std::runtime_error(
                "efmm_solver failed with status " + std::to_string(status));
        }
        const std::size_t count = static_cast<std::size_t>(model.nx) * model.nz;
        std::vector<float> traveltime(solver.tt, solver.tt + count);
        efmm_free(&solver);
        return traveltime;
    } catch (...) {
        efmm_free(&solver);
        throw;
    }
}

std::vector<double> fmm_to_matlab_order(const FmmModel& model,
                                        const std::vector<float>& input)
{
    std::vector<double> output(input.size());
    for (int ix = 0; ix < model.nx; ++ix) {
        for (int iz = 0; iz < model.nz; ++iz) {
            output[static_cast<std::size_t>(iz) +
                   static_cast<std::size_t>(ix) * model.nz] =
                input[model.index(ix, iz)];
        }
    }
    return output;
}

class SimplifiedKirchhoffFilter {
public:
    SimplifiedKirchhoffFilter(float frequency,
                              float sample_interval,
                              float filter_length,
                              float maximum_traveltime,
                              int lookup_subsamples)
        : frequency_(frequency),
          sample_interval_(sample_interval),
          filter_length_(filter_length),
          omega_(2.0f * static_cast<float>(kPi) * frequency)
    {
        if (!(frequency_ > 0.0f) || !(sample_interval_ > 0.0f) ||
            !(filter_length_ > 0.0f) || !(maximum_traveltime > 0.0f) ||
            lookup_subsamples < 1) {
            throw std::invalid_argument("invalid Kirchhoff filter parameters");
        }
        lookup_step_ = sample_interval_ / lookup_subsamples;
        const std::size_t count = static_cast<std::size_t>(
            std::ceil(maximum_traveltime / lookup_step_)) + 2;
        lookup_.resize(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(count); ++i) {
            const float tau = std::max(
                lookup_step_ * static_cast<float>(i),
                0.25f * sample_interval_);
            lookup_[static_cast<std::size_t>(i)] = exact_response(tau);
        }
    }

    ComplexF response(float tau) const
    {
        if (!(tau > 0.0f) || !std::isfinite(tau)) return ComplexF(0.0f, 0.0f);
        const float coordinate = tau / lookup_step_;
        if (coordinate <= 0.0f) return lookup_.front();
        const std::size_t left = static_cast<std::size_t>(coordinate);
        if (left + 1 >= lookup_.size()) return lookup_.back();
        const float fraction = coordinate - static_cast<float>(left);
        return (1.0f - fraction) * lookup_[left] +
               fraction * lookup_[left + 1];
    }

    float omega() const { return omega_; }

private:
    ComplexF exact_response(float tau) const
    {
        const int nsam = static_cast<int>(filter_length_ / sample_interval_) + 2;
        std::vector<float> coefficients(static_cast<std::size_t>(nsam - 1), 0.0f);
        float previous2 = 0.0f;
        float previous1 = std::sqrt(
            ((tau + sample_interval_) / tau) *
            ((tau + sample_interval_) / tau) - 1.0f);
        coefficients[0] = previous1;
        for (int k = 1; k < nsam - 1; ++k) {
            const float t =
                (tau + static_cast<float>(k + 1) * sample_interval_) / tau;
            const float current = std::sqrt(t * t - 1.0f);
            coefficients[static_cast<std::size_t>(k)] =
                current - 2.0f * previous1 + previous2;
            previous2 = previous1;
            previous1 = current;
        }
        const float digital_omega = omega_ * sample_interval_;
        const ComplexF step = std::exp(ComplexF(0.0f, -digital_omega));
        ComplexF power(1.0f, 0.0f);
        ComplexF sum(0.0f, 0.0f);
        for (int k = 0; k <= nsam - 2; ++k) {
            sum += (coefficients[static_cast<std::size_t>(k)] /
                    sample_interval_) * power;
            power *= step;
        }
        const float q = tau / sample_interval_;
        const int m = static_cast<int>(std::ceil(q));
        const float delta = static_cast<float>(m) - q;
        const ComplexF linear_delay =
            (1.0f - delta) * std::exp(ComplexF(
                0.0f, -digital_omega * static_cast<float>(m))) +
            delta * std::exp(ComplexF(
                0.0f, -digital_omega * static_cast<float>(m - 1)));
        return linear_delay * sum;
    }

    float frequency_ = 0.0f;
    float sample_interval_ = 0.001f;
    float filter_length_ = 0.025f;
    float lookup_step_ = 0.0f;
    float omega_ = 0.0f;
    std::vector<ComplexF> lookup_;
};

std::size_t travel_table_index(int source,
                               int target_x,
                               int local_z,
                               int nx,
                               int depth_count)
{
    return (static_cast<std::size_t>(source) * nx + target_x) *
               depth_count + local_z;
}

struct TravelTable {
    int nx = 0;
    int depth_count = 0;
    std::vector<float> values;
    std::vector<float> middle_source_full_field;
    double seconds = 0.0;
    float minimum = 0.0f;
    float maximum = 0.0f;
};

TravelTable build_true_fmm_table(const FmmModel& model,
                                 const RecursiveBlock& block,
                                 int requested_threads)
{
    TravelTable table;
    table.nx = model.nx;
    table.depth_count = block.integration_depth_count();
    table.values.resize(
        static_cast<std::size_t>(model.nx) * model.nx * table.depth_count);
    const auto started = Clock::now();
    std::atomic<int> completed{0};
    std::atomic<bool> failed{false};
    std::mutex failure_mutex;
    std::string failure_message;
    const int workers = requested_threads > 0 ? requested_threads :
#ifdef _OPENMP
        omp_get_max_threads();
#else
        1;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
#endif
    for (int source = 0; source < model.nx; ++source) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            const std::vector<float> tau = solve_repository_fmm(
                model, source, block.datum_iz);
            for (int target_x = 0; target_x < model.nx; ++target_x) {
                for (int local_z = 0; local_z < table.depth_count; ++local_z) {
                    const int target_z = block.integration_start_iz() + local_z;
                    table.values[travel_table_index(
                        source, target_x, local_z,
                        model.nx, table.depth_count)] =
                        std::max(tau[model.index(target_x, target_z)], 1.0e-8f);
                }
            }
            if (source == model.nx / 2) table.middle_source_full_field = tau;
            const int done = completed.fetch_add(1) + 1;
            if (done == model.nx || done % std::max(1, model.nx / 8) == 0) {
#ifdef _OPENMP
#pragma omp critical(recursive_fmm_progress)
#endif
                std::cout << "    FMM sources " << done << '/' << model.nx << '\n';
            }
        } catch (const std::exception& error) {
            bool expected = false;
            if (failed.compare_exchange_strong(expected, true)) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                failure_message = error.what();
            }
        }
    }
    if (failed.load()) {
        throw std::runtime_error("parallel FMM table failed: " + failure_message);
    }
    table.seconds = elapsed_seconds(started);
    const auto range = std::minmax_element(table.values.begin(), table.values.end());
    table.minimum = *range.first;
    table.maximum = *range.second;
    return table;
}

ComplexF simplified_kernel_entry(const FmmModel& model,
                                 const RecursiveBlock& block,
                                 const TravelTable& table,
                                 const SimplifiedKirchhoffFilter& filter,
                                 int target_x,
                                 int source_x,
                                 int local_z)
{
    const float tau = table.values[travel_table_index(
        source_x, target_x, local_z, model.nx, table.depth_count)];
    const float horizontal = (target_x - source_x) * model.dx;
    const float vertical =
        (block.integration_start_iz() + local_z - block.datum_iz) * model.dz;
    const float distance_squared = horizontal * horizontal + vertical * vertical;
    const float geometry = vertical * tau /
        (static_cast<float>(kPi) * std::max(distance_squared, 1.0e-20f));
    const float quadrature =
        (source_x == 0 || source_x == model.nx - 1)
            ? 0.5f * model.dx
            : model.dx;
    return quadrature * geometry * filter.response(tau);
}

struct FactorDeleterRecursive {
    void operator()(BFStrictSegmentedFactor* factor) const
    {
        bf1d_strict_segmented_destroy(factor);
    }
};

using RecursiveFactor =
    std::unique_ptr<BFStrictSegmentedFactor, FactorDeleterRecursive>;

BlockMetric propagate_block_with_butterfly(
    const FmmModel& model,
    const RecursiveBlock& block,
    const RecursiveSettings& settings,
    const TravelTable& table,
    float frequency,
    std::vector<ComplexF>& wavefield)
{
    BlockMetric metric;
    metric.block = block;
    metric.fmm_sources = model.nx;
    metric.table_megabytes =
        table.values.size() * sizeof(float) / 1.0e6;
    metric.fmm_seconds = table.seconds;
    metric.minimum_traveltime = table.minimum;
    metric.maximum_traveltime = table.maximum;

    const SimplifiedKirchhoffFilter filter(
        frequency, settings.filter_dt, settings.filter_length,
        table.maximum, settings.filter_lookup_subsamples);
    const int depths = table.depth_count;
    std::vector<RecursiveFactor> factors(static_cast<std::size_t>(depths));
    std::atomic<bool> failed{false};
    const int workers = settings.threads > 0 ? settings.threads :
#ifdef _OPENMP
        omp_get_max_threads();
#else
        1;
#endif

    auto started = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
#endif
    for (int local_z = 0; local_z < depths; ++local_z) {
        if (failed.load(std::memory_order_relaxed)) continue;
        std::vector<float> tau_storage(
            static_cast<std::size_t>(model.nx) * model.nx);
        std::vector<float*> tau_rows(static_cast<std::size_t>(model.nx));
        std::vector<float> amplitude(
            2 * static_cast<std::size_t>(model.nx) * model.nx);
        for (int target_x = 0; target_x < model.nx; ++target_x) {
            tau_rows[static_cast<std::size_t>(target_x)] =
                tau_storage.data() + static_cast<std::size_t>(target_x) * model.nx;
            for (int source_x = 0; source_x < model.nx; ++source_x) {
                const std::size_t matrix_index =
                    static_cast<std::size_t>(target_x) * model.nx + source_x;
                const float tau = table.values[travel_table_index(
                    source_x, target_x, local_z, model.nx, depths)];
                tau_storage[matrix_index] = tau;
                const ComplexF entry = simplified_kernel_entry(
                    model, block, table, filter,
                    target_x, source_x, local_z);
                const ComplexF phase_removed = entry * std::exp(
                    ComplexF(0.0f, filter.omega() * tau));
                amplitude[2 * matrix_index] = phase_removed.real();
                amplitude[2 * matrix_index + 1] = phase_removed.imag();
            }
        }
        BFStrictSegmentedFactor* factor =
            bf1d_strict_segmented_create_phase_amp(
                model.nx, tau_rows.data(),
                reinterpret_cast<const fftwf_complex*>(amplitude.data()),
                filter.omega(), settings.bf_p, settings.bf_leaf,
                settings.bf_panel_levels, settings.bf_amp_eps,
                settings.bf_phase_tol);
        if (factor == nullptr) {
            failed.store(true);
        } else {
            factors[static_cast<std::size_t>(local_z)].reset(factor);
        }
    }
    if (failed.load()) {
        throw std::runtime_error("bf1d factor construction returned null");
    }
    metric.butterfly_build_seconds = elapsed_seconds(started);

    std::vector<float> packed_input(2 * static_cast<std::size_t>(model.nx));
    for (int source_x = 0; source_x < model.nx; ++source_x) {
        const ComplexF value = wavefield[model.index(source_x, block.datum_iz)];
        packed_input[2 * static_cast<std::size_t>(source_x)] = value.real();
        packed_input[2 * static_cast<std::size_t>(source_x) + 1] = value.imag();
    }
    std::vector<ComplexF> target_values(
        static_cast<std::size_t>(model.nx) * depths);
    started = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
#endif
    for (int local_z = 0; local_z < depths; ++local_z) {
        std::vector<float> output(2 * static_cast<std::size_t>(model.nx));
        bf1d_strict_segmented_apply(
            factors[static_cast<std::size_t>(local_z)].get(),
            reinterpret_cast<const fftwf_complex*>(packed_input.data()),
            reinterpret_cast<fftwf_complex*>(output.data()));
        for (int target_x = 0; target_x < model.nx; ++target_x) {
            target_values[static_cast<std::size_t>(target_x) * depths + local_z] =
                ComplexF(output[2 * static_cast<std::size_t>(target_x)],
                         output[2 * static_cast<std::size_t>(target_x) + 1]);
        }
    }
    metric.butterfly_apply_seconds = elapsed_seconds(started);

    if (settings.validate_direct) {
        started = Clock::now();
        const std::set<int> checked_depths = {0, depths / 2, depths - 1};
        long double numerator = 0.0;
        long double denominator = 0.0;
        for (int local_z : checked_depths) {
            std::vector<ComplexF> direct(static_cast<std::size_t>(model.nx));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
            for (int target_x = 0; target_x < model.nx; ++target_x) {
                ComplexF sum(0.0f, 0.0f);
                for (int source_x = 0; source_x < model.nx; ++source_x) {
                    const ComplexF input(
                        packed_input[2 * static_cast<std::size_t>(source_x)],
                        packed_input[2 * static_cast<std::size_t>(source_x) + 1]);
                    sum += simplified_kernel_entry(
                        model, block, table, filter,
                        target_x, source_x, local_z) * input;
                }
                direct[static_cast<std::size_t>(target_x)] = sum;
            }
            for (int target_x = 0; target_x < model.nx; ++target_x) {
                const ComplexF bf = target_values[
                    static_cast<std::size_t>(target_x) * depths + local_z];
                const ComplexF exact = direct[static_cast<std::size_t>(target_x)];
                numerator += std::norm(bf - exact);
                denominator += std::norm(exact);
            }
        }
        metric.direct_relative_error = denominator > 0.0
            ? std::sqrt(static_cast<double>(numerator / denominator))
            : 0.0;
        metric.direct_check_seconds = elapsed_seconds(started);
    }

    // The integration target includes the rows shared with the preceding
    // block.  Use those independently computed values to estimate one complex
    // block scale, then cross-fade through the overlap before accepting the
    // recursively propagated values in the new output rows.  The datum itself
    // is shared exactly and is excluded from the target matrix because the
    // value-only kernel is singular at zero vertical separation.
    const int overlap_depths =
        block.output_start_iz - block.integration_start_iz();
    if (overlap_depths > 0) {
        std::complex<long double> numerator(0.0L, 0.0L);
        long double denominator = 0.0;
        long double existing_energy = 0.0;
        long double mismatch_before = 0.0;
        for (int target_x = 0; target_x < model.nx; ++target_x) {
            for (int local_z = 0; local_z < overlap_depths; ++local_z) {
                const ComplexF computed = target_values[
                    static_cast<std::size_t>(target_x) * depths + local_z];
                const ComplexF existing = wavefield[model.index(
                    target_x, block.integration_start_iz() + local_z)];
                numerator += std::conj(std::complex<long double>(computed)) *
                    std::complex<long double>(existing);
                denominator += std::norm(computed);
                existing_energy += std::norm(existing);
                mismatch_before += std::norm(computed - existing);
            }
        }
        if (denominator > 0.0) {
            const std::complex<long double> fitted = numerator / denominator;
            metric.overlap_scale = ComplexF(
                static_cast<float>(fitted.real()),
                static_cast<float>(fitted.imag()));
        }
        long double mismatch_after = 0.0;
        for (int target_x = 0; target_x < model.nx; ++target_x) {
            for (int local_z = 0; local_z < overlap_depths; ++local_z) {
                const ComplexF computed = metric.overlap_scale * target_values[
                    static_cast<std::size_t>(target_x) * depths + local_z];
                const ComplexF existing = wavefield[model.index(
                    target_x, block.integration_start_iz() + local_z)];
                mismatch_after += std::norm(computed - existing);
            }
        }
        if (existing_energy > 0.0) {
            metric.overlap_mismatch_before = std::sqrt(
                static_cast<double>(mismatch_before / existing_energy));
            metric.overlap_mismatch_after = std::sqrt(
                static_cast<double>(mismatch_after / existing_energy));
        }
    }

    for (int target_x = 0; target_x < model.nx; ++target_x) {
        for (int local_z = 0; local_z < depths; ++local_z) {
            const int target_z = block.integration_start_iz() + local_z;
            const ComplexF propagated = metric.overlap_scale * target_values[
                static_cast<std::size_t>(target_x) * depths + local_z];
            if (target_z < block.output_start_iz) {
                const float weight = static_cast<float>(
                    target_z - block.datum_iz) /
                    static_cast<float>(block.output_start_iz - block.datum_iz);
                wavefield[model.index(target_x, target_z)] =
                    (1.0f - weight) * wavefield[model.index(target_x, target_z)] +
                    weight * propagated;
            } else {
                wavefield[model.index(target_x, target_z)] = propagated;
            }
        }
    }
    return metric;
}

std::vector<ComplexF> initialize_local_primary_field(
    const FmmModel& model,
    const RecursiveBlock& first,
    const std::vector<float>& traveltime,
    int source_x,
    int source_z,
    float frequency)
{
    std::vector<ComplexF> wavefield(
        static_cast<std::size_t>(model.nx) * model.nz,
        ComplexF(0.0f, 0.0f));
    const float omega = 2.0f * static_cast<float>(kPi) * frequency;
    const float minimum_time = 0.10f * std::min(model.dx, model.dz) /
        model.velocity[model.index(source_x, source_z)];
    for (int iz = first.output_start_iz; iz <= first.output_end_iz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            const float argument = omega * std::max(
                traveltime[model.index(ix, iz)], minimum_time);
            const double j0 = std::cyl_bessel_j(0.0, argument);
            const double y0 = std::cyl_neumann(0.0, argument);
            const ComplexF outgoing(
                static_cast<float>(-0.25 * y0),
                static_cast<float>(0.25 * j0));
            // bf1d propagates exp(-i*omega*tau); store that convention during
            // recursion and conjugate once before the Helmholtz solve.
            wavefield[model.index(ix, iz)] = std::conj(outgoing);
        }
    }
    return wavefield;
}

RecursiveResult build_recursive_initial(const FmmModel& model,
                                        int source_x,
                                        int source_z,
                                        int physical_z0,
                                        int physical_nz,
                                        float frequency,
                                        const RecursiveSettings& settings)
{
    const auto total_started = Clock::now();
    RecursiveResult result;
    const std::vector<RecursiveBlock> blocks = make_four_blocks(
        model, source_z, settings.overlap_rows, physical_z0, physical_nz);
    std::cout << "\nFour-block recursive layout:\n";
    for (const RecursiveBlock& block : blocks) {
        std::cout << "  block " << block.id + 1
                  << ": computational output=" << block.output_start_iz
                  << ".." << block.output_end_iz
                  << ", datum=" << block.datum_iz
                  << "; physical-z coordinates="
                  << block.output_start_iz - physical_z0
                  << ".." << block.output_end_iz - physical_z0
                  << ", datum=" << block.datum_iz - physical_z0 << '\n';
    }

    auto started = Clock::now();
    const std::vector<float> primary = solve_repository_fmm(
        model, source_x, source_z);
    result.primary_fmm_seconds = elapsed_seconds(started);
    result.primary_traveltime = fmm_to_matlab_order(model, primary);
    result.one_way_field = initialize_local_primary_field(
        model, blocks.front(), primary, source_x, source_z, frequency);
    result.block_snapshots.push_back(result.one_way_field);

    BlockMetric first_metric;
    first_metric.block = blocks.front();
    first_metric.fmm_sources = 1;
    first_metric.fmm_seconds = result.primary_fmm_seconds;
    const auto primary_range = std::minmax_element(primary.begin(), primary.end());
    first_metric.minimum_traveltime = *primary_range.first;
    first_metric.maximum_traveltime = *primary_range.second;
    result.block_metrics.push_back(first_metric);

    for (std::size_t index = 1; index < blocks.size(); ++index) {
        const RecursiveBlock& block = blocks[index];
        std::cout << "\nBlock " << block.id + 1
                  << ": true eFMM table at datum z=" << block.datum_iz << '\n';
        TravelTable table = build_true_fmm_table(
            model, block, settings.threads);
        std::cout << "  table="
                  << table.values.size() * sizeof(float) / 1.0e6
                  << " MB, FMM=" << table.seconds << " s, tau=["
                  << table.minimum << ',' << table.maximum << "] s\n";
        BlockMetric metric = propagate_block_with_butterfly(
            model, block, settings, table, frequency, result.one_way_field);
        std::cout << "  bf1d build=" << metric.butterfly_build_seconds
                  << " s, apply=" << metric.butterfly_apply_seconds
                  << " s, direct-check error=" << std::scientific
                  << metric.direct_relative_error
                  << ", overlap mismatch=" << metric.overlap_mismatch_before
                  << " -> " << metric.overlap_mismatch_after
                  << std::fixed << '\n';
        if (settings.save_fmm_samples &&
            !table.middle_source_full_field.empty()) {
            result.datum_middle_source_traveltimes.push_back(
                fmm_to_matlab_order(model, table.middle_source_full_field));
        }
        result.block_metrics.push_back(metric);
        result.block_snapshots.push_back(result.one_way_field);
    }
    result.total_seconds = elapsed_seconds(total_started);
    return result;
}

Vector one_way_to_helmholtz(const FmmModel& fmm_model,
                            const Model& model,
                            const Parameters& p,
                            const std::vector<ComplexF>& one_way)
{
    Vector output(model.nx * model.nz);
    for (int ix = 0; ix < model.nx; ++ix) {
        for (int iz = 0; iz < model.nz; ++iz) {
            output[model.index(iz, ix)] = static_cast<Complex>(
                std::conj(one_way[fmm_model.index(ix, iz)]));
        }
    }
    if (p.use_pml_taper && p.nabs > 0) {
        const int effective = std::min({p.nabs, model.nx, model.nz});
        for (int ii = 0; ii < effective; ++ii) {
            const double sine = std::sin(
                0.5 * kPi * static_cast<double>(ii + 1) /
                static_cast<double>(p.nabs + 1));
            const double factor = sine * sine;
            for (int ix = 0; ix < model.nx; ++ix) {
                output[model.index(ii, ix)] *= factor;
                output[model.index(model.nz - ii - 1, ix)] *= factor;
            }
            for (int iz = 0; iz < model.nz; ++iz) {
                output[model.index(iz, ii)] *= factor;
                output[model.index(iz, model.nx - ii - 1)] *= factor;
            }
        }
    }
    return output;
}

struct KirchhoffCarrierDiagnostics {
    Vector carrier;
    std::vector<double> phase_regularization_weight;
    double field_rms_amplitude = 0.0;
    double phase_epsilon = 0.0;
    double minimum_amplitude = 0.0;
    double maximum_amplitude = 0.0;
    std::size_t below_epsilon_samples = 0;
    std::size_t zero_samples = 0;
};

CoarseSpace build_kirchhoff_field_coarse_space(
    const SparseMatrix& matrix,
    const Model& model,
    const Parameters& p,
    const Vector& kirchhoff_field,
    double phase_epsilon_rms_fraction,
    KirchhoffCarrierDiagnostics& diagnostics)
{
    if (kirchhoff_field.size() != model.nx * model.nz) {
        throw std::invalid_argument(
            "Kirchhoff coarse carrier has the wrong field size");
    }
    if (!(phase_epsilon_rms_fraction > 0.0) ||
        !std::isfinite(phase_epsilon_rms_fraction)) {
        throw std::invalid_argument(
            "coarse_phase_epsilon_rms_fraction must be finite and positive");
    }

    const auto started = Clock::now();
    Interpolation1D px = interp1_matrix(model.nx, p.q_coarse);
    Interpolation1D pz = interp1_matrix(model.nz, p.q_coarse);

    CoarseSpace coarse;
    coarse.nx_coarse = static_cast<int>(px.nodes.size());
    coarse.nz_coarse = static_cast<int>(pz.nodes.size());
    coarse.interpolation = Eigen::kroneckerProduct(px.matrix, pz.matrix).eval();
    coarse.interpolation.makeCompressed();

    long double squared_amplitude_sum = 0.0L;
    diagnostics.minimum_amplitude = std::numeric_limits<double>::infinity();
    diagnostics.maximum_amplitude = 0.0;
    for (Eigen::Index index = 0; index < kirchhoff_field.size(); ++index) {
        const double amplitude = std::abs(kirchhoff_field[index]);
        if (!std::isfinite(amplitude)) {
            throw std::runtime_error(
                "recursive Kirchhoff field contains a non-finite sample");
        }
        diagnostics.minimum_amplitude = std::min(
            diagnostics.minimum_amplitude, amplitude);
        diagnostics.maximum_amplitude = std::max(
            diagnostics.maximum_amplitude, amplitude);
        squared_amplitude_sum += static_cast<long double>(amplitude) * amplitude;
    }
    diagnostics.field_rms_amplitude = std::sqrt(static_cast<double>(
        squared_amplitude_sum / static_cast<long double>(kirchhoff_field.size())));
    diagnostics.phase_epsilon = std::max(
        phase_epsilon_rms_fraction * diagnostics.field_rms_amplitude,
        std::numeric_limits<double>::min());
    diagnostics.carrier.resize(kirchhoff_field.size());
    diagnostics.phase_regularization_weight.resize(
        static_cast<std::size_t>(kirchhoff_field.size()), 0.0);

    for (Eigen::Index index = 0; index < kirchhoff_field.size(); ++index) {
        const Complex sample = kirchhoff_field[index];
        const double amplitude = std::abs(sample);
        if (amplitude == 0.0) {
            diagnostics.carrier[index] = Complex(0.0, 0.0);
            ++diagnostics.zero_samples;
            continue;
        }
        const double regularization_weight =
            amplitude / (amplitude + diagnostics.phase_epsilon);
        diagnostics.phase_regularization_weight[
            static_cast<std::size_t>(index)] = regularization_weight;
        diagnostics.carrier[index] =
            sample / (amplitude + diagnostics.phase_epsilon);
        if (amplitude < diagnostics.phase_epsilon) {
            ++diagnostics.below_epsilon_samples;
        }
    }

    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(coarse.interpolation.nonZeros()));
    for (int column = 0; column < coarse.interpolation.outerSize(); ++column) {
        for (SparseMatrix::InnerIterator value(coarse.interpolation, column);
             value; ++value) {
            const Complex weighted_carrier =
                diagnostics.carrier[value.row()] * value.value();
            if (weighted_carrier != Complex(0.0, 0.0)) {
                entries.emplace_back(value.row(), value.col(), weighted_carrier);
            }
        }
    }
    coarse.basis.resize(coarse.interpolation.rows(), coarse.interpolation.cols());
    coarse.basis.setFromTriplets(entries.begin(), entries.end());
    coarse.basis.makeCompressed();
    std::cout << "  carrier source: recursive butterfly-Kirchhoff U_K only\n"
              << "  carrier formula: U_K/(abs(U_K)+epsilon), epsilon="
              << std::scientific << diagnostics.phase_epsilon
              << " (" << phase_epsilon_rms_fraction
              << " * RMS), below epsilon="
              << diagnostics.below_epsilon_samples << '/'
              << kirchhoff_field.size() << std::fixed << '\n'
              << "  P/Z_K: " << coarse.basis.rows() << " x "
              << coarse.basis.cols() << ", nnz="
              << coarse.basis.nonZeros() << '\n';

    coarse.action_basis = matrix * coarse.basis;
    coarse.action_basis.prune(
        [](int, int, const Complex& value) {
            return value != Complex(0.0, 0.0);
        });
    coarse.action_basis.makeCompressed();
    std::cout << "  A*Z_K: nnz=" << coarse.action_basis.nonZeros() << '\n';
    coarse.coarse_operator = coarse.basis.adjoint() * coarse.action_basis;
    coarse.coarse_operator.prune(
        [](int, int, const Complex& value) {
            return value != Complex(0.0, 0.0);
        });
    coarse.coarse_operator.makeCompressed();
    std::cout << "  Z_K^H*A*Z_K: " << coarse.coarse_operator.rows() << " x "
              << coarse.coarse_operator.cols() << ", nnz="
              << coarse.coarse_operator.nonZeros() << "; factorizing\n";
    coarse.solver = std::make_unique<SparseLU>();
    coarse.solver->analyzePattern(coarse.coarse_operator);
    coarse.solver->factorize(coarse.coarse_operator);
    if (coarse.solver->info() != Eigen::Success) {
        throw std::runtime_error(
            "Kirchhoff-field coarse LU factorization failed");
    }
    coarse.setup_seconds = elapsed_seconds(started);
    std::cout << "  Kirchhoff-field coarse LU complete in "
              << coarse.setup_seconds << " s\n";
    return coarse;
}

struct KirchhoffCoarseStageMetric {
    int outer = 0;
    int fine_steps_before = 0;
    double residual_before = 0.0;
    double coarse_relative_residual = 0.0;
    double coarse_true_relative_residual = 0.0;
    double coarse_relative_error = std::numeric_limits<double>::quiet_NaN();
    Complex alpha = Complex(0.0, 0.0);
    double correction_norm_over_reference =
        std::numeric_limits<double>::quiet_NaN();
};

HybridResult run_kirchhoff_field_hybrid(
    const SparseMatrix& matrix,
    const Vector& rhs,
    const Vector& initial,
    CoarseSpace& coarse,
    const Vector* reference,
    const Parameters& p,
    const fs::path& field_directory,
    const fs::path& extended_field_directory,
    const Model& physical_model,
    const Model& computational_model,
    const ExplicitAbsorbingGrid& absorbing_grid,
    std::vector<KirchhoffCoarseStageMetric>& coarse_stages)
{
    const double rhs_norm = rhs.norm();
    const double reference_norm = reference ? reference->norm() :
        std::numeric_limits<double>::quiet_NaN();
    Vector current = initial;
    Vector residual = rhs - matrix * current;
    HybridResult result;
    coarse_stages.clear();

    StateMetric initial_metric;
    initial_metric.relative_residual = residual.norm() / rhs_norm;
    initial_metric.true_relative_residual = initial_metric.relative_residual;
    initial_metric.relative_error = relative_error(current, reference);
    result.metrics.push_back(initial_metric);

    KirchhoffCoarseStageMetric initial_stage;
    initial_stage.residual_before = initial_metric.relative_residual;
    initial_stage.coarse_relative_residual = initial_metric.relative_residual;
    initial_stage.coarse_true_relative_residual = initial_metric.relative_residual;
    initial_stage.coarse_relative_error = initial_metric.relative_error;
    coarse_stages.push_back(initial_stage);
    if (p.save_frame_fields) {
        write_physical_complex_field(
            field_directory / "hybrid_outer_000", current,
            physical_model, computational_model, absorbing_grid);
        write_physical_complex_field(
            field_directory / "kirchhoff_coarsecorr_outer_000",
            Vector::Zero(current.size()),
            physical_model, computational_model, absorbing_grid);
        write_physical_complex_field(
            field_directory / "gmrescorr_outer_000",
            Vector::Zero(current.size()),
            physical_model, computational_model, absorbing_grid);
    }

    std::cout << "\nKIRCHHOFF-FIELD HYBRID: initial residual="
              << std::scientific << initial_metric.relative_residual
              << ", error=" << initial_metric.relative_error << '\n';
    const auto started = Clock::now();
    int fine_steps = 0;
    for (int outer = 1; outer <= p.outer_max; ++outer) {
        KirchhoffCoarseStageMetric stage;
        stage.outer = outer;
        stage.fine_steps_before = fine_steps;
        stage.residual_before = residual.norm() / rhs_norm;

        const Vector coarse_rhs = coarse.basis.adjoint() * residual;
        const Vector envelope = coarse.solver->solve(coarse_rhs);
        if (coarse.solver->info() != Eigen::Success) {
            throw std::runtime_error(
                "Kirchhoff-field coarse triangular solve failed");
        }
        const Vector coarse_correction = coarse.basis * envelope;
        const Vector coarse_action = coarse.action_basis * envelope;
        const Complex denominator = coarse_action.dot(coarse_action);
        Complex alpha = Complex(0.0, 0.0);
        if (std::abs(denominator) > std::numeric_limits<double>::epsilon()) {
            alpha = coarse_action.dot(residual) / denominator;
        }
        if (std::abs(alpha) > p.alpha_cap) {
            alpha = p.alpha_cap * alpha / std::abs(alpha);
        }
        Vector half = current + alpha * coarse_correction;
        Vector residual_half = residual - alpha * coarse_action;
        if (residual_half.norm() > residual.norm() * (1.0 + 1.0e-10)) {
            alpha *= 0.5;
            half = current + alpha * coarse_correction;
            residual_half = residual - alpha * coarse_action;
        }

        stage.alpha = alpha;
        stage.coarse_relative_residual = residual_half.norm() / rhs_norm;
        stage.coarse_true_relative_residual =
            (rhs - matrix * half).norm() / rhs_norm;
        stage.coarse_relative_error = relative_error(half, reference);
        if (reference && reference_norm > 0.0) {
            stage.correction_norm_over_reference =
                (alpha * coarse_correction).norm() / reference_norm;
        }
        coarse_stages.push_back(stage);

        const FixedGmresResult gmres = gmres_correction_fixed(
            matrix, residual_half, p.gmres_steps_per_outer);
        current = half + gmres.correction;
        residual = gmres.residual_after;
        fine_steps += gmres.steps;
        const Vector true_residual = rhs - matrix * current;

        StateMetric metric;
        metric.outer = outer;
        metric.fine_steps = fine_steps;
        metric.relative_residual = residual.norm() / rhs_norm;
        metric.true_relative_residual = true_residual.norm() / rhs_norm;
        metric.relative_error = relative_error(current, reference);
        metric.alpha = alpha;
        if (reference && reference_norm > 0.0) {
            metric.coarse_correction_norm_over_reference =
                (alpha * coarse_correction).norm() / reference_norm;
            metric.gmres_correction_norm_over_reference =
                gmres.correction.norm() / reference_norm;
        } else {
            metric.coarse_correction_norm_over_reference =
                std::numeric_limits<double>::quiet_NaN();
            metric.gmres_correction_norm_over_reference =
                std::numeric_limits<double>::quiet_NaN();
        }
        metric.elapsed = elapsed_seconds(started);
        result.metrics.push_back(metric);
        if (p.save_frame_fields) {
            const std::string id = three_digits(outer);
            write_physical_complex_field(
                field_directory / ("hybrid_outer_" + id), current,
                physical_model, computational_model, absorbing_grid);
            write_physical_complex_field(
                field_directory / ("kirchhoff_coarsecorr_outer_" + id),
                alpha * coarse_correction,
                physical_model, computational_model, absorbing_grid);
            write_physical_complex_field(
                field_directory / ("gmrescorr_outer_" + id),
                gmres.correction,
                physical_model, computational_model, absorbing_grid);
            if (outer == 1) {
                write_real_imag_field(
                    extended_field_directory /
                        "kirchhoff_coarsecorr_outer_001",
                    alpha * coarse_correction);
            }
        }
        std::cout << "  outer=" << std::setw(2) << outer
                  << ", fine=" << std::setw(3) << fine_steps
                  << ", before=" << std::scientific << stage.residual_before
                  << ", coarse=" << stage.coarse_relative_residual
                  << ", combined=" << metric.relative_residual
                  << ", error=" << metric.relative_error
                  << ", |alpha|=" << std::abs(alpha) << '\n';
        if (metric.relative_residual <= p.target_relres) break;
    }
    result.final = std::move(current);
    result.residual = std::move(residual);
    return result;
}

SnapshotResult continuous_gmres_selected_physical_outputs(
    const SparseMatrix& matrix,
    const Vector& rhs,
    const Vector& initial,
    const std::vector<int>& requested_steps,
    const Vector* reference,
    const fs::path& field_directory,
    const std::string& prefix,
    bool save_fields,
    const Model& physical_model,
    const Model& computational_model,
    const ExplicitAbsorbingGrid& absorbing_grid)
{
    if (requested_steps.empty() || requested_steps.front() != 0) {
        throw std::invalid_argument("selected GMRES steps must start at zero");
    }
    const int maximum_steps = requested_steps.back();
    const int n = static_cast<int>(rhs.size());
    const double rhs_norm = rhs.norm();
    const Vector initial_residual = rhs - matrix * initial;
    const double beta = initial_residual.norm();
    SnapshotResult result;
    result.metrics.push_back({0, beta / rhs_norm, beta / rhs_norm,
                              relative_error(initial, reference)});
    if (save_fields) {
        write_physical_complex_field(
            field_directory / (prefix + "_step_000"), initial,
            physical_model, computational_model, absorbing_grid);
    }
    result.final = initial;
    if (maximum_steps == 0 || beta == 0.0) return result;

    DenseMatrix basis = DenseMatrix::Zero(n, maximum_steps + 1);
    DenseMatrix hessenberg =
        DenseMatrix::Zero(maximum_steps + 1, maximum_steps);
    basis.col(0) = initial_residual / beta;
    std::size_t next = 1;
    for (int j = 0; j < maximum_steps; ++j) {
        Vector vector = matrix * basis.col(j);
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i <= j; ++i) {
                const Complex coefficient = basis.col(i).dot(vector);
                hessenberg(i, j) += coefficient;
                vector -= coefficient * basis.col(i);
            }
        }
        hessenberg(j + 1, j) = vector.norm();
        if (std::abs(hessenberg(j + 1, j)) > 0.0) {
            basis.col(j + 1) = vector / hessenberg(j + 1, j);
        }
        const int step = j + 1;
        if (next < requested_steps.size() && step == requested_steps[next]) {
            Vector small_rhs = Vector::Zero(step + 1);
            small_rhs[0] = beta;
            const DenseMatrix small =
                hessenberg.block(0, 0, step + 1, step);
            const Vector coefficients = least_squares(small, small_rhs);
            const Vector current =
                initial + basis.leftCols(step) * coefficients;
            const Vector small_residual = small_rhs - small * coefficients;
            const double arnoldi_relative =
                small_residual.norm() / rhs_norm;
            const double true_relative =
                (rhs - matrix * current).norm() / rhs_norm;
            result.metrics.push_back(
                {step, arnoldi_relative, true_relative,
                 relative_error(current, reference)});
            result.final = current;
            if (save_fields) {
                write_physical_complex_field(
                    field_directory /
                        (prefix + "_step_" + three_digits(step)),
                    current,
                    physical_model, computational_model, absorbing_grid);
            }
            ++next;
        }
        if (std::abs(hessenberg(j + 1, j)) == 0.0) break;
    }
    return result;
}

Vector load_split_float_reference(const std::string& real_file,
                                  const std::string& imag_file,
                                  int count)
{
    const auto read_component = [count](const std::string& filename) {
        std::ifstream stream(filename, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot open reference: " + filename);
        std::vector<float> values(static_cast<std::size_t>(count));
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (stream.gcount() != static_cast<std::streamsize>(
                values.size() * sizeof(float))) {
            throw std::runtime_error("split reference has the wrong size: " + filename);
        }
        char extra = 0;
        if (stream.read(&extra, 1)) {
            throw std::runtime_error("split reference has extra samples: " + filename);
        }
        return values;
    };
    const std::vector<float> real = read_component(real_file);
    const std::vector<float> imag = read_component(imag_file);
    Vector output(count);
    for (int index = 0; index < count; ++index) {
        output[index] = Complex(real[static_cast<std::size_t>(index)],
                                imag[static_cast<std::size_t>(index)]);
    }
    return output;
}

void write_block_metrics(const fs::path& filename,
                         const std::vector<BlockMetric>& metrics,
                         int physical_z0)
{
    std::ofstream stream(filename);
    stream << std::setprecision(17)
           << "block,output_start_iz,output_end_iz,datum_iz,fmm_sources,"
              "computational_output_start_iz,computational_output_end_iz,"
              "computational_datum_iz,"
              "table_megabytes,fmm_seconds,butterfly_build_seconds,"
              "butterfly_apply_seconds,direct_check_seconds,"
              "direct_relative_error,overlap_scale_real,overlap_scale_imag,"
              "overlap_mismatch_before,overlap_mismatch_after,min_tau,max_tau\n";
    for (const BlockMetric& metric : metrics) {
        stream << metric.block.id + 1 << ','
               << metric.block.output_start_iz - physical_z0 << ','
               << metric.block.output_end_iz - physical_z0 << ','
               << metric.block.datum_iz - physical_z0 << ','
               << metric.fmm_sources << ','
               << metric.block.output_start_iz << ','
               << metric.block.output_end_iz << ','
               << metric.block.datum_iz << ','
               << metric.table_megabytes << ','
               << metric.fmm_seconds << ','
               << metric.butterfly_build_seconds << ','
               << metric.butterfly_apply_seconds << ','
               << metric.direct_check_seconds << ','
               << metric.direct_relative_error << ','
               << metric.overlap_scale.real() << ','
               << metric.overlap_scale.imag() << ','
               << metric.overlap_mismatch_before << ','
               << metric.overlap_mismatch_after << ','
               << metric.minimum_traveltime << ','
               << metric.maximum_traveltime << '\n';
    }
}

void write_recursive_hybrid_csv(const fs::path& filename,
                                const std::vector<StateMetric>& metrics)
{
    std::ofstream stream(filename);
    stream << std::setprecision(17)
           << "outer,fine_gmres_steps,hybrid_residual,hybrid_true_residual,"
              "hybrid_error,alpha_real,alpha_imag,abs_alpha,"
              "kirchhoff_field_coarse_norm_ref,gmres_correction_norm_ref,"
              "hybrid_cumulative_time\n";
    for (const StateMetric& metric : metrics) {
        stream << metric.outer << ',' << metric.fine_steps << ','
               << metric.relative_residual << ','
               << metric.true_relative_residual << ','
               << metric.relative_error << ',' << metric.alpha.real() << ','
               << metric.alpha.imag() << ',' << std::abs(metric.alpha) << ','
               << metric.coarse_correction_norm_over_reference << ','
               << metric.gmres_correction_norm_over_reference << ','
               << metric.elapsed << '\n';
    }
}

void write_kirchhoff_coarse_stage_csv(
    const fs::path& filename,
    const std::vector<KirchhoffCoarseStageMetric>& metrics)
{
    std::ofstream stream(filename);
    stream << std::setprecision(17)
           << "outer,fine_steps_before,residual_before,"
              "coarse_residual,coarse_true_residual,coarse_error,"
              "alpha_real,alpha_imag,abs_alpha,"
              "coarse_correction_norm_ref\n";
    for (const KirchhoffCoarseStageMetric& metric : metrics) {
        stream << metric.outer << ',' << metric.fine_steps_before << ','
               << metric.residual_before << ','
               << metric.coarse_relative_residual << ','
               << metric.coarse_true_relative_residual << ','
               << metric.coarse_relative_error << ','
               << metric.alpha.real() << ',' << metric.alpha.imag() << ','
               << std::abs(metric.alpha) << ','
               << metric.correction_norm_over_reference << '\n';
    }
}

void write_recursive_restart_csv(const fs::path& filename,
                                 const RestartedResult& zero,
                                 const RestartedResult& kirchhoff)
{
    std::ofstream stream(filename);
    stream << std::setprecision(17)
           << "method,inner_iteration,relative_residual\n";
    for (std::size_t i = 0; i < zero.history_steps.size(); ++i) {
        stream << "zero_start," << zero.history_steps[i] << ','
               << zero.history_residuals[i] << '\n';
    }
    for (std::size_t i = 0; i < kirchhoff.history_steps.size(); ++i) {
        stream << "recursive_kirchhoff_start,"
               << kirchhoff.history_steps[i] << ','
               << kirchhoff.history_residuals[i] << '\n';
    }
}

struct AlgorithmTiming {
    double fmm_seconds = 0.0;
    double butterfly_build_seconds = 0.0;
    double butterfly_apply_seconds = 0.0;
    double butterfly_only_seconds = 0.0;
    double recursive_initial_core_seconds = 0.0;
    double recursive_initial_production_seconds = 0.0;
    double recursive_measured_wall_seconds = 0.0;
    double direct_validation_seconds = 0.0;
    double coarse_setup_seconds = 0.0;
    double hybrid_iteration_seconds = 0.0;
    double recursive_plus_iteration_seconds = 0.0;
};

AlgorithmTiming summarize_algorithm_timing(
    const RecursiveResult& recursive,
    const HybridResult& hybrid,
    double coarse_setup_seconds)
{
    AlgorithmTiming timing;
    for (const BlockMetric& metric : recursive.block_metrics) {
        timing.fmm_seconds += metric.fmm_seconds;
        timing.butterfly_build_seconds += metric.butterfly_build_seconds;
        timing.butterfly_apply_seconds += metric.butterfly_apply_seconds;
        timing.direct_validation_seconds += metric.direct_check_seconds;
    }
    timing.butterfly_only_seconds =
        timing.butterfly_build_seconds + timing.butterfly_apply_seconds;
    timing.recursive_initial_core_seconds =
        timing.fmm_seconds + timing.butterfly_only_seconds;
    timing.recursive_measured_wall_seconds = recursive.total_seconds;
    timing.recursive_initial_production_seconds = std::max(
        0.0,
        timing.recursive_measured_wall_seconds -
            timing.direct_validation_seconds);
    timing.coarse_setup_seconds = coarse_setup_seconds;
    if (!hybrid.metrics.empty()) {
        timing.hybrid_iteration_seconds = hybrid.metrics.back().elapsed;
    }
    timing.recursive_plus_iteration_seconds =
        timing.recursive_initial_production_seconds +
        timing.coarse_setup_seconds +
        timing.hybrid_iteration_seconds;
    return timing;
}

void write_algorithm_timing(const fs::path& csv_filename,
                            const fs::path& text_filename,
                            const AlgorithmTiming& timing,
                            int completed_outer_iterations,
                            int completed_fine_gmres_steps)
{
    std::ofstream csv(csv_filename);
    if (!csv) {
        throw std::runtime_error(
            "cannot create timing output: " + csv_filename.string());
    }
    csv << std::setprecision(17)
        << "metric,seconds\n"
        << "fmm_total," << timing.fmm_seconds << '\n'
        << "butterfly_build," << timing.butterfly_build_seconds << '\n'
        << "butterfly_apply," << timing.butterfly_apply_seconds << '\n'
        << "butterfly_only_build_plus_apply,"
        << timing.butterfly_only_seconds << '\n'
        << "recursive_initial_core_fmm_plus_butterfly,"
        << timing.recursive_initial_core_seconds << '\n'
        << "recursive_initial_production_excluding_validation,"
        << timing.recursive_initial_production_seconds << '\n'
        << "kirchhoff_field_coarse_setup," << timing.coarse_setup_seconds << '\n'
        << "global_hybrid_iteration," << timing.hybrid_iteration_seconds << '\n'
        << "recursive_initial_plus_global_iteration,"
        << timing.recursive_plus_iteration_seconds << '\n'
        << "recursive_measured_wall_including_validation,"
        << timing.recursive_measured_wall_seconds << '\n'
        << "direct_validation_not_in_core,"
        << timing.direct_validation_seconds << '\n';

    std::ofstream text_output(text_filename);
    if (!text_output) {
        throw std::runtime_error(
            "cannot create timing output: " + text_filename.string());
    }
    text_output << std::fixed << std::setprecision(6)
        << "Algorithm timing summary (seconds)\n"
        << "==================================\n"
        << "FMM total                         : "
        << timing.fmm_seconds << '\n'
        << "Butterfly build                   : "
        << timing.butterfly_build_seconds << '\n'
        << "Butterfly apply                   : "
        << timing.butterfly_apply_seconds << '\n'
        << "Butterfly only (build + apply)    : "
        << timing.butterfly_only_seconds << '\n'
        << "Recursive initial core (FMM + BF) : "
        << timing.recursive_initial_core_seconds << '\n'
        << "Recursive initial production       : "
        << timing.recursive_initial_production_seconds << '\n'
        << "Kirchhoff-field coarse setup      : "
        << timing.coarse_setup_seconds << '\n'
        << "Global hybrid iteration           : "
        << timing.hybrid_iteration_seconds << '\n'
        << "Initial production + setup + iter : "
        << timing.recursive_plus_iteration_seconds << '\n'
        << "Completed outer iterations        : "
        << completed_outer_iterations << '\n'
        << "Completed fine GMRES steps        : "
        << completed_fine_gmres_steps << "\n\n"
        << "Notes:\n"
        << "1. Butterfly-only time excludes every FMM solve.\n"
        << "2. Recursive-initial core is the sum of measured FMM and butterfly times.\n"
        << "3. Recursive-initial production is the measured recursive wall time\n"
        << "   after removing optional direct-summation validation.\n"
        << "4. Initial+iteration uses the production time and includes the\n"
        << "   Kirchhoff-field coarse-space setup, but excludes\n"
        << "   sparse-direct reference, zero-start comparisons, plotting, and I/O.\n"
        << "5. Global iteration is timed in a no-snapshot pass, so field-file I/O\n"
        << "   is not charged to the iteration algorithm.\n"
        << "6. Recursive measured wall time is "
        << timing.recursive_measured_wall_seconds
        << " s and includes " << timing.direct_validation_seconds
        << " s of optional direct-summation validation.\n";
}

std::string json_number(double value)
{
    if (!std::isfinite(value)) return "null";
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

void write_recursive_metadata(
    const fs::path& filename,
    const Model& physical_model,
    const Model& computational_model,
    const ExplicitAbsorbingGrid& absorbing_grid,
    const AbsorbingAudit& absorbing_audit,
    const Parameters& p,
    const RecursiveSettings& settings,
    const RecursiveResult& recursive,
    Complex scale,
    double unscaled_residual,
    double initial_residual,
    double initial_error,
    double reference_residual,
    const HybridResult& hybrid,
    const RestartedResult& zero,
    const RestartedResult& kirchhoff,
    const std::vector<KirchhoffCoarseStageMetric>& coarse_stages,
    const KirchhoffCarrierDiagnostics& carrier,
    const AlgorithmTiming& timing)
{
    std::ofstream stream(filename);
    stream << std::setprecision(17)
           << "{\n"
           << "  \"method\": \"explicitly_padded_four_block_true_efmm_bf1d_kirchhoff_field_phase_coarse_gmres\",\n"
           << "  \"nx\": " << physical_model.nx << ",\n"
           << "  \"nz\": " << physical_model.nz << ",\n"
           << "  \"computational_nx\": " << computational_model.nx << ",\n"
           << "  \"computational_nz\": " << computational_model.nz << ",\n"
           << "  \"physical_x0_in_computational\": "
           << absorbing_grid.physical_x0 << ",\n"
           << "  \"physical_z0_in_computational\": "
           << absorbing_grid.physical_z0 << ",\n"
           << "  \"absorbing_padding_left\": "
           << absorbing_grid.padding_left << ",\n"
           << "  \"absorbing_padding_right\": "
           << absorbing_grid.padding_right << ",\n"
           << "  \"absorbing_padding_top\": "
           << absorbing_grid.padding_top << ",\n"
           << "  \"absorbing_padding_bottom\": "
           << absorbing_grid.padding_bottom << ",\n"
           << "  \"nabs\": " << p.nabs << ",\n"
           << "  \"damp_max\": " << p.damp_max << ",\n"
           << "  \"absorbing_boundary_kind\": "
              "\"explicit_edge_padded_quadratic_complex_sponge\",\n"
           << "  \"absorbing_velocity_extension\": "
              "\"constant_nearest_edge\",\n"
           << "  \"standard_field_grid\": \"physical_cropped\",\n"
           << "  \"verification_field_grid\": "
              "\"computational_with_absorbing_padding\",\n"
           << "  \"dx\": " << p.dx << ",\n"
           << "  \"dz\": " << p.dz << ",\n"
           << "  \"frequency\": " << p.frequency << ",\n"
           << "  \"source_x_matlab\": "
           << physical_model.source_x + 1 << ",\n"
           << "  \"source_z_matlab\": "
           << physical_model.source_z + 1 << ",\n"
           << "  \"source_x_computational_matlab\": "
           << computational_model.source_x + 1 << ",\n"
           << "  \"source_z_computational_matlab\": "
           << computational_model.source_z + 1 << ",\n"
           << "  \"source_absorbing_eta\": "
           << absorbing_audit.source_eta << ",\n"
           << "  \"maximum_absorbing_eta_in_physical_model\": "
           << absorbing_audit.maximum_physical_eta << ",\n"
           << "  \"source_inside_absorbing_layer\": false,\n"
           << "  \"damped_padding_samples\": "
           << absorbing_audit.damped_padding_samples << ",\n"
           << "  \"expected_padding_samples\": "
           << absorbing_audit.expected_padding_samples << ",\n"
           << "  \"block_count\": " << settings.block_count << ",\n"
           << "  \"overlap_rows\": " << settings.overlap_rows << ",\n"
           << "  \"fmm_backend\": \"SERECKIRCH efmm_solver\",\n"
           << "  \"butterfly_backend\": \"bf1d_strict_segmented_phase_amp\",\n"
           << "  \"kirchhoff_form\": \"simplified one-way value-only integral\",\n"
           << "  \"overlap_stitch\": \"complex least-squares fit plus linear cross-fade\",\n"
           << "  \"coarse_phase_source\": \"recursive_butterfly_kirchhoff_complex_field_only\",\n"
           << "  \"coarse_carrier_formula\": \"U_K/(abs(U_K)+epsilon)\",\n"
           << "  \"coarse_support_mask\": \"full_domain_chi_equals_one\",\n"
           << "  \"coarse_phase_uses_primary_fmm_traveltime\": false,\n"
           << "  \"coarse_phase_uses_exp_iomegaT\": false,\n"
           << "  \"coarse_phase_uses_reference_solution\": false,\n"
           << "  \"coarse_phase_epsilon_rms_fraction\": "
           << settings.coarse_phase_epsilon_rms_fraction << ",\n"
           << "  \"coarse_field_rms_amplitude\": "
           << carrier.field_rms_amplitude << ",\n"
           << "  \"coarse_phase_epsilon\": "
           << carrier.phase_epsilon << ",\n"
           << "  \"coarse_carrier_below_epsilon_samples\": "
           << carrier.below_epsilon_samples << ",\n"
           << "  \"coarse_carrier_zero_samples\": "
           << carrier.zero_samples << ",\n"
           << "  \"bf_p\": " << settings.bf_p << ",\n"
           << "  \"bf_leaf\": " << settings.bf_leaf << ",\n"
           << "  \"bf_phase_tol\": " << settings.bf_phase_tol << ",\n"
           << "  \"fixed_outer_iterations\": "
           << (settings.fixed_outer_iterations ? "true" : "false") << ",\n"
           << "  \"requested_outer_iterations\": " << p.outer_max << ",\n"
           << "  \"recursive_seconds\": " << recursive.total_seconds << ",\n"
           << "  \"primary_fmm_seconds\": " << recursive.primary_fmm_seconds << ",\n"
           << "  \"fmm_total_seconds\": " << timing.fmm_seconds << ",\n"
           << "  \"butterfly_build_seconds\": "
           << timing.butterfly_build_seconds << ",\n"
           << "  \"butterfly_apply_seconds\": "
           << timing.butterfly_apply_seconds << ",\n"
           << "  \"butterfly_only_seconds\": "
           << timing.butterfly_only_seconds << ",\n"
           << "  \"recursive_initial_core_seconds\": "
           << timing.recursive_initial_core_seconds << ",\n"
           << "  \"recursive_initial_production_seconds\": "
           << timing.recursive_initial_production_seconds << ",\n"
           << "  \"initial_scale_real\": " << scale.real() << ",\n"
           << "  \"initial_scale_imag\": " << scale.imag() << ",\n"
           << "  \"unscaled_initial_residual\": "
           << json_number(unscaled_residual) << ",\n"
           << "  \"scaled_initial_residual\": "
           << json_number(initial_residual) << ",\n"
           << "  \"scaled_initial_error\": "
           << json_number(initial_error) << ",\n"
           << "  \"reference_residual\": "
           << json_number(reference_residual) << ",\n"
           << "  \"coarse_setup_seconds\": "
           << timing.coarse_setup_seconds << ",\n"
           << "  \"hybrid_iteration_seconds\": "
           << timing.hybrid_iteration_seconds << ",\n"
           << "  \"recursive_plus_iteration_seconds\": "
           << timing.recursive_plus_iteration_seconds << ",\n"
           << "  \"hybrid_outer_iterations\": "
           << hybrid.metrics.back().outer << ",\n"
           << "  \"hybrid_fine_gmres_steps\": "
           << hybrid.metrics.back().fine_steps << ",\n"
           << "  \"hybrid_final_residual\": "
           << hybrid.metrics.back().relative_residual << ",\n"
           << "  \"hybrid_final_error\": "
           << json_number(hybrid.metrics.back().relative_error) << ",\n"
           << "  \"first_coarse_only_residual\": "
           << json_number(coarse_stages.size() > 1
                  ? coarse_stages[1].coarse_relative_residual
                  : std::numeric_limits<double>::quiet_NaN()) << ",\n"
           << "  \"first_coarse_only_error\": "
           << json_number(coarse_stages.size() > 1
                  ? coarse_stages[1].coarse_relative_error
                  : std::numeric_limits<double>::quiet_NaN()) << ",\n"
           << "  \"zero_gmres_iterations\": " << zero.inner_iterations << ",\n"
           << "  \"zero_gmres_final_residual\": "
           << zero.relative_residual << ",\n"
           << "  \"zero_gmres_final_error\": "
           << json_number(zero.relative_error) << ",\n"
           << "  \"kirchhoff_gmres_iterations\": "
           << kirchhoff.inner_iterations << ",\n"
           << "  \"kirchhoff_gmres_final_residual\": "
           << kirchhoff.relative_residual << ",\n"
           << "  \"kirchhoff_gmres_final_error\": "
           << json_number(kirchhoff.relative_error) << "\n"
           << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Arguments arguments(argc, argv);
        if (arguments.help()) {
            print_recursive_help();
            return 0;
        }
        Parameters p = parse_parameters(arguments);
        // The shared MATLAB-replica parser intentionally retains sz=31.
        // This recursive executable instead defaults to a physical surface
        // source; an explicit sz=... argument still takes precedence.
        p.sz_matlab = arguments.get_int("sz", 1);
        if (p.sz_matlab < 1 || p.sz_matlab > p.nz) {
            throw std::invalid_argument(
                "physical MATLAB source depth sz is outside the model");
        }
        const RecursiveSettings settings = parse_recursive_settings(arguments);
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        if (settings.threads > 0) omp_set_num_threads(settings.threads);
#endif
        const fs::path output = fs::absolute(p.output_dir);
        const fs::path fields = output / "fields";
        const fs::path extended_fields = output / "fields_extended";
        fs::create_directories(fields);
        fs::create_directories(extended_fields);

        std::cout << "Recursive true-FMM/bf1d Kirchhoff initial field + "
                     "Kirchhoff-field coarse correction + GMRES\n"
                  << "  model=" << p.model_file << "\n"
                  << "  requested frequency=" << p.frequency << " Hz\n";
        const Model physical_model = read_model(p);
        const Model computational_model =
            add_explicit_absorbing_padding(physical_model, p.nabs);
        const ExplicitAbsorbingGrid absorbing_grid = make_absorbing_grid(
            physical_model, computational_model, p.nabs);
        const int n = computational_model.nx * computational_model.nz;
        const FmmModel fmm_model = make_fmm_model(computational_model, p);
        write_float_field(
            fields / "velocity_m_per_s.f32", physical_model.velocity);
        write_float_field(
            extended_fields / "velocity_m_per_s.f32",
            computational_model.velocity);

        std::cout
            << "  physical grid=" << physical_model.nx << 'x'
            << physical_model.nz << ", source MATLAB physical=("
            << physical_model.source_x + 1 << ','
            << physical_model.source_z + 1 << ")\n"
            << "  computational grid=" << computational_model.nx << 'x'
            << computational_model.nz << ", source MATLAB computational=("
            << computational_model.source_x + 1 << ','
            << computational_model.source_z + 1 << ")\n"
            << "  explicit absorbing padding=" << p.nabs
            << " cells on every side\n";

        auto started = Clock::now();
        HelmholtzSystem system = build_helmholtz(computational_model, p);
        const double matrix_seconds = elapsed_seconds(started);
        const AbsorbingAudit absorbing_audit =
            audit_explicit_absorbing_grid(
                physical_model, computational_model, absorbing_grid, system);
        write_float_field(
            extended_fields / "absorbing_eta.f32", system.eta);
        const double rhs_norm = system.rhs.norm();
        std::cout << "Helmholtz: N=" << n
                  << ", nnz=" << system.matrix.nonZeros()
                  << ", build=" << matrix_seconds << " s\n"
                  << "  absorbing audit: eta(source)="
                  << absorbing_audit.source_eta
                  << ", max eta(physical)="
                  << absorbing_audit.maximum_physical_eta
                  << ", damped padding="
                  << absorbing_audit.damped_padding_samples << '/'
                  << absorbing_audit.expected_padding_samples << "\n";

        Vector reference;
        const Vector* reference_pointer = nullptr;
        if (!settings.reference_real.empty()) {
            reference = load_split_float_reference(
                settings.reference_real, settings.reference_imag, n);
            reference_pointer = &reference;
            std::cout << "Loaded split float32 Helmholtz reference.\n";
        } else if (p.reference_mode == "read") {
            reference = load_reference(p.reference_file, n);
            reference_pointer = &reference;
        } else if (p.reference_mode == "direct") {
            std::cout << "Computing sparse-direct reference...\n";
            reference = direct_reference(system.matrix, system.rhs);
            reference_pointer = &reference;
        }
        double reference_residual = std::numeric_limits<double>::quiet_NaN();
        if (reference_pointer) {
            reference_residual =
                (system.rhs - system.matrix * reference).norm() / rhs_norm;
            write_physical_complex_field(
                fields / "reference", reference,
                physical_model, computational_model, absorbing_grid);
            write_real_imag_field(
                extended_fields / "reference", reference);
            std::cout << "Reference relative residual=" << std::scientific
                      << reference_residual << std::fixed << '\n';
        }

        RecursiveResult recursive = build_recursive_initial(
            fmm_model,
            computational_model.source_x,
            computational_model.source_z,
            absorbing_grid.physical_z0,
            absorbing_grid.physical_nz,
            static_cast<float>(p.frequency), settings);
        write_physical_real_field(
            fields / "fmm_primary_traveltime_s.f32",
            recursive.primary_traveltime,
            physical_model, computational_model, absorbing_grid);
        for (std::size_t index = 0;
             index < recursive.datum_middle_source_traveltimes.size(); ++index) {
            write_physical_real_field(
                fields / ("fmm_block_" + three_digits(
                    static_cast<int>(index + 2)) +
                    "_middle_datum_source_traveltime_s.f32"),
                recursive.datum_middle_source_traveltimes[index],
                physical_model, computational_model, absorbing_grid);
        }

        // Traveltimes are required to construct the recursive Kirchhoff field,
        // but they are diagnostics only after that field exists. Release every
        // saved FMM field before coarse-space construction so the correction
        // stage cannot consume them, even accidentally.
        recursive.primary_traveltime.clear();
        recursive.primary_traveltime.shrink_to_fit();
        recursive.datum_middle_source_traveltimes.clear();
        recursive.datum_middle_source_traveltimes.shrink_to_fit();
        std::cout << "All FMM diagnostic arrays released before coarse setup.\n";

        Vector unscaled = one_way_to_helmholtz(
            fmm_model, computational_model, p, recursive.one_way_field);
        const double unscaled_residual =
            (system.rhs - system.matrix * unscaled).norm() / rhs_norm;
        const Vector action = system.matrix * unscaled;
        const Complex denominator = action.dot(action);
        Complex scale(1.0, 0.0);
        if (std::abs(denominator) > std::numeric_limits<double>::epsilon()) {
            scale = action.dot(system.rhs) / denominator;
        }
        const Vector initial = scale * unscaled;
        const double initial_residual =
            (system.rhs - system.matrix * initial).norm() / rhs_norm;
        const double initial_error = relative_error(initial, reference_pointer);
        write_physical_complex_field(
            fields / "kirchhoff_initial_unscaled", unscaled,
            physical_model, computational_model, absorbing_grid);
        write_physical_complex_field(
            fields / "kirchhoff_initial", initial,
            physical_model, computational_model, absorbing_grid);
        write_real_imag_field(
            extended_fields / "kirchhoff_initial_unscaled", unscaled);
        write_real_imag_field(
            extended_fields / "kirchhoff_initial", initial);
        for (std::size_t index = 0;
             index < recursive.block_snapshots.size(); ++index) {
            Vector snapshot = one_way_to_helmholtz(
                fmm_model, computational_model, p,
                recursive.block_snapshots[index]);
            snapshot *= scale;
            write_physical_complex_field(
                fields / ("kirchhoff_block_" + three_digits(
                    static_cast<int>(index + 1))), snapshot,
                physical_model, computational_model, absorbing_grid);
        }
        std::cout << "Recursive Kirchhoff initial: raw residual="
                  << std::scientific << unscaled_residual
                  << ", scaled residual=" << initial_residual
                  << ", error=" << initial_error
                  << ", scale=" << scale << std::fixed << '\n';

        std::cout << "Building recursive-Kirchhoff-field coarse space q="
                  << p.q_coarse << "...\n";
        KirchhoffCarrierDiagnostics carrier;
        CoarseSpace coarse = build_kirchhoff_field_coarse_space(
            system.matrix, computational_model, p, unscaled,
            settings.coarse_phase_epsilon_rms_fraction, carrier);
        write_physical_complex_field(
            fields / "kirchhoff_coarse_carrier", carrier.carrier,
            physical_model, computational_model, absorbing_grid);
        write_physical_real_field(
            fields / "kirchhoff_coarse_phase_weight.f32",
            carrier.phase_regularization_weight,
            physical_model, computational_model, absorbing_grid);
        write_real_imag_field(
            extended_fields / "kirchhoff_coarse_carrier",
            carrier.carrier);
        write_float_field(
            extended_fields / "kirchhoff_coarse_phase_weight.f32",
            carrier.phase_regularization_weight);
        Parameters hybrid_parameters = p;
        if (settings.fixed_outer_iterations) {
            // Keep target_relres unchanged for the restarted-GMRES comparison,
            // but prevent run_hybrid from stopping before outer_max.
            hybrid_parameters.target_relres =
                std::numeric_limits<double>::min();
            std::cout << "Running exactly " << p.outer_max
                      << " hybrid outer iterations (early stopping disabled).\n";
        }
        Parameters timing_parameters = hybrid_parameters;
        timing_parameters.save_frame_fields = false;
        std::cout << "Timing global hybrid iteration without snapshot I/O...\n";
        std::vector<KirchhoffCoarseStageMetric> timed_coarse_stages;
        const HybridResult timed_hybrid = run_kirchhoff_field_hybrid(
            system.matrix, system.rhs, initial, coarse,
            reference_pointer, timing_parameters, fields, extended_fields,
            physical_model, computational_model, absorbing_grid,
            timed_coarse_stages);

        HybridResult hybrid;
        std::vector<KirchhoffCoarseStageMetric> coarse_stages;
        if (p.save_frame_fields) {
            std::cout << "Repeating deterministic hybrid iteration to write "
                         "the requested plot frames...\n";
            hybrid = run_kirchhoff_field_hybrid(
                system.matrix, system.rhs, initial, coarse,
                reference_pointer, hybrid_parameters, fields, extended_fields,
                physical_model, computational_model, absorbing_grid,
                coarse_stages);
        } else {
            hybrid = timed_hybrid;
            coarse_stages = timed_coarse_stages;
        }
        write_physical_complex_field(
            fields / "hybrid_final", hybrid.final,
            physical_model, computational_model, absorbing_grid);
        write_real_imag_field(
            extended_fields / "hybrid_final", hybrid.final);

        std::vector<int> matched_steps;
        for (const StateMetric& metric : hybrid.metrics) {
            matched_steps.push_back(metric.fine_steps);
        }
        const SnapshotResult zero_matched =
            continuous_gmres_selected_physical_outputs(
            system.matrix, system.rhs, Vector::Zero(n), matched_steps,
            reference_pointer, fields, "zero", p.save_frame_fields,
            physical_model, computational_model, absorbing_grid);

        const RestartedResult zero_target = restarted_gmres_to_target(
            system.matrix, system.rhs, Vector::Zero(n), p.zero_restart,
            p.target_relres, p.zero_max_cycles, reference_pointer,
            "zero-start GMRES");
        const RestartedResult kirchhoff_target = restarted_gmres_to_target(
            system.matrix, system.rhs, initial, p.zero_restart,
            p.target_relres, p.zero_max_cycles, reference_pointer,
            "recursive-Kirchhoff-start GMRES");
        write_physical_complex_field(
            fields / "zero_restarted_target", zero_target.final,
            physical_model, computational_model, absorbing_grid);
        write_physical_complex_field(
            fields / "kirchhoff_restarted_target", kirchhoff_target.final,
            physical_model, computational_model, absorbing_grid);
        write_real_imag_field(
            extended_fields / "zero_restarted_target", zero_target.final);
        write_real_imag_field(
            extended_fields / "kirchhoff_restarted_target",
            kirchhoff_target.final);

        const AlgorithmTiming timing = summarize_algorithm_timing(
            recursive, timed_hybrid, coarse.setup_seconds);

        write_block_metrics(
            output / "recursive_block_metrics.csv",
            recursive.block_metrics, absorbing_grid.physical_z0);
        write_recursive_hybrid_csv(output / "hybrid_outer_metrics.csv",
                                   hybrid.metrics);
        write_kirchhoff_coarse_stage_csv(
            output / "kirchhoff_coarse_stage_metrics.csv",
            coarse_stages);
        write_matched_csv(output / "matched_step_comparison.csv",
                          hybrid.metrics, zero_matched);
        write_recursive_restart_csv(
            output / "restarted_gmres_histories.csv",
            zero_target, kirchhoff_target);
        write_algorithm_timing(
            output / "algorithm_timing.csv",
            output / "algorithm_timing.txt",
            timing,
            hybrid.metrics.back().outer,
            hybrid.metrics.back().fine_steps);
        write_recursive_metadata(
            output / "metadata.json",
            physical_model, computational_model,
            absorbing_grid, absorbing_audit,
            p, settings, recursive, scale,
            unscaled_residual, initial_residual, initial_error,
            reference_residual, hybrid, zero_target, kirchhoff_target,
            coarse_stages, carrier, timing);

        std::ofstream log(output / "results.txt");
        log << std::setprecision(12) << std::scientific
            << "recursive_kirchhoff_initial_residual " << initial_residual << '\n'
            << "recursive_kirchhoff_initial_error " << initial_error << '\n'
            << "hybrid_final_residual "
            << hybrid.metrics.back().relative_residual << '\n'
            << "hybrid_final_error "
            << hybrid.metrics.back().relative_error << '\n'
            << "hybrid_outer_iterations "
            << hybrid.metrics.back().outer << '\n'
            << "hybrid_fine_gmres_steps "
            << hybrid.metrics.back().fine_steps << '\n'
            << "first_coarse_only_residual "
            << (coarse_stages.size() > 1
                    ? coarse_stages[1].coarse_relative_residual
                    : std::numeric_limits<double>::quiet_NaN()) << '\n'
            << "first_coarse_only_error "
            << (coarse_stages.size() > 1
                    ? coarse_stages[1].coarse_relative_error
                    : std::numeric_limits<double>::quiet_NaN()) << '\n'
            << "coarse_phase_uses_primary_fmm_traveltime 0\n"
            << "coarse_phase_uses_reference_solution 0\n"
            << "coarse_field_rms_amplitude "
            << carrier.field_rms_amplitude << '\n'
            << "coarse_phase_epsilon "
            << carrier.phase_epsilon << '\n'
            << "coarse_carrier_below_epsilon_samples "
            << carrier.below_epsilon_samples << '\n'
            << "physical_grid_nx_nz "
            << physical_model.nx << ' ' << physical_model.nz << '\n'
            << "computational_grid_nx_nz "
            << computational_model.nx << ' '
            << computational_model.nz << '\n'
            << "physical_source_matlab "
            << physical_model.source_x + 1 << ' '
            << physical_model.source_z + 1 << '\n'
            << "computational_source_matlab "
            << computational_model.source_x + 1 << ' '
            << computational_model.source_z + 1 << '\n'
            << "source_absorbing_eta "
            << absorbing_audit.source_eta << '\n'
            << "maximum_absorbing_eta_in_physical_model "
            << absorbing_audit.maximum_physical_eta << '\n'
            << "butterfly_only_seconds "
            << timing.butterfly_only_seconds << '\n'
            << "recursive_initial_core_seconds "
            << timing.recursive_initial_core_seconds << '\n'
            << "recursive_initial_production_seconds "
            << timing.recursive_initial_production_seconds << '\n'
            << "hybrid_iteration_seconds "
            << timing.hybrid_iteration_seconds << '\n'
            << "recursive_plus_iteration_seconds "
            << timing.recursive_plus_iteration_seconds << '\n'
            << "zero_target_steps " << zero_target.inner_iterations << '\n'
            << "kirchhoff_target_steps "
            << kirchhoff_target.inner_iterations << '\n';

        std::cout << "\n================ RECURSIVE KIRCHHOFF SUMMARY ================\n"
                  << "FMM backend: SERECKIRCH efmm_solver\n"
                  << "Butterfly backend: bf1d_strict_segmented_phase_amp\n"
                  << "Coarse phase source: recursive Kirchhoff complex field only\n"
                  << "Primary-FMM/reference phase used by coarse setup: no/no\n"
                  << "Physical/computational grid: "
                  << physical_model.nx << 'x' << physical_model.nz << " / "
                  << computational_model.nx << 'x'
                  << computational_model.nz << '\n'
                  << "Physical/computational source MATLAB: ("
                  << physical_model.source_x + 1 << ','
                  << physical_model.source_z + 1 << ") / ("
                  << computational_model.source_x + 1 << ','
                  << computational_model.source_z + 1 << ")\n"
                  << "Absorbing eta at source/max physical: "
                  << absorbing_audit.source_eta << '/'
                  << absorbing_audit.maximum_physical_eta << '\n'
                  << "Blocks/overlap: 4/" << settings.overlap_rows << " rows\n"
                  << "Initial residual/error: " << std::scientific
                  << initial_residual << '/' << initial_error << '\n'
                  << "Hybrid final residual/error: "
                  << hybrid.metrics.back().relative_residual << '/'
                  << hybrid.metrics.back().relative_error << '\n'
                  << "Completed outer/fine iterations: "
                  << hybrid.metrics.back().outer << '/'
                  << hybrid.metrics.back().fine_steps << '\n'
                  << "Butterfly only time: "
                  << timing.butterfly_only_seconds << " s\n"
                  << "Recursive production + iteration time: "
                  << timing.recursive_plus_iteration_seconds << " s\n"
                  << "Zero/Kirchhoff restarted steps: "
                  << zero_target.inner_iterations << '/'
                  << kirchhoff_target.inner_iterations << '\n'
                  << "Outputs: " << output << '\n'
                  << "==============================================================\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
