/*
 * ray_helmholtz_iteration.cpp
 *
 * Purpose
 * -------
 * 1. Read the existing block information and precomputed traveltime tables.
 * 2. Use the repository's 1-D butterfly factorization to propagate only the
 *    wavefield on the upper datum of each block to the lower datum.
 * 3. Reconstruct the volume wavefield inside the block with a local
 *    Helmholtz solve.  The butterfly traces are therefore used as interface
 *    data, instead of evaluating a Kirchhoff integral at every volume point.
 * 4. March downward block by block.  A positive overlap is recommended: the
 *    next datum then lies inside the previously corrected block, so the local
 *    Helmholtz correction is fed back into the next butterfly propagation.
 * 5. Write the assembled wavefield and diagnostic residual information.
 *
 * This is deliberately a prototype of the interface-trace idea.  It tests the
 * most useful part of the polarized-trace paper for this repository: solve in
 * the volume from interface data, while keeping the expensive oscillatory
 * propagation on lower-dimensional interfaces.  It is not yet the full
 * bidirectional polarized-trace GMRES system of Zepeda-Nunez & Demanet (2016).
 */

#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_fs.h>
#include <SEFILESYSTEM/include/se_par_sep.h>
#include <SERECKIRCH/include/bf1d.h>
#include <SERECKIRCH/include/huygens_sweep.hpp>
#include <se_eigen.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using Complex = se::huygens::Complex;
using EigenVector = se::eigen::DenseVector<double>;
using EigenMatrix = se::eigen::SparseMatrix<double>;
using FFTStorage = std::vector<float>;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct FactorDeleter {
    void operator()(BFStrictSegmentedFactor* factor) const
    {
        bf1d_strict_segmented_destroy(factor);
    }
};
using FactorPointer = std::unique_ptr<BFStrictSegmentedFactor, FactorDeleter>;

struct LocalHelmholtzSystem {
    int top_iz = 0;
    int bottom_iz = 0;
    int nx_interior = 0;
    int nz_interior = 0;
    double coefficient_x = 0.0;
    double coefficient_z = 0.0;
    EigenMatrix matrix;

    int unknowns() const noexcept
    {
        return nx_interior * nz_interior;
    }

    int row(int ix, int iz) const
    {
        return (iz - top_iz - 1) * nx_interior + (ix - 1);
    }
};

struct CorrectionMetrics {
    double residual_before = 0.0;
    double residual_after = 0.0;
    double residual_ratio = 1.0;
    double relative_update = 0.0;
    double relaxation = 0.0;
    int real_iterations = 0;
    int imag_iterations = 0;
    se::eigen::SolverStatus real_status = se::eigen::SolverStatus::success;
    se::eigen::SolverStatus imag_status = se::eigen::SolverStatus::success;
};

struct ResidualSummary {
    double relative_residual = 0.0;
    double bulk_rms = 0.0;
    double seam_rms = 0.0;
    double seam_to_bulk = 0.0;
};

static double elapsed_seconds(const Clock::time_point& started)
{
    return std::chrono::duration<double>(Clock::now() - started).count();
}

static void print_help(const char* program)
{
    std::printf(
        "Usage:\n"
        "  %s velocity=... block_file=... table_prefix=... [key=value ...]\n\n"
        "Required:\n"
        "  velocity=MODEL.rsf\n"
        "  block_file=block_info.dat\n"
        "  table_prefix=huygens_tt/travel\n\n"
        "Main parameters:\n"
        "  output_prefix=output/ray_helmholtz_iteration\n"
        "  frequency=25\n"
        "  source_ix=-1                 (-1 means nx/2)\n"
        "  source_iz=0                  (prototype requires surface source)\n"
        "  source_amplitude=1\n"
        "  source_radius=0\n\n"
        "Butterfly parameters:\n"
        "  bf_p=12\n"
        "  bf_n_leaf=16\n"
        "  bf_panel_levels=1\n"
        "  bf_amp_eps=1e-20\n"
        "  bf_phase_tol=1\n"
        "  filter_dt=0.001\n"
        "  filter_length=0.025\n"
        "  filter_lookup_subsamples=64\n\n"
        "Local Helmholtz correction:\n"
        "  iter_cycles=3\n"
        "  iter_iterations=10\n"
        "  iter_restart=10\n"
        "  iter_tolerance=1e-4\n"
        "  iter_preconditioner=diagonal   (identity, diagonal, ilut)\n"
        "  iter_ilut_drop_tolerance=1e-3\n"
        "  iter_ilut_fill_factor=10\n"
        "  iter_relaxation=-1             (-1 = residual-minimizing)\n"
        "  iter_max_relaxation=1\n\n"
        "Output/debug:\n"
        "  write_interfaces=1\n"
        "  write_each_block=0\n"
        "  threads=0                     (0 keeps OpenMP default)\n\n",
        program);
}

static bool help_requested(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == nullptr) continue;
        if (std::strcmp(argv[i], "-h") == 0 ||
            std::strcmp(argv[i], "--help") == 0) {
            return true;
        }
    }
    return se_have_par("help") && se_get_par_int("help") != 0;
}

static void set_threads(int threads)
{
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif
}

static void copy_to_fftw(const std::vector<Complex>& input, FFTStorage& output)
{
    output.resize(2 * input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[2 * i] = input[i].real();
        output[2 * i + 1] = input[i].imag();
    }
}

static void write_trace_component_rsf(const std::string& filename,
                                      const se::huygens::Model2D& model,
                                      const std::vector<Complex>& trace,
                                      float frequency,
                                      int block_id,
                                      int iz,
                                      const char* kind,
                                      bool imaginary)
{
    if (static_cast<int>(trace.size()) != model.nx) {
        throw std::invalid_argument("interface trace length does not equal nx");
    }

    se::huygens::ensure_parent_directory(filename);
    std::vector<float> values(trace.size());
    for (std::size_t i = 0; i < trace.size(); ++i) {
        values[i] = imaginary ? trace[i].imag() : trace[i].real();
    }

    sep_t* output = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (output == nullptr || output->headers == nullptr ||
        output->data == nullptr || output->data->io == nullptr) {
        throw std::runtime_error("cannot create interface RSF: " + filename);
    }

    output->headers->ndim = 1;
    output->headers->n[0] = model.nx;
    output->headers->d[0] = model.dx;
    output->headers->o[0] = model.ox;
    output->headers->esize = 4;
    output->headers->le = 1;
    sep_set_header(output, "data_format", "native_float");
    sep_set_header_int(output, "esize", 4);
    sep_set_header(output, "label1", "Distance");
    sep_set_header(output, "unit1", "m");
    sep_set_header(output, "trace_kind", kind);
    sep_set_header(output, "component", imaginary ? "imaginary" : "real");
    sep_set_header_int(output, "block_id", block_id);
    sep_set_header_int(output, "iz", iz);
    sep_set_header_float(output, "depth", model.z(iz));
    sep_set_header_float(output, "frequency", frequency);
    se_fsio_write_float(output->data->io, values.data(), values.size());
    sep_close(output);
}

static std::string block_trace_filename(const std::string& prefix,
                                        int block_id,
                                        const char* kind,
                                        const char* component)
{
    std::ostringstream stream;
    stream << prefix << "_block_" << std::setw(3) << std::setfill('0')
           << block_id << '_' << kind << '_' << component << ".rsf";
    return stream.str();
}

static std::string block_field_filename(const std::string& prefix,
                                        int block_id,
                                        const char* component)
{
    std::ostringstream stream;
    stream << prefix << "_after_block_" << std::setw(3) << std::setfill('0')
           << block_id << '_' << component << ".rsf";
    return stream.str();
}

static FactorPointer build_bottom_interface_factor(
    const se::huygens::Model2D& model,
    const se::huygens::Block& block,
    const se::huygens::LayerGeometry& geometry,
    const se::huygens::OneWayLayerTables& tables,
    const se::huygens::FrequencyKirchhoffFilter& filter,
    float omega,
    int p,
    int leaf,
    int panel_levels,
    float amp_eps,
    float phase_tol)
{
    if (static_cast<int>(geometry.source_ix.size()) != model.nx ||
        static_cast<int>(geometry.target_ix.size()) != model.nx) {
        throw std::runtime_error(
            "interface butterfly requires one source/target per lateral grid point");
    }

    const auto target_it = std::find(
        geometry.target_iz.begin(), geometry.target_iz.end(), block.target_end_iz);
    if (target_it == geometry.target_iz.end()) {
        throw std::runtime_error("target_end_iz is absent from the traveltime table");
    }
    const int target_depth_index = static_cast<int>(
        std::distance(geometry.target_iz.begin(), target_it));
    const int target_depth_count = static_cast<int>(geometry.target_iz.size());

    const std::vector<float> weights =
        se::huygens::trapezoidal_weights(model, geometry.source_ix);

    se::huygens::OneWayKernelData kernel;
    kernel.rows = static_cast<int>(geometry.targets.size());
    kernel.sources = model.nx;
    kernel.source_z = model.z(block.source_iz);
    kernel.quadrature_weights = &weights;
    kernel.tables = &tables;
    kernel.filter = &filter;

    std::vector<float> tau_storage(
        static_cast<std::size_t>(model.nx) * model.nx);
    std::vector<float*> tau_rows(static_cast<std::size_t>(model.nx));
    FFTStorage amplitude(
        2 * static_cast<std::size_t>(model.nx) * model.nx);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ix_target = 0; ix_target < model.nx; ++ix_target) {
        tau_rows[static_cast<std::size_t>(ix_target)] =
            tau_storage.data() + static_cast<std::size_t>(ix_target) * model.nx;
        const int row = ix_target * target_depth_count + target_depth_index;

        for (int source = 0; source < model.nx; ++source) {
            const std::size_t matrix_index =
                static_cast<std::size_t>(ix_target) * model.nx + source;
            const float tau = kernel.traveltime(row, source);
            const Complex value = kernel.phase_removed_amplitude(row, source, tau);
            tau_storage[matrix_index] = tau;
            amplitude[2 * matrix_index] = value.real();
            amplitude[2 * matrix_index + 1] = value.imag();
        }
    }

    BFStrictSegmentedFactor* raw = bf1d_strict_segmented_create_phase_amp(
        model.nx,
        tau_rows.data(),
        reinterpret_cast<const fftwf_complex*>(amplitude.data()),
        omega,
        p,
        leaf,
        panel_levels,
        amp_eps,
        phase_tol);

    if (raw == nullptr) {
        throw std::runtime_error("butterfly factor construction returned null");
    }
    return FactorPointer(raw);
}

static std::vector<Complex> apply_bottom_interface_factor(
    const BFStrictSegmentedFactor* factor,
    const std::vector<Complex>& top)
{
    if (factor == nullptr) {
        throw std::invalid_argument("null butterfly factor");
    }

    FFTStorage input;
    copy_to_fftw(top, input);
    FFTStorage output(2 * top.size(), 0.0f);

    bf1d_strict_segmented_apply(
        factor,
        reinterpret_cast<const fftwf_complex*>(input.data()),
        reinterpret_cast<fftwf_complex*>(output.data()));

    std::vector<Complex> bottom(top.size());
    for (std::size_t ix = 0; ix < bottom.size(); ++ix) {
        bottom[ix] = Complex(output[2 * ix], output[2 * ix + 1]);
    }
    return bottom;
}

static LocalHelmholtzSystem build_local_system(
    const se::huygens::Model2D& model,
    const se::huygens::Block& block,
    float frequency)
{
    LocalHelmholtzSystem system;
    system.top_iz = block.source_iz;
    system.bottom_iz = block.target_end_iz;
    system.nx_interior = model.nx - 2;
    system.nz_interior = system.bottom_iz - system.top_iz - 1;

    if (system.nx_interior < 1 || system.nz_interior < 1) {
        throw std::invalid_argument("local block has no Helmholtz interior");
    }

    system.coefficient_x = 1.0 /
        (static_cast<double>(model.dx) * model.dx);
    system.coefficient_z = 1.0 /
        (static_cast<double>(model.dz) * model.dz);
    const double omega = 2.0 * kPi * frequency;

    std::vector<se::eigen::Triplet<double>> entries;
    entries.reserve(static_cast<std::size_t>(system.unknowns()) * 5);

    for (int iz = system.top_iz + 1; iz < system.bottom_iz; ++iz) {
        for (int ix = 1; ix < model.nx - 1; ++ix) {
            const int row = system.row(ix, iz);
            const double velocity = model.velocity[model.index(ix, iz)];
            const double wavenumber = omega / velocity;
            const double diagonal = wavenumber * wavenumber -
                2.0 * system.coefficient_x -
                2.0 * system.coefficient_z;

            entries.emplace_back(row, row, diagonal);
            if (ix > 1) {
                entries.emplace_back(
                    row, system.row(ix - 1, iz), system.coefficient_x);
            }
            if (ix + 1 < model.nx - 1) {
                entries.emplace_back(
                    row, system.row(ix + 1, iz), system.coefficient_x);
            }
            if (iz > system.top_iz + 1) {
                entries.emplace_back(
                    row, system.row(ix, iz - 1), system.coefficient_z);
            }
            if (iz + 1 < system.bottom_iz) {
                entries.emplace_back(
                    row, system.row(ix, iz + 1), system.coefficient_z);
            }
        }
    }

    system.matrix = se::eigen::make_sparse_matrix<double>(
        system.unknowns(), system.unknowns(), entries);
    return system;
}

static void extract_local_vectors(
    const se::huygens::Model2D& model,
    const LocalHelmholtzSystem& system,
    const std::vector<Complex>& wavefield,
    EigenVector& real_values,
    EigenVector& imag_values,
    EigenVector& real_boundary_rhs,
    EigenVector& imag_boundary_rhs)
{
    const int count = system.unknowns();
    real_values.resize(count);
    imag_values.resize(count);
    real_boundary_rhs.setZero(count);
    imag_boundary_rhs.setZero(count);

    const auto add_boundary = [&](int row, double coefficient, const Complex& value) {
        real_boundary_rhs[row] -= coefficient * static_cast<double>(value.real());
        imag_boundary_rhs[row] -= coefficient * static_cast<double>(value.imag());
    };

    for (int iz = system.top_iz + 1; iz < system.bottom_iz; ++iz) {
        for (int ix = 1; ix < model.nx - 1; ++ix) {
            const int row = system.row(ix, iz);
            const Complex value = wavefield[model.index(ix, iz)];
            real_values[row] = value.real();
            imag_values[row] = value.imag();

            if (ix == 1) {
                add_boundary(row, system.coefficient_x,
                             wavefield[model.index(0, iz)]);
            }
            if (ix == model.nx - 2) {
                add_boundary(row, system.coefficient_x,
                             wavefield[model.index(model.nx - 1, iz)]);
            }
            if (iz == system.top_iz + 1) {
                add_boundary(row, system.coefficient_z,
                             wavefield[model.index(ix, system.top_iz)]);
            }
            if (iz == system.bottom_iz - 1) {
                add_boundary(row, system.coefficient_z,
                             wavefield[model.index(ix, system.bottom_iz)]);
            }
        }
    }
}

static void write_local_values(
    const se::huygens::Model2D& model,
    const LocalHelmholtzSystem& system,
    const EigenVector& real_values,
    const EigenVector& imag_values,
    std::vector<Complex>& wavefield)
{
    for (int iz = system.top_iz + 1; iz < system.bottom_iz; ++iz) {
        for (int ix = 1; ix < model.nx - 1; ++ix) {
            const int row = system.row(ix, iz);
            wavefield[model.index(ix, iz)] = Complex(
                static_cast<float>(real_values[row]),
                static_cast<float>(imag_values[row]));
        }
    }
}

static bool all_finite(const EigenVector& values)
{
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

static se::eigen::SolverResult<double> solve_component(
    const EigenMatrix& matrix,
    const EigenVector& rhs,
    const se::eigen::SolverOptions& options)
{
    if (rhs.norm() <= std::numeric_limits<double>::min()) {
        se::eigen::SolverResult<double> result;
        result.solution = EigenVector::Zero(rhs.size());
        result.report.status = se::eigen::SolverStatus::success;
        result.report.relative_residual = 0.0;
        result.report.estimated_error = 0.0;
        return result;
    }

    return se::eigen::solve_iterative(
        matrix, rhs, se::eigen::IterativeMethod::gmres, options);
}

static CorrectionMetrics correct_local_block(
    const se::huygens::Model2D& model,
    const LocalHelmholtzSystem& system,
    std::vector<Complex>& wavefield,
    const se::eigen::SolverOptions& options,
    double requested_relaxation,
    double maximum_relaxation)
{
    EigenVector real_values;
    EigenVector imag_values;
    EigenVector real_boundary_rhs;
    EigenVector imag_boundary_rhs;
    extract_local_vectors(
        model, system, wavefield,
        real_values, imag_values,
        real_boundary_rhs, imag_boundary_rhs);

    const EigenVector real_residual =
        system.matrix * real_values - real_boundary_rhs;
    const EigenVector imag_residual =
        system.matrix * imag_values - imag_boundary_rhs;

    CorrectionMetrics metrics;
    metrics.residual_before = std::sqrt(
        real_residual.squaredNorm() + imag_residual.squaredNorm());
    metrics.residual_after = metrics.residual_before;

    if (metrics.residual_before <= std::numeric_limits<double>::min()) {
        return metrics;
    }

    const auto real_result = solve_component(
        system.matrix, -real_residual, options);
    const auto imag_result = solve_component(
        system.matrix, -imag_residual, options);

    metrics.real_iterations = real_result.report.iterations;
    metrics.imag_iterations = imag_result.report.iterations;
    metrics.real_status = real_result.report.status;
    metrics.imag_status = imag_result.report.status;

    if (!all_finite(real_result.solution) || !all_finite(imag_result.solution)) {
        metrics.real_status = se::eigen::SolverStatus::numerical_issue;
        metrics.imag_status = se::eigen::SolverStatus::numerical_issue;
        return metrics;
    }

    const EigenVector real_action = system.matrix * real_result.solution;
    const EigenVector imag_action = system.matrix * imag_result.solution;
    const double action_norm_squared =
        real_action.squaredNorm() + imag_action.squaredNorm();

    double relaxation = requested_relaxation;
    if (requested_relaxation < 0.0) {
        if (action_norm_squared > std::numeric_limits<double>::min()) {
            relaxation = -(
                real_residual.dot(real_action) +
                imag_residual.dot(imag_action)) / action_norm_squared;
        } else {
            relaxation = 0.0;
        }
        relaxation = std::clamp(relaxation, 0.0, maximum_relaxation);
    }
    if (!std::isfinite(relaxation)) relaxation = 0.0;
    metrics.relaxation = relaxation;

    const EigenVector corrected_real =
        real_values + relaxation * real_result.solution;
    const EigenVector corrected_imag =
        imag_values + relaxation * imag_result.solution;
    const EigenVector real_after = real_residual + relaxation * real_action;
    const EigenVector imag_after = imag_residual + relaxation * imag_action;

    metrics.residual_after = std::sqrt(
        real_after.squaredNorm() + imag_after.squaredNorm());
    metrics.residual_ratio = metrics.residual_after /
        std::max(metrics.residual_before, std::numeric_limits<double>::min());

    const double field_norm = std::sqrt(
        real_values.squaredNorm() + imag_values.squaredNorm());
    const double update_norm = std::abs(relaxation) * std::sqrt(
        real_result.solution.squaredNorm() +
        imag_result.solution.squaredNorm());
    metrics.relative_update = update_norm /
        std::max(field_norm, std::numeric_limits<double>::min());

    if (all_finite(corrected_real) && all_finite(corrected_imag)) {
        write_local_values(
            model, system, corrected_real, corrected_imag, wavefield);
    }
    return metrics;
}

static std::vector<Complex> gather_row(
    const se::huygens::Model2D& model,
    const std::vector<Complex>& wavefield,
    int iz)
{
    if (iz < 0 || iz >= model.nz) {
        throw std::invalid_argument("gather_row iz is outside model");
    }
    std::vector<Complex> row(static_cast<std::size_t>(model.nx));
    for (int ix = 0; ix < model.nx; ++ix) {
        row[static_cast<std::size_t>(ix)] = wavefield[model.index(ix, iz)];
    }
    return row;
}

static void inject_row(const se::huygens::Model2D& model,
                       std::vector<Complex>& wavefield,
                       int iz,
                       const std::vector<Complex>& row)
{
    if (iz < 0 || iz >= model.nz || static_cast<int>(row.size()) != model.nx) {
        throw std::invalid_argument("inject_row dimensions are invalid");
    }
    for (int ix = 0; ix < model.nx; ++ix) {
        wavefield[model.index(ix, iz)] = row[static_cast<std::size_t>(ix)];
    }
}

static void initialize_block_guess(
    const se::huygens::Model2D& model,
    const se::huygens::Block& block,
    const std::vector<Complex>& global_wavefield,
    std::vector<Complex>& local_wavefield)
{
    const int top = block.source_iz;
    const int bottom = block.target_end_iz;
    const double denominator = std::max(1, bottom - top);

    for (int iz = top + 1; iz < bottom; ++iz) {
        const double t = static_cast<double>(iz - top) / denominator;
        const bool keep_existing = iz < block.target_start_iz;

        for (int ix = 0; ix < model.nx; ++ix) {
            const std::size_t index = model.index(ix, iz);
            if (keep_existing) {
                local_wavefield[index] = global_wavefield[index];
                continue;
            }

            const Complex top_value = local_wavefield[model.index(ix, top)];
            const Complex bottom_value = local_wavefield[model.index(ix, bottom)];
            local_wavefield[index] =
                static_cast<float>(1.0 - t) * top_value +
                static_cast<float>(t) * bottom_value;
        }
    }
}

static void commit_block_output(
    const se::huygens::Model2D& model,
    const se::huygens::Block& block,
    const std::vector<Complex>& local_wavefield,
    std::vector<Complex>& global_wavefield)
{
    const int first = (block.id == 0) ? block.source_iz : block.target_start_iz;
    for (int iz = first; iz <= block.target_end_iz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            global_wavefield[model.index(ix, iz)] =
                local_wavefield[model.index(ix, iz)];
        }
    }
}

static ResidualSummary evaluate_global_residual(
    const se::huygens::Model2D& model,
    const se::huygens::BlockInfo& info,
    const std::vector<Complex>& wavefield,
    float frequency)
{
    const double cx = 1.0 /
        (static_cast<double>(model.dx) * model.dx);
    const double cz = 1.0 /
        (static_cast<double>(model.dz) * model.dz);
    const double omega = 2.0 * kPi * frequency;

    std::vector<unsigned char> seam_row(static_cast<std::size_t>(model.nz), 0);
    for (std::size_t iblock = 0; iblock + 1 < info.blocks.size(); ++iblock) {
        const int seam = info.blocks[iblock].target_end_iz;
        if (seam >= 1 && seam < model.nz - 1) seam_row[seam] = 1;
        if (seam + 1 >= 1 && seam + 1 < model.nz - 1) seam_row[seam + 1] = 1;
    }

    long double total_r2 = 0.0L;
    long double total_scale2 = 0.0L;
    long double bulk_r2 = 0.0L;
    long double seam_r2 = 0.0L;
    std::size_t bulk_count = 0;
    std::size_t seam_count = 0;

    for (int iz = 1; iz < model.nz - 1; ++iz) {
        for (int ix = 1; ix < model.nx - 1; ++ix) {
            const std::size_t index = model.index(ix, iz);
            const Complex center = wavefield[index];
            const double velocity = model.velocity[index];
            const double k2 = (omega / velocity) * (omega / velocity);

            const Complex residual =
                static_cast<float>(cx) *
                    (wavefield[model.index(ix - 1, iz)] - 2.0f * center +
                     wavefield[model.index(ix + 1, iz)]) +
                static_cast<float>(cz) *
                    (wavefield[model.index(ix, iz - 1)] - 2.0f * center +
                     wavefield[model.index(ix, iz + 1)]) +
                static_cast<float>(k2) * center;

            const long double r2 = static_cast<long double>(std::norm(residual));
            const long double scale2 = static_cast<long double>(
                std::norm(static_cast<float>(k2) * center));
            total_r2 += r2;
            total_scale2 += scale2;

            if (seam_row[static_cast<std::size_t>(iz)] != 0) {
                seam_r2 += r2;
                ++seam_count;
            } else {
                bulk_r2 += r2;
                ++bulk_count;
            }
        }
    }

    ResidualSummary result;
    result.relative_residual = std::sqrt(
        static_cast<double>(total_r2 /
            std::max(total_scale2, static_cast<long double>(1.0e-30))));
    result.bulk_rms = std::sqrt(
        static_cast<double>(bulk_r2 /
            std::max<std::size_t>(bulk_count, 1)));
    result.seam_rms = std::sqrt(
        static_cast<double>(seam_r2 /
            std::max<std::size_t>(seam_count, 1)));
    result.seam_to_bulk = result.seam_rms /
        std::max(result.bulk_rms, std::numeric_limits<double>::min());
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    se_par_init(argc, argv);

    try {
        if (help_requested(argc, argv)) {
            print_help(argv[0]);
            return 0;
        }

        if (!se_have_par("velocity")) {
            ERROR(("Need velocity= RSF model"));
        }
        if (!se_have_par("block_file")) {
            ERROR(("Need block_file= block_info.dat"));
        }
        if (!se_have_par("table_prefix")) {
            ERROR(("Need table_prefix= precomputed traveltime prefix"));
        }

        const std::string velocity = se_get_par_str("velocity");
        const std::string block_file = se_get_par_str("block_file");
        const std::string table_prefix = se_get_par_str("table_prefix");
        const std::string output_prefix = se_have_par("output_prefix")
            ? std::string(se_get_par_str("output_prefix"))
            : std::string("output/ray_helmholtz_iteration");

        const float frequency = se_have_par("frequency")
            ? se_get_par_float("frequency") : 25.0f;
        const int source_ix_parameter = se_have_par("source_ix")
            ? se_get_par_int("source_ix") : -1;
        const int source_iz = se_have_par("source_iz")
            ? se_get_par_int("source_iz") : 0;
        const float source_amplitude = se_have_par("source_amplitude")
            ? se_get_par_float("source_amplitude") : 1.0f;
        const float source_radius = se_have_par("source_radius")
            ? se_get_par_float("source_radius") : 0.0f;

        const float filter_dt = se_have_par("filter_dt")
            ? se_get_par_float("filter_dt") : 0.001f;
        const float filter_length = se_have_par("filter_length")
            ? se_get_par_float("filter_length") : 0.025f;
        const int filter_lookup_subsamples =
            se_have_par("filter_lookup_subsamples")
            ? se_get_par_int("filter_lookup_subsamples") : 64;

        const int bf_p = se_have_par("bf_p")
            ? se_get_par_int("bf_p") : 12;
        const int bf_leaf = se_have_par("bf_n_leaf")
            ? se_get_par_int("bf_n_leaf") : 16;
        const int bf_panel_levels = se_have_par("bf_panel_levels")
            ? se_get_par_int("bf_panel_levels") : 1;
        const float bf_amp_eps = se_have_par("bf_amp_eps")
            ? se_get_par_float("bf_amp_eps") : 1.0e-20f;
        const float bf_phase_tol = se_have_par("bf_phase_tol")
            ? se_get_par_float("bf_phase_tol") : 1.0f;

        const int iteration_cycles = se_have_par("iter_cycles")
            ? se_get_par_int("iter_cycles") : 3;
        const int iteration_count = se_have_par("iter_iterations")
            ? se_get_par_int("iter_iterations") : 10;
        const int iteration_restart = se_have_par("iter_restart")
            ? se_get_par_int("iter_restart") : 10;
        const double iteration_tolerance = se_have_par("iter_tolerance")
            ? static_cast<double>(se_get_par_float("iter_tolerance")) : 1.0e-4;
        const std::string preconditioner_text =
            se_have_par("iter_preconditioner")
            ? std::string(se_get_par_str("iter_preconditioner"))
            : std::string("diagonal");
        const double ilut_drop_tolerance =
            se_have_par("iter_ilut_drop_tolerance")
            ? static_cast<double>(se_get_par_float("iter_ilut_drop_tolerance"))
            : 1.0e-3;
        const int ilut_fill_factor = se_have_par("iter_ilut_fill_factor")
            ? se_get_par_int("iter_ilut_fill_factor") : 10;
        const double requested_relaxation = se_have_par("iter_relaxation")
            ? static_cast<double>(se_get_par_float("iter_relaxation")) : -1.0;
        const double maximum_relaxation = se_have_par("iter_max_relaxation")
            ? static_cast<double>(se_get_par_float("iter_max_relaxation")) : 1.0;

        const int write_interfaces = se_have_par("write_interfaces")
            ? se_get_par_int("write_interfaces") : 1;
        const int write_each_block = se_have_par("write_each_block")
            ? se_get_par_int("write_each_block") : 0;
        const int threads = se_have_par("threads")
            ? se_get_par_int("threads") : 0;

        if (!(frequency > 0.0f) || !(filter_dt > 0.0f) ||
            !(filter_length > 0.0f) || filter_lookup_subsamples < 1 ||
            bf_p < 2 || bf_leaf < 4 || bf_p >= bf_leaf ||
            bf_panel_levels < 0 || bf_amp_eps < 0.0f || !(bf_phase_tol > 0.0f) ||
            iteration_cycles < 1 || iteration_count < 1 || iteration_restart < 1 ||
            !(iteration_tolerance > 0.0) || ilut_drop_tolerance < 0.0 ||
            ilut_fill_factor < 1 || maximum_relaxation < 0.0 ||
            requested_relaxation > maximum_relaxation) {
            throw std::invalid_argument("invalid butterfly/iteration parameters");
        }

        set_threads(threads);

        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(velocity);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(block_file);
        se::huygens::validate_block_info(info, &model);

        if (source_iz != 0) {
            throw std::invalid_argument(
                "this prototype currently requires source_iz=0");
        }
        if (info.overlap_rows < 1) {
            std::cerr
                << "WARNING: overlap_rows=0. The program will run, but the next "
                   "datum cannot reuse a corrected row inside the previous block.\n";
        }

        const int source_ix = source_ix_parameter >= 0
            ? source_ix_parameter : model.nx / 2;
        if (source_ix < 0 || source_ix >= model.nx) {
            throw std::invalid_argument("source_ix is outside model");
        }

        se::eigen::SolverOptions solver_options;
        solver_options.max_iterations = iteration_count;
        solver_options.tolerance = iteration_tolerance;
        solver_options.restart = iteration_restart;
        solver_options.preconditioner =
            se::eigen::preconditioner_from_string(preconditioner_text);
        solver_options.ilut_drop_tolerance = ilut_drop_tolerance;
        solver_options.ilut_fill_factor = ilut_fill_factor;

        std::vector<Complex> wavefield;
        se::huygens::initialize_first_block_hankel_one_way(
            model, info, frequency, source_ix, source_iz,
            source_amplitude, source_radius, wavefield);

        se::huygens::ensure_parent_directory(output_prefix + "_metrics.csv");
        std::ofstream metrics(output_prefix + "_metrics.csv");
        if (!metrics) {
            throw std::runtime_error("cannot create metrics CSV");
        }
        metrics << std::setprecision(12)
                << "block_id,cycle,source_iz,target_start_iz,target_end_iz,unknowns,"
                   "bf_build_seconds,bf_apply_seconds,helmholtz_build_seconds,"
                   "correction_seconds,real_iterations,imag_iterations,"
                   "real_status,imag_status,residual_before,residual_after,"
                   "residual_ratio,relative_update,relaxation\n";

        const float omega =
            2.0f * static_cast<float>(kPi) * frequency;

        std::cout
            << "Interface-Butterfly + block Helmholtz iteration\n"
            << "frequency=" << frequency << " Hz, blocks=" << info.blocks.size()
            << ", overlap_rows=" << info.overlap_rows << '\n'
            << "BF: p=" << bf_p << ", leaf=" << bf_leaf
            << ", phase_tol=" << bf_phase_tol << '\n'
            << "Local GMRES: cycles=" << iteration_cycles
            << ", max_iter/cycle=" << iteration_count
            << ", restart=" << iteration_restart
            << ", tol=" << iteration_tolerance
            << ", preconditioner="
            << se::eigen::to_string(solver_options.preconditioner) << '\n';

        for (const se::huygens::Block& block : info.blocks) {
            const se::huygens::LayerGeometry geometry =
                se::huygens::make_layer_geometry(model, block, 1, 1, 1);
            const se::huygens::OneWayLayerTables tables =
                se::huygens::read_one_way_layer_tables(table_prefix, block.id);
            se::huygens::validate_one_way_layer_tables(tables, geometry, block);

            const float maximum_tau = *std::max_element(
                tables.traveltime.values.begin(), tables.traveltime.values.end());
            const se::huygens::FrequencyKirchhoffFilter filter(
                frequency, filter_dt, filter_length,
                maximum_tau, filter_lookup_subsamples);

            const std::vector<Complex> top_trace =
                gather_row(model, wavefield, block.source_iz);

            const auto bf_build_started = Clock::now();
            FactorPointer factor = build_bottom_interface_factor(
                model, block, geometry, tables, filter, omega,
                bf_p, bf_leaf, bf_panel_levels, bf_amp_eps, bf_phase_tol);
            const double bf_build_seconds = elapsed_seconds(bf_build_started);

            const auto bf_apply_started = Clock::now();
            const std::vector<Complex> bottom_trace =
                apply_bottom_interface_factor(factor.get(), top_trace);
            const double bf_apply_seconds = elapsed_seconds(bf_apply_started);
            factor.reset();

            std::vector<Complex> local_wavefield = wavefield;
            inject_row(model, local_wavefield, block.source_iz, top_trace);
            inject_row(model, local_wavefield, block.target_end_iz, bottom_trace);
            initialize_block_guess(
                model, block, wavefield, local_wavefield);

            const auto matrix_started = Clock::now();
            const LocalHelmholtzSystem local_system =
                build_local_system(model, block, frequency);
            const double matrix_seconds = elapsed_seconds(matrix_started);

            std::cout
                << "Block " << block.id
                << ": top=" << block.source_iz
                << ", output=[" << block.target_start_iz
                << ',' << block.target_end_iz << ']'
                << ", unknowns=" << local_system.unknowns()
                << ", BF(build/apply)=" << bf_build_seconds
                << '/' << bf_apply_seconds << " s\n";

            double first_residual = -1.0;
            for (int cycle = 0; cycle < iteration_cycles; ++cycle) {
                const auto correction_started = Clock::now();
                const CorrectionMetrics correction = correct_local_block(
                    model, local_system, local_wavefield,
                    solver_options,
                    requested_relaxation,
                    maximum_relaxation);
                const double correction_seconds =
                    elapsed_seconds(correction_started);

                if (cycle == 0) first_residual = correction.residual_before;

                metrics << block.id << ','
                        << cycle << ','
                        << block.source_iz << ','
                        << block.target_start_iz << ','
                        << block.target_end_iz << ','
                        << local_system.unknowns() << ','
                        << bf_build_seconds << ','
                        << bf_apply_seconds << ','
                        << matrix_seconds << ','
                        << correction_seconds << ','
                        << correction.real_iterations << ','
                        << correction.imag_iterations << ','
                        << se::eigen::to_string(correction.real_status) << ','
                        << se::eigen::to_string(correction.imag_status) << ','
                        << correction.residual_before << ','
                        << correction.residual_after << ','
                        << correction.residual_ratio << ','
                        << correction.relative_update << ','
                        << correction.relaxation << '\n';

                std::cout
                    << "  cycle " << cycle
                    << ": residual " << correction.residual_before
                    << " -> " << correction.residual_after
                    << ", ratio=" << correction.residual_ratio
                    << ", alpha=" << correction.relaxation
                    << ", GMRES=" << correction.real_iterations
                    << '/' << correction.imag_iterations << '\n';

                const double target = iteration_tolerance *
                    std::max(first_residual, std::numeric_limits<double>::min());
                if (correction.residual_after <= target) break;
            }

            commit_block_output(
                model, block, local_wavefield, wavefield);

            if (write_interfaces != 0) {
                write_trace_component_rsf(
                    block_trace_filename(
                        output_prefix, block.id, "top", "real"),
                    model, top_trace, frequency,
                    block.id, block.source_iz, "top", false);
                write_trace_component_rsf(
                    block_trace_filename(
                        output_prefix, block.id, "top", "imag"),
                    model, top_trace, frequency,
                    block.id, block.source_iz, "top", true);
                write_trace_component_rsf(
                    block_trace_filename(
                        output_prefix, block.id, "bottom_bf", "real"),
                    model, bottom_trace, frequency,
                    block.id, block.target_end_iz, "bottom_bf", false);
                write_trace_component_rsf(
                    block_trace_filename(
                        output_prefix, block.id, "bottom_bf", "imag"),
                    model, bottom_trace, frequency,
                    block.id, block.target_end_iz, "bottom_bf", true);
            }

            if (write_each_block != 0) {
                se::huygens::write_wavefield_component_rsf(
                    block_field_filename(output_prefix, block.id, "real"),
                    model, wavefield, frequency,
                    "interface_butterfly_block_helmholtz", false);
                se::huygens::write_wavefield_component_rsf(
                    block_field_filename(output_prefix, block.id, "imag"),
                    model, wavefield, frequency,
                    "interface_butterfly_block_helmholtz", true);
            }
        }

        metrics.close();

        se::huygens::write_wavefield_component_rsf(
            output_prefix + "_real.rsf",
            model, wavefield, frequency,
            "interface_butterfly_block_helmholtz", false);
        se::huygens::write_wavefield_component_rsf(
            output_prefix + "_imag.rsf",
            model, wavefield, frequency,
            "interface_butterfly_block_helmholtz", true);

        const ResidualSummary residual = evaluate_global_residual(
            model, info, wavefield, frequency);

        std::ofstream summary(output_prefix + "_summary.txt");
        if (summary) {
            summary << std::setprecision(12)
                    << "relative_helmholtz_residual "
                    << residual.relative_residual << '\n'
                    << "bulk_residual_rms " << residual.bulk_rms << '\n'
                    << "seam_residual_rms " << residual.seam_rms << '\n'
                    << "seam_to_bulk " << residual.seam_to_bulk << '\n';
        }

        std::cout
            << "Completed.\n"
            << "  output: " << output_prefix << "_real.rsf / _imag.rsf\n"
            << "  relative Helmholtz residual = "
            << residual.relative_residual << '\n'
            << "  seam/bulk residual ratio = "
            << residual.seam_to_bulk << '\n'
            << "  metrics: " << output_prefix << "_metrics.csv\n";

        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ray_helmholtz_iteration: %s\n", error.what());
        return 1;
    }
}
