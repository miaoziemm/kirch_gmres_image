#include "huygens_cli.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/bf1d.h>
#include <SERECKIRCH/include/huygens_sweep.hpp>
#include <se_eigen.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Complex = se::huygens::Complex;
using EigenVector = se::eigen::DenseVector<double>;
using EigenMatrix = se::eigen::SparseMatrix<double>;

struct FactorDeleter {
    void operator()(BFStrictSegmentedFactor* factor) const
    {
        bf1d_strict_segmented_destroy(factor);
    }
};
using FactorPointer = std::unique_ptr<BFStrictSegmentedFactor, FactorDeleter>;
using FFTStorage = std::vector<float>;

struct MethodSpec {
    std::string name;
    se::eigen::IterativeMethod method = se::eigen::IterativeMethod::gmres;
};

struct Variant {
    std::string name;
    bool corrected = false;
    se::eigen::IterativeMethod method = se::eigen::IterativeMethod::gmres;
    std::vector<std::vector<Complex>> wavefields;
    double total_apply_seconds = 0.0;
    double total_correction_seconds = 0.0;
};

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

std::vector<int> make_rhs_sources(int nx,
                                  int rhs_count,
                                  int requested_source_ix)
{
    if (rhs_count < 1) throw std::invalid_argument("rhs_count must be positive");
    if (rhs_count == 1) {
        const int source = requested_source_ix >= 0 ? requested_source_ix : nx / 2;
        if (source < 0 || source >= nx) {
            throw std::invalid_argument("source_ix is outside model");
        }
        return {source};
    }
    std::vector<int> sources(static_cast<std::size_t>(rhs_count));
    for (int rhs = 0; rhs < rhs_count; ++rhs) {
        sources[static_cast<std::size_t>(rhs)] = static_cast<int>(
            std::llround((nx - 1.0) * rhs / (rhs_count - 1.0)));
    }
    return sources;
}

void copy_to_fftw(const std::vector<Complex>& input, FFTStorage& output)
{
    output.resize(2 * input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[2 * i] = input[i].real();
        output[2 * i + 1] = input[i].imag();
    }
}

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string short_method_name(se::eigen::IterativeMethod method)
{
    switch (method) {
        case se::eigen::IterativeMethod::conjugate_gradient: return "cg";
        case se::eigen::IterativeMethod::bicgstab: return "bicgstab";
        case se::eigen::IterativeMethod::gmres: return "gmres";
        case se::eigen::IterativeMethod::dgmres: return "dgmres";
        case se::eigen::IterativeMethod::minres: return "minres";
        case se::eigen::IterativeMethod::least_squares_conjugate_gradient:
            return "lscg";
    }
    throw std::invalid_argument("unknown Eigen iterative method");
}

std::vector<MethodSpec> parse_methods(const std::string& text)
{
    // The project parameter parser treats commas as array separators and
    // optional_string() returns the first array item. Accept '+' and ';' so a
    // complete method list can travel through one scalar command-line value.
    std::string normalized = text;
    std::replace(normalized.begin(), normalized.end(), '+', ',');
    std::replace(normalized.begin(), normalized.end(), ';', ',');
    std::replace(normalized.begin(), normalized.end(), '|', ',');

    std::vector<MethodSpec> result;
    std::set<std::string> seen;
    std::stringstream stream(normalized);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = trim(token);
        if (token.empty()) continue;
        const auto method = se::eigen::iterative_method_from_string(token);
        const std::string name = short_method_name(method);
        if (seen.insert(name).second) result.push_back({name, method});
    }
    if (result.empty()) {
        throw std::invalid_argument("iter_methods must contain at least one method");
    }
    return result;
}

LocalHelmholtzSystem build_local_system(const se::huygens::Model2D& model,
                                        const se::huygens::Block& block,
                                        float frequency)
{
    LocalHelmholtzSystem system;
    system.top_iz = block.source_iz;
    system.bottom_iz = block.target_end_iz;
    system.nx_interior = model.nx - 2;
    system.nz_interior = system.bottom_iz - system.top_iz - 1;
    if (system.nx_interior < 1 || system.nz_interior < 1) {
        throw std::invalid_argument("local Helmholtz block has no interior unknowns");
    }

    system.coefficient_x = 1.0 /
        (static_cast<double>(model.dx) * model.dx);
    system.coefficient_z = 1.0 /
        (static_cast<double>(model.dz) * model.dz);
    const double omega = 2.0 * 3.1415926535897932384626433832795 * frequency;

    std::vector<se::eigen::Triplet<double>> entries;
    entries.reserve(static_cast<std::size_t>(system.unknowns()) * 5);
    for (int iz = system.top_iz + 1; iz < system.bottom_iz; ++iz) {
        for (int ix = 1; ix < model.nx - 1; ++ix) {
            const int row = system.row(ix, iz);
            const double velocity = model.velocity[model.index(ix, iz)];
            const double wavenumber = omega / velocity;
            const double diagonal = wavenumber * wavenumber -
                2.0 * system.coefficient_x - 2.0 * system.coefficient_z;
            entries.emplace_back(row, row, diagonal);
            if (ix > 1) {
                entries.emplace_back(row, system.row(ix - 1, iz),
                                     system.coefficient_x);
            }
            if (ix + 1 < model.nx - 1) {
                entries.emplace_back(row, system.row(ix + 1, iz),
                                     system.coefficient_x);
            }
            if (iz > system.top_iz + 1) {
                entries.emplace_back(row, system.row(ix, iz - 1),
                                     system.coefficient_z);
            }
            if (iz + 1 < system.bottom_iz) {
                entries.emplace_back(row, system.row(ix, iz + 1),
                                     system.coefficient_z);
            }
        }
    }
    system.matrix = se::eigen::make_sparse_matrix<double>(
        system.unknowns(), system.unknowns(), entries);
    return system;
}

void extract_local_system_vectors(const se::huygens::Model2D& model,
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

void write_local_values(const se::huygens::Model2D& model,
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

bool all_finite(const EigenVector& values)
{
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

se::eigen::SolverResult<double> solve_component(
    const EigenMatrix& matrix,
    const EigenVector& rhs,
    se::eigen::IterativeMethod method,
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
    return se::eigen::solve_iterative(matrix, rhs, method, options);
}

CorrectionMetrics evaluate_or_correct(
    const se::huygens::Model2D& model,
    const LocalHelmholtzSystem& system,
    std::vector<Complex>& wavefield,
    bool apply_correction,
    se::eigen::IterativeMethod method,
    const se::eigen::SolverOptions& options,
    double requested_relaxation,
    double maximum_relaxation)
{
    EigenVector real_values;
    EigenVector imag_values;
    EigenVector real_boundary_rhs;
    EigenVector imag_boundary_rhs;
    extract_local_system_vectors(
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
    if (!apply_correction || metrics.residual_before <=
            std::numeric_limits<double>::min()) {
        return metrics;
    }

    const auto real_result = solve_component(
        system.matrix, -real_residual, method, options);
    const auto imag_result = solve_component(
        system.matrix, -imag_residual, method, options);
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

std::string output_filename(const std::string& prefix,
                            const std::string& method,
                            const std::string& component)
{
    return prefix + "_" + method + "_" + component + ".rsf";
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
        const std::string output_prefix = huygens_cli::optional_string(
            "output_prefix", "wave_frequency_bf_oneway_iter");
        const std::string metrics_csv = huygens_cli::optional_string(
            "metrics_csv", output_prefix + "_metrics.csv");
        const std::string iter_methods_text = huygens_cli::optional_string(
            "iter_methods", "cg,bicgstab,gmres,dgmres,minres,lscg");
        const std::string iter_preconditioner_text = huygens_cli::optional_string(
            "iter_preconditioner", "diagonal");

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

        const int iteration_count = huygens_cli::optional_int("iter_iterations", 5);
        const double iteration_tolerance = static_cast<double>(
            huygens_cli::optional_float("iter_tolerance", 1.0e-20f));
        const int iteration_restart = huygens_cli::optional_int("iter_restart", 20);
        const double requested_relaxation = static_cast<double>(
            huygens_cli::optional_float("iter_relaxation", -1.0f));
        const double maximum_relaxation = static_cast<double>(
            huygens_cli::optional_float("iter_max_relaxation", 1.0f));
        const double ilut_drop_tolerance = static_cast<double>(
            huygens_cli::optional_float("iter_ilut_drop_tolerance", 1.0e-3f));
        const int ilut_fill_factor =
            huygens_cli::optional_int("iter_ilut_fill_factor", 10);

        if (!(frequency > 0.0f) || !(filter_dt > 0.0f) ||
            !(filter_length > 0.0f) || filter_lookup_subsamples < 1 ||
            rhs_count < 1 || p < 2 || leaf < 4 || p >= leaf ||
            panel_levels < 0 || !(phase_tol > 0.0f) || amp_eps < 0.0f ||
            iteration_count < 1 || !(iteration_tolerance > 0.0) ||
            iteration_restart < 1 || maximum_relaxation < 0.0 ||
            requested_relaxation > maximum_relaxation ||
            ilut_drop_tolerance < 0.0 || ilut_fill_factor < 1) {
            throw std::invalid_argument("invalid Butterfly or iteration parameters");
        }
        if (source_stride != 1 || target_x_stride != 1 || target_z_stride != 1) {
            throw std::invalid_argument(
                "phase-aware bf1d path currently requires all spatial strides to equal one");
        }
        huygens_cli::set_openmp_threads(threads);

        const std::vector<MethodSpec> methods = parse_methods(iter_methods_text);
        se::eigen::SolverOptions solver_options;
        solver_options.max_iterations = iteration_count;
        solver_options.tolerance = iteration_tolerance;
        solver_options.restart = iteration_restart;
        solver_options.preconditioner =
            se::eigen::preconditioner_from_string(iter_preconditioner_text);
        solver_options.ilut_drop_tolerance = ilut_drop_tolerance;
        solver_options.ilut_fill_factor = ilut_fill_factor;

        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(velocity);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(block_file);
        se::huygens::validate_block_info(info, &model);
        if (source_iz != 0) {
            throw std::invalid_argument("source_iz must be zero in this prototype");
        }
        if (info.overlap_rows < 1) {
            throw std::invalid_argument(
                "iterative correction requires overlap_rows>=1 so the next datum lies inside the corrected block");
        }

        const std::vector<int> source_positions =
            make_rhs_sources(model.nx, rhs_count, source_ix_parameter);
        const int output_rhs = output_rhs_parameter >= 0
            ? output_rhs_parameter
            : rhs_count / 2;
        if (output_rhs < 0 || output_rhs >= rhs_count) {
            throw std::invalid_argument("output_rhs is outside [0,rhs_count)");
        }

        std::vector<std::vector<Complex>> analytic_first_block(
            static_cast<std::size_t>(rhs_count));
        for (int rhs = 0; rhs < rhs_count; ++rhs) {
            se::huygens::initialize_first_block_hankel_one_way(
                model, info, frequency,
                source_positions[static_cast<std::size_t>(rhs)], source_iz,
                source_amplitude, source_radius,
                analytic_first_block[static_cast<std::size_t>(rhs)]);
        }

        std::vector<Variant> variants;
        variants.push_back({"bf", false, se::eigen::IterativeMethod::gmres,
                            analytic_first_block});
        for (const MethodSpec& spec : methods) {
            variants.push_back({spec.name, true, spec.method,
                                analytic_first_block});
        }

        se::huygens::ensure_parent_directory(metrics_csv);
        std::ofstream metrics(metrics_csv);
        if (!metrics) throw std::runtime_error("cannot create metrics CSV: " + metrics_csv);
        metrics << std::setprecision(12)
                << "method,block_id,rhs,unknowns,requested_iterations,"
                   "real_iterations,imag_iterations,real_status,imag_status,"
                   "residual_before,residual_after,residual_ratio,relative_update,"
                   "relaxation,build_seconds,apply_seconds,correction_seconds\n";

        std::cout << "One-way Butterfly plus blockwise Helmholtz defect correction\n"
                  << "Fourier convention: exp(-i*omega*tau)\n"
                  << "Blocks: " << info.blocks.size()
                  << ", overlap rows: " << info.overlap_rows << '\n'
                  << "Correction iterations: " << iteration_count
                  << ", preconditioner: "
                  << se::eigen::to_string(solver_options.preconditioner)
                  << ", relaxation: "
                  << (requested_relaxation < 0.0 ? "residual-minimizing" : "fixed")
                  << '\n';

        double total_build_seconds = 0.0;
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
                    "bf1d requires one datum source and target per lateral grid point");
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

            std::cout << "Block " << block.id
                      << ": source row=" << block.source_iz
                      << ", target rows=[" << block.target_start_iz
                      << ',' << block.target_end_iz << ']'
                      << ", BF matrices=" << target_depths
                      << ", variants=" << variants.size() << '\n';

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
                        const Complex value =
                            kernel.phase_removed_amplitude(row, source);
                        amplitude[2 * matrix_index] = value.real();
                        amplitude[2 * matrix_index + 1] = value.imag();
                    }
                }
                factors[static_cast<std::size_t>(iz_local)].reset(
                    bf1d_strict_segmented_create_phase_amp(
                        model.nx, tau_rows.data(),
                        reinterpret_cast<const fftwf_complex*>(amplitude.data()),
                        omega, p, leaf, panel_levels, amp_eps, phase_tol));
                if (!factors[static_cast<std::size_t>(iz_local)]) {
                    throw std::runtime_error("bf1d factor construction returned null");
                }
            }
            const double build_seconds = std::chrono::duration<double>(
                Clock::now() - build_started).count();
            total_build_seconds += build_seconds;

            const auto matrix_started = Clock::now();
            const LocalHelmholtzSystem local_system =
                build_local_system(model, block, frequency);
            const double matrix_seconds = std::chrono::duration<double>(
                Clock::now() - matrix_started).count();
            std::cout << "  BF build=" << build_seconds
                      << " s, Helmholtz matrix=" << matrix_seconds
                      << " s, unknowns=" << local_system.unknowns()
                      << ", nnz=" << local_system.matrix.nonZeros() << '\n';

            for (Variant& variant : variants) {
                std::vector<FFTStorage> packed_inputs(
                    static_cast<std::size_t>(rhs_count));
                for (int rhs = 0; rhs < rhs_count; ++rhs) {
                    const std::vector<Complex> boundary =
                        se::huygens::gather_boundary_values(
                            model, block, geometry.source_ix,
                            variant.wavefields[static_cast<std::size_t>(rhs)]);
                    copy_to_fftw(boundary, packed_inputs[static_cast<std::size_t>(rhs)]);
                }

                std::vector<std::vector<Complex>> target_values(
                    static_cast<std::size_t>(rhs_count),
                    std::vector<Complex>(geometry.targets.size()));
                const auto apply_started = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (int iz_local = 0; iz_local < target_depths; ++iz_local) {
                    FFTStorage output(2 * static_cast<std::size_t>(model.nx));
                    for (int rhs = 0; rhs < rhs_count; ++rhs) {
                        const auto& input = packed_inputs[static_cast<std::size_t>(rhs)];
                        bf1d_strict_segmented_apply(
                            factors[static_cast<std::size_t>(iz_local)].get(),
                            reinterpret_cast<const fftwf_complex*>(input.data()),
                            reinterpret_cast<fftwf_complex*>(output.data()));
                        for (int ix_target = 0; ix_target < model.nx; ++ix_target) {
                            const std::size_t row =
                                static_cast<std::size_t>(ix_target) * target_depths +
                                iz_local;
                            target_values[static_cast<std::size_t>(rhs)][row] =
                                Complex(
                                    output[2 * static_cast<std::size_t>(ix_target)],
                                    output[2 * static_cast<std::size_t>(ix_target) + 1]);
                        }
                    }
                }
                const double apply_seconds = std::chrono::duration<double>(
                    Clock::now() - apply_started).count();
                variant.total_apply_seconds += apply_seconds;

                for (int rhs = 0; rhs < rhs_count; ++rhs) {
                    se::huygens::inject_target_values(
                        model, geometry.targets,
                        target_values[static_cast<std::size_t>(rhs)],
                        variant.wavefields[static_cast<std::size_t>(rhs)]);

                    const auto correction_started = Clock::now();
                    const CorrectionMetrics correction = evaluate_or_correct(
                        model, local_system,
                        variant.wavefields[static_cast<std::size_t>(rhs)],
                        variant.corrected, variant.method, solver_options,
                        requested_relaxation, maximum_relaxation);
                    const double correction_seconds = std::chrono::duration<double>(
                        Clock::now() - correction_started).count();
                    if (variant.corrected) {
                        variant.total_correction_seconds += correction_seconds;
                    }

                    metrics << variant.name << ','
                            << block.id << ','
                            << rhs << ','
                            << local_system.unknowns() << ','
                            << iteration_count << ','
                            << correction.real_iterations << ','
                            << correction.imag_iterations << ','
                            << se::eigen::to_string(correction.real_status) << ','
                            << se::eigen::to_string(correction.imag_status) << ','
                            << correction.residual_before << ','
                            << correction.residual_after << ','
                            << correction.residual_ratio << ','
                            << correction.relative_update << ','
                            << correction.relaxation << ','
                            << build_seconds << ','
                            << apply_seconds << ','
                            << (variant.corrected ? correction_seconds : 0.0)
                            << '\n';

                    std::cout << "  " << variant.name
                              << " rhs=" << rhs
                              << ": apply=" << apply_seconds << " s"
                              << ", correction="
                              << (variant.corrected ? correction_seconds : 0.0)
                              << " s, residual ratio="
                              << correction.residual_ratio
                              << ", alpha=" << correction.relaxation
                              << ", iterations="
                              << correction.real_iterations << '/'
                              << correction.imag_iterations << '\n';
                }
            }
        }

        metrics.close();
        for (const Variant& variant : variants) {
            const std::vector<Complex>& output_wavefield =
                variant.wavefields[static_cast<std::size_t>(output_rhs)];
            const std::string method = variant.corrected
                ? "phase_aware_butterfly_one_way_plus_helmholtz_" + variant.name
                : "phase_aware_butterfly_one_way_kirchhoff";
            se::huygens::write_wavefield_component_rsf(
                output_filename(output_prefix, variant.name, "real"),
                model, output_wavefield, frequency, method, false);
            se::huygens::write_wavefield_component_rsf(
                output_filename(output_prefix, variant.name, "imag"),
                model, output_wavefield, frequency, method, true);
        }

        std::cout << "Butterfly/iterative sweep completed.\n"
                  << "Total shared BF build time: " << total_build_seconds << " s\n"
                  << "Metrics: " << metrics_csv << '\n';
        for (const Variant& variant : variants) {
            std::cout << "  " << variant.name
                      << ": apply=" << variant.total_apply_seconds
                      << " s, correction=" << variant.total_correction_seconds
                      << " s\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "wave_frequency_bf_oneway_iter: " << error.what() << '\n';
        return 1;
    }
}
