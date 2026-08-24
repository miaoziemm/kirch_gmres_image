#include "huygens_cli.hpp"
#include "program_help.hpp"

#include <SERECKIRCH/include/bf1d.h>
#include <SERECKIRCH/include/huygens_sweep.hpp>

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/IterativeSolvers>

extern "C" {
#include <slu_zdefs.h>
}
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using ComplexF = se::huygens::Complex;
using ComplexD = std::complex<double>;
using SparseMatrix = Eigen::SparseMatrix<ComplexD, Eigen::ColMajor, int>;
using Vector = Eigen::Matrix<ComplexD, Eigen::Dynamic, 1>;
using Triplet = Eigen::Triplet<ComplexD, int>;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct FactorDeleter {
    void operator()(BFStrictSegmentedFactor* factor) const
    {
        bf1d_strict_segmented_destroy(factor);
    }
};
using FactorPointer = std::unique_ptr<BFStrictSegmentedFactor, FactorDeleter>;
using FFTStorage = std::vector<float>;

std::string indexed_filename(const std::string& prefix,
                             int iteration,
                             const std::string& component)
{
    std::ostringstream stream;
    stream << prefix << "_iter" << std::setw(2) << std::setfill('0')
           << iteration << '_' << component << ".rsf";
    return stream.str();
}

std::string component_filename(const std::string& prefix,
                               const std::string& component)
{
    return prefix + '_' + component + ".rsf";
}

void ensure_parent(const std::string& filename)
{
    const std::filesystem::path path(filename);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

std::vector<int> make_rhs_sources(int nx,
                                  int rhs_count,
                                  int requested_source_ix)
{
    if (rhs_count < 1) {
        throw std::invalid_argument("rhs_count must be positive");
    }
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

void copy_to_fftw(const std::vector<ComplexF>& input, FFTStorage& output)
{
    output.resize(2 * input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[2 * i] = input[i].real();
        output[2 * i + 1] = input[i].imag();
    }
}

std::vector<ComplexD> to_double(const std::vector<ComplexF>& input)
{
    std::vector<ComplexD> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = ComplexD(
            static_cast<double>(input[i].real()),
            static_cast<double>(input[i].imag()));
    }
    return output;
}

std::vector<ComplexF> to_float(const std::vector<ComplexD>& input)
{
    std::vector<ComplexF> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = ComplexF(
            static_cast<float>(input[i].real()),
            static_cast<float>(input[i].imag()));
    }
    return output;
}

double vector_norm(const std::vector<ComplexD>& values)
{
    long double sum = 0.0L;
    for (const ComplexD& value : values) {
        sum += static_cast<long double>(std::norm(value));
    }
    return std::sqrt(static_cast<double>(std::max(sum, 0.0L)));
}

double vector_norm(const Vector& values)
{
    return values.norm();
}

double difference_norm(const std::vector<ComplexD>& lhs,
                       const std::vector<ComplexD>& rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("difference_norm size mismatch");
    }
    long double sum = 0.0L;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        sum += static_cast<long double>(std::norm(lhs[i] - rhs[i]));
    }
    return std::sqrt(static_cast<double>(std::max(sum, 0.0L)));
}

int reflect_index(int index, int size)
{
    if (size < 1) {
        throw std::invalid_argument("reflect_index requires positive size");
    }
    if (size == 1) {
        return 0;
    }
    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index - 1;
        } else {
            index = 2 * size - index - 1;
        }
    }
    return index;
}

std::vector<double> gaussian_kernel(double sigma)
{
    if (!(sigma > 0.0)) {
        return {1.0};
    }
    const int radius = std::max(1, static_cast<int>(4.0 * sigma + 0.5));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const double coordinate = static_cast<double>(offset) / sigma;
        const double value = std::exp(-0.5 * coordinate * coordinate);
        kernel[static_cast<std::size_t>(offset + radius)] = value;
        sum += value;
    }
    for (double& value : kernel) {
        value /= sum;
    }
    return kernel;
}

void gaussian_filter_axis(const std::vector<ComplexD>& input,
                          std::vector<ComplexD>& output,
                          int nz,
                          int nx,
                          const std::vector<double>& kernel,
                          bool vertical)
{
    output.assign(input.size(), ComplexD(0.0, 0.0));
    const int radius = static_cast<int>(kernel.size() / 2);
    for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            ComplexD sum(0.0, 0.0);
            for (int offset = -radius; offset <= radius; ++offset) {
                const int sample_z = vertical ? reflect_index(iz + offset, nz) : iz;
                const int sample_x = vertical ? ix : reflect_index(ix + offset, nx);
                sum += kernel[static_cast<std::size_t>(offset + radius)] *
                       input[static_cast<std::size_t>(sample_z) * nx + sample_x];
            }
            output[static_cast<std::size_t>(iz) * nx + ix] = sum;
        }
    }
}

void gaussian_filter_in_place(std::vector<ComplexD>& values,
                              int nz,
                              int nx,
                              double sigma)
{
    if (!(sigma > 0.0)) {
        return;
    }
    const std::vector<double> kernel = gaussian_kernel(sigma);
    std::vector<ComplexD> temporary;
    std::vector<ComplexD> filtered;
    // scipy.ndimage.gaussian_filter processes axis 0 and then axis 1.
    gaussian_filter_axis(values, temporary, nz, nx, kernel, true);
    gaussian_filter_axis(temporary, filtered, nz, nx, kernel, false);
    values.swap(filtered);
}

double percentile_linear(std::vector<double> values, double percentile)
{
    if (values.empty()) {
        throw std::invalid_argument("percentile of empty vector");
    }
    percentile = std::clamp(percentile, 0.0, 100.0);
    std::sort(values.begin(), values.end());
    const double rank = percentile * 0.01 * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
    const double fraction = rank - static_cast<double>(lower);
    return (1.0 - fraction) * values[lower] + fraction * values[upper];
}

double seam_ratio(const std::vector<ComplexD>& field,
                  int nz,
                  int nx,
                  const std::vector<int>& boundaries,
                  bool imaginary)
{
    long double ordinary_sum = 0.0L;
    std::size_t ordinary_count = 0;
    for (int iz = 0; iz < nz - 1; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            const ComplexD difference =
                field[static_cast<std::size_t>(iz + 1) * nx + ix] -
                field[static_cast<std::size_t>(iz) * nx + ix];
            ordinary_sum += std::abs(imaginary ? difference.imag() : difference.real());
            ++ordinary_count;
        }
    }

    long double seam_sum = 0.0L;
    std::size_t seam_count = 0;
    for (const int boundary : boundaries) {
        if (boundary < 0 || boundary + 1 >= nz) {
            continue;
        }
        for (int ix = 0; ix < nx; ++ix) {
            const ComplexD difference =
                field[static_cast<std::size_t>(boundary + 1) * nx + ix] -
                field[static_cast<std::size_t>(boundary) * nx + ix];
            seam_sum += std::abs(imaginary ? difference.imag() : difference.real());
            ++seam_count;
        }
    }

    const double ordinary = static_cast<double>(ordinary_sum /
        std::max<std::size_t>(ordinary_count, 1));
    const double seam = static_cast<double>(seam_sum /
        std::max<std::size_t>(seam_count, 1));
    return seam / std::max(ordinary, std::numeric_limits<double>::min());
}

class SuperLUIncompleteLUT
    : public Eigen::SparseSolverBase<SuperLUIncompleteLUT> {
protected:
    using Base = Eigen::SparseSolverBase<SuperLUIncompleteLUT>;
    using Base::m_isInitialized;

public:
    using Scalar = ComplexD;
    using RealScalar = double;
    using StorageIndex = int;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic
    };

    SuperLUIncompleteLUT() = default;
    ~SuperLUIncompleteLUT() { clear(); }

    SuperLUIncompleteLUT(const SuperLUIncompleteLUT&) = delete;
    SuperLUIncompleteLUT& operator=(const SuperLUIncompleteLUT&) = delete;

    SuperLUIncompleteLUT& configure(double shift,
                                    double drop_tolerance,
                                    int fill_factor)
    {
        shift_ = shift;
        drop_tolerance_ = drop_tolerance;
        fill_factor_ = static_cast<double>(fill_factor);
        return *this;
    }

    template <typename MatrixType>
    SuperLUIncompleteLUT& compute(const MatrixType& matrix)
    {
        clear();

        SparseMatrix shifted = matrix;
        for (int index = 0; index < shifted.rows(); ++index) {
            const double diagonal =
                std::max(std::abs(matrix.coeff(index, index)), 1.0);
            shifted.coeffRef(index, index) +=
                ComplexD(0.0, shift_ * diagonal);
        }
        shifted.makeCompressed();

        size_ = static_cast<int>(shifted.rows());
        values_.resize(static_cast<std::size_t>(shifted.nonZeros()));
        row_indices_.resize(static_cast<std::size_t>(shifted.nonZeros()));
        column_offsets_.resize(static_cast<std::size_t>(size_ + 1));

        for (int index = 0; index <= size_; ++index) {
            column_offsets_[static_cast<std::size_t>(index)] =
                shifted.outerIndexPtr()[index];
        }
        for (int index = 0; index < shifted.nonZeros(); ++index) {
            row_indices_[static_cast<std::size_t>(index)] =
                shifted.innerIndexPtr()[index];
            const ComplexD value = shifted.valuePtr()[index];
            values_[static_cast<std::size_t>(index)].r = value.real();
            values_[static_cast<std::size_t>(index)].i = value.imag();
        }

        zCreate_CompCol_Matrix(
            &matrix_, size_, size_, shifted.nonZeros(), values_.data(),
            row_indices_.data(), column_offsets_.data(),
            SLU_NC, SLU_Z, SLU_GE);

        ilu_set_default_options(&options_);
        options_.ColPerm = COLAMD;
        options_.DiagPivotThresh = 0.0;
        options_.ILU_DropTol = drop_tolerance_;
        options_.ILU_FillFactor = fill_factor_;
        options_.PrintStat = NO;

        const int panel_size = sp_ienv(1);
        const int relax = sp_ienv(2);
        column_permutation_.resize(static_cast<std::size_t>(size_));
        row_permutation_.resize(static_cast<std::size_t>(size_));
        elimination_tree_.resize(static_cast<std::size_t>(size_));

        get_perm_c(options_.ColPerm, &matrix_, column_permutation_.data());
        sp_preorder(
            &options_, &matrix_, column_permutation_.data(),
            elimination_tree_.data(), &preordered_matrix_);

        SuperLUStat_t statistics;
        StatInit(&statistics);
        int_t info = 0;
        zgsitrf(
            &options_, &preordered_matrix_, relax, panel_size,
            elimination_tree_.data(), nullptr, 0,
            column_permutation_.data(), row_permutation_.data(),
            &lower_, &upper_, &global_lu_, &statistics, &info);
        Destroy_CompCol_Permuted(&preordered_matrix_);
        preordered_matrix_.Store = nullptr;
        StatFree(&statistics);

        m_info = info == 0 ? Eigen::Success : Eigen::NumericalIssue;
        m_isInitialized = true;
        return *this;
    }

    Eigen::Index rows() const noexcept { return size_; }
    Eigen::Index cols() const noexcept { return size_; }

    Eigen::ComputationInfo info() const
    {
        return m_info;
    }

    template <typename Rhs, typename Dest>
    void _solve_impl(const Rhs& rhs, Dest& solution) const
    {
        solution = rhs;
        std::vector<doublecomplex> buffer(static_cast<std::size_t>(size_));
        for (int index = 0; index < size_; ++index) {
            buffer[static_cast<std::size_t>(index)].r = solution[index].real();
            buffer[static_cast<std::size_t>(index)].i = solution[index].imag();
        }

        SuperMatrix dense_rhs;
        zCreate_Dense_Matrix(
            &dense_rhs, size_, 1, buffer.data(), size_,
            SLU_DN, SLU_Z, SLU_GE);
        SuperLUStat_t statistics;
        StatInit(&statistics);
        int solve_info = 0;
        zgstrs(
            NOTRANS,
            const_cast<SuperMatrix*>(&lower_),
            const_cast<SuperMatrix*>(&upper_),
            const_cast<int*>(column_permutation_.data()),
            const_cast<int*>(row_permutation_.data()),
            &dense_rhs, &statistics, &solve_info);
        StatFree(&statistics);
        Destroy_SuperMatrix_Store(&dense_rhs);
        if (solve_info != 0) {
            throw std::runtime_error("SuperLU ILUT triangular solve failed");
        }

        for (int index = 0; index < size_; ++index) {
            const doublecomplex value = buffer[static_cast<std::size_t>(index)];
            solution[index] = ComplexD(value.r, value.i);
        }
    }

private:
    void clear()
    {
        if (lower_.Store != nullptr) {
            Destroy_SuperNode_Matrix(&lower_);
            lower_.Store = nullptr;
        }
        if (upper_.Store != nullptr) {
            Destroy_CompCol_Matrix(&upper_);
            upper_.Store = nullptr;
        }
        if (preordered_matrix_.Store != nullptr) {
            Destroy_CompCol_Permuted(&preordered_matrix_);
            preordered_matrix_.Store = nullptr;
        }
        if (matrix_.Store != nullptr) {
            Destroy_SuperMatrix_Store(&matrix_);
            matrix_.Store = nullptr;
        }
        size_ = 0;
        m_isInitialized = false;
    }

    int size_ = 0;
    double shift_ = 0.10;
    double drop_tolerance_ = 1.0e-2;
    double fill_factor_ = 6.0;
    Eigen::ComputationInfo m_info = Eigen::Success;

    std::vector<doublecomplex> values_;
    std::vector<int_t> row_indices_;
    std::vector<int_t> column_offsets_;
    std::vector<int> column_permutation_;
    std::vector<int> row_permutation_;
    std::vector<int> elimination_tree_;

    SuperMatrix matrix_{};
    SuperMatrix preordered_matrix_{};
    SuperMatrix lower_{};
    SuperMatrix upper_{};
    superlu_options_t options_{};
    GlobalLU_t global_lu_{};
};

using GmresSolver = Eigen::GMRES<SparseMatrix, SuperLUIncompleteLUT>;

struct LocalSystem {
    se::huygens::Block block;
    SparseMatrix matrix;
    std::unique_ptr<GmresSolver> solver;
    int nx_unknown = 0;
    int nz_unknown = 0;
    double preconditioner_seconds = 0.0;
};

SparseMatrix build_local_operator(const se::huygens::Model2D& model,
                                  int top,
                                  int bottom,
                                  double omega,
                                  int& nx_unknown,
                                  int& nz_unknown)
{
    nz_unknown = bottom - top - 1;
    nx_unknown = model.nx - 2;
    if (nz_unknown < 1 || nx_unknown < 1) {
        throw std::invalid_argument("local block is too small");
    }

    const int count = nz_unknown * nx_unknown;
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(5) * count);
    const double inv_dx2 = 1.0 / (static_cast<double>(model.dx) * model.dx);
    const double inv_dz2 = 1.0 / (static_cast<double>(model.dz) * model.dz);

    for (int local_z = 0; local_z < nz_unknown; ++local_z) {
        const int iz = top + 1 + local_z;
        for (int local_x = 0; local_x < nx_unknown; ++local_x) {
            const int ix = local_x + 1;
            const int row = local_z * nx_unknown + local_x;
            const double velocity = model.velocity[model.index(ix, iz)];
            const double k2 = (omega / velocity) * (omega / velocity);
            entries.emplace_back(row, row,
                ComplexD(k2 - 2.0 * inv_dx2 - 2.0 * inv_dz2, 0.0));
            if (local_x > 0) {
                entries.emplace_back(row, row - 1, ComplexD(inv_dx2, 0.0));
            }
            if (local_x + 1 < nx_unknown) {
                entries.emplace_back(row, row + 1, ComplexD(inv_dx2, 0.0));
            }
            if (local_z > 0) {
                entries.emplace_back(row, row - nx_unknown, ComplexD(inv_dz2, 0.0));
            }
            if (local_z + 1 < nz_unknown) {
                entries.emplace_back(row, row + nx_unknown, ComplexD(inv_dz2, 0.0));
            }
        }
    }

    SparseMatrix matrix(count, count);
    matrix.setFromTriplets(entries.begin(), entries.end());
    matrix.makeCompressed();
    return matrix;
}

std::vector<ComplexD> extract_local(const std::vector<ComplexD>& field,
                                    int nx,
                                    int top,
                                    int bottom)
{
    const int local_nz = bottom - top + 1;
    std::vector<ComplexD> local(static_cast<std::size_t>(local_nz) * nx);
    for (int local_z = 0; local_z < local_nz; ++local_z) {
        const std::size_t source = static_cast<std::size_t>(top + local_z) * nx;
        const std::size_t target = static_cast<std::size_t>(local_z) * nx;
        std::copy_n(field.begin() + static_cast<std::ptrdiff_t>(source),
                    nx,
                    local.begin() + static_cast<std::ptrdiff_t>(target));
    }
    return local;
}

Vector interior_vector(const std::vector<ComplexD>& local,
                       int local_nz,
                       int nx,
                       int nx_unknown,
                       int nz_unknown)
{
    Vector output(nx_unknown * nz_unknown);
    for (int iz = 0; iz < nz_unknown; ++iz) {
        for (int ix = 0; ix < nx_unknown; ++ix) {
            output[iz * nx_unknown + ix] =
                local[static_cast<std::size_t>(iz + 1) * nx + ix + 1];
        }
    }
    return output;
}

Vector boundary_rhs(const std::vector<ComplexD>& local,
                    int local_nz,
                    int nx,
                    int nx_unknown,
                    int nz_unknown,
                    double dx,
                    double dz)
{
    Vector rhs = Vector::Zero(nx_unknown * nz_unknown);
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dz2 = 1.0 / (dz * dz);

    for (int ix = 0; ix < nx_unknown; ++ix) {
        rhs[ix] -= local[ix + 1] * inv_dz2;
        rhs[(nz_unknown - 1) * nx_unknown + ix] -=
            local[static_cast<std::size_t>(local_nz - 1) * nx + ix + 1] * inv_dz2;
    }
    for (int iz = 0; iz < nz_unknown; ++iz) {
        const int row = iz * nx_unknown;
        rhs[row] -= local[static_cast<std::size_t>(iz + 1) * nx] * inv_dx2;
        rhs[row + nx_unknown - 1] -=
            local[static_cast<std::size_t>(iz + 1) * nx + nx - 1] * inv_dx2;
    }
    return rhs;
}

std::vector<ComplexD> vector_to_grid(const Vector& values,
                                     int nz_unknown,
                                     int nx_unknown)
{
    std::vector<ComplexD> output(static_cast<std::size_t>(nz_unknown) * nx_unknown);
    for (int iz = 0; iz < nz_unknown; ++iz) {
        for (int ix = 0; ix < nx_unknown; ++ix) {
            output[static_cast<std::size_t>(iz) * nx_unknown + ix] =
                values[iz * nx_unknown + ix];
        }
    }
    return output;
}

void write_csv_header(std::ofstream& stream,
                      int corrected_blocks,
                      bool block_metrics)
{
    stream << std::setprecision(17);
    if (block_metrics) {
        stream << "iteration,block,source_iz,target_end_iz,gmres_info,relaxation,"
                  "relative_update,residual_before,residual_after,residual_ratio,"
                  "changed_norm_from_bf\n";
        return;
    }

    stream << "iteration";
    for (int block = 1; block < 1 + corrected_blocks; ++block) {
        stream << ",block" << block << "_update"
               << ",block" << block << "_residual_ratio";
    }
    stream << ",total_change_from_bf,step_change,seam_ratio_real,"
              "seam_ratio_imag,seconds\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) {
        return 0;
    }

    try {
        huygens_cli::initialize(argc, argv);

        const std::string velocity = huygens_cli::required_string("velocity");
        const std::string block_file = huygens_cli::required_string("block_file");
        const std::string table_prefix = huygens_cli::required_string("table_prefix");
        const std::string bf_output_real = huygens_cli::optional_string(
            "bf_output_real", "wave_bf_real.rsf");
        const std::string bf_output_imag = huygens_cli::optional_string(
            "bf_output_imag", "wave_bf_imag.rsf");
        const std::string output_prefix = huygens_cli::optional_string(
            "output_prefix", "wave_gmres");
        const std::string summary_csv = huygens_cli::optional_string(
            "summary_csv", "summary.csv");
        const std::string block_metrics_csv = huygens_cli::optional_string(
            "block_metrics_csv", "block_updates.csv");
        const std::string timing_file = huygens_cli::optional_string(
            "timing", "wave_bf_gmres_timing.rsf");

        const float frequency = huygens_cli::optional_float("frequency", 10.0f);
        const float filter_dt = huygens_cli::optional_float("filter_dt", 0.001f);
        const float filter_length = huygens_cli::optional_float("filter_length", 0.025f);
        const int filter_lookup_subsamples =
            huygens_cli::optional_int("filter_lookup_subsamples", 32);
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
        const float phase_tol = huygens_cli::optional_float("bf_phase_tol", 0.5f);
        const int threads = huygens_cli::optional_int("threads", 0);

        const int iterations = huygens_cli::optional_int("iterations", 3);
        const int restart = huygens_cli::optional_int("gmres_restart", 30);
        const int gmres_cycles = huygens_cli::optional_int("gmres_cycles", 1);
        const double tolerance = huygens_cli::optional_float("gmres_tolerance", 1.0e-4f);
        const double shift = huygens_cli::optional_float("shift", 0.10f);
        const double ilut_drop_tolerance =
            huygens_cli::optional_float("ilut_drop_tolerance", 1.0e-2f);
        const int ilut_fill_factor = huygens_cli::optional_int("ilut_fill_factor", 6);
        const double minimum_relaxation =
            huygens_cli::optional_float("minimum_relaxation", 0.25f);
        const double maximum_relaxation =
            huygens_cli::optional_float("maximum_relaxation", 1.05f);
        const double correction_sigma =
            huygens_cli::optional_float("correction_sigma", 0.5f);
        const double clip_fraction =
            huygens_cli::optional_float("clip_fraction", 0.50f);
        const double clip_percentile =
            huygens_cli::optional_float("clip_percentile", 35.0f);

        if (!(frequency > 0.0f) || !(filter_dt > 0.0f) ||
            !(filter_length > 0.0f) || filter_lookup_subsamples < 1 ||
            rhs_count < 1 || p < 2 || leaf < 4 || p >= leaf ||
            panel_levels < 0 || !(phase_tol > 0.0f) || amp_eps < 0.0f ||
            iterations < 1 || restart < 1 || gmres_cycles < 1 ||
            !(tolerance > 0.0) || shift < 0.0 || ilut_drop_tolerance < 0.0 ||
            ilut_fill_factor < 1 || minimum_relaxation > maximum_relaxation ||
            correction_sigma < 0.0 || clip_fraction < 0.0 ||
            clip_percentile < 0.0 || clip_percentile > 100.0) {
            throw std::invalid_argument("invalid BF or GMRES parameters");
        }
        if (source_stride != 1 || target_x_stride != 1 || target_z_stride != 1) {
            throw std::invalid_argument(
                "phase-aware bf1d path currently requires all spatial strides to equal one");
        }
        huygens_cli::set_openmp_threads(threads);

        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(velocity);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(block_file);
        se::huygens::validate_block_info(info, &model);
        if (source_iz != 0) {
            throw std::invalid_argument("source_iz must be zero in this prototype");
        }
        if (info.blocks.size() < 2) {
            throw std::invalid_argument("at least two blocks are required");
        }

        const std::vector<int> source_positions =
            make_rhs_sources(model.nx, rhs_count, source_ix_parameter);
        const int output_rhs = output_rhs_parameter >= 0
            ? output_rhs_parameter
            : rhs_count / 2;
        if (output_rhs < 0 || output_rhs >= rhs_count) {
            throw std::invalid_argument("output_rhs is outside [0,rhs_count)");
        }

        std::cout << "Combined phase-aware ButterflyPACK + Eigen ILUT-GMRES\n"
                  << "Fourier convention: launch exp(-i*k*r), propagation "
                     "exp(-i*omega*tau).\n";

        std::vector<std::vector<ComplexF>> wavefields(
            static_cast<std::size_t>(rhs_count));
        for (int rhs = 0; rhs < rhs_count; ++rhs) {
            se::huygens::initialize_first_block_hankel_one_way(
                model, info, frequency,
                source_positions[static_cast<std::size_t>(rhs)], source_iz,
                source_amplitude, source_radius,
                wavefields[static_cast<std::size_t>(rhs)]);
        }

        const int propagation_blocks = static_cast<int>(info.blocks.size()) - 1;
        const int bf_metrics = 7;
        std::vector<float> timing_values(
            static_cast<std::size_t>(bf_metrics) * propagation_blocks, 0.0f);
        double total_build_seconds = 0.0;
        double total_apply_seconds = 0.0;
        const float omega_float = static_cast<float>(2.0 * kPi * frequency);

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
            const std::vector<float> quadrature_weights =
                se::huygens::trapezoidal_weights(model, geometry.source_ix);

            se::huygens::OneWayKernelData kernel;
            kernel.rows = static_cast<int>(geometry.targets.size());
            kernel.sources = model.nx;
            kernel.source_z = model.z(block.source_iz);
            kernel.quadrature_weights = &quadrature_weights;
            kernel.tables = &tables;
            kernel.filter = &filter;

            const int target_depths = static_cast<int>(geometry.target_iz.size());
            std::vector<FactorPointer> factors(
                static_cast<std::size_t>(target_depths));

            std::cout << "Butterfly block " << block.id
                      << ": " << target_depths << " matrices "
                      << model.nx << " x " << model.nx << '\n';

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
                        const ComplexF value =
                            kernel.phase_removed_amplitude(row, source);
                        amplitude[2 * matrix_index] = value.real();
                        amplitude[2 * matrix_index + 1] = value.imag();
                    }
                }
                factors[static_cast<std::size_t>(iz_local)].reset(
                    bf1d_strict_segmented_create_phase_amp(
                        model.nx,
                        tau_rows.data(),
                        reinterpret_cast<const fftwf_complex*>(amplitude.data()),
                        omega_float,
                        p,
                        leaf,
                        panel_levels,
                        amp_eps,
                        phase_tol));
                if (!factors[static_cast<std::size_t>(iz_local)]) {
                    throw std::runtime_error("bf1d factor construction returned null");
                }
            }
            const double build_seconds = std::chrono::duration<double>(
                Clock::now() - build_started).count();
            total_build_seconds += build_seconds;

            std::vector<std::vector<ComplexF>> boundary_inputs(
                static_cast<std::size_t>(rhs_count));
            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                boundary_inputs[static_cast<std::size_t>(rhs)] =
                    se::huygens::gather_boundary_values(
                        model, block, geometry.source_ix,
                        wavefields[static_cast<std::size_t>(rhs)]);
            }
            std::vector<FFTStorage> packed_inputs(
                static_cast<std::size_t>(rhs_count));
            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                copy_to_fftw(boundary_inputs[static_cast<std::size_t>(rhs)],
                             packed_inputs[static_cast<std::size_t>(rhs)]);
            }
            std::vector<std::vector<ComplexF>> target_values(
                static_cast<std::size_t>(rhs_count),
                std::vector<ComplexF>(geometry.targets.size()));

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
                            ComplexF(
                                output[2 * static_cast<std::size_t>(ix_target)],
                                output[2 * static_cast<std::size_t>(ix_target) + 1]);
                    }
                }
            }
            const double apply_seconds = std::chrono::duration<double>(
                Clock::now() - apply_started).count();
            total_apply_seconds += apply_seconds;

            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                se::huygens::inject_target_values(
                    model,
                    geometry.targets,
                    target_values[static_cast<std::size_t>(rhs)],
                    wavefields[static_cast<std::size_t>(rhs)]);
            }

            const double dense_megabytes_per_depth =
                static_cast<double>(model.nx) * model.nx *
                (sizeof(float) + sizeof(fftwf_complex)) / 1.0e6;
            const int column = static_cast<int>(iblock) - 1;
            timing_values[static_cast<std::size_t>(column) * bf_metrics + 0] =
                static_cast<float>(build_seconds);
            timing_values[static_cast<std::size_t>(column) * bf_metrics + 1] =
                static_cast<float>(apply_seconds);
            timing_values[static_cast<std::size_t>(column) * bf_metrics + 2] =
                static_cast<float>(apply_seconds / rhs_count);
            timing_values[static_cast<std::size_t>(column) * bf_metrics + 3] =
                static_cast<float>(model.nx);
            timing_values[static_cast<std::size_t>(column) * bf_metrics + 4] =
                static_cast<float>(kernel.rows);
            timing_values[static_cast<std::size_t>(column) * bf_metrics + 5] =
                static_cast<float>(rhs_count);
            timing_values[static_cast<std::size_t>(column) * bf_metrics + 6] =
                static_cast<float>(dense_megabytes_per_depth);
        }

        const std::vector<ComplexF>& bf_wavefield_float =
            wavefields[static_cast<std::size_t>(output_rhs)];
        se::huygens::write_wavefield_component_rsf(
            bf_output_real,
            model,
            bf_wavefield_float,
            frequency,
            "phase_aware_butterfly_one_way_kirchhoff",
            false);
        se::huygens::write_wavefield_component_rsf(
            bf_output_imag,
            model,
            bf_wavefield_float,
            frequency,
            "phase_aware_butterfly_one_way_kirchhoff",
            true);

        const std::vector<ComplexD> original = to_double(bf_wavefield_float);
        std::vector<ComplexD> current = original;
        std::vector<ComplexD> previous = original;
        const double omega = 2.0 * kPi * static_cast<double>(frequency);

        std::vector<std::unique_ptr<LocalSystem>> systems;
        systems.reserve(info.blocks.size());
        for (std::size_t iblock = 0; iblock < info.blocks.size(); ++iblock) {
            auto system = std::make_unique<LocalSystem>();
            system->block = info.blocks[iblock];
            system->matrix = build_local_operator(
                model,
                system->block.source_iz,
                system->block.target_end_iz,
                omega,
                system->nx_unknown,
                system->nz_unknown);
            system->solver = std::make_unique<GmresSolver>();
            system->solver->set_restart(restart);
            system->solver->setMaxIterations(restart * gmres_cycles);
            system->solver->setTolerance(tolerance);
            system->solver->preconditioner().configure(
                shift,
                ilut_drop_tolerance,
                ilut_fill_factor);
            const auto started = Clock::now();
            system->solver->compute(system->matrix);
            system->preconditioner_seconds = std::chrono::duration<double>(
                Clock::now() - started).count();
            if (system->solver->info() != Eigen::Success) {
                throw std::runtime_error(
                    "SuperLU shifted ILUT-GMRES preconditioner construction failed");
            }
            std::cout << "GMRES block " << (iblock + 1)
                      << ": source=" << system->block.source_iz
                      << ", bottom=" << system->block.target_end_iz
                      << ", unknowns=" << system->matrix.rows()
                      << ", preconditioner=" << system->preconditioner_seconds
                      << " s\n";
            systems.push_back(std::move(system));
        }

        ensure_parent(summary_csv);
        ensure_parent(block_metrics_csv);
        ensure_parent(output_prefix + "_placeholder");
        std::ofstream summary_stream(summary_csv);
        std::ofstream block_stream(block_metrics_csv);
        if (!summary_stream || !block_stream) {
            throw std::runtime_error("failed to open GMRES CSV outputs");
        }
        write_csv_header(summary_stream, static_cast<int>(systems.size()), false);
        write_csv_header(block_stream, static_cast<int>(systems.size()), true);

        std::vector<int> boundaries;
        boundaries.reserve(info.blocks.size() - 1);
        for (std::size_t index = 0; index + 1 < info.blocks.size(); ++index) {
            boundaries.push_back(info.blocks[index].target_end_iz);
        }
        const double original_norm = std::max(
            vector_norm(original),
            std::numeric_limits<double>::min());

        for (int iteration = 1; iteration <= iterations; ++iteration) {
            const auto iteration_started = Clock::now();
            std::vector<double> block_updates;
            std::vector<double> block_ratios;

            for (std::size_t system_index = 0;
                 system_index < systems.size();
                 ++system_index) {
                LocalSystem& system = *systems[system_index];
                const int top = system.block.source_iz;
                const int bottom = system.block.target_end_iz;
                const int target_start = system.block.target_start_iz;
                const int local_nz = bottom - top + 1;

                std::vector<ComplexD> local =
                    extract_local(current, model.nx, top, bottom);
                const Vector guess = interior_vector(
                    local,
                    local_nz,
                    model.nx,
                    system.nx_unknown,
                    system.nz_unknown);
                const Vector rhs = boundary_rhs(
                    local,
                    local_nz,
                    model.nx,
                    system.nx_unknown,
                    system.nz_unknown,
                    model.dx,
                    model.dz);
                const Vector defect = rhs - system.matrix * guess;
                const double residual_before = vector_norm(defect);

                Vector correction = system.solver->solve(defect);
                const int solver_iterations =
                    static_cast<int>(system.solver->iterations());
                const Eigen::ComputationInfo solver_info = system.solver->info();
                const Vector action = system.matrix * correction;
                const double denominator = std::real(action.dot(action));
                double relaxation = 1.0;
                if (denominator > std::numeric_limits<double>::min()) {
                    relaxation = std::real(action.dot(defect)) / denominator;
                }
                relaxation = std::clamp(
                    relaxation,
                    minimum_relaxation,
                    maximum_relaxation);
                correction *= relaxation;

                std::vector<ComplexD> correction_grid = vector_to_grid(
                    correction,
                    system.nz_unknown,
                    system.nx_unknown);
                gaussian_filter_in_place(
                    correction_grid,
                    system.nz_unknown,
                    system.nx_unknown,
                    correction_sigma);

                std::vector<double> amplitude;
                amplitude.reserve(correction_grid.size());
                for (int iz = 0; iz < system.nz_unknown; ++iz) {
                    for (int ix = 0; ix < system.nx_unknown; ++ix) {
                        amplitude.push_back(std::abs(
                            local[static_cast<std::size_t>(iz + 1) * model.nx + ix + 1]));
                    }
                }
                const double amplitude_floor =
                    percentile_linear(amplitude, clip_percentile);
                for (int iz = 0; iz < system.nz_unknown; ++iz) {
                    for (int ix = 0; ix < system.nx_unknown; ++ix) {
                        const std::size_t correction_index =
                            static_cast<std::size_t>(iz) * system.nx_unknown + ix;
                        const double local_amplitude = amplitude[correction_index];
                        const double limit = clip_fraction *
                            std::max(local_amplitude, amplitude_floor);
                        const ComplexD value = correction_grid[correction_index];
                        correction_grid[correction_index] = ComplexD(
                            std::clamp(value.real(), -limit, limit),
                            std::clamp(value.imag(), -limit, limit));
                    }
                }

                std::vector<ComplexD> updated = local;
                for (int iz = 0; iz < system.nz_unknown; ++iz) {
                    for (int ix = 0; ix < system.nx_unknown; ++ix) {
                        updated[static_cast<std::size_t>(iz + 1) * model.nx + ix + 1] +=
                            correction_grid[
                                static_cast<std::size_t>(iz) * system.nx_unknown + ix];
                    }
                }

                for (int global_z = top; global_z <= bottom; ++global_z) {
                    double weight = 1.0;
                    if (global_z < target_start) {
                        const double parameter =
                            static_cast<double>(global_z - top) /
                            std::max(target_start - top, 1);
                        weight = 0.5 * (1.0 - std::cos(kPi * parameter));
                    }
                    if (global_z == top) {
                        weight = 0.0;
                    }
                    const int local_z = global_z - top;
                    for (int ix = 0; ix < model.nx; ++ix) {
                        const std::size_t global_index = model.index(ix, global_z);
                        const std::size_t local_index =
                            static_cast<std::size_t>(local_z) * model.nx + ix;
                        current[global_index] =
                            (1.0 - weight) * current[global_index] +
                            weight * updated[local_index];
                    }
                }

                const std::vector<ComplexD> corrected_local =
                    extract_local(current, model.nx, top, bottom);
                const Vector corrected_guess = interior_vector(
                    corrected_local,
                    local_nz,
                    model.nx,
                    system.nx_unknown,
                    system.nz_unknown);
                const Vector corrected_rhs = boundary_rhs(
                    corrected_local,
                    local_nz,
                    model.nx,
                    system.nx_unknown,
                    system.nz_unknown,
                    model.dx,
                    model.dz);
                const double residual_after = vector_norm(
                    corrected_rhs - system.matrix * corrected_guess);

                std::vector<ComplexD> guess_grid = vector_to_grid(
                    guess,
                    system.nz_unknown,
                    system.nx_unknown);
                const double relative_update =
                    vector_norm(correction_grid) /
                    std::max(vector_norm(guess_grid),
                             std::numeric_limits<double>::min());

                const std::vector<ComplexD> current_local =
                    extract_local(current, model.nx, top, bottom);
                const std::vector<ComplexD> original_local =
                    extract_local(original, model.nx, top, bottom);
                const double changed_from_bf =
                    difference_norm(current_local, original_local) /
                    std::max(vector_norm(original_local),
                             std::numeric_limits<double>::min());
                const double residual_ratio = residual_after /
                    std::max(residual_before,
                             std::numeric_limits<double>::min());

                block_updates.push_back(relative_update);
                block_ratios.push_back(residual_ratio);
                block_stream
                    << iteration << ',' << (system_index + 1) << ','
                    << top << ',' << bottom << ','
                    << (solver_info == Eigen::Success ? 0 : solver_iterations) << ','
                    << relaxation << ',' << relative_update << ','
                    << residual_before << ',' << residual_after << ','
                    << residual_ratio << ',' << changed_from_bf << '\n';
            }

            const double total_change =
                difference_norm(current, original) / original_norm;
            const double step_change =
                difference_norm(current, previous) / original_norm;
            const double seam_real = seam_ratio(
                current, model.nz, model.nx, boundaries, false);
            const double seam_imag = seam_ratio(
                current, model.nz, model.nx, boundaries, true);
            const double iteration_seconds = std::chrono::duration<double>(
                Clock::now() - iteration_started).count();

            summary_stream << iteration;
            for (std::size_t index = 0; index < block_updates.size(); ++index) {
                summary_stream << ',' << block_updates[index]
                               << ',' << block_ratios[index];
            }
            summary_stream << ',' << total_change
                           << ',' << step_change
                           << ',' << seam_real
                           << ',' << seam_imag
                           << ',' << iteration_seconds << '\n';

            const std::vector<ComplexF> current_float = to_float(current);
            se::huygens::write_wavefield_component_rsf(
                indexed_filename(output_prefix, iteration, "real"),
                model,
                current_float,
                frequency,
                "phase_aware_butterfly_one_way_eigen_ilut_gmres",
                false);
            se::huygens::write_wavefield_component_rsf(
                indexed_filename(output_prefix, iteration, "imag"),
                model,
                current_float,
                frequency,
                "phase_aware_butterfly_one_way_eigen_ilut_gmres",
                true);

            previous = current;
            std::cout << "iteration=" << std::setw(2) << std::setfill('0')
                      << iteration << std::setfill(' ')
                      << " total_change=" << total_change
                      << " step_change=" << step_change
                      << " block1_update=" << block_updates.front() << '\n';
        }

        const std::vector<ComplexF> final_float = to_float(current);
        se::huygens::write_wavefield_component_rsf(
            component_filename(output_prefix, "real"),
            model,
            final_float,
            frequency,
            "phase_aware_butterfly_one_way_eigen_ilut_gmres",
            false);
        se::huygens::write_wavefield_component_rsf(
            component_filename(output_prefix, "imag"),
            model,
            final_float,
            frequency,
            "phase_aware_butterfly_one_way_eigen_ilut_gmres",
            true);

        se::huygens::write_timing_rsf(
            timing_file,
            timing_values,
            bf_metrics,
            propagation_blocks,
            "phase_aware_butterfly_one_way_eigen_ilut_gmres",
            {"build_seconds", "apply_seconds", "apply_seconds_per_rhs",
             "source_points", "target_points", "rhs_count",
             "dense_workspace_megabytes_per_depth"},
            {{"total_build_seconds", static_cast<float>(total_build_seconds)},
             {"total_apply_seconds", static_cast<float>(total_apply_seconds)},
             {"frequency_hz", frequency},
             {"rhs_count", static_cast<float>(rhs_count)},
             {"gmres_iterations", static_cast<float>(iterations)},
             {"gmres_restart", static_cast<float>(restart)},
             {"gmres_cycles", static_cast<float>(gmres_cycles)},
             {"ilut_drop_tolerance", static_cast<float>(ilut_drop_tolerance)},
             {"ilut_fill_factor", static_cast<float>(ilut_fill_factor)},
             {"shift", static_cast<float>(shift)}});

        std::cout << "Combined ButterflyPACK + Eigen ILUT-GMRES completed.\n"
                  << "BF build time: " << total_build_seconds << " s\n"
                  << "BF apply time: " << total_apply_seconds << " s\n"
                  << "Summary CSV: " << summary_csv << '\n'
                  << "Block CSV: " << block_metrics_csv << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "wave_frequency_bf_oneway_gmres: " << error.what() << '\n';
        return 1;
    }
}
