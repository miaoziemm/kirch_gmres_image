#ifndef GLOBAL_PRECONDITIONED_GMRES_HPP
#define GLOBAL_PRECONDITIONED_GMRES_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/IterativeSolvers>

extern "C" {
#include <slu_zdefs.h>
}

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace global_preconditioned_gmres {

using Complex = std::complex<double>;
using Sparse = Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>;
using Vector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using Triplet = Eigen::Triplet<Complex, int>;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct Parameters {
    int iterations = 10;
    int restart = 30;
    double tolerance = 1.0e-8;
    int absorbing_rows = 25;
    double damp_max = 2.0;
    double smooth_sigma = 1.0;
    double shift_beta = 0.30;
    double ilut_drop_tolerance = 3.0e-2;
    int ilut_fill_factor = 12;
    double ilut_pivot_threshold = 0.10;
    int receiver_mask_rows = 2;
};

struct Metric {
    int gmres_steps = 0;
    double residual_before = 0.0;
    double residual_after = 0.0;
    double correction_norm = 0.0;
};

inline int index(int iz, int ix, int nx) { return ix + iz * nx; }

inline std::vector<double> gaussian_kernel(double sigma)
{
    if (!(sigma > 0.0)) return {1.0};
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double q = static_cast<double>(i) / sigma;
        const double value = std::exp(-0.5 * q * q);
        kernel[static_cast<std::size_t>(i + radius)] = value;
        sum += value;
    }
    for (double& value : kernel) value /= sum;
    return kernel;
}

inline std::vector<float> smooth_velocity(const std::vector<float>& input,
                                          int nz, int nx, double sigma)
{
    if (static_cast<int>(input.size()) != nz * nx) {
        throw std::invalid_argument("global smoothing velocity size mismatch");
    }
    if (!(sigma > 0.0)) return input;

    const std::vector<double> kernel = gaussian_kernel(sigma);
    const int radius = static_cast<int>((kernel.size() - 1) / 2);
    std::vector<float> temp(input.size(), 0.0f);
    std::vector<float> output(input.size(), 0.0f);

    for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            double sum = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                const int jx = std::clamp(ix + k, 0, nx - 1);
                sum += kernel[static_cast<std::size_t>(k + radius)] *
                       input[static_cast<std::size_t>(index(iz, jx, nx))];
            }
            temp[static_cast<std::size_t>(index(iz, ix, nx))] =
                static_cast<float>(sum);
        }
    }
    for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            double sum = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                const int jz = std::clamp(iz + k, 0, nz - 1);
                sum += kernel[static_cast<std::size_t>(k + radius)] *
                       temp[static_cast<std::size_t>(index(jz, ix, nx))];
            }
            output[static_cast<std::size_t>(index(iz, ix, nx))] =
                static_cast<float>(sum);
        }
    }
    return output;
}

inline double absorbing_eta(int iz, int ix, int nz, int nx,
                            int nabs, double damp_max)
{
    if (nabs <= 0 || !(damp_max > 0.0)) return 0.0;
    const int distance = std::min({iz, nz - 1 - iz, ix, nx - 1 - ix});
    if (distance >= nabs) return 0.0;
    const double q = static_cast<double>(nabs - distance) /
                     static_cast<double>(std::max(nabs, 1));
    return damp_max * q * q;
}

struct OperatorData {
    Sparse matrix;
    std::vector<double> k2;
};

inline OperatorData build_helmholtz(const std::vector<float>& velocity,
                                    int nz, int nx,
                                    double dz, double dx,
                                    double frequency,
                                    int nabs, double damp_max)
{
    if (nz < 3 || nx < 3 || static_cast<int>(velocity.size()) != nz * nx ||
        !(dx > 0.0) || !(dz > 0.0) || !(frequency > 0.0) || nabs < 0) {
        throw std::invalid_argument("invalid global Helmholtz grid");
    }
    if (nabs > 0 && (2 * nabs >= nz || 2 * nabs >= nx)) {
        throw std::invalid_argument("global absorbing layer is too thick");
    }

    const double omega = 2.0 * kPi * frequency;
    const double idx2 = 1.0 / (dx * dx);
    const double idz2 = 1.0 / (dz * dz);
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(5) * nz * nx);
    std::vector<double> k2(static_cast<std::size_t>(nz) * nx, 0.0);

    for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            const int row = index(iz, ix, nx);
            const double v = std::max(
                static_cast<double>(velocity[static_cast<std::size_t>(row)]),
                1.0);
            const double local_k2 = (omega / v) * (omega / v);
            k2[static_cast<std::size_t>(row)] = local_k2;
            const double eta = absorbing_eta(
                iz, ix, nz, nx, nabs, damp_max);

            // Production Kirchhoff convention uses exp(-i omega t), so the
            // absorbing term has the negative imaginary sign.
            triplets.emplace_back(
                row, row,
                Complex(local_k2 - 2.0 * idx2 - 2.0 * idz2,
                        -local_k2 * eta));
            if (iz > 0)
                triplets.emplace_back(row, index(iz - 1, ix, nx), Complex(idz2, 0.0));
            if (iz + 1 < nz)
                triplets.emplace_back(row, index(iz + 1, ix, nx), Complex(idz2, 0.0));
            if (ix > 0)
                triplets.emplace_back(row, index(iz, ix - 1, nx), Complex(idx2, 0.0));
            if (ix + 1 < nx)
                triplets.emplace_back(row, index(iz, ix + 1, nx), Complex(idx2, 0.0));
        }
    }

    Sparse matrix(nz * nx, nz * nx);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return {std::move(matrix), std::move(k2)};
}

class ShiftedSuperLUIncompleteLUT
    : public Eigen::SparseSolverBase<ShiftedSuperLUIncompleteLUT> {
protected:
    using Base = Eigen::SparseSolverBase<ShiftedSuperLUIncompleteLUT>;
    using Base::m_isInitialized;

public:
    using Scalar = Complex;
    using RealScalar = double;
    using StorageIndex = int;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic
    };

    ShiftedSuperLUIncompleteLUT() = default;
    ~ShiftedSuperLUIncompleteLUT() { clear(); }
    ShiftedSuperLUIncompleteLUT(const ShiftedSuperLUIncompleteLUT&) = delete;
    ShiftedSuperLUIncompleteLUT& operator=(const ShiftedSuperLUIncompleteLUT&) = delete;

    ShiftedSuperLUIncompleteLUT& configure(
        const std::vector<double>* k2,
        double beta,
        double drop_tolerance,
        int fill_factor,
        double pivot_threshold)
    {
        k2_ = k2;
        beta_ = beta;
        drop_tolerance_ = drop_tolerance;
        fill_factor_ = static_cast<double>(fill_factor);
        pivot_threshold_ = pivot_threshold;
        return *this;
    }

    template <typename MatrixType>
    ShiftedSuperLUIncompleteLUT& compute(const MatrixType& matrix)
    {
        clear();
        if (k2_ == nullptr || static_cast<int>(k2_->size()) != matrix.rows()) {
            throw std::runtime_error("shifted ILUT k2 vector is not configured");
        }

        Sparse shifted = matrix;
        for (int i = 0; i < shifted.rows(); ++i) {
            // Same sign as the production absorbing convention:
            // M = H - i beta k^2.
            shifted.coeffRef(i, i) +=
                Complex(0.0, -beta_ * (*k2_)[static_cast<std::size_t>(i)]);
        }
        shifted.makeCompressed();

        size_ = static_cast<int>(shifted.rows());
        values_.resize(static_cast<std::size_t>(shifted.nonZeros()));
        row_indices_.resize(static_cast<std::size_t>(shifted.nonZeros()));
        column_offsets_.resize(static_cast<std::size_t>(size_ + 1));
        for (int i = 0; i <= size_; ++i)
            column_offsets_[static_cast<std::size_t>(i)] = shifted.outerIndexPtr()[i];
        for (int i = 0; i < shifted.nonZeros(); ++i) {
            row_indices_[static_cast<std::size_t>(i)] = shifted.innerIndexPtr()[i];
            const Complex value = shifted.valuePtr()[i];
            values_[static_cast<std::size_t>(i)].r = value.real();
            values_[static_cast<std::size_t>(i)].i = value.imag();
        }

        zCreate_CompCol_Matrix(
            &matrix_, size_, size_, shifted.nonZeros(), values_.data(),
            row_indices_.data(), column_offsets_.data(),
            SLU_NC, SLU_Z, SLU_GE);

        ilu_set_default_options(&options_);
        options_.ColPerm = COLAMD;
        options_.DiagPivotThresh = pivot_threshold_;
        options_.ILU_DropTol = drop_tolerance_;
        options_.ILU_FillFactor = fill_factor_;
        options_.PrintStat = NO;

        const int panel_size = sp_ienv(1);
        const int relax = sp_ienv(2);
        column_permutation_.resize(static_cast<std::size_t>(size_));
        row_permutation_.resize(static_cast<std::size_t>(size_));
        elimination_tree_.resize(static_cast<std::size_t>(size_));

        get_perm_c(options_.ColPerm, &matrix_, column_permutation_.data());
        sp_preorder(&options_, &matrix_, column_permutation_.data(),
                    elimination_tree_.data(), &preordered_matrix_);

        SuperLUStat_t statistics;
        StatInit(&statistics);
        int_t info = 0;
        zgsitrf(&options_, &preordered_matrix_, relax, panel_size,
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
    Eigen::ComputationInfo info() const { return m_info; }

    template <typename Rhs, typename Dest>
    void _solve_impl(const Rhs& rhs, Dest& solution) const
    {
        solution = rhs;
        std::vector<doublecomplex> buffer(static_cast<std::size_t>(size_));
        for (int i = 0; i < size_; ++i) {
            buffer[static_cast<std::size_t>(i)].r = solution[i].real();
            buffer[static_cast<std::size_t>(i)].i = solution[i].imag();
        }

        SuperMatrix dense_rhs;
        zCreate_Dense_Matrix(
            &dense_rhs, size_, 1, buffer.data(), size_,
            SLU_DN, SLU_Z, SLU_GE);
        SuperLUStat_t statistics;
        StatInit(&statistics);
        int solve_info = 0;
        zgstrs(NOTRANS,
               const_cast<SuperMatrix*>(&lower_),
               const_cast<SuperMatrix*>(&upper_),
               const_cast<int*>(column_permutation_.data()),
               const_cast<int*>(row_permutation_.data()),
               &dense_rhs, &statistics, &solve_info);
        StatFree(&statistics);
        Destroy_SuperMatrix_Store(&dense_rhs);
        if (solve_info != 0)
            throw std::runtime_error("SuperLU ILUT triangular solve failed");

        for (int i = 0; i < size_; ++i) {
            const doublecomplex value = buffer[static_cast<std::size_t>(i)];
            solution[i] = Complex(value.r, value.i);
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

    const std::vector<double>* k2_ = nullptr;
    int size_ = 0;
    double beta_ = 0.30;
    double drop_tolerance_ = 3.0e-2;
    double fill_factor_ = 12.0;
    double pivot_threshold_ = 0.10;
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

using Solver = Eigen::GMRES<Sparse, ShiftedSuperLUIncompleteLUT>;

struct PreparedSystem {
    Sparse matrix;
    std::vector<double> k2;
    std::unique_ptr<Solver> solver;
    int nz = 0;
    int nx = 0;
    int physical_nz = 0;
    int physical_nx = 0;
    int padding = 0;
};

inline Vector embed_physical_field(const PreparedSystem& system,
                                   const Vector& physical)
{
    if (physical.size() != system.physical_nz * system.physical_nx) {
        throw std::invalid_argument("physical GMRES field size mismatch");
    }
    Vector padded = Vector::Zero(system.nz * system.nx);
    for (int iz = 0; iz < system.physical_nz; ++iz)
        for (int ix = 0; ix < system.physical_nx; ++ix)
            padded[index(iz + system.padding, ix + system.padding, system.nx)] =
                physical[index(iz, ix, system.physical_nx)];
    return padded;
}

inline Vector crop_physical_field(const PreparedSystem& system,
                                  const Vector& padded)
{
    if (padded.size() != system.nz * system.nx) {
        throw std::invalid_argument("padded GMRES field size mismatch");
    }
    Vector physical(system.physical_nz * system.physical_nx);
    for (int iz = 0; iz < system.physical_nz; ++iz)
        for (int ix = 0; ix < system.physical_nx; ++ix)
            physical[index(iz, ix, system.physical_nx)] =
                padded[index(iz + system.padding, ix + system.padding, system.nx)];
    return physical;
}

inline PreparedSystem prepare_system(const std::vector<float>& velocity,
                                     int nz, int nx,
                                     double dz, double dx,
                                     double frequency,
                                     const Parameters& p)
{
    if (p.iterations < 1 || p.restart < 1 || !(p.tolerance > 0.0) ||
        p.absorbing_rows < 0 || !(p.damp_max >= 0.0) ||
        p.smooth_sigma < 0.0 || !(p.shift_beta >= 0.0) ||
        !(p.ilut_drop_tolerance > 0.0) || p.ilut_fill_factor < 1 ||
        !(p.ilut_pivot_threshold >= 0.0) || p.receiver_mask_rows < 1) {
        throw std::invalid_argument("invalid global preconditioned GMRES parameters");
    }

    const std::vector<float> smooth =
        smooth_velocity(velocity, nz, nx, p.smooth_sigma);
    // The absorbing layer is appended outside the physical model. Edge
    // velocities are extended into it; solved fields are cropped back by
    // crop_physical_field() before imaging.
    const int padding = p.absorbing_rows;
    const int padded_nz = nz + 2 * padding;
    const int padded_nx = nx + 2 * padding;
    std::vector<float> padded_velocity(
        static_cast<std::size_t>(padded_nz) * padded_nx);
    for (int iz = 0; iz < padded_nz; ++iz) {
        const int physical_iz = std::clamp(iz - padding, 0, nz - 1);
        for (int ix = 0; ix < padded_nx; ++ix) {
            const int physical_ix = std::clamp(ix - padding, 0, nx - 1);
            padded_velocity[static_cast<std::size_t>(index(iz, ix, padded_nx))] =
                smooth[static_cast<std::size_t>(index(physical_iz, physical_ix, nx))];
        }
    }
    OperatorData op = build_helmholtz(
        padded_velocity, padded_nz, padded_nx, dz, dx, frequency,
        p.absorbing_rows, p.damp_max);

    PreparedSystem system;
    system.matrix = std::move(op.matrix);
    system.k2 = std::move(op.k2);
    system.nz = padded_nz;
    system.nx = padded_nx;
    system.physical_nz = nz;
    system.physical_nx = nx;
    system.padding = padding;
    system.solver = std::make_unique<Solver>();
    system.solver->set_restart(p.restart);
    system.solver->setMaxIterations(p.iterations);
    system.solver->setTolerance(p.tolerance);
    system.solver->preconditioner().configure(
        &system.k2, p.shift_beta, p.ilut_drop_tolerance,
        p.ilut_fill_factor, p.ilut_pivot_threshold);
    system.solver->compute(system.matrix);
    if (system.solver->info() != Eigen::Success) {
        throw std::runtime_error(
            "global shifted-Laplacian SuperLU-ILUT preconditioner construction failed");
    }
    return system;
}

inline Vector make_source_rhs(const PreparedSystem& system,
                              int source_ix,
                              Complex source_spectrum,
                              double dx, double dz)
{
    if (source_ix < 0 || source_ix >= system.physical_nx) {
        throw std::invalid_argument("global source index is outside model");
    }
    Vector b = Vector::Zero(system.nz * system.nx);
    b[index(system.padding, source_ix + system.padding, system.nx)] =
        source_spectrum / (dx * dz);
    return b;
}

inline Vector receiver_equivalent_source(const PreparedSystem& system,
                                         const Vector& receiver_initial,
                                         int transition_rows)
{
    if (receiver_initial.size() != system.matrix.rows()) {
        throw std::invalid_argument("receiver initial field size mismatch");
    }
    const int rows = std::clamp(
        transition_rows, 1, std::max(1, system.physical_nz - 2));
    Vector chi = Vector::Zero(receiver_initial.size());
    for (int iz = 0; iz < system.physical_nz; ++iz) {
        const double value = iz < rows
            ? static_cast<double>(iz) / static_cast<double>(rows)
            : 1.0;
        for (int ix = 0; ix < system.physical_nx; ++ix)
            chi[index(iz + system.padding, ix + system.padding, system.nx)] =
                Complex(value, 0.0);
    }
    const Vector x = chi.array() * receiver_initial.array();
    return system.matrix * x -
        (chi.array() * (system.matrix * receiver_initial).array()).matrix();
}

inline Vector apply_receiver_mask(const PreparedSystem& system,
                                  const Vector& receiver_initial,
                                  int transition_rows)
{
    const int rows = std::clamp(
        transition_rows, 1, std::max(1, system.physical_nz - 2));
    Vector x = Vector::Zero(receiver_initial.size());
    for (int iz = 0; iz < system.physical_nz; ++iz) {
        const double value = static_cast<double>(iz) /
            static_cast<double>(rows);
        const double weight = iz < rows ? value : 1.0;
        for (int ix = 0; ix < system.physical_nx; ++ix) {
            const int padded = index(
                iz + system.padding, ix + system.padding, system.nx);
            x[padded] = receiver_initial[padded] * weight;
        }
    }
    return x;
}

inline Vector solve(PreparedSystem& system,
                    const Vector& rhs,
                    const Vector& initial,
                    Metric* metric = nullptr)
{
    if (rhs.size() != system.matrix.rows() || initial.size() != rhs.size()) {
        throw std::invalid_argument("global GMRES vector size mismatch");
    }
    const double rhs_norm = std::max(
        rhs.norm(), std::numeric_limits<double>::epsilon());
    const Vector residual_before = rhs - system.matrix * initial;
    const Vector corrected = system.solver->solveWithGuess(rhs, initial);
    if (system.solver->info() != Eigen::Success &&
        system.solver->info() != Eigen::NoConvergence) {
        throw std::runtime_error("global preconditioned GMRES failed");
    }
    if (metric != nullptr) {
        metric->gmres_steps = static_cast<int>(system.solver->iterations());
        metric->residual_before = residual_before.norm() / rhs_norm;
        metric->residual_after =
            (rhs - system.matrix * corrected).norm() / rhs_norm;
        metric->correction_norm = (corrected - initial).norm();
    }
    return corrected;
}

} // namespace global_preconditioned_gmres

#endif
