#ifndef SE_EIGEN_SOLVER_IMPL_HPP
#define SE_EIGEN_SOLVER_IMPL_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <unsupported/Eigen/IterativeSolvers>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace se {
namespace eigen {

template <typename Scalar>
using DenseVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using DenseMatrix =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename Scalar>
using SparseMatrix =
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

template <typename Scalar>
using Triplet = Eigen::Triplet<Scalar, int>;

template <typename Scalar>
using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

enum class SolverStatus {
    success,
    numerical_issue,
    no_convergence,
    invalid_input
};

enum class IterativeMethod {
    conjugate_gradient,
    bicgstab,
    gmres,
    dgmres,
    minres,
    least_squares_conjugate_gradient
};

enum class Preconditioner {
    identity,
    diagonal,
    incomplete_lut
};

struct SolverOptions {
    int max_iterations = 1000;
    double tolerance = 1.0e-8;
    int restart = 40;
    int dgmres_eigenvalues = 0;
    int dgmres_max_eigenvalues = 20;
    double ilut_drop_tolerance = 1.0e-3;
    int ilut_fill_factor = 10;
    Preconditioner preconditioner = Preconditioner::diagonal;
};

struct SolverReport {
    SolverStatus status = SolverStatus::invalid_input;
    int iterations = 0;
    double estimated_error = std::numeric_limits<double>::infinity();
    double relative_residual = std::numeric_limits<double>::infinity();

    bool converged() const noexcept { return status == SolverStatus::success; }
};

template <typename Scalar>
struct SolverResult {
    DenseVector<Scalar> solution;
    SolverReport report;
};

const char* to_string(SolverStatus status) noexcept;
const char* to_string(IterativeMethod method) noexcept;
const char* to_string(Preconditioner preconditioner) noexcept;

IterativeMethod iterative_method_from_string(const std::string& value);
Preconditioner preconditioner_from_string(const std::string& value);
SolverStatus solver_status_from_eigen(Eigen::ComputationInfo info) noexcept;

namespace detail {

inline std::string normalized_name(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        if (c == '-' || c == ' ') return '_';
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline void validate_options(const SolverOptions& options) {
    if (options.max_iterations < 1) {
        throw std::invalid_argument("max_iterations must be positive");
    }
    if (!(options.tolerance > 0.0) || !std::isfinite(options.tolerance)) {
        throw std::invalid_argument("tolerance must be positive and finite");
    }
    if (options.restart < 1) {
        throw std::invalid_argument("restart must be positive");
    }
    if (options.dgmres_eigenvalues < 0 ||
        options.dgmres_max_eigenvalues < 1 ||
        options.dgmres_eigenvalues > options.dgmres_max_eigenvalues) {
        throw std::invalid_argument("invalid DGMRES deflation parameters");
    }
    if (!(options.ilut_drop_tolerance >= 0.0) ||
        !std::isfinite(options.ilut_drop_tolerance)) {
        throw std::invalid_argument(
            "ilut_drop_tolerance must be nonnegative and finite");
    }
    if (options.ilut_fill_factor < 1) {
        throw std::invalid_argument("ilut_fill_factor must be positive");
    }
}

template <typename Scalar>
void validate_system(const SparseMatrix<Scalar>& matrix,
                     const DenseVector<Scalar>& rhs,
                     bool require_square) {
    if (matrix.rows() < 1 || matrix.cols() < 1) {
        throw std::invalid_argument("matrix must be nonempty");
    }
    if (require_square && matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("solver requires a square matrix");
    }
    if (rhs.size() != matrix.rows()) {
        throw std::invalid_argument("right-hand side size does not match matrix rows");
    }
}

template <typename Scalar>
void validate_guess(const SparseMatrix<Scalar>& matrix,
                    const DenseVector<Scalar>* initial_guess) {
    if (initial_guess != nullptr && initial_guess->size() != matrix.cols()) {
        throw std::invalid_argument(
            "initial guess size does not match matrix columns");
    }
}

template <typename Derived>
double norm_as_double(const Eigen::MatrixBase<Derived>& value) {
    return static_cast<double>(value.norm());
}

template <typename Scalar>
double relative_residual_value(const SparseMatrix<Scalar>& matrix,
                               const DenseVector<Scalar>& solution,
                               const DenseVector<Scalar>& rhs) {
    const DenseVector<Scalar> residual = rhs - matrix * solution;
    const double rhs_norm = norm_as_double(rhs);
    const double denominator = rhs_norm > std::numeric_limits<double>::min()
        ? rhs_norm
        : 1.0;
    return norm_as_double(residual) / denominator;
}

template <typename Solver, typename Scalar>
SolverResult<Scalar> run_solver(
    Solver& solver,
    const SparseMatrix<Scalar>& matrix,
    const DenseVector<Scalar>& rhs,
    const SolverOptions& options,
    const DenseVector<Scalar>* initial_guess) {
    solver.setMaxIterations(options.max_iterations);
    solver.setTolerance(static_cast<RealScalar<Scalar>>(options.tolerance));
    solver.compute(matrix);

    SolverResult<Scalar> result;
    if (solver.info() != Eigen::Success) {
        result.solution = DenseVector<Scalar>::Zero(matrix.cols());
        result.report.status = solver_status_from_eigen(solver.info());
        return result;
    }

    if (initial_guess != nullptr) {
        result.solution = solver.solveWithGuess(rhs, *initial_guess);
    } else {
        result.solution = solver.solve(rhs);
    }

    result.report.status = solver_status_from_eigen(solver.info());
    result.report.iterations = static_cast<int>(solver.iterations());
    result.report.estimated_error = static_cast<double>(solver.error());
    result.report.relative_residual =
        relative_residual_value(matrix, result.solution, rhs);
    return result;
}

template <typename Scalar>
SolverResult<Scalar> run_cg(const SparseMatrix<Scalar>& matrix,
                            const DenseVector<Scalar>& rhs,
                            const SolverOptions& options,
                            const DenseVector<Scalar>* guess) {
    if (options.preconditioner == Preconditioner::identity) {
        using Solver = Eigen::ConjugateGradient<
            SparseMatrix<Scalar>,
            Eigen::Lower | Eigen::Upper,
            Eigen::IdentityPreconditioner>;
        Solver solver;
        return run_solver(solver, matrix, rhs, options, guess);
    }
    if (options.preconditioner == Preconditioner::diagonal) {
        using Solver = Eigen::ConjugateGradient<
            SparseMatrix<Scalar>,
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<Scalar>>;
        Solver solver;
        return run_solver(solver, matrix, rhs, options, guess);
    }
    throw std::invalid_argument(
        "ConjugateGradient supports identity or diagonal preconditioning");
}

template <typename Scalar>
SolverResult<Scalar> run_bicgstab(const SparseMatrix<Scalar>& matrix,
                                  const DenseVector<Scalar>& rhs,
                                  const SolverOptions& options,
                                  const DenseVector<Scalar>* guess) {
    if (options.preconditioner == Preconditioner::identity) {
        using Solver = Eigen::BiCGSTAB<
            SparseMatrix<Scalar>, Eigen::IdentityPreconditioner>;
        Solver solver;
        return run_solver(solver, matrix, rhs, options, guess);
    }
    if (options.preconditioner == Preconditioner::diagonal) {
        using Solver = Eigen::BiCGSTAB<
            SparseMatrix<Scalar>, Eigen::DiagonalPreconditioner<Scalar>>;
        Solver solver;
        return run_solver(solver, matrix, rhs, options, guess);
    }

    using Solver = Eigen::BiCGSTAB<
        SparseMatrix<Scalar>, Eigen::IncompleteLUT<Scalar, int>>;
    Solver solver;
    solver.preconditioner().setDroptol(
        static_cast<RealScalar<Scalar>>(options.ilut_drop_tolerance));
    solver.preconditioner().setFillfactor(options.ilut_fill_factor);
    return run_solver(solver, matrix, rhs, options, guess);
}

template <typename Scalar>
SolverResult<Scalar> run_gmres(const SparseMatrix<Scalar>& matrix,
                               const DenseVector<Scalar>& rhs,
                               const SolverOptions& options,
                               const DenseVector<Scalar>* guess) {
    if (options.preconditioner == Preconditioner::identity) {
        using Solver = Eigen::GMRES<
            SparseMatrix<Scalar>, Eigen::IdentityPreconditioner>;
        Solver solver;
        solver.set_restart(options.restart);
        return run_solver(solver, matrix, rhs, options, guess);
    }
    if (options.preconditioner == Preconditioner::diagonal) {
        using Solver = Eigen::GMRES<
            SparseMatrix<Scalar>, Eigen::DiagonalPreconditioner<Scalar>>;
        Solver solver;
        solver.set_restart(options.restart);
        return run_solver(solver, matrix, rhs, options, guess);
    }

    using Solver = Eigen::GMRES<
        SparseMatrix<Scalar>, Eigen::IncompleteLUT<Scalar, int>>;
    Solver solver;
    solver.set_restart(options.restart);
    solver.preconditioner().setDroptol(
        static_cast<RealScalar<Scalar>>(options.ilut_drop_tolerance));
    solver.preconditioner().setFillfactor(options.ilut_fill_factor);
    return run_solver(solver, matrix, rhs, options, guess);
}

template <typename Scalar>
SolverResult<Scalar> run_dgmres(const SparseMatrix<Scalar>& matrix,
                                const DenseVector<Scalar>& rhs,
                                const SolverOptions& options,
                                const DenseVector<Scalar>* guess) {
    auto configure = [&options](auto& solver) {
        solver.set_restart(options.restart);
        solver.setEigenv(options.dgmres_eigenvalues);
        solver.setMaxEigenv(options.dgmres_max_eigenvalues);
    };

    if (options.preconditioner == Preconditioner::identity) {
        using Solver = Eigen::DGMRES<
            SparseMatrix<Scalar>, Eigen::IdentityPreconditioner>;
        Solver solver;
        configure(solver);
        return run_solver(solver, matrix, rhs, options, guess);
    }
    if (options.preconditioner == Preconditioner::diagonal) {
        using Solver = Eigen::DGMRES<
            SparseMatrix<Scalar>, Eigen::DiagonalPreconditioner<Scalar>>;
        Solver solver;
        configure(solver);
        return run_solver(solver, matrix, rhs, options, guess);
    }

    using Solver = Eigen::DGMRES<
        SparseMatrix<Scalar>, Eigen::IncompleteLUT<Scalar, int>>;
    Solver solver;
    configure(solver);
    solver.preconditioner().setDroptol(
        static_cast<RealScalar<Scalar>>(options.ilut_drop_tolerance));
    solver.preconditioner().setFillfactor(options.ilut_fill_factor);
    return run_solver(solver, matrix, rhs, options, guess);
}

template <typename Scalar>
SolverResult<Scalar> run_minres(const SparseMatrix<Scalar>& matrix,
                                const DenseVector<Scalar>& rhs,
                                const SolverOptions& options,
                                const DenseVector<Scalar>* guess) {
    if (options.preconditioner == Preconditioner::identity) {
        using Solver = Eigen::MINRES<
            SparseMatrix<Scalar>,
            Eigen::Lower | Eigen::Upper,
            Eigen::IdentityPreconditioner>;
        Solver solver;
        return run_solver(solver, matrix, rhs, options, guess);
    }
    if (options.preconditioner == Preconditioner::diagonal) {
        using Solver = Eigen::MINRES<
            SparseMatrix<Scalar>,
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<Scalar>>;
        Solver solver;
        return run_solver(solver, matrix, rhs, options, guess);
    }
    throw std::invalid_argument(
        "MINRES supports identity or diagonal preconditioning");
}

template <typename Scalar>
SolverResult<Scalar> run_lscg(const SparseMatrix<Scalar>& matrix,
                              const DenseVector<Scalar>& rhs,
                              const SolverOptions& options,
                              const DenseVector<Scalar>* guess) {
    if (options.preconditioner == Preconditioner::identity) {
        using Solver = Eigen::LeastSquaresConjugateGradient<
            SparseMatrix<Scalar>, Eigen::IdentityPreconditioner>;
        Solver solver;
        return run_solver(solver, matrix, rhs, options, guess);
    }
    if (options.preconditioner == Preconditioner::diagonal) {
        using Solver = Eigen::LeastSquaresConjugateGradient<
            SparseMatrix<Scalar>,
            Eigen::LeastSquareDiagonalPreconditioner<Scalar>>;
        Solver solver;
        return run_solver(solver, matrix, rhs, options, guess);
    }
    throw std::invalid_argument(
        "LeastSquaresConjugateGradient supports identity or diagonal "
        "preconditioning");
}

template <typename Scalar>
SolverResult<Scalar> dispatch_iterative(
    const SparseMatrix<Scalar>& matrix,
    const DenseVector<Scalar>& rhs,
    IterativeMethod method,
    const SolverOptions& options,
    const DenseVector<Scalar>* guess) {
    switch (method) {
        case IterativeMethod::conjugate_gradient:
            return run_cg(matrix, rhs, options, guess);
        case IterativeMethod::bicgstab:
            return run_bicgstab(matrix, rhs, options, guess);
        case IterativeMethod::gmres:
            return run_gmres(matrix, rhs, options, guess);
        case IterativeMethod::dgmres:
            return run_dgmres(matrix, rhs, options, guess);
        case IterativeMethod::minres:
            return run_minres(matrix, rhs, options, guess);
        case IterativeMethod::least_squares_conjugate_gradient:
            return run_lscg(matrix, rhs, options, guess);
    }
    throw std::invalid_argument("unknown iterative method");
}

}  // namespace detail

template <typename Scalar>
SparseMatrix<Scalar> make_sparse_matrix(
    Eigen::Index rows,
    Eigen::Index cols,
    const std::vector<Triplet<Scalar>>& entries) {
    if (rows < 1 || cols < 1) {
        throw std::invalid_argument("sparse matrix dimensions must be positive");
    }
    if (rows > std::numeric_limits<int>::max() ||
        cols > std::numeric_limits<int>::max()) {
        throw std::overflow_error("sparse matrix exceeds 32-bit storage index");
    }
    SparseMatrix<Scalar> matrix(rows, cols);
    matrix.setFromTriplets(
        entries.begin(), entries.end(),
        [](const Scalar& lhs, const Scalar& rhs) { return lhs + rhs; });
    matrix.makeCompressed();
    return matrix;
}

template <typename Scalar>
SparseMatrix<Scalar> dense_to_sparse(
    const DenseMatrix<Scalar>& matrix,
    RealScalar<Scalar> reference = RealScalar<Scalar>(0)) {
    SparseMatrix<Scalar> sparse = matrix.sparseView(reference);
    sparse.makeCompressed();
    return sparse;
}

template <typename Scalar>
DenseVector<Scalar> multiply(const SparseMatrix<Scalar>& matrix,
                             const DenseVector<Scalar>& vector) {
    if (vector.size() != matrix.cols()) {
        throw std::invalid_argument("vector size does not match matrix columns");
    }
    return matrix * vector;
}

template <typename Scalar>
DenseVector<Scalar> residual(const SparseMatrix<Scalar>& matrix,
                             const DenseVector<Scalar>& solution,
                             const DenseVector<Scalar>& rhs) {
    detail::validate_system(matrix, rhs, false);
    if (solution.size() != matrix.cols()) {
        throw std::invalid_argument("solution size does not match matrix columns");
    }
    return rhs - matrix * solution;
}

template <typename Scalar>
double relative_residual(const SparseMatrix<Scalar>& matrix,
                         const DenseVector<Scalar>& solution,
                         const DenseVector<Scalar>& rhs) {
    detail::validate_system(matrix, rhs, false);
    if (solution.size() != matrix.cols()) {
        throw std::invalid_argument("solution size does not match matrix columns");
    }
    return detail::relative_residual_value(matrix, solution, rhs);
}

template <typename Scalar>
double relative_error(const DenseVector<Scalar>& value,
                      const DenseVector<Scalar>& reference) {
    if (value.size() != reference.size()) {
        throw std::invalid_argument("vector sizes differ");
    }
    const double reference_norm = detail::norm_as_double(reference);
    const double denominator =
        reference_norm > std::numeric_limits<double>::min()
            ? reference_norm
            : 1.0;
    return detail::norm_as_double(value - reference) / denominator;
}

template <typename Scalar>
SolverResult<Scalar> solve_dense_lu(const DenseMatrix<Scalar>& matrix,
                                    const DenseVector<Scalar>& rhs) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("dense LU requires a square matrix");
    }
    if (rhs.size() != matrix.rows()) {
        throw std::invalid_argument("right-hand side size does not match matrix");
    }
    Eigen::FullPivLU<DenseMatrix<Scalar>> solver(matrix);
    SolverResult<Scalar> result;
    result.solution = solver.solve(rhs);
    result.report.status = solver.isInvertible()
        ? SolverStatus::success
        : SolverStatus::numerical_issue;
    result.report.relative_residual =
        detail::norm_as_double(rhs - matrix * result.solution) /
        std::max(detail::norm_as_double(rhs), 1.0);
    result.report.estimated_error = result.report.relative_residual;
    return result;
}

template <typename Scalar>
SolverResult<Scalar> solve_dense_qr(const DenseMatrix<Scalar>& matrix,
                                    const DenseVector<Scalar>& rhs) {
    if (matrix.rows() < 1 || matrix.cols() < 1) {
        throw std::invalid_argument("matrix must be nonempty");
    }
    if (rhs.size() != matrix.rows()) {
        throw std::invalid_argument("right-hand side size does not match matrix");
    }
    Eigen::ColPivHouseholderQR<DenseMatrix<Scalar>> solver(matrix);
    SolverResult<Scalar> result;
    result.solution = solver.solve(rhs);
    result.report.status = solver.rank() == std::min(matrix.rows(), matrix.cols())
        ? SolverStatus::success
        : SolverStatus::numerical_issue;
    result.report.relative_residual =
        detail::norm_as_double(rhs - matrix * result.solution) /
        std::max(detail::norm_as_double(rhs), 1.0);
    result.report.estimated_error = result.report.relative_residual;
    return result;
}

template <typename Scalar>
SolverResult<Scalar> solve_sparse_lu(const SparseMatrix<Scalar>& matrix,
                                     const DenseVector<Scalar>& rhs) {
    detail::validate_system(matrix, rhs, true);
    Eigen::SparseLU<
        SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(matrix);
    solver.factorize(matrix);
    SolverResult<Scalar> result;
    if (solver.info() != Eigen::Success) {
        result.solution = DenseVector<Scalar>::Zero(matrix.cols());
        result.report.status = solver_status_from_eigen(solver.info());
        return result;
    }
    result.solution = solver.solve(rhs);
    result.report.status = solver_status_from_eigen(solver.info());
    result.report.relative_residual =
        detail::relative_residual_value(matrix, result.solution, rhs);
    result.report.estimated_error = result.report.relative_residual;
    return result;
}

template <typename Scalar>
SolverResult<Scalar> solve_sparse_qr(const SparseMatrix<Scalar>& matrix,
                                     const DenseVector<Scalar>& rhs) {
    detail::validate_system(matrix, rhs, false);
    Eigen::SparseQR<
        SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(matrix);
    SolverResult<Scalar> result;
    if (solver.info() != Eigen::Success) {
        result.solution = DenseVector<Scalar>::Zero(matrix.cols());
        result.report.status = solver_status_from_eigen(solver.info());
        return result;
    }
    result.solution = solver.solve(rhs);
    result.report.status = solver_status_from_eigen(solver.info());
    result.report.relative_residual =
        detail::relative_residual_value(matrix, result.solution, rhs);
    result.report.estimated_error = result.report.relative_residual;
    return result;
}

template <typename Scalar>
SolverResult<Scalar> solve_iterative(
    const SparseMatrix<Scalar>& matrix,
    const DenseVector<Scalar>& rhs,
    IterativeMethod method,
    const SolverOptions& options = SolverOptions()) {
    detail::validate_options(options);
    detail::validate_system(
        matrix,
        rhs,
        method != IterativeMethod::least_squares_conjugate_gradient);
    return detail::dispatch_iterative(
        matrix,
        rhs,
        method,
        options,
        static_cast<const DenseVector<Scalar>*>(nullptr));
}

template <typename Scalar>
SolverResult<Scalar> solve_iterative_with_guess(
    const SparseMatrix<Scalar>& matrix,
    const DenseVector<Scalar>& rhs,
    const DenseVector<Scalar>& initial_guess,
    IterativeMethod method,
    const SolverOptions& options = SolverOptions()) {
    detail::validate_options(options);
    detail::validate_system(
        matrix,
        rhs,
        method != IterativeMethod::least_squares_conjugate_gradient);
    detail::validate_guess(matrix, &initial_guess);
    return detail::dispatch_iterative(
        matrix, rhs, method, options, &initial_guess);
}

}  // namespace eigen
}  // namespace se

#endif  // SE_EIGEN_SOLVER_IMPL_HPP
