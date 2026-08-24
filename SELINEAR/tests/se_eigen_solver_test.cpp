#include "se_eigen.hpp"

#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Scalar>
void require_close(const se::eigen::DenseVector<Scalar>& value,
                   const se::eigen::DenseVector<Scalar>& reference,
                   double tolerance,
                   const char* label) {
    const double error = se::eigen::relative_error(value, reference);
    if (!(error <= tolerance)) {
        throw std::runtime_error(
            std::string(label) + " relative error=" + std::to_string(error));
    }
}

void test_real_direct_and_iterative() {
    using Scalar = double;
    using namespace se::eigen;

    const std::vector<Triplet<Scalar>> entries = {
        {0, 0, 4.0}, {0, 1, -1.0},
        {1, 0, -1.0}, {1, 1, 4.0}, {1, 2, -1.0},
        {2, 1, -1.0}, {2, 2, 3.0},
    };
    const SparseMatrix<Scalar> matrix = make_sparse_matrix<Scalar>(3, 3, entries);

    DenseVector<Scalar> reference(3);
    reference << 1.0, 2.0, -1.0;
    const DenseVector<Scalar> rhs = multiply(matrix, reference);

    const SolverResult<Scalar> direct = solve_sparse_lu(matrix, rhs);
    if (!direct.report.converged()) {
        throw std::runtime_error("SparseLU did not converge");
    }
    require_close(direct.solution, reference, 1.0e-12, "SparseLU");

    SolverOptions options;
    options.max_iterations = 100;
    options.tolerance = 1.0e-12;
    options.preconditioner = Preconditioner::diagonal;

    for (const IterativeMethod method : {
             IterativeMethod::conjugate_gradient,
             IterativeMethod::bicgstab,
             IterativeMethod::gmres,
             IterativeMethod::dgmres}) {
        const SolverResult<Scalar> result =
            solve_iterative(matrix, rhs, method, options);
        if (!result.report.converged()) {
            throw std::runtime_error(
                std::string(to_string(method)) + " did not converge");
        }
        require_close(result.solution, reference, 1.0e-9, to_string(method));
    }

    DenseVector<Scalar> guess = DenseVector<Scalar>::Constant(3, 0.5);
    const SolverResult<Scalar> guessed = solve_iterative_with_guess(
        matrix, rhs, guess, IterativeMethod::gmres, options);
    require_close(guessed.solution, reference, 1.0e-9, "GMRES with guess");
}

void test_minres() {
    using Scalar = double;
    using namespace se::eigen;

    const std::vector<Triplet<Scalar>> entries = {
        {0, 0, 2.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, -2.0},
    };
    const SparseMatrix<Scalar> matrix = make_sparse_matrix<Scalar>(2, 2, entries);
    DenseVector<Scalar> reference(2);
    reference << 1.5, -0.5;
    const DenseVector<Scalar> rhs = matrix * reference;

    SolverOptions options;
    options.max_iterations = 100;
    options.tolerance = 1.0e-12;
    options.preconditioner = Preconditioner::identity;

    const SolverResult<Scalar> result =
        solve_iterative(matrix, rhs, IterativeMethod::minres, options);
    if (!result.report.converged()) {
        throw std::runtime_error("MINRES did not converge");
    }
    require_close(result.solution, reference, 1.0e-9, "MINRES");
}

void test_least_squares() {
    using Scalar = double;
    using namespace se::eigen;

    const std::vector<Triplet<Scalar>> entries = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 2.0}, {1, 1, -1.0},
        {2, 0, -1.0}, {2, 1, 2.0},
        {3, 0, 3.0}, {3, 1, 1.0},
    };
    const SparseMatrix<Scalar> matrix = make_sparse_matrix<Scalar>(4, 2, entries);
    DenseVector<Scalar> reference(2);
    reference << 2.0, -1.0;
    const DenseVector<Scalar> rhs = matrix * reference;

    const SolverResult<Scalar> qr = solve_sparse_qr(matrix, rhs);
    require_close(qr.solution, reference, 1.0e-10, "SparseQR");

    SolverOptions options;
    options.max_iterations = 100;
    options.tolerance = 1.0e-12;
    options.preconditioner = Preconditioner::diagonal;
    const SolverResult<Scalar> iterative = solve_iterative(
        matrix,
        rhs,
        IterativeMethod::least_squares_conjugate_gradient,
        options);
    if (!iterative.report.converged()) {
        throw std::runtime_error("LeastSquaresCG did not converge");
    }
    require_close(iterative.solution, reference, 1.0e-9, "LeastSquaresCG");
}

void test_complex_helmholtz_like_system() {
    using Scalar = std::complex<double>;
    using namespace se::eigen;

    const Scalar diagonal(4.0, 0.4);
    const Scalar off_diagonal(-1.0, 0.1);
    const std::vector<Triplet<Scalar>> entries = {
        {0, 0, diagonal}, {0, 1, off_diagonal},
        {1, 0, off_diagonal}, {1, 1, diagonal}, {1, 2, off_diagonal},
        {2, 1, off_diagonal}, {2, 2, diagonal},
    };
    const SparseMatrix<Scalar> matrix = make_sparse_matrix<Scalar>(3, 3, entries);

    DenseVector<Scalar> reference(3);
    reference << Scalar(1.0, 0.5), Scalar(-0.25, 1.0), Scalar(0.5, -0.75);
    const DenseVector<Scalar> rhs = matrix * reference;

    SolverOptions options;
    options.max_iterations = 200;
    options.tolerance = 1.0e-12;
    options.restart = 10;
    options.preconditioner = Preconditioner::incomplete_lut;
    options.ilut_drop_tolerance = 1.0e-6;
    options.ilut_fill_factor = 10;

    for (const IterativeMethod method : {
             IterativeMethod::bicgstab,
             IterativeMethod::gmres,
             IterativeMethod::dgmres}) {
        const SolverResult<Scalar> result =
            solve_iterative(matrix, rhs, method, options);
        if (!result.report.converged()) {
            throw std::runtime_error(
                std::string("complex ") + to_string(method) +
                " did not converge");
        }
        require_close(result.solution, reference, 1.0e-8, to_string(method));
    }
}

void test_dense_helpers_and_string_parsing() {
    using namespace se::eigen;

    DenseMatrix<double> matrix(2, 2);
    matrix << 3.0, 1.0,
              1.0, 2.0;
    DenseVector<double> reference(2);
    reference << 2.0, -1.0;
    const DenseVector<double> rhs = matrix * reference;

    require_close(
        solve_dense_lu(matrix, rhs).solution,
        reference,
        1.0e-12,
        "DenseLU");
    require_close(
        solve_dense_qr(matrix, rhs).solution,
        reference,
        1.0e-12,
        "DenseQR");

    if (iterative_method_from_string("GMRES") != IterativeMethod::gmres ||
        preconditioner_from_string("ILUT") != Preconditioner::incomplete_lut) {
        throw std::runtime_error("string parser returned the wrong enum");
    }
}

}  // namespace

int main() {
    try {
        test_real_direct_and_iterative();
        test_minres();
        test_least_squares();
        test_complex_helmholtz_like_system();
        test_dense_helpers_and_string_parsing();
        std::cout << "Eigen solver API tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Eigen solver API test failed: " << error.what() << '\n';
        return 1;
    }
}
