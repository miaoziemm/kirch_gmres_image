#ifndef GLOBAL_PRECONDITIONED_GMRES_HPP
#define GLOBAL_PRECONDITIONED_GMRES_HPP

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

namespace global_preconditioned_gmres {

using Complex = std::complex<double>;
using Sparse = Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>;
using Vector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using Triplet = Eigen::Triplet<Complex, int>;

constexpr double kPi = 3.14159265358979323846;
constexpr int kPml = 25;
constexpr int kDefaultRestart = 30;
constexpr int kDefaultOuter = 10;
constexpr double kTolerance = 1.0e-12;

inline int index(int iz, int ix, int nx)
{
    return iz * nx + ix;
}

inline double damping(int iz, int ix, int nz, int nx)
{
    const int distance = std::min({iz, nz - 1 - iz, ix, nx - 1 - ix});
    if (distance >= kPml) return 0.0;

    const double t = static_cast<double>(kPml - distance) / kPml;
    return 2.0 * t * t;
}

inline void add_second_derivative(std::vector<Triplet>& terms,
                                  int row, int coordinate,
                                  int count, int stride, double h)
{
    const double scale = 1.0 / (h * h);
    const auto add = [&](int offset, double value) {
        terms.emplace_back(row, row + offset * stride, Complex(value * scale));
    };

    if (coordinate < 4 || coordinate + 4 >= count) {
        if (coordinate == 0) {
            add(0, 2.0); add(1, -5.0); add(2, 4.0); add(3, -1.0);
        } else if (coordinate == count - 1) {
            add(-3, -1.0); add(-2, 4.0); add(-1, -5.0); add(0, 2.0);
        } else {
            add(-1, 1.0); add(0, -2.0); add(1, 1.0);
        }
        return;
    }

    add(-4, -1.0 / 560.0);
    add(-3,  8.0 / 315.0);
    add(-2, -1.0 / 5.0);
    add(-1,  8.0 / 5.0);
    add( 0, -205.0 / 72.0);
    add( 1,  8.0 / 5.0);
    add( 2, -1.0 / 5.0);
    add( 3,  8.0 / 315.0);
    add( 4, -1.0 / 560.0);
}

inline Sparse build_helmholtz(const std::vector<float>& velocity,
                              int nz, int nx, double dz, double dx,
                              double frequency, bool adjoint_operator = false)
{
    const double omega = 2.0 * kPi * frequency;
    std::vector<Triplet> terms;
    terms.reserve(static_cast<std::size_t>(17) * nz * nx);

    for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            const int row = index(iz, ix, nx);
            const double k2 = std::pow(omega / velocity[row], 2.0);

            add_second_derivative(terms, row, iz, nz, nx, dz);
            add_second_derivative(terms, row, ix, nx, 1, dx);
            const double pml = k2 * damping(iz, ix, nz, nx);
            // The receiver spectrum is conjugated before continuation, so
            // its global correction uses the adjoint Helmholtz operator.  The
            // imaging condition subsequently correlates source with the
            // conjugate of this corrected receiver field.
            terms.emplace_back(
                row, row, Complex(k2, adjoint_operator ? -pml : pml));
        }
    }

    Sparse matrix(nz * nx, nz * nx);
    matrix.setFromTriplets(terms.begin(), terms.end());
    matrix.makeCompressed();
    return matrix;
}

struct Result {
    Vector field;
    int steps = 0;
    double residual_before = 0.0;
    double residual_after = 0.0;
};

inline Result solve(const Sparse& matrix, const Vector& rhs,
                    const Vector& initial, int restart, int outer)
{
    const double rhs_norm = std::max(
        rhs.norm(), std::numeric_limits<double>::epsilon());

    Vector field = initial;
    Vector residual = rhs - matrix * field;
    const double residual_before = residual.norm();
    int steps = 0;
    const int max_iterations = restart * outer;

    while (steps < max_iterations &&
           residual.norm() > kTolerance * rhs_norm) {
        const int columns = std::min(restart, max_iterations - steps);
        const double beta = residual.norm();
        Eigen::MatrixXcd basis(rhs.size(), columns + 1);
        Eigen::MatrixXcd hessenberg =
            Eigen::MatrixXcd::Zero(columns + 1, columns);
        basis.col(0) = residual / beta;

        int used = 0;
        for (int j = 0; j < columns; ++j) {
            Vector w = matrix * basis.col(j);
            for (int i = 0; i <= j; ++i) {
                const Complex h = basis.col(i).dot(w);
                hessenberg(i, j) = h;
                w -= h * basis.col(i);
            }

            const double hnext = w.norm();
            hessenberg(j + 1, j) = hnext;
            used = j + 1;
            if (hnext <= std::numeric_limits<double>::epsilon()) break;
            basis.col(j + 1) = w / hnext;
        }

        Vector g = Vector::Zero(used + 1);
        g[0] = beta;
        const Vector y = hessenberg.topLeftCorner(used + 1, used)
                             .householderQr()
                             .solve(g);
        field += basis.leftCols(used) * y;
        residual = rhs - matrix * field;
        steps += used;
    }

    return {std::move(field), steps,
            residual_before / rhs_norm, residual.norm() / rhs_norm};
}

} // namespace global_preconditioned_gmres

#endif