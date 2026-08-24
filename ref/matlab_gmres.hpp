#ifndef MATLAB_GMRES_HPP
#define MATLAB_GMRES_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/IterativeSolvers>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

namespace matlab_gmres {

using Complex = std::complex<double>;
using Sparse = Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>;
using Vector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using Triplet = Eigen::Triplet<Complex, int>;

struct Parameters {
    int outer_iterations = 3;
    int gmres_restart = 30;
    int gmres_cycles = 1;
    int absorbing_rows = 25;
    double gmres_tolerance = 1.0e-12;
    double damp_max = 2.0;
};

struct IterationMetric {
    int outer = 0;
    int gmres_steps = 0;
    double residual_before = 0.0;
    double residual_after_gmres = 0.0;
    double gmres_correction_norm = 0.0;
    bool gmres_accepted = false;
};

inline int index(int iz, int ix, int nz) { return iz + ix * nz; }

inline Sparse build_helmholtz(const std::vector<float>& velocity,
                              int nz, int nx, double dz, double dx,
                              double frequency, int nabs, double damp_max)
{
    if (nz < 3 || nx < 3 || static_cast<int>(velocity.size()) != nz * nx ||
        !(dx > 0.0) || !(dz > 0.0) || !(frequency > 0.0) || nabs < 0) {
        throw std::invalid_argument("invalid MATLAB Helmholtz grid");
    }
    const double omega = 2.0 * 3.14159265358979323846 * frequency;
    const double idx2 = 1.0 / (dx * dx);
    const double idz2 = 1.0 / (dz * dz);
    std::vector<Triplet> t;
    t.reserve(static_cast<std::size_t>(5) * nz * nx);
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            double eta = 0.0;
            for (int ii = 0; ii < nabs; ++ii) {
                const double value = std::pow(
                    static_cast<double>(nabs - ii) / std::max(nabs, 1), 2.0) * damp_max;
                if (iz == ii || iz == nz - 1 - ii || ix == ii || ix == nx - 1 - ii) {
                    eta = std::max(eta, value);
                }
            }
            const int row = index(iz, ix, nz);
            const double v = velocity[static_cast<std::size_t>(row)];
            const double k2 = (omega / v) * (omega / v);
            t.emplace_back(row, row, Complex(k2 - 2.0 * idx2 - 2.0 * idz2, k2 * eta));
            if (iz > 0) t.emplace_back(row, index(iz - 1, ix, nz), Complex(idz2, 0.0));
            if (iz + 1 < nz) t.emplace_back(row, index(iz + 1, ix, nz), Complex(idz2, 0.0));
            if (ix > 0) t.emplace_back(row, index(iz, ix - 1, nz), Complex(idx2, 0.0));
            if (ix + 1 < nx) t.emplace_back(row, index(iz, ix + 1, nz), Complex(idx2, 0.0));
        }
    }
    Sparse a(nz * nx, nz * nx);
    a.setFromTriplets(t.begin(), t.end());
    a.makeCompressed();
    return a;
}

inline Vector equivalent_source(const Sparse& a, const Vector& u, const Vector& chi)
{
    const Vector x0 = chi.array() * u.array();
    const Vector au = a * u;
    const Vector achiu = a * x0;
    return achiu - (chi.array() * au.array()).matrix();
}

struct PreparedSystem {
    Sparse helmholtz;
    Vector mask;
};

inline PreparedSystem prepare_system(const std::vector<float>& velocity,
                                     int nz, int nx, double dz, double dx,
                                     double frequency, const Parameters& p)
{
    if (p.absorbing_rows < 0) {
        throw std::invalid_argument("absorbing_rows must be non-negative");
    }
    if (p.absorbing_rows > 0 &&
        (2 * p.absorbing_rows >= nz || 2 * p.absorbing_rows >= nx)) {
        throw std::invalid_argument(
            "external GMRES absorbing layer is too thick for the padded grid");
    }

    PreparedSystem system{
        build_helmholtz(velocity, nz, nx, dz, dx, frequency,
                        p.absorbing_rows, p.damp_max),
        Vector::Zero(nz * nx)};

    /*
     * chi=1 only on the physical block and chi=0 in the artificial
     * absorbing pad.  The caller pads the physical velocity/wavefield by
     * exactly absorbing_rows samples on all four sides before this function
     * is called.  Therefore the equivalent interface source is generated on
     * the complete four-side physical/PML interface rather than only at the
     * top and bottom boundaries.
     */
    const int collar = p.absorbing_rows;
    for (int ix = collar; ix < nx - collar; ++ix) {
        for (int iz = collar; iz < nz - collar; ++iz) {
            system.mask[index(iz, ix, nz)] = Complex(1.0, 0.0);
        }
    }
    return system;
}

inline Vector correct(const PreparedSystem& system,
                      const Vector& kirchhoff, const Parameters& p,
                      std::vector<IterationMetric>* metrics = nullptr)
{
    const Sparse& a = system.helmholtz;
    const Vector& chi = system.mask;
    Vector x = chi.array() * kirchhoff.array();
    const Vector b = equivalent_source(a, kirchhoff, chi);
    const double bnorm = std::max(b.norm(), std::numeric_limits<double>::epsilon());
    // The matrix and all GMRES controls are invariant across nonlinear outer
    // corrections.  Eigen::GMRES::compute() mainly initializes the solver,
    // but repeating it also discards/recreates internal work storage.
    Eigen::GMRES<Sparse, Eigen::IdentityPreconditioner> gmres;
    gmres.set_restart(p.gmres_restart);
    gmres.setMaxIterations(p.gmres_restart * p.gmres_cycles);
    gmres.setTolerance(p.gmres_tolerance);
    gmres.compute(a);
    for (int outer = 1; outer <= p.outer_iterations; ++outer) {
        IterationMetric m;
        m.outer = outer;
        Vector r = b - a * x;
        m.residual_before = r.norm() / bnorm;
        const Vector x_before_gmres = x;
        const double residual_before_norm = r.norm();
        const Vector candidate = gmres.solveWithGuess(b, x_before_gmres);
        const Vector gmres_r = b - a * candidate;
        m.gmres_steps = static_cast<int>(gmres.iterations());
        if (gmres_r.norm() <= residual_before_norm * (1.0 + 1.0e-12)) {
            m.gmres_correction_norm = (candidate - x_before_gmres).norm();
            x = candidate; m.gmres_accepted = true;
            m.residual_after_gmres = gmres_r.norm() / bnorm;
        } else {
            x = x_before_gmres;
            m.residual_after_gmres = m.residual_before;
        }
        if (metrics) metrics->push_back(m);
    }
    return x;
}

inline Vector correct(const std::vector<float>& velocity,
                      const Vector& kirchhoff, int nz, int nx,
                      double dz, double dx, double frequency,
                      const Parameters& p,
                      std::vector<IterationMetric>* metrics = nullptr)
{
    return correct(prepare_system(velocity, nz, nx, dz, dx, frequency, p),
                   kirchhoff, p, metrics);
}

} // namespace matlab_gmres

#endif
