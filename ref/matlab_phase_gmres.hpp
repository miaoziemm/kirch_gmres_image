#ifndef MATLAB_PHASE_GMRES_HPP
#define MATLAB_PHASE_GMRES_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/IterativeSolvers>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

namespace matlab_phase_gmres {

using Complex = std::complex<double>;
using Sparse = Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>;
using Vector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using Triplet = Eigen::Triplet<Complex, int>;

struct Parameters {
    int outer_iterations = 3;
    int coarse_outer_max = 3;
    int q_coarse = 2;
    int gmres_restart = 30;
    int gmres_cycles = 1;
    int absorbing_rows = 6;
    double gmres_tolerance = 1.0e-12;
    double alpha_cap = 0.8;
    double phase_amp_rel_floor = 1.0e-4;
    double damp_max = 2.0;
};

struct IterationMetric {
    int outer = 0;
    int gmres_steps = 0;
    double residual_before = 0.0;
    double residual_after_phase = 0.0;
    double residual_after_gmres = 0.0;
    double phase_correction_norm = 0.0;
    double gmres_correction_norm = 0.0;
    Complex alpha{0.0, 0.0};
    bool phase_accepted = false;
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

using RealRowSparse = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

inline RealRowSparse interp1_matrix(int n, int q)
{
    std::vector<int> nodes;
    for (int i = 0; i < n; i += q) nodes.push_back(i);
    if (nodes.back() != n - 1) nodes.push_back(n - 1);
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(static_cast<std::size_t>(2) * n);
    for (int i = 0; i < n; ++i) {
        const auto upper = std::lower_bound(nodes.begin(), nodes.end(), i);
        if (upper == nodes.begin()) t.emplace_back(i, 0, 1.0);
        else if (upper == nodes.end()) t.emplace_back(i, static_cast<int>(nodes.size()) - 1, 1.0);
        else if (*upper == i) t.emplace_back(i, static_cast<int>(upper - nodes.begin()), 1.0);
        else {
            const int right = static_cast<int>(upper - nodes.begin());
            const int left = right - 1;
            const double w = static_cast<double>(i - nodes[left]) /
                             static_cast<double>(nodes[right] - nodes[left]);
            t.emplace_back(i, left, 1.0 - w);
            t.emplace_back(i, right, w);
        }
    }
    RealRowSparse p(n, static_cast<int>(nodes.size()));
    p.setFromTriplets(t.begin(), t.end());
    return p;
}

inline Sparse build_phase_basis(const Vector& kirchhoff,
                                const Vector& chi,
                                int nz, int nx, int q,
                                double rel_floor)
{
    const auto pz = interp1_matrix(nz, q);
    const auto px = interp1_matrix(nx, q);
    const int ncz = pz.cols();
    const int ncx = px.cols();
    double amax = 0.0;
    for (int i = 0; i < kirchhoff.size(); ++i) amax = std::max(amax, std::abs(kirchhoff[i]));
    const double floor = std::max(rel_floor * amax, std::numeric_limits<double>::epsilon());
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(4) * nz * nx);
    std::vector<double> energy(static_cast<std::size_t>(ncz) * ncx, 0.0);
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            const int row = index(iz, ix, nz);
            Complex phase(1.0, 0.0);
            const double amp = std::abs(kirchhoff[row]);
            if (amp >= floor && std::isfinite(amp)) phase = kirchhoff[row] / amp;
            const Complex carrier = chi[row] * phase;
            for (RealRowSparse::InnerIterator xit(px, ix); xit; ++xit) {
                for (RealRowSparse::InnerIterator zit(pz, iz); zit; ++zit) {
                    const int col = zit.col() + xit.col() * ncz;
                    const Complex value = carrier * (xit.value() * zit.value());
                    if (value != Complex(0.0, 0.0)) {
                        entries.emplace_back(row, col, value);
                        energy[static_cast<std::size_t>(col)] += std::norm(value);
                    }
                }
            }
        }
    }
    const double max_energy = *std::max_element(energy.begin(), energy.end());
    std::vector<int> remap(energy.size(), -1);
    int active = 0;
    for (std::size_t i = 0; i < energy.size(); ++i)
        if (energy[i] > 1.0e-20 * max_energy) remap[i] = active++;
    std::vector<Triplet> filtered;
    filtered.reserve(entries.size());
    for (const Triplet& value : entries) {
        const int col = remap[static_cast<std::size_t>(value.col())];
        if (col >= 0) filtered.emplace_back(value.row(), col, value.value());
    }
    Sparse z(nz * nx, active);
    z.setFromTriplets(filtered.begin(), filtered.end());
    z.makeCompressed();
    return z;
}

inline Vector equivalent_source(const Sparse& a, const Vector& u, const Vector& chi)
{
    const Vector x0 = chi.array() * u.array();
    const Vector au = a * u;
    const Vector achiu = a * x0;
    return achiu - (chi.array() * au.array()).matrix();
}

inline Vector correct(const std::vector<float>& velocity,
                      const Vector& kirchhoff, int nz, int nx,
                      double dz, double dx, double frequency,
                      const Parameters& p,
                      std::vector<IterationMetric>* metrics = nullptr)
{
    const Sparse a = build_helmholtz(velocity, nz, nx, dz, dx, frequency,
                                     p.absorbing_rows, p.damp_max);
    Vector chi = Vector::Zero(nz * nx);
    const int collar = std::min(p.absorbing_rows, std::max(0, (nz - 1) / 3));
    for (int ix = 0; ix < nx; ++ix)
        for (int iz = collar; iz < nz - collar; ++iz)
            chi[index(iz, ix, nz)] = Complex(1.0, 0.0);
    Vector x = chi.array() * kirchhoff.array();
    const Vector b = equivalent_source(a, kirchhoff, chi);
    const double bnorm = std::max(b.norm(), std::numeric_limits<double>::epsilon());
    const Sparse z = build_phase_basis(kirchhoff, chi, nz, nx,
                                       p.q_coarse, p.phase_amp_rel_floor);
    const Sparse az = a * z;
    Sparse ar = z.adjoint() * az;
    ar.makeCompressed();
    Eigen::SparseLU<Sparse, Eigen::COLAMDOrdering<int>> coarse_lu;
    coarse_lu.compute(ar);
    if (coarse_lu.info() != Eigen::Success) throw std::runtime_error("phase coarse LU failed");

    for (int outer = 1; outer <= p.outer_iterations; ++outer) {
        IterationMetric m;
        m.outer = outer;
        Vector r = b - a * x;
        m.residual_before = r.norm() / bnorm;
        Vector xhalf = x;
        Vector rhalf = r;
        if (outer <= p.coarse_outer_max && z.cols() > 0) {
            const Vector ah = coarse_lu.solve(z.adjoint() * r);
            if (coarse_lu.info() != Eigen::Success) throw std::runtime_error("phase coarse solve failed");
            const Vector cr = z * ah;
            const Vector acr = az * ah;
            const Complex den = acr.dot(acr);
            if (std::abs(den) > std::numeric_limits<double>::epsilon()) {
                Complex alpha = acr.dot(r) / den;
                if (std::abs(alpha) > p.alpha_cap) alpha *= p.alpha_cap / std::abs(alpha);
                const Vector trial = x + alpha * cr;
                const Vector trial_r = b - a * trial;
                m.alpha = alpha;
                if (trial_r.norm() < r.norm()) {
                    xhalf = trial; rhalf = trial_r; m.phase_accepted = true;
                    m.phase_correction_norm = (alpha * cr).norm();
                }
            }
        }
        m.residual_after_phase = rhalf.norm() / bnorm;
        Eigen::GMRES<Sparse, Eigen::IdentityPreconditioner> gmres;
        gmres.set_restart(p.gmres_restart);
        gmres.setMaxIterations(p.gmres_restart * p.gmres_cycles);
        gmres.setTolerance(p.gmres_tolerance);
        gmres.compute(a);
        const Vector candidate = gmres.solveWithGuess(b, xhalf);
        const Vector gmres_r = b - a * candidate;
        m.gmres_steps = static_cast<int>(gmres.iterations());
        if (gmres_r.norm() <= rhalf.norm() * (1.0 + 1.0e-12)) {
            m.gmres_correction_norm = (candidate - xhalf).norm();
            x = candidate; m.gmres_accepted = true;
            m.residual_after_gmres = gmres_r.norm() / bnorm;
        } else {
            x = xhalf;
            m.residual_after_gmres = m.residual_after_phase;
        }
        if (metrics) metrics->push_back(m);
    }
    return x;
}

} // namespace matlab_phase_gmres

#endif
