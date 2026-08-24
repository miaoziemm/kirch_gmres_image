// Exact C++ translation of test_hybrid_rayphase_residual_gmres_marmousi.m.
//
// Numerical conventions intentionally retained from MATLAB:
//   * column-major/z-fast indexing: linear_index = iz + ix*nz;
//   * second-order five-point Helmholtz operator;
//   * quadratic complex sponge k^2*(1+i*eta);
//   * first-order four-direction fast sweeping;
//   * 0.25*i*H_0^(1)(omega*T) first-arrival field and global complex scale;
//   * P = kron(Px,Pz), Z = diag(exp(i*omega*T))*P, AR = Z^H*A*Z;
//   * residual-minimizing complex coarse correction with |alpha| cap;
//   * exactly three fresh, unrestarted, twice-reorthogonalized GMRES steps;
//   * one continuous zero-initial GMRES process at matched step counts;
//   * restarted zero-initial GMRES for the common residual target.
//
// Plotting is intentionally kept separate from the numerical executable.  The
// companion script plot_matlab_hybrid_rayphase.py consumes the binary snapshots
// and reproduces the MATLAB tiled animation and convergence figures.

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/KroneckerProduct>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Complex = std::complex<double>;
using Vector = Eigen::VectorXcd;
using DenseMatrix = Eigen::MatrixXcd;
using SparseMatrix = Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>;
using Triplet = Eigen::Triplet<Complex, int>;
using SparseLU = Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>>;
using Clock = std::chrono::steady_clock;

constexpr double kPi = 3.141592653589793238462643383279502884;

double elapsed_seconds(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool parse_bool(const std::string& text)
{
    const std::string value = lower_copy(text);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    throw std::invalid_argument("invalid Boolean value: " + text);
}

class Arguments {
public:
    Arguments(int argc, char** argv)
    {
        for (int i = 1; i < argc; ++i) {
            std::string token = argv[i];
            if (token == "-h" || token == "--help" || token == "help") {
                help_ = true;
                continue;
            }
            if (token.rfind("--", 0) == 0) token.erase(0, 2);
            const auto equal = token.find('=');
            if (equal != std::string::npos) {
                values_[token.substr(0, equal)] = token.substr(equal + 1);
                continue;
            }
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                if (next.rfind("--", 0) != 0 && next.find('=') == std::string::npos) {
                    values_[token] = next;
                    ++i;
                    continue;
                }
            }
            values_[token] = "1";
        }
    }

    bool help() const { return help_; }

    std::string get(const std::string& key, const std::string& fallback) const
    {
        const auto found = values_.find(key);
        return found == values_.end() ? fallback : found->second;
    }

    int get_int(const std::string& key, int fallback) const
    {
        const auto found = values_.find(key);
        return found == values_.end() ? fallback : std::stoi(found->second);
    }

    double get_double(const std::string& key, double fallback) const
    {
        const auto found = values_.find(key);
        return found == values_.end() ? fallback : std::stod(found->second);
    }

    bool get_bool(const std::string& key, bool fallback) const
    {
        const auto found = values_.find(key);
        return found == values_.end() ? fallback : parse_bool(found->second);
    }

private:
    std::map<std::string, std::string> values_;
    bool help_ = false;
};

void print_help()
{
    std::cout
        << "MATLAB-equivalent ray-phase coarse correction + GMRES\n\n"
        << "Required:\n"
        << "  model=FILE                   raw float32 velocity, MATLAB [nz,nx] order\n\n"
        << "Exact MATLAB defaults:\n"
        << "  nx=1024 nz=298 dx=10.5 dz=10.5 frequency=15\n"
        << "  sx=513 sz=31 nabs=25 damp_max=2\n"
        << "  q_coarse=2 alpha_cap=0.8 gmres_steps_per_outer=3\n"
        << "  outer_max=25 target_relres=0.05\n"
        << "  zero_restart=30 zero_max_cycles=300\n"
        << "  ray_source_regularization=0.10 use_pml_taper=1\n\n"
        << "Reference and output:\n"
        << "  reference_mode=direct|read|none   default direct\n"
        << "  reference_file=FILE               .npy c8/c16 or raw c8/c16\n"
        << "  output_dir=matlab_hybrid_cpp\n"
        << "  save_frame_fields=1               fields needed by plot script\n"
        << "  run_ray_start_gmres=1             additional fair initial-guess control\n\n"
        << "Optional smoke-test crop (not part of MATLAB default):\n"
        << "  crop_nx=NX crop_nz=NZ\n";
}

struct Parameters {
    std::string model_file;
    std::string output_dir = "matlab_hybrid_cpp";
    std::string reference_mode = "direct";
    std::string reference_file;
    int nx = 1024;
    int nz = 298;
    double dx = 10.5;
    double dz = 10.5;
    double frequency = 15.0;
    int sx_matlab = 513;
    int sz_matlab = 31;
    int nabs = 25;
    double damp_max = 2.0;
    int q_coarse = 2;
    double alpha_cap = 0.8;
    int gmres_steps_per_outer = 3;
    int outer_max = 25;
    double target_relres = 5.0e-2;
    int zero_restart = 30;
    int zero_max_cycles = 300;
    double ray_source_regularization = 0.10;
    bool use_pml_taper = true;
    int eikonal_sweeps = 12;
    double velocity_scale = 1000.0;
    bool save_frame_fields = true;
    bool run_ray_start_gmres = true;
    int crop_nx = 0;
    int crop_nz = 0;
};

Parameters parse_parameters(const Arguments& arguments)
{
    Parameters p;
    p.model_file = arguments.get("model", arguments.get("velocity", ""));
    p.output_dir = arguments.get("output_dir", p.output_dir);
    p.reference_mode = lower_copy(arguments.get("reference_mode", p.reference_mode));
    p.reference_file = arguments.get("reference_file", "");
    p.nx = arguments.get_int("nx", p.nx);
    p.nz = arguments.get_int("nz", p.nz);
    p.dx = arguments.get_double("dx", p.dx);
    p.dz = arguments.get_double("dz", p.dz);
    p.frequency = arguments.get_double("frequency", arguments.get_double("freq", p.frequency));
    p.sx_matlab = arguments.get_int("sx", p.sx_matlab);
    p.sz_matlab = arguments.get_int("sz", p.sz_matlab);
    p.nabs = arguments.get_int("nabs", p.nabs);
    p.damp_max = arguments.get_double("damp_max", p.damp_max);
    p.q_coarse = arguments.get_int("q_coarse", p.q_coarse);
    p.alpha_cap = arguments.get_double("alpha_cap", p.alpha_cap);
    p.gmres_steps_per_outer = arguments.get_int(
        "gmres_steps_per_outer", p.gmres_steps_per_outer);
    p.outer_max = arguments.get_int("outer_max", p.outer_max);
    p.target_relres = arguments.get_double("target_relres", p.target_relres);
    p.zero_restart = arguments.get_int("zero_restart", p.zero_restart);
    p.zero_max_cycles = arguments.get_int("zero_max_cycles", p.zero_max_cycles);
    p.ray_source_regularization = arguments.get_double(
        "ray_source_regularization", p.ray_source_regularization);
    p.use_pml_taper = arguments.get_bool("use_pml_taper", p.use_pml_taper);
    p.eikonal_sweeps = arguments.get_int("eikonal_sweeps", p.eikonal_sweeps);
    p.velocity_scale = arguments.get_double("velocity_scale", p.velocity_scale);
    p.save_frame_fields = arguments.get_bool("save_frame_fields", p.save_frame_fields);
    p.run_ray_start_gmres = arguments.get_bool(
        "run_ray_start_gmres", p.run_ray_start_gmres);
    p.crop_nx = arguments.get_int("crop_nx", 0);
    p.crop_nz = arguments.get_int("crop_nz", 0);

    if (p.model_file.empty()) throw std::invalid_argument("model=FILE is required");
    if (p.nx < 3 || p.nz < 3 || p.dx <= 0.0 || p.dz <= 0.0 || p.frequency <= 0.0) {
        throw std::invalid_argument("invalid model geometry or frequency");
    }
    if (p.sx_matlab < 1 || p.sx_matlab > p.nx ||
        p.sz_matlab < 1 || p.sz_matlab > p.nz) {
        throw std::invalid_argument("MATLAB source index is outside the model");
    }
    if (p.nabs < 0 || p.q_coarse < 1 || p.gmres_steps_per_outer < 0 ||
        p.outer_max < 0 || p.zero_restart < 1 || p.zero_max_cycles < 1 ||
        p.target_relres <= 0.0 || p.alpha_cap < 0.0) {
        throw std::invalid_argument("invalid iteration or coarse-space parameter");
    }
    if (p.reference_mode != "direct" && p.reference_mode != "read" &&
        p.reference_mode != "none") {
        throw std::invalid_argument("reference_mode must be direct, read, or none");
    }
    if (!p.reference_file.empty()) p.reference_mode = "read";
    if (p.reference_mode == "read" && p.reference_file.empty()) {
        throw std::invalid_argument("reference_mode=read requires reference_file");
    }
    return p;
}

struct Model {
    int nx = 0;
    int nz = 0;
    int full_nx = 0;
    int full_nz = 0;
    int crop_x0 = 0;
    int crop_z0 = 0;
    int source_x = 0;
    int source_z = 0;
    std::vector<double> velocity;

    int index(int iz, int ix) const { return iz + ix * nz; }
};

Model read_model(const Parameters& p)
{
    std::ifstream stream(p.model_file, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open model: " + p.model_file);
    const std::size_t count = static_cast<std::size_t>(p.nx) * p.nz;
    std::vector<float> raw(count);
    stream.read(reinterpret_cast<char*>(raw.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
    if (stream.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
        throw std::runtime_error("model has fewer samples than nx*nz");
    }
    char extra = 0;
    if (stream.read(&extra, 1)) {
        throw std::runtime_error("model has more samples than nx*nz");
    }

    Model model;
    model.full_nx = p.nx;
    model.full_nz = p.nz;
    model.nx = p.crop_nx > 0 ? p.crop_nx : p.nx;
    model.nz = p.crop_nz > 0 ? p.crop_nz : p.nz;
    if (model.nx < 3 || model.nx > p.nx || model.nz < 3 || model.nz > p.nz) {
        throw std::invalid_argument("crop dimensions are outside the full model");
    }
    const int full_sx = p.sx_matlab - 1;
    const int full_sz = p.sz_matlab - 1;
    model.crop_x0 = std::max(0, std::min(p.nx - model.nx, full_sx - model.nx / 2));
    model.crop_z0 = 0;
    model.source_x = full_sx - model.crop_x0;
    model.source_z = full_sz - model.crop_z0;
    if (model.source_x < 0 || model.source_x >= model.nx ||
        model.source_z < 0 || model.source_z >= model.nz) {
        throw std::invalid_argument("crop excludes the source");
    }
    model.velocity.resize(static_cast<std::size_t>(model.nx) * model.nz);
    for (int ix = 0; ix < model.nx; ++ix) {
        for (int iz = 0; iz < model.nz; ++iz) {
            const int full_index = (iz + model.crop_z0) +
                (ix + model.crop_x0) * p.nz;
            model.velocity[model.index(iz, ix)] =
                static_cast<double>(raw[full_index]) * p.velocity_scale;
        }
    }
    return model;
}

struct HelmholtzSystem {
    SparseMatrix matrix;
    std::vector<double> eta;
    Vector rhs;
};

HelmholtzSystem build_helmholtz(const Model& model, const Parameters& p)
{
    const int n = model.nx * model.nz;
    const double idx2 = 1.0 / (p.dx * p.dx);
    const double idz2 = 1.0 / (p.dz * p.dz);
    const double omega = 2.0 * kPi * p.frequency;
    std::vector<double> eta(static_cast<std::size_t>(n), 0.0);
    const int effective = std::min({p.nabs, model.nx, model.nz});
    for (int ii = 0; ii < effective; ++ii) {
        const double ratio = static_cast<double>(p.nabs - ii) / p.nabs;
        const double value = ratio * ratio * p.damp_max;
        for (int ix = 0; ix < model.nx; ++ix) {
            eta[model.index(ii, ix)] = std::max(eta[model.index(ii, ix)], value);
            eta[model.index(model.nz - ii - 1, ix)] =
                std::max(eta[model.index(model.nz - ii - 1, ix)], value);
        }
        for (int iz = 0; iz < model.nz; ++iz) {
            eta[model.index(iz, ii)] = std::max(eta[model.index(iz, ii)], value);
            eta[model.index(iz, model.nx - ii - 1)] =
                std::max(eta[model.index(iz, model.nx - ii - 1)], value);
        }
    }

    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(5) * n);
    for (int ix = 0; ix < model.nx; ++ix) {
        for (int iz = 0; iz < model.nz; ++iz) {
            const int row = model.index(iz, ix);
            const double k2 = std::pow(omega / model.velocity[row], 2.0);
            entries.emplace_back(
                row, row,
                Complex(-2.0 * idx2 - 2.0 * idz2 + k2, k2 * eta[row]));
            if (iz > 0) entries.emplace_back(row, model.index(iz - 1, ix), Complex(idz2, 0.0));
            if (iz + 1 < model.nz) entries.emplace_back(row, model.index(iz + 1, ix), Complex(idz2, 0.0));
            if (ix > 0) entries.emplace_back(row, model.index(iz, ix - 1), Complex(idx2, 0.0));
            if (ix + 1 < model.nx) entries.emplace_back(row, model.index(iz, ix + 1), Complex(idx2, 0.0));
        }
    }
    SparseMatrix matrix(n, n);
    matrix.setFromTriplets(entries.begin(), entries.end());
    matrix.makeCompressed();

    Vector rhs = Vector::Zero(n);
    rhs[model.index(model.source_z, model.source_x)] = 1.0 / (p.dx * p.dz);
    return {std::move(matrix), std::move(eta), std::move(rhs)};
}

std::vector<double> fast_sweep_eikonal(const Model& model, const Parameters& p)
{
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> traveltime(
        static_cast<std::size_t>(model.nx) * model.nz, infinity);
    traveltime[model.index(model.source_z, model.source_x)] = 0.0;

    const auto sweep = [&](int z0, int z1, int zs, int x0, int x1, int xs,
                           double& maximum_change) {
        for (int iz = z0; iz != z1; iz += zs) {
            for (int ix = x0; ix != x1; ix += xs) {
                if (iz == model.source_z && ix == model.source_x) continue;
                double ax = infinity;
                double bz = infinity;
                if (ix > 0) ax = std::min(ax, traveltime[model.index(iz, ix - 1)]);
                if (ix + 1 < model.nx) ax = std::min(ax, traveltime[model.index(iz, ix + 1)]);
                if (iz > 0) bz = std::min(bz, traveltime[model.index(iz - 1, ix)]);
                if (iz + 1 < model.nz) bz = std::min(bz, traveltime[model.index(iz + 1, ix)]);
                if (!std::isfinite(ax) && !std::isfinite(bz)) continue;

                const double sh = p.dx / model.velocity[model.index(iz, ix)];
                double candidate = infinity;
                if (std::abs(ax - bz) >= sh) {
                    candidate = std::min(ax, bz) + sh;
                } else {
                    const double discriminant = std::max(
                        0.0, 2.0 * sh * sh - (ax - bz) * (ax - bz));
                    candidate = 0.5 * (ax + bz + std::sqrt(discriminant));
                }
                const int index = model.index(iz, ix);
                const double old = traveltime[index];
                if (candidate < old) {
                    if (std::isfinite(old)) {
                        maximum_change = std::max(maximum_change, old - candidate);
                    }
                    traveltime[index] = candidate;
                }
            }
        }
    };

    for (int iteration = 0; iteration < p.eikonal_sweeps; ++iteration) {
        double maximum_change = 0.0;
        sweep(0, model.nz, 1, 0, model.nx, 1, maximum_change);
        sweep(0, model.nz, 1, model.nx - 1, -1, -1, maximum_change);
        sweep(model.nz - 1, -1, -1, 0, model.nx, 1, maximum_change);
        sweep(model.nz - 1, -1, -1, model.nx - 1, -1, -1, maximum_change);
        std::cout << "  eikonal sweep " << (iteration + 1)
                  << ": max finite change=" << std::scientific
                  << maximum_change << " s\n";
        if (maximum_change < 1.0e-11) break;
    }
    return traveltime;
}

struct RayField {
    Vector field;
    Complex scale = 1.0;
};

RayField build_first_arrival_ray(const SparseMatrix& matrix,
                                 const Vector& rhs,
                                 const Model& model,
                                 const Parameters& p,
                                 const std::vector<double>& traveltime)
{
    const int n = model.nx * model.nz;
    const double omega = 2.0 * kPi * p.frequency;
    const double minimum_time = p.ray_source_regularization *
        std::min(p.dx, p.dz) /
        model.velocity[model.index(model.source_z, model.source_x)];
    std::vector<double> taper(static_cast<std::size_t>(n), 1.0);
    if (p.use_pml_taper) {
        const int effective = std::min({p.nabs, model.nx, model.nz});
        for (int ii = 0; ii < effective; ++ii) {
            const double sine = std::sin(
                0.5 * kPi * static_cast<double>(ii + 1) /
                static_cast<double>(p.nabs + 1));
            const double factor = sine * sine;
            for (int ix = 0; ix < model.nx; ++ix) {
                taper[model.index(ii, ix)] *= factor;
                taper[model.index(model.nz - ii - 1, ix)] *= factor;
            }
            for (int iz = 0; iz < model.nz; ++iz) {
                taper[model.index(iz, ii)] *= factor;
                taper[model.index(iz, model.nx - ii - 1)] *= factor;
            }
        }
    }

    Vector candidate(n);
    for (int index = 0; index < n; ++index) {
        const double argument = omega * std::max(traveltime[index], minimum_time);
        const double j0 = std::cyl_bessel_j(0.0, argument);
        const double y0 = std::cyl_neumann(0.0, argument);
        // 0.25*i*(J0+i*Y0) = -0.25*Y0 + 0.25*i*J0.
        candidate[index] = Complex(-0.25 * y0, 0.25 * j0) * taper[index];
    }
    const Vector action = matrix * candidate;
    const Complex denominator = action.dot(action);
    Complex scale = 1.0;
    if (std::abs(denominator) > std::numeric_limits<double>::epsilon()) {
        scale = action.dot(rhs) / denominator;
    }
    return {scale * candidate, scale};
}

struct Interpolation1D {
    SparseMatrix matrix;
    std::vector<int> nodes;
};

Interpolation1D interp1_matrix(int n, int q)
{
    Interpolation1D result;
    for (int index = 0; index < n; index += q) result.nodes.push_back(index);
    if (result.nodes.back() != n - 1) result.nodes.push_back(n - 1);
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(2) * n);
    int left = 0;
    for (int fine = 0; fine < n; ++fine) {
        while (left + 1 < static_cast<int>(result.nodes.size()) &&
               result.nodes[left + 1] < fine) {
            ++left;
        }
        if (fine <= result.nodes.front()) {
            entries.emplace_back(fine, 0, Complex(1.0, 0.0));
        } else if (fine >= result.nodes.back()) {
            entries.emplace_back(
                fine, static_cast<int>(result.nodes.size()) - 1,
                Complex(1.0, 0.0));
        } else if (result.nodes[left] == fine) {
            entries.emplace_back(fine, left, Complex(1.0, 0.0));
        } else {
            const double weight = static_cast<double>(fine - result.nodes[left]) /
                static_cast<double>(result.nodes[left + 1] - result.nodes[left]);
            entries.emplace_back(fine, left, Complex(1.0 - weight, 0.0));
            entries.emplace_back(fine, left + 1, Complex(weight, 0.0));
        }
    }
    result.matrix.resize(n, static_cast<int>(result.nodes.size()));
    result.matrix.setFromTriplets(entries.begin(), entries.end());
    result.matrix.makeCompressed();
    return result;
}

struct CoarseSpace {
    SparseMatrix interpolation;
    SparseMatrix basis;
    SparseMatrix action_basis;
    SparseMatrix coarse_operator;
    std::unique_ptr<SparseLU> solver;
    int nx_coarse = 0;
    int nz_coarse = 0;
    double setup_seconds = 0.0;
};

CoarseSpace build_phase_coarse_space(const SparseMatrix& matrix,
                                     const Model& model,
                                     const Parameters& p,
                                     const std::vector<double>& traveltime)
{
    const auto started = Clock::now();
    Interpolation1D px = interp1_matrix(model.nx, p.q_coarse);
    Interpolation1D pz = interp1_matrix(model.nz, p.q_coarse);

    CoarseSpace coarse;
    coarse.nx_coarse = static_cast<int>(px.nodes.size());
    coarse.nz_coarse = static_cast<int>(pz.nodes.size());
    // This is the literal MATLAB P = kron(Px,Pz), retaining z-fast order.
    coarse.interpolation = Eigen::kroneckerProduct(px.matrix, pz.matrix).eval();
    coarse.interpolation.makeCompressed();

    const double omega = 2.0 * kPi * p.frequency;
    std::vector<Triplet> entries;
    entries.reserve(static_cast<std::size_t>(coarse.interpolation.nonZeros()));
    for (int column = 0; column < coarse.interpolation.outerSize(); ++column) {
        for (SparseMatrix::InnerIterator value(coarse.interpolation, column);
             value; ++value) {
            const Complex phase = std::exp(Complex(0.0, omega * traveltime[value.row()]));
            entries.emplace_back(
                value.row(), value.col(), phase * value.value());
        }
    }
    coarse.basis.resize(coarse.interpolation.rows(), coarse.interpolation.cols());
    coarse.basis.setFromTriplets(entries.begin(), entries.end());
    coarse.basis.makeCompressed();
    std::cout << "  P/Z: " << coarse.basis.rows() << " x " << coarse.basis.cols()
              << ", nnz=" << coarse.basis.nonZeros() << "\n";

    // Literal MATLAB AZ=A*Z and AR=Z'*AZ; both are retained and reused.
    coarse.action_basis = matrix * coarse.basis;
    coarse.action_basis.prune(
        [](int, int, const Complex& value) { return value != Complex(0.0, 0.0); });
    coarse.action_basis.makeCompressed();
    std::cout << "  A*Z: nnz=" << coarse.action_basis.nonZeros() << "\n";
    coarse.coarse_operator = coarse.basis.adjoint() * coarse.action_basis;
    coarse.coarse_operator.prune(
        [](int, int, const Complex& value) { return value != Complex(0.0, 0.0); });
    coarse.coarse_operator.makeCompressed();
    std::cout << "  Z^H*A*Z: " << coarse.coarse_operator.rows() << " x "
              << coarse.coarse_operator.cols() << ", nnz="
              << coarse.coarse_operator.nonZeros() << "; factorizing\n";
    coarse.solver = std::make_unique<SparseLU>();
    coarse.solver->analyzePattern(coarse.coarse_operator);
    coarse.solver->factorize(coarse.coarse_operator);
    if (coarse.solver->info() != Eigen::Success) {
        throw std::runtime_error("ray-phase coarse LU factorization failed");
    }
    coarse.setup_seconds = elapsed_seconds(started);
    std::cout << "  coarse LU complete in " << std::fixed
              << coarse.setup_seconds << " s\n";
    return coarse;
}

Vector load_npy_complex(const std::string& filename, int expected_count)
{
    std::ifstream stream(filename, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open reference: " + filename);
    char magic[6] = {};
    stream.read(magic, 6);
    if (stream.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("not an NPY file: " + filename);
    }
    std::uint8_t major = 0;
    std::uint8_t minor = 0;
    stream.read(reinterpret_cast<char*>(&major), 1);
    stream.read(reinterpret_cast<char*>(&minor), 1);
    std::uint32_t header_length = 0;
    if (major == 1) {
        std::uint16_t length16 = 0;
        stream.read(reinterpret_cast<char*>(&length16), 2);
        header_length = length16;
    } else {
        stream.read(reinterpret_cast<char*>(&header_length), 4);
    }
    std::string header(header_length, '\0');
    stream.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!stream) throw std::runtime_error("truncated NPY header: " + filename);
    const bool c8 = header.find("<c8") != std::string::npos ||
                    header.find("|c8") != std::string::npos;
    const bool c16 = header.find("<c16") != std::string::npos ||
                     header.find("|c16") != std::string::npos;
    if (!c8 && !c16) {
        throw std::runtime_error("NPY reference must contain complex64 or complex128");
    }
    const std::regex number_expression("([0-9]+)");
    const auto shape_position = header.find("shape");
    if (shape_position == std::string::npos) {
        throw std::runtime_error("NPY shape is missing");
    }
    std::smatch match;
    const std::string shape_text = header.substr(shape_position);
    if (!std::regex_search(shape_text, match, number_expression)) {
        throw std::runtime_error("cannot parse NPY shape");
    }
    const int count = std::stoi(match[1].str());
    if (count != expected_count) {
        throw std::runtime_error("reference length does not match the active model");
    }
    Vector result(count);
    if (c8) {
        std::vector<float> data(static_cast<std::size_t>(2) * count);
        stream.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(data.size() * sizeof(float)));
        if (!stream) throw std::runtime_error("truncated complex64 NPY reference");
        for (int i = 0; i < count; ++i) result[i] = Complex(data[2 * i], data[2 * i + 1]);
    } else {
        std::vector<double> data(static_cast<std::size_t>(2) * count);
        stream.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(data.size() * sizeof(double)));
        if (!stream) throw std::runtime_error("truncated complex128 NPY reference");
        for (int i = 0; i < count; ++i) result[i] = Complex(data[2 * i], data[2 * i + 1]);
    }
    return result;
}

Vector load_raw_complex(const std::string& filename, int expected_count)
{
    const auto size = fs::file_size(filename);
    std::ifstream stream(filename, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open reference: " + filename);
    Vector result(expected_count);
    if (size == static_cast<std::uintmax_t>(expected_count) * 2 * sizeof(float)) {
        std::vector<float> data(static_cast<std::size_t>(2) * expected_count);
        stream.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(data.size() * sizeof(float)));
        for (int i = 0; i < expected_count; ++i) {
            result[i] = Complex(data[2 * i], data[2 * i + 1]);
        }
    } else if (size == static_cast<std::uintmax_t>(expected_count) * 2 * sizeof(double)) {
        std::vector<double> data(static_cast<std::size_t>(2) * expected_count);
        stream.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(data.size() * sizeof(double)));
        for (int i = 0; i < expected_count; ++i) {
            result[i] = Complex(data[2 * i], data[2 * i + 1]);
        }
    } else {
        throw std::runtime_error("raw reference size is neither complex64 nor complex128");
    }
    return result;
}

Vector load_reference(const std::string& filename, int expected_count)
{
    return fs::path(filename).extension() == ".npy"
        ? load_npy_complex(filename, expected_count)
        : load_raw_complex(filename, expected_count);
}

Vector direct_reference(const SparseMatrix& matrix, const Vector& rhs)
{
    SparseLU solver;
    solver.analyzePattern(matrix);
    solver.factorize(matrix);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("direct Helmholtz LU factorization failed");
    }
    Vector reference = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("direct Helmholtz LU solve failed");
    }
    return reference;
}

double relative_error(const Vector& field, const Vector* reference)
{
    if (reference == nullptr || reference->norm() == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (field - *reference).norm() / reference->norm();
}

struct FixedGmresResult {
    Vector correction;
    Vector residual_after;
    int steps = 0;
};

Vector least_squares(const DenseMatrix& matrix, const Vector& rhs)
{
    // MATLAB's tall complex backslash uses a pivoted QR route for this small
    // Hessenberg least-squares problem.
    return matrix.colPivHouseholderQr().solve(rhs);
}

FixedGmresResult gmres_correction_fixed(const SparseMatrix& matrix,
                                        const Vector& residual,
                                        int maximum_steps)
{
    const int n = static_cast<int>(residual.size());
    const double beta = residual.norm();
    FixedGmresResult result{Vector::Zero(n), residual, 0};
    if (beta == 0.0 || maximum_steps <= 0) return result;

    DenseMatrix basis = DenseMatrix::Zero(n, maximum_steps + 1);
    DenseMatrix hessenberg = DenseMatrix::Zero(maximum_steps + 1, maximum_steps);
    basis.col(0) = residual / beta;
    for (int j = 0; j < maximum_steps; ++j) {
        Vector vector = matrix * basis.col(j);
        // Literal MATLAB modified Gram-Schmidt plus one reorthogonalization.
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
        Vector small_rhs = Vector::Zero(j + 2);
        small_rhs[0] = beta;
        const DenseMatrix small = hessenberg.block(0, 0, j + 2, j + 1);
        const Vector coefficients = least_squares(small, small_rhs);
        result.correction = basis.leftCols(j + 1) * coefficients;
        const Vector small_residual = small_rhs - small * coefficients;
        result.residual_after = basis.leftCols(j + 2) * small_residual;
        result.steps = j + 1;
        if (std::abs(hessenberg(j + 1, j)) == 0.0) break;
    }
    return result;
}

void write_real_imag_field(const fs::path& prefix, const Vector& field)
{
    fs::create_directories(prefix.parent_path());
    std::ofstream real_stream(prefix.string() + "_real.f32", std::ios::binary);
    std::ofstream imag_stream(prefix.string() + "_imag.f32", std::ios::binary);
    if (!real_stream || !imag_stream) {
        throw std::runtime_error("cannot create field output: " + prefix.string());
    }
    std::vector<float> real(static_cast<std::size_t>(field.size()));
    std::vector<float> imag(static_cast<std::size_t>(field.size()));
    for (Eigen::Index i = 0; i < field.size(); ++i) {
        real[static_cast<std::size_t>(i)] = static_cast<float>(field[i].real());
        imag[static_cast<std::size_t>(i)] = static_cast<float>(field[i].imag());
    }
    real_stream.write(reinterpret_cast<const char*>(real.data()),
                      static_cast<std::streamsize>(real.size() * sizeof(float)));
    imag_stream.write(reinterpret_cast<const char*>(imag.data()),
                      static_cast<std::streamsize>(imag.size() * sizeof(float)));
}

void write_float_field(const fs::path& filename, const std::vector<double>& values)
{
    fs::create_directories(filename.parent_path());
    std::vector<float> output(values.size());
    std::transform(values.begin(), values.end(), output.begin(),
                   [](double value) { return static_cast<float>(value); });
    std::ofstream stream(filename, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot create: " + filename.string());
    stream.write(reinterpret_cast<const char*>(output.data()),
                 static_cast<std::streamsize>(output.size() * sizeof(float)));
}

std::string three_digits(int value)
{
    std::ostringstream stream;
    stream << std::setw(3) << std::setfill('0') << value;
    return stream.str();
}

struct StateMetric {
    int outer = 0;
    int fine_steps = 0;
    double relative_residual = 0.0;
    double true_relative_residual = 0.0;
    double relative_error = std::numeric_limits<double>::quiet_NaN();
    Complex alpha = 0.0;
    double coarse_correction_norm_over_reference = 0.0;
    double gmres_correction_norm_over_reference = 0.0;
    double elapsed = 0.0;
};

struct HybridResult {
    Vector final;
    Vector residual;
    std::vector<StateMetric> metrics;
};

HybridResult run_hybrid(const SparseMatrix& matrix,
                        const Vector& rhs,
                        const Vector& initial,
                        CoarseSpace& coarse,
                        const Vector* reference,
                        const Parameters& p,
                        const fs::path& field_directory)
{
    const double rhs_norm = rhs.norm();
    const double reference_norm = reference ? reference->norm() :
        std::numeric_limits<double>::quiet_NaN();
    Vector current = initial;
    Vector residual = rhs - matrix * current;
    HybridResult result;
    StateMetric initial_metric;
    initial_metric.relative_residual = residual.norm() / rhs_norm;
    initial_metric.true_relative_residual = initial_metric.relative_residual;
    initial_metric.relative_error = relative_error(current, reference);
    result.metrics.push_back(initial_metric);
    if (p.save_frame_fields) {
        write_real_imag_field(field_directory / "hybrid_outer_000", current);
        write_real_imag_field(field_directory / "raycorr_outer_000", Vector::Zero(current.size()));
        write_real_imag_field(field_directory / "gmrescorr_outer_000", Vector::Zero(current.size()));
    }

    std::cout << "\nHYBRID: initial residual=" << std::scientific
              << initial_metric.relative_residual << ", error="
              << initial_metric.relative_error << "\n";
    const auto started = Clock::now();
    int fine_steps = 0;
    for (int outer = 1; outer <= p.outer_max; ++outer) {
        const Vector coarse_rhs = coarse.basis.adjoint() * residual;
        const Vector envelope = coarse.solver->solve(coarse_rhs);
        if (coarse.solver->info() != Eigen::Success) {
            throw std::runtime_error("ray-phase coarse triangular solve failed");
        }
        const Vector coarse_correction = coarse.basis * envelope;
        const Vector coarse_action = coarse.action_basis * envelope;
        const Complex denominator = coarse_action.dot(coarse_action);
        Complex alpha = 0.0;
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

        const FixedGmresResult gmres = gmres_correction_fixed(
            matrix, residual_half, p.gmres_steps_per_outer);
        current = half + gmres.correction;
        residual = gmres.residual_after; // exactly as carried by MATLAB
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
            write_real_imag_field(field_directory / ("hybrid_outer_" + id), current);
            write_real_imag_field(
                field_directory / ("raycorr_outer_" + id), alpha * coarse_correction);
            write_real_imag_field(
                field_directory / ("gmrescorr_outer_" + id), gmres.correction);
        }
        std::cout << "  outer=" << std::setw(2) << outer
                  << ", fine=" << std::setw(3) << fine_steps
                  << ", residual=" << std::scientific << metric.relative_residual
                  << ", true=" << metric.true_relative_residual
                  << ", error=" << metric.relative_error
                  << ", |alpha|=" << std::abs(alpha) << "\n";
        if (metric.relative_residual <= p.target_relres) break;
    }
    result.final = std::move(current);
    result.residual = std::move(residual);
    return result;
}

struct SnapshotMetric {
    int step = 0;
    double relative_residual = 0.0;
    double true_relative_residual = 0.0;
    double relative_error = std::numeric_limits<double>::quiet_NaN();
};

struct SnapshotResult {
    std::vector<SnapshotMetric> metrics;
    Vector final;
};

SnapshotResult continuous_gmres_selected(const SparseMatrix& matrix,
                                         const Vector& rhs,
                                         const Vector& initial,
                                         const std::vector<int>& requested_steps,
                                         const Vector* reference,
                                         const fs::path& field_directory,
                                         const std::string& prefix,
                                         bool save_fields)
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
    if (save_fields) write_real_imag_field(field_directory / (prefix + "_step_000"), initial);
    result.final = initial;
    if (maximum_steps == 0 || beta == 0.0) return result;

    DenseMatrix basis = DenseMatrix::Zero(n, maximum_steps + 1);
    DenseMatrix hessenberg = DenseMatrix::Zero(maximum_steps + 1, maximum_steps);
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
            const DenseMatrix small = hessenberg.block(0, 0, step + 1, step);
            const Vector coefficients = least_squares(small, small_rhs);
            const Vector current = initial + basis.leftCols(step) * coefficients;
            const Vector small_residual = small_rhs - small * coefficients;
            const double arnoldi_relative = small_residual.norm() / rhs_norm;
            const double true_relative = (rhs - matrix * current).norm() / rhs_norm;
            result.metrics.push_back({step, arnoldi_relative, true_relative,
                                      relative_error(current, reference)});
            result.final = current;
            if (save_fields) {
                write_real_imag_field(
                    field_directory / (prefix + "_step_" + three_digits(step)), current);
            }
            ++next;
        }
        if (std::abs(hessenberg(j + 1, j)) == 0.0) break;
    }
    return result;
}

struct RestartedResult {
    Vector final;
    int inner_iterations = 0;
    int cycles = 0;
    double relative_residual = 0.0;
    double relative_error = std::numeric_limits<double>::quiet_NaN();
    double elapsed = 0.0;
    std::vector<int> history_steps;
    std::vector<double> history_residuals;
};

RestartedResult restarted_gmres_to_target(const SparseMatrix& matrix,
                                          const Vector& rhs,
                                          const Vector& initial,
                                          int restart,
                                          double target,
                                          int maximum_cycles,
                                          const Vector* reference,
                                          const std::string& label)
{
    Vector current = initial;
    const double rhs_norm = rhs.norm();
    RestartedResult result;
    Vector residual = rhs - matrix * current;
    result.history_steps.push_back(0);
    result.history_residuals.push_back(residual.norm() / rhs_norm);
    const auto started = Clock::now();
    int total_steps = 0;
    for (int cycle = 1; cycle <= maximum_cycles; ++cycle) {
        residual = rhs - matrix * current;
        const double beta = residual.norm();
        if (beta / rhs_norm <= target) break;
        const int n = static_cast<int>(rhs.size());
        DenseMatrix basis = DenseMatrix::Zero(n, restart + 1);
        DenseMatrix hessenberg = DenseMatrix::Zero(restart + 1, restart);
        basis.col(0) = residual / beta;
        Vector candidate = current;
        for (int j = 0; j < restart; ++j) {
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
            Vector small_rhs = Vector::Zero(j + 2);
            small_rhs[0] = beta;
            const DenseMatrix small = hessenberg.block(0, 0, j + 2, j + 1);
            const Vector coefficients = least_squares(small, small_rhs);
            candidate = current + basis.leftCols(j + 1) * coefficients;
            ++total_steps;
            const double true_relative = (rhs - matrix * candidate).norm() / rhs_norm;
            result.history_steps.push_back(total_steps);
            result.history_residuals.push_back(true_relative);
            if (true_relative <= target || std::abs(hessenberg(j + 1, j)) == 0.0) {
                current = candidate;
                result.inner_iterations = total_steps;
                result.cycles = cycle;
                result.relative_residual = true_relative;
                result.relative_error = relative_error(current, reference);
                result.elapsed = elapsed_seconds(started);
                result.final = current;
                std::cout << label << ": target at step=" << total_steps
                          << ", cycle=" << cycle << ", residual="
                          << std::scientific << true_relative << "\n";
                return result;
            }
        }
        current = candidate;
        result.cycles = cycle;
        if (cycle == 1 || cycle % 10 == 0) {
            std::cout << label << ": cycle=" << cycle << ", step="
                      << total_steps << ", residual=" << std::scientific
                      << result.history_residuals.back() << "\n";
        }
    }
    const Vector final_residual = rhs - matrix * current;
    result.final = current;
    result.inner_iterations = total_steps;
    result.relative_residual = final_residual.norm() / rhs_norm;
    result.relative_error = relative_error(current, reference);
    result.elapsed = elapsed_seconds(started);
    return result;
}

void write_hybrid_csv(const fs::path& filename,
                      const std::vector<StateMetric>& metrics)
{
    std::ofstream stream(filename);
    stream << std::setprecision(17);
    stream << "outer,fine_gmres_steps,hybrid_residual,hybrid_true_residual,"
              "hybrid_error,alpha_real,alpha_imag,abs_alpha,"
              "raycorr_norm_ref,gmrescorr_norm_ref,hybrid_cumulative_time\n";
    for (const StateMetric& metric : metrics) {
        stream << metric.outer << ',' << metric.fine_steps << ','
               << metric.relative_residual << ',' << metric.true_relative_residual << ','
               << metric.relative_error << ',' << metric.alpha.real() << ','
               << metric.alpha.imag() << ',' << std::abs(metric.alpha) << ','
               << metric.coarse_correction_norm_over_reference << ','
               << metric.gmres_correction_norm_over_reference << ','
               << metric.elapsed << '\n';
    }
}

void write_matched_csv(const fs::path& filename,
                       const std::vector<StateMetric>& hybrid,
                       const SnapshotResult& zero)
{
    if (hybrid.size() != zero.metrics.size()) {
        throw std::runtime_error("hybrid and matched zero histories differ in length");
    }
    std::ofstream stream(filename);
    stream << std::setprecision(17);
    stream << "outer,fine_gmres_steps,hybrid_error,hybrid_residual,"
              "hybrid_true_residual,zero_error,zero_residual,zero_true_residual\n";
    for (std::size_t i = 0; i < hybrid.size(); ++i) {
        stream << hybrid[i].outer << ',' << hybrid[i].fine_steps << ','
               << hybrid[i].relative_error << ',' << hybrid[i].relative_residual << ','
               << hybrid[i].true_relative_residual << ','
               << zero.metrics[i].relative_error << ','
               << zero.metrics[i].relative_residual << ','
               << zero.metrics[i].true_relative_residual << '\n';
    }
}

void write_restarted_history(const fs::path& filename,
                             const RestartedResult& zero,
                             const RestartedResult* ray)
{
    std::ofstream stream(filename);
    stream << std::setprecision(17);
    stream << "method,inner_iteration,relative_residual\n";
    for (std::size_t i = 0; i < zero.history_steps.size(); ++i) {
        stream << "zero_start," << zero.history_steps[i] << ','
               << zero.history_residuals[i] << '\n';
    }
    if (ray) {
        for (std::size_t i = 0; i < ray->history_steps.size(); ++i) {
            stream << "ray_start," << ray->history_steps[i] << ','
                   << ray->history_residuals[i] << '\n';
        }
    }
}

void write_metadata(const fs::path& filename,
                    const Model& model,
                    const Parameters& p,
                    const std::vector<StateMetric>& hybrid,
                    double reference_residual,
                    double ray_error,
                    double ray_residual,
                    Complex ray_scale,
                    double eikonal_seconds,
                    double ray_seconds,
                    double coarse_seconds,
                    const RestartedResult& zero,
                    const RestartedResult* ray)
{
    std::ofstream stream(filename);
    stream << std::setprecision(17);
    stream << "{\n"
           << "  \"nx\": " << model.nx << ",\n"
           << "  \"nz\": " << model.nz << ",\n"
           << "  \"full_nx\": " << model.full_nx << ",\n"
           << "  \"full_nz\": " << model.full_nz << ",\n"
           << "  \"crop_x0\": " << model.crop_x0 << ",\n"
           << "  \"crop_z0\": " << model.crop_z0 << ",\n"
           << "  \"source_x_matlab_local\": " << (model.source_x + 1) << ",\n"
           << "  \"source_z_matlab_local\": " << (model.source_z + 1) << ",\n"
           << "  \"dx\": " << p.dx << ",\n"
           << "  \"dz\": " << p.dz << ",\n"
           << "  \"frequency\": " << p.frequency << ",\n"
           << "  \"q_coarse\": " << p.q_coarse << ",\n"
           << "  \"gmres_steps_per_outer\": " << p.gmres_steps_per_outer << ",\n"
           << "  \"target_relres\": " << p.target_relres << ",\n"
           << "  \"reference_mode\": \"" << p.reference_mode << "\",\n"
           << "  \"reference_residual\": " << reference_residual << ",\n"
           << "  \"raw_ray_error\": " << ray_error << ",\n"
           << "  \"raw_ray_residual\": " << ray_residual << ",\n"
           << "  \"ray_scale_real\": " << ray_scale.real() << ",\n"
           << "  \"ray_scale_imag\": " << ray_scale.imag() << ",\n"
           << "  \"eikonal_seconds\": " << eikonal_seconds << ",\n"
           << "  \"ray_seconds\": " << ray_seconds << ",\n"
           << "  \"coarse_setup_seconds\": " << coarse_seconds << ",\n"
           << "  \"hybrid_outer_iterations\": " << hybrid.back().outer << ",\n"
           << "  \"hybrid_fine_gmres_steps\": " << hybrid.back().fine_steps << ",\n"
           << "  \"hybrid_final_residual\": " << hybrid.back().relative_residual << ",\n"
           << "  \"hybrid_final_error\": " << hybrid.back().relative_error << ",\n"
           << "  \"zero_gmres_iterations\": " << zero.inner_iterations << ",\n"
           << "  \"zero_gmres_final_residual\": " << zero.relative_residual << ",\n"
           << "  \"zero_gmres_final_error\": " << zero.relative_error;
    if (ray) {
        stream << ",\n  \"ray_gmres_iterations\": " << ray->inner_iterations
               << ",\n  \"ray_gmres_final_residual\": " << ray->relative_residual
               << ",\n  \"ray_gmres_final_error\": " << ray->relative_error;
    }
    stream << "\n}\n";
}

void write_matlab_text_log(const fs::path& filename,
                           const Model& model,
                           const Parameters& p,
                           const HybridResult& hybrid,
                           const SnapshotResult& zero_matched,
                           double ray_error,
                           double ray_residual,
                           Complex ray_scale,
                           double eikonal_seconds,
                           double ray_seconds,
                           double coarse_seconds,
                           const RestartedResult& zero)
{
    std::ofstream stream(filename);
    stream << std::setprecision(10) << std::scientific;
    stream << "Model nx=" << model.nx << " nz=" << model.nz
           << " dx=dz=" << std::fixed << std::setprecision(6) << p.dx
           << " m freq=" << p.frequency << " Hz\n";
    stream << "q_coarse=" << p.q_coarse
           << " gmres_steps_per_outer=" << p.gmres_steps_per_outer
           << " target_relres=" << std::scientific << std::setprecision(8)
           << p.target_relres << "\n";
    stream << "raw_ray_error=" << std::setprecision(10) << ray_error
           << " raw_ray_residual=" << ray_residual << "\n";
    stream << "ray_scale=" << std::setprecision(12) << ray_scale.real()
           << (ray_scale.imag() >= 0.0 ? " +" : " ") << ray_scale.imag() << "i\n";
    stream << "eikonal_time=" << std::fixed << std::setprecision(8)
           << eikonal_seconds << " ray_build_time=" << ray_seconds
           << " coarse_setup_time=" << coarse_seconds << "\n\n";
    stream << "outer fine_gmres_steps hybrid_error hybrid_residual zero_error "
              "zero_residual abs_alpha raycorr_norm_ref gmrescorr_norm_ref "
              "hybrid_cumulative_time\n";
    for (std::size_t i = 0; i < hybrid.metrics.size(); ++i) {
        const StateMetric& h = hybrid.metrics[i];
        const SnapshotMetric& z = zero_matched.metrics[i];
        stream << h.outer << ' ' << h.fine_steps << ' ' << std::scientific
               << std::setprecision(10) << h.relative_error << ' '
               << h.relative_residual << ' ' << z.relative_error << ' '
               << z.relative_residual << ' ' << std::abs(h.alpha) << ' '
               << h.coarse_correction_norm_over_reference << ' '
               << h.gmres_correction_norm_over_reference << ' '
               << std::fixed << std::setprecision(8) << h.elapsed << '\n';
    }
    stream << "\nTARGET COMPARISON\n" << std::scientific;
    stream << "zero_inner_iterations=" << zero.inner_iterations
           << " zero_time=" << std::fixed << std::setprecision(8) << zero.elapsed
           << " zero_rr=" << std::scientific << std::setprecision(10)
           << zero.relative_residual << " zero_err=" << zero.relative_error << "\n";
    const StateMetric& last = hybrid.metrics.back();
    stream << "hybrid_outer_iterations=" << last.outer
           << " hybrid_fine_gmres_steps=" << last.fine_steps
           << " hybrid_iteration_time=" << std::fixed << std::setprecision(8)
           << last.elapsed << " hybrid_rr=" << std::scientific
           << std::setprecision(10) << last.relative_residual
           << " hybrid_err=" << last.relative_error << "\n";
    stream << "hybrid_preparation_time=" << std::fixed << std::setprecision(8)
           << (eikonal_seconds + ray_seconds + coarse_seconds)
           << " hybrid_end_to_end_time="
           << (eikonal_seconds + ray_seconds + coarse_seconds + last.elapsed) << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Arguments arguments(argc, argv);
        if (arguments.help()) {
            print_help();
            return 0;
        }
        const Parameters p = parse_parameters(arguments);
        const fs::path output = fs::absolute(p.output_dir);
        const fs::path fields = output / "fields";
        fs::create_directories(fields);

        std::cout << "Exact MATLAB hybrid translation\n"
                  << "  source: test_hybrid_rayphase_residual_gmres_marmousi.m\n"
                  << "  model: " << p.model_file << "\n";
        const Model model = read_model(p);
        const int n = model.nx * model.nz;
        const auto [vmin, vmax] = std::minmax_element(
            model.velocity.begin(), model.velocity.end());
        std::cout << "  active model: nx=" << model.nx << ", nz=" << model.nz
                  << ", source MATLAB=(" << (model.source_x + 1) << ','
                  << (model.source_z + 1) << ")\n"
                  << "  velocity=" << *vmin << "--" << *vmax << " m/s\n";
        write_float_field(fields / "velocity_m_per_s.f32", model.velocity);

        auto started = Clock::now();
        HelmholtzSystem system = build_helmholtz(model, p);
        const double matrix_seconds = elapsed_seconds(started);
        const double rhs_norm = system.rhs.norm();
        std::cout << "A: N=" << n << ", nnz=" << system.matrix.nonZeros()
                  << ", build=" << std::fixed << matrix_seconds << " s\n";

        Vector reference;
        const Vector* reference_pointer = nullptr;
        double reference_residual = std::numeric_limits<double>::quiet_NaN();
        if (p.reference_mode == "read") {
            std::cout << "Loading validation reference: " << p.reference_file << "\n";
            reference = load_reference(p.reference_file, n);
            reference_pointer = &reference;
        } else if (p.reference_mode == "direct") {
            std::cout << "Computing sparse-direct Helmholtz reference...\n";
            started = Clock::now();
            reference = direct_reference(system.matrix, system.rhs);
            std::cout << "Direct reference complete in " << std::fixed
                      << elapsed_seconds(started) << " s\n";
            reference_pointer = &reference;
        }
        if (reference_pointer) {
            reference_residual =
                (system.rhs - system.matrix * reference).norm() / rhs_norm;
            std::cout << "Reference relative residual=" << std::scientific
                      << reference_residual << "\n";
            write_real_imag_field(fields / "reference", reference);
        }

        std::cout << "\nComputing first-arrival eikonal traveltime...\n";
        started = Clock::now();
        const std::vector<double> traveltime = fast_sweep_eikonal(model, p);
        const double eikonal_seconds = elapsed_seconds(started);
        write_float_field(fields / "traveltime_s.f32", traveltime);
        std::cout << "Eikonal complete in " << std::fixed << eikonal_seconds
                  << " s, Tmax="
                  << *std::max_element(traveltime.begin(), traveltime.end()) << " s\n";

        std::cout << "\nConstructing raw first-arrival ray field...\n";
        started = Clock::now();
        const RayField ray = build_first_arrival_ray(
            system.matrix, system.rhs, model, p, traveltime);
        const double ray_seconds = elapsed_seconds(started);
        const double ray_residual =
            (system.rhs - system.matrix * ray.field).norm() / rhs_norm;
        const double ray_error = relative_error(ray.field, reference_pointer);
        std::cout << "Raw ray: residual=" << std::scientific << ray_residual
                  << ", error=" << ray_error << ", scale=" << ray.scale
                  << ", build=" << std::fixed << ray_seconds << " s\n";
        write_real_imag_field(fields / "ray_initial", ray.field);

        std::cout << "\nBuilding literal MATLAB ray-phase coarse space q="
                  << p.q_coarse << "...\n";
        CoarseSpace coarse = build_phase_coarse_space(
            system.matrix, model, p, traveltime);

        HybridResult hybrid = run_hybrid(
            system.matrix, system.rhs, ray.field, coarse, reference_pointer,
            p, fields);
        write_real_imag_field(fields / "hybrid_final", hybrid.final);

        std::vector<int> matched_steps;
        matched_steps.reserve(hybrid.metrics.size());
        for (const StateMetric& metric : hybrid.metrics) {
            matched_steps.push_back(metric.fine_steps);
        }
        std::cout << "\nContinuous zero-initial GMRES at matched steps...\n";
        const SnapshotResult zero_matched = continuous_gmres_selected(
            system.matrix, system.rhs, Vector::Zero(n), matched_steps,
            reference_pointer, fields, "zero", p.save_frame_fields);

        std::cout << "\nRestarted GMRES target comparison...\n";
        const RestartedResult zero_target = restarted_gmres_to_target(
            system.matrix, system.rhs, Vector::Zero(n), p.zero_restart,
            p.target_relres, p.zero_max_cycles, reference_pointer,
            "zero-start GMRES");
        write_real_imag_field(fields / "zero_restarted_target", zero_target.final);

        RestartedResult ray_target;
        RestartedResult* ray_target_pointer = nullptr;
        if (p.run_ray_start_gmres) {
            ray_target = restarted_gmres_to_target(
                system.matrix, system.rhs, ray.field, p.zero_restart,
                p.target_relres, p.zero_max_cycles, reference_pointer,
                "ray-start GMRES");
            ray_target_pointer = &ray_target;
            write_real_imag_field(fields / "ray_restarted_target", ray_target.final);
        }

        write_hybrid_csv(output / "hybrid_outer_metrics.csv", hybrid.metrics);
        write_matched_csv(output / "matched_step_comparison.csv",
                          hybrid.metrics, zero_matched);
        write_restarted_history(output / "restarted_gmres_histories.csv",
                                zero_target, ray_target_pointer);
        write_metadata(output / "metadata.json", model, p, hybrid.metrics,
                       reference_residual, ray_error, ray_residual, ray.scale,
                       eikonal_seconds, ray_seconds, coarse.setup_seconds,
                       zero_target, ray_target_pointer);
        write_matlab_text_log(
            output / "hybrid_rayphase_residual_gmres_results.txt",
            model, p, hybrid, zero_matched, ray_error, ray_residual,
            ray.scale, eikonal_seconds, ray_seconds, coarse.setup_seconds,
            zero_target);

        std::cout << "\n================ C++ MATLAB REPLICA SUMMARY ================\n"
                  << "Hybrid: outer=" << hybrid.metrics.back().outer
                  << ", fine GMRES=" << hybrid.metrics.back().fine_steps
                  << ", residual=" << std::scientific
                  << hybrid.metrics.back().relative_residual
                  << ", error=" << hybrid.metrics.back().relative_error << "\n"
                  << "Zero-start restarted GMRES: steps="
                  << zero_target.inner_iterations << ", residual="
                  << zero_target.relative_residual << "\n";
        if (ray_target_pointer) {
            std::cout << "Ray-start restarted GMRES:  steps="
                      << ray_target.inner_iterations << ", residual="
                      << ray_target.relative_residual << "\n";
        }
        std::cout << "Outputs: " << output << "\n"
                  << "=============================================================\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
