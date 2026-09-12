/*
 * ray_helmholtz_iteration.cpp
 *
 * Native ButterflyPACK interface propagation + block Helmholtz correction.
 *
 * The Butterfly part intentionally follows src/test_bf_kir.cpp:
 *   c_c_bpack_construct_init
 *   c_c_bf_construct_init
 *   c_c_bf_construct_element_compute
 *   c_c_bf_mult
 *
 * For every block:
 *   1) take the wavefield on the block upper interface;
 *   2) use the precomputed tau table and ButterflyPACK to evaluate the
 *      Kirchhoff integral only on the lower interface;
 *   3) use upper/lower interface values as Dirichlet data for a local
 *      Helmholtz defect-correction solve;
 *   4) copy the corrected block field to the global field and continue.
 *
 * Block 0 is NOT initialized with a Hankel field.  Its upper interface is a
 * spatial point source, exactly like the input used by test_bf_kir.cpp.
 */

#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_fs.h>
#include <SEFILESYSTEM/include/se_par_sep.h>
#include <SERECKIRCH/include/huygens_sweep.hpp>
#include <se_eigen.hpp>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "cBPACK_wrapper.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Complex = std::complex<float>;
using NativeComplex = __complex__ float;
using Clock = std::chrono::steady_clock;
using EigenVector = se::eigen::DenseVector<double>;
using EigenMatrix = se::eigen::SparseMatrix<double>;

constexpr double PI = 3.141592653589793238462643383279502884;


/* ============================================================
 * General helpers
 * ============================================================ */

double elapsed(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void configure_threads(int requested_threads)
{
    int n = requested_threads > 0 ? requested_threads : 32;

    /* Keep compatibility with test_bf_kir.cpp. */
    if (const char* value = std::getenv("BF_THREADS")) {
        const int from_env = std::atoi(value);
        if (from_env > 0) n = from_env;
    }

    const std::string ns = std::to_string(n);
    ::setenv("OMP_NUM_THREADS", ns.c_str(), 1);
    ::setenv("OMP_DYNAMIC", "FALSE", 1);
    ::setenv("OMP_PROC_BIND", "close", 1);
    ::setenv("OMP_PLACES", "cores", 1);
    ::setenv("OPENBLAS_NUM_THREADS", "1", 1);
    ::setenv("GOTO_NUM_THREADS", "1", 1);

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(n);
#endif
}

NativeComplex to_native(const Complex& z)
{
    NativeComplex out;
    __real__ out = z.real();
    __imag__ out = z.imag();
    return out;
}

Complex from_native(NativeComplex z)
{
    return Complex(static_cast<float>(__real__ z),
                   static_cast<float>(__imag__ z));
}

float max_tau(const se::huygens::Table3D& tau)
{
    if (tau.values.empty()) {
        throw std::runtime_error("empty traveltime table");
    }
    return *std::max_element(tau.values.begin(), tau.values.end());
}


/* ============================================================
 * Native ButterflyPACK: copied/adapted from test_bf_kir.cpp
 * ============================================================ */

struct BPackContext {
    const se::huygens::Table3D* tau = nullptr;
    const std::vector<float>* weights = nullptr;
    const se::huygens::FrequencyKirchhoffFilter* filter = nullptr;

    float source_z = 0.0f;
    int target_depth_index = 0;

    double bf_tol = 1.0e-4;
    double bf_sample = 4.0;
    int bf_leaf = 64;
    int bf_knn = 0;

    int rows = 0;
    int cols = 0;

    std::vector<int> row_new2old;
    std::vector<int> col_new2old;
};

struct BPackHandles {
    F2Cptr ptree = nullptr;
    F2Cptr option = nullptr;
    F2Cptr row_stats = nullptr;
    F2Cptr col_stats = nullptr;
    F2Cptr bf_stats = nullptr;
    F2Cptr row_mat = nullptr;
    F2Cptr col_mat = nullptr;
    F2Cptr row_mesh = nullptr;
    F2Cptr col_mesh = nullptr;
    F2Cptr bf_mesh = nullptr;
    F2Cptr row_ker = nullptr;
    F2Cptr col_ker = nullptr;
    F2Cptr bf_ker = nullptr;
    F2Cptr bf = nullptr;

    int row_local = 0;
    int col_local = 0;
};

std::vector<float> make_weights(const se::huygens::Table3D& tau)
{
    if (tau.nsource < 1) {
        throw std::runtime_error("traveltime table has no sources");
    }

    std::vector<float> w(static_cast<std::size_t>(tau.nsource), tau.dx_source);
    if (tau.nsource > 1) {
        w.front() *= 0.5f;
        w.back() *= 0.5f;
    }
    return w;
}

int find_target_depth_index(const se::huygens::Model2D& model,
                            const se::huygens::Block& block,
                            const se::huygens::Table3D& tau)
{
    const double z = static_cast<double>(model.z(block.target_end_iz));
    const double q = (z - static_cast<double>(tau.oz_target)) /
                     static_cast<double>(tau.dz_target);
    const int iz = static_cast<int>(std::llround(q));

    if (iz < 0 || iz >= tau.nz_target) {
        throw std::runtime_error("block lower interface is outside tau target depth range");
    }

    const double z_table = static_cast<double>(tau.oz_target) +
                           static_cast<double>(iz) * tau.dz_target;
    const double tolerance = 1.0e-3 * std::max(1.0, std::abs(static_cast<double>(tau.dz_target)));
    if (std::abs(z_table - z) > tolerance) {
        std::ostringstream message;
        message << "cannot match block lower interface z=" << z
                << " to tau target grid; nearest z=" << z_table;
        throw std::runtime_error(message.str());
    }

    return iz;
}

std::vector<double> make_bottom_row_coord(const se::huygens::Table3D& tau,
                                          int target_depth_index)
{
    std::vector<double> coord(static_cast<std::size_t>(2 * tau.nx_target));
    const double z = static_cast<double>(tau.oz_target) +
                     static_cast<double>(target_depth_index) * tau.dz_target;

    for (int row = 0; row < tau.nx_target; ++row) {
        coord[2 * row] = static_cast<double>(tau.ox_target) +
                         static_cast<double>(row) * tau.dx_target;
        coord[2 * row + 1] = z;
    }
    return coord;
}

std::vector<double> make_col_coord(const se::huygens::Table3D& tau,
                                   float source_z)
{
    std::vector<double> coord(static_cast<std::size_t>(2 * tau.nsource));

    for (int src = 0; src < tau.nsource; ++src) {
        coord[2 * src] = static_cast<double>(tau.ox_source) +
                         static_cast<double>(src) * tau.dx_source;
        coord[2 * src + 1] = source_z;
    }
    return coord;
}

/*
 * Same Kirchhoff matrix element as test_bf_kir.cpp, except row now means one
 * point on the selected lower interface rather than one point in the volume.
 */
Complex kirchhoff_entry(int row, int src, const BPackContext& ctx)
{
    const auto& tau = *ctx.tau;
    const int ix = row;
    const int iz = ctx.target_depth_index;

    const float t = std::max(tau.values[tau.index(src, ix, iz)], 1.0e-8f);
    const float xt = tau.ox_target + ix * tau.dx_target;
    const float zt = tau.oz_target + iz * tau.dz_target;
    const float xs = tau.ox_source + src * tau.dx_source;

    const float dx = xt - xs;
    const float dz = zt - ctx.source_z;
    const float r2 = std::max(dx * dx + dz * dz, 1.0e-20f);

    return (*ctx.weights)[static_cast<std::size_t>(src)] *
           dz * t / (static_cast<float>(PI) * r2) *
           ctx.filter->response(t);
}

void dummy_distance(int*, int*, double* value, C2Fptr)
{
    *value = 0.0;
}

void dummy_near_far(int*, int*, int* value, C2Fptr)
{
    *value = 0;
}

void set_bpack_options(F2Cptr* option, const BPackContext& ctx)
{
    auto D = [&](const char* key, double value) {
        c_c_bpack_set_D_option(option, key, value);
    };
    auto I = [&](const char* key, int value) {
        c_c_bpack_set_I_option(option, key, value);
    };

    /* Keep the options identical to test_bf_kir.cpp. */
    D("tol_comp", ctx.bf_tol);
    D("tol_rand", ctx.bf_tol);
    D("tol_Rdetect", 0.1 * ctx.bf_tol);
    D("sample_para", ctx.bf_sample);
    D("sample_para_outer", ctx.bf_sample);

    I("nogeo", 0);
    I("Nmin_leaf", ctx.bf_leaf);
    I("RecLR_leaf", 5);
    I("xyzsort", 1);
    I("cpp", 1);
    I("LRlevel", 100);
    I("forwardN15flag", 0);
    I("knn", ctx.bf_knn);
    I("verbosity", -1);
    I("less_adapt", 1);
    I("pat_comp", 3);
    I("BACA_Batch", 16);
    I("LR_BLK_NUM", 1);
    I("itermax", 10);
    I("ErrFillFull", 0);
    I("ErrSol", 0);
    I("elem_extract", 2);
    I("format", 1);
}

void bpack_sample(int* a, int* b, NativeComplex* value, C2Fptr ptr)
{
    auto* ctx = static_cast<BPackContext*>(ptr);

    const int row_new = (*a > 0) ? *a : *b;
    const int col_new = (*a > 0) ? -*b : -*a;

    const int row = ctx->row_new2old[static_cast<std::size_t>(row_new - 1)] - 1;
    const int col = ctx->col_new2old[static_cast<std::size_t>(col_new - 1)] - 1;

    *value = to_native(kirchhoff_entry(row, col, *ctx));
}

void bpack_sample_block(int* nblock, int*, int*, std::int64_t*,
                        int* all_rows, int* all_cols, NativeComplex* values,
                        int* nrows, int* ncols, int*, int*, int*, C2Fptr ptr)
{
    auto* ctx = static_cast<BPackContext*>(ptr);
    std::int64_t roff = 0;
    std::int64_t coff = 0;
    std::int64_t voff = 0;

    for (int ib = 0; ib < *nblock; ++ib) {
        const int nr = nrows[ib];
        const int nc = ncols[ib];
        std::vector<int> rows(static_cast<std::size_t>(nr));
        std::vector<int> cols(static_cast<std::size_t>(nc));

        for (int r = 0; r < nr; ++r) {
            rows[static_cast<std::size_t>(r)] =
                ctx->row_new2old[static_cast<std::size_t>(all_rows[roff + r] - 1)] - 1;
        }
        for (int c = 0; c < nc; ++c) {
            cols[static_cast<std::size_t>(c)] =
                ctx->col_new2old[static_cast<std::size_t>(all_cols[coff + c] - 1)] - 1;
        }

        const std::int64_t count = static_cast<std::int64_t>(nr) * nc;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096)
#endif
        for (std::int64_t k = 0; k < count; ++k) {
            const int c = static_cast<int>(k / nr);
            const int r = static_cast<int>(k % nr);
            values[voff + k] = to_native(
                kirchhoff_entry(rows[static_cast<std::size_t>(r)],
                                cols[static_cast<std::size_t>(c)], *ctx));
        }

        roff += nr;
        coff += nc;
        voff += count;
    }
}

void destroy_bpack(BPackHandles& h)
{
    if (h.bf) c_c_bf_deletebf(&h.bf);
    if (h.bf_mesh) c_c_bpack_deletemesh(&h.bf_mesh);
    if (h.bf_ker) c_c_bpack_deletekernelquant(&h.bf_ker);

    if (h.row_mat) c_c_bpack_delete(&h.row_mat);
    if (h.col_mat) c_c_bpack_delete(&h.col_mat);
    if (h.row_mesh) c_c_bpack_deletemesh(&h.row_mesh);
    if (h.col_mesh) c_c_bpack_deletemesh(&h.col_mesh);
    if (h.row_ker) c_c_bpack_deletekernelquant(&h.row_ker);
    if (h.col_ker) c_c_bpack_deletekernelquant(&h.col_ker);

    if (h.row_stats) c_c_bpack_deletestats(&h.row_stats);
    if (h.col_stats) c_c_bpack_deletestats(&h.col_stats);
    if (h.bf_stats) c_c_bpack_deletestats(&h.bf_stats);

    if (h.option) c_c_bpack_deleteoption(&h.option);
    if (h.ptree) c_c_bpack_deleteproctree(&h.ptree);
}

void build_butterfly(BPackContext& ctx,
                     BPackHandles& h,
                     std::vector<double>& row_coord,
                     std::vector<double>& col_coord)
{
    int nproc = 1;
    int groups[1] = {0};
    MPI_Fint comm = static_cast<MPI_Fint>(321);

    c_c_bpack_createptree(&nproc, groups, &comm, &h.ptree);
    c_c_bpack_createoption(&h.option);
    c_c_bpack_createstats(&h.row_stats);
    c_c_bpack_createstats(&h.col_stats);
    c_c_bpack_createstats(&h.bf_stats);
    set_bpack_options(&h.option, ctx);

    ctx.row_new2old.resize(static_cast<std::size_t>(ctx.rows));
    ctx.col_new2old.resize(static_cast<std::size_t>(ctx.cols));

    int row_counts[1] = {ctx.rows};
    int col_counts[1] = {ctx.cols};
    int row_tree[1] = {ctx.rows};
    int col_tree[1] = {ctx.cols};

    int level = 0;
    int ndim = 2;
    int rows = ctx.rows;
    int cols = ctx.cols;

    c_c_bpack_construct_init(
        &rows, &ndim, row_coord.data(), row_counts, &level, row_tree,
        ctx.row_new2old.data(), &h.row_local, &h.row_mat, &h.option,
        &h.row_stats, &h.row_mesh, &h.row_ker, &h.ptree,
        &dummy_distance, &dummy_near_far, &ctx);

    level = 0;
    c_c_bpack_construct_init(
        &cols, &ndim, col_coord.data(), col_counts, &level, col_tree,
        ctx.col_new2old.data(), &h.col_local, &h.col_mat, &h.option,
        &h.col_stats, &h.col_mesh, &h.col_ker, &h.ptree,
        &dummy_distance, &dummy_near_far, &ctx);

    c_c_bf_construct_init(
        &rows, &cols, &h.row_local, &h.col_local,
        row_counts, col_counts, &h.row_mesh, &h.col_mesh,
        &h.bf, &h.option, &h.bf_stats, &h.bf_mesh, &h.bf_ker, &h.ptree,
        &dummy_distance, &dummy_near_far, &ctx);

    c_c_bf_construct_element_compute(
        &h.bf, &h.option, &h.bf_stats, &h.bf_mesh, &h.bf_ker, &h.ptree,
        &bpack_sample, &bpack_sample_block, &ctx);
}

std::vector<Complex> apply_butterfly(const BPackContext& ctx,
                                     BPackHandles& h,
                                     const std::vector<Complex>& input)
{
    if (static_cast<int>(input.size()) != ctx.cols) {
        throw std::invalid_argument("Butterfly input size does not equal tau.nsource");
    }

    std::vector<NativeComplex> x(static_cast<std::size_t>(ctx.cols));
    std::vector<NativeComplex> y(static_cast<std::size_t>(ctx.rows));

    for (int i = 0; i < ctx.cols; ++i) {
        x[static_cast<std::size_t>(i)] =
            to_native(input[static_cast<std::size_t>(ctx.col_new2old[i] - 1)]);
    }

    char trans = 'N';
    int nin = ctx.cols;
    int nout = ctx.rows;
    int nrhs = 1;

    c_c_bf_mult(&trans, x.data(), y.data(),
                &nin, &nout, &nrhs,
                &h.bf, &h.option, &h.bf_stats, &h.ptree);

    std::vector<Complex> output(static_cast<std::size_t>(ctx.rows));
    for (int i = 0; i < ctx.rows; ++i) {
        output[static_cast<std::size_t>(ctx.row_new2old[i] - 1)] =
            from_native(y[static_cast<std::size_t>(i)]);
    }
    return output;
}

std::vector<Complex> propagate_interface_butterfly(
    const se::huygens::Model2D& model,
    const se::huygens::Block& block,
    const se::huygens::Table3D& tau,
    const se::huygens::FrequencyKirchhoffFilter& filter,
    const std::vector<Complex>& top_trace,
    double bf_tol,
    int bf_leaf,
    double bf_sample,
    int bf_knn,
    double& build_seconds,
    double& apply_seconds)
{
    /*
     * The block Helmholtz solve needs a complete lower boundary.  Therefore
     * this prototype requires one tau source and one tau target per model x.
     */
    if (tau.nsource != model.nx || tau.nx_target != model.nx) {
        std::ostringstream message;
        message << "interface Butterfly currently requires tau.nsource == "
                << "tau.nx_target == model.nx; got "
                << tau.nsource << ", " << tau.nx_target << ", " << model.nx;
        throw std::runtime_error(message.str());
    }

    if (std::abs(tau.dx_source - model.dx) > 1.0e-4f * std::max(1.0f, model.dx) ||
        std::abs(tau.dx_target - model.dx) > 1.0e-4f * std::max(1.0f, model.dx)) {
        throw std::runtime_error("tau source/target x spacing does not match model dx");
    }

    const int target_depth_index = find_target_depth_index(model, block, tau);
    const std::vector<float> weights = make_weights(tau);

    BPackContext ctx;
    ctx.tau = &tau;
    ctx.weights = &weights;
    ctx.filter = &filter;
    ctx.source_z = model.z(block.source_iz);
    ctx.target_depth_index = target_depth_index;
    ctx.bf_tol = bf_tol;
    ctx.bf_leaf = bf_leaf;
    ctx.bf_sample = bf_sample;
    ctx.bf_knn = bf_knn;
    ctx.rows = tau.nx_target;
    ctx.cols = tau.nsource;

    auto row_coord = make_bottom_row_coord(tau, target_depth_index);
    auto col_coord = make_col_coord(tau, ctx.source_z);

    BPackHandles handles;
    try {
        const auto t_build = Clock::now();
        build_butterfly(ctx, handles, row_coord, col_coord);
        build_seconds = elapsed(t_build);

        const auto t_apply = Clock::now();
        std::vector<Complex> bottom = apply_butterfly(ctx, handles, top_trace);
        apply_seconds = elapsed(t_apply);

        destroy_bpack(handles);
        return bottom;
    } catch (...) {
        destroy_bpack(handles);
        throw;
    }
}


/* ============================================================
 * Block Helmholtz defect correction
 * ============================================================ */

struct LocalHelmholtzSystem {
    int top_iz = 0;
    int bottom_iz = 0;
    int nx_interior = 0;
    int nz_interior = 0;
    double cx = 0.0;
    double cz = 0.0;
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
        throw std::runtime_error("local block has no Helmholtz interior");
    }

    system.cx = 1.0 / (static_cast<double>(model.dx) * model.dx);
    system.cz = 1.0 / (static_cast<double>(model.dz) * model.dz);
    const double omega = 2.0 * PI * frequency;

    std::vector<se::eigen::Triplet<double>> entries;
    entries.reserve(static_cast<std::size_t>(system.unknowns()) * 5);

    for (int iz = system.top_iz + 1; iz < system.bottom_iz; ++iz) {
        for (int ix = 1; ix < model.nx - 1; ++ix) {
            const int row = system.row(ix, iz);
            const double velocity = model.velocity[model.index(ix, iz)];
            const double k = omega / velocity;
            const double diagonal = k * k - 2.0 * system.cx - 2.0 * system.cz;

            entries.emplace_back(row, row, diagonal);
            if (ix > 1)
                entries.emplace_back(row, system.row(ix - 1, iz), system.cx);
            if (ix + 1 < model.nx - 1)
                entries.emplace_back(row, system.row(ix + 1, iz), system.cx);
            if (iz > system.top_iz + 1)
                entries.emplace_back(row, system.row(ix, iz - 1), system.cz);
            if (iz + 1 < system.bottom_iz)
                entries.emplace_back(row, system.row(ix, iz + 1), system.cz);
        }
    }

    system.matrix = se::eigen::make_sparse_matrix<double>(
        system.unknowns(), system.unknowns(), entries);
    return system;
}

void extract_local_vectors(const se::huygens::Model2D& model,
                           const LocalHelmholtzSystem& system,
                           const std::vector<Complex>& wavefield,
                           EigenVector& real_values,
                           EigenVector& imag_values,
                           EigenVector& real_rhs,
                           EigenVector& imag_rhs)
{
    const int n = system.unknowns();
    real_values.resize(n);
    imag_values.resize(n);
    real_rhs.setZero(n);
    imag_rhs.setZero(n);

    auto add_boundary = [&](int row, double coefficient, const Complex& value) {
        real_rhs[row] -= coefficient * static_cast<double>(value.real());
        imag_rhs[row] -= coefficient * static_cast<double>(value.imag());
    };

    for (int iz = system.top_iz + 1; iz < system.bottom_iz; ++iz) {
        for (int ix = 1; ix < model.nx - 1; ++ix) {
            const int row = system.row(ix, iz);
            const Complex value = wavefield[model.index(ix, iz)];
            real_values[row] = value.real();
            imag_values[row] = value.imag();

            if (ix == 1)
                add_boundary(row, system.cx, wavefield[model.index(0, iz)]);
            if (ix == model.nx - 2)
                add_boundary(row, system.cx, wavefield[model.index(model.nx - 1, iz)]);
            if (iz == system.top_iz + 1)
                add_boundary(row, system.cz,
                             wavefield[model.index(ix, system.top_iz)]);
            if (iz == system.bottom_iz - 1)
                add_boundary(row, system.cz,
                             wavefield[model.index(ix, system.bottom_iz)]);
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

CorrectionMetrics correct_local_block(
    const se::huygens::Model2D& model,
    const LocalHelmholtzSystem& system,
    std::vector<Complex>& wavefield,
    const se::eigen::SolverOptions& options,
    double requested_relaxation,
    double maximum_relaxation)
{
    EigenVector real_values, imag_values, real_rhs, imag_rhs;
    extract_local_vectors(model, system, wavefield,
                          real_values, imag_values, real_rhs, imag_rhs);

    const EigenVector real_residual = system.matrix * real_values - real_rhs;
    const EigenVector imag_residual = system.matrix * imag_values - imag_rhs;

    CorrectionMetrics m;
    m.residual_before = std::sqrt(real_residual.squaredNorm() +
                                  imag_residual.squaredNorm());
    m.residual_after = m.residual_before;

    if (m.residual_before <= std::numeric_limits<double>::min()) return m;

    const auto real_result = solve_component(system.matrix, -real_residual, options);
    const auto imag_result = solve_component(system.matrix, -imag_residual, options);

    m.real_iterations = real_result.report.iterations;
    m.imag_iterations = imag_result.report.iterations;
    m.real_status = real_result.report.status;
    m.imag_status = imag_result.report.status;

    if (!all_finite(real_result.solution) || !all_finite(imag_result.solution)) {
        m.real_status = se::eigen::SolverStatus::numerical_issue;
        m.imag_status = se::eigen::SolverStatus::numerical_issue;
        return m;
    }

    const EigenVector real_action = system.matrix * real_result.solution;
    const EigenVector imag_action = system.matrix * imag_result.solution;
    const double action2 = real_action.squaredNorm() + imag_action.squaredNorm();

    double alpha = requested_relaxation;
    if (requested_relaxation < 0.0) {
        if (action2 > std::numeric_limits<double>::min()) {
            alpha = -(real_residual.dot(real_action) +
                      imag_residual.dot(imag_action)) / action2;
        } else {
            alpha = 0.0;
        }
        alpha = std::clamp(alpha, 0.0, maximum_relaxation);
    }
    if (!std::isfinite(alpha)) alpha = 0.0;
    m.relaxation = alpha;

    const EigenVector corrected_real = real_values + alpha * real_result.solution;
    const EigenVector corrected_imag = imag_values + alpha * imag_result.solution;
    const EigenVector after_real = real_residual + alpha * real_action;
    const EigenVector after_imag = imag_residual + alpha * imag_action;

    m.residual_after = std::sqrt(after_real.squaredNorm() + after_imag.squaredNorm());
    m.residual_ratio = m.residual_after /
        std::max(m.residual_before, std::numeric_limits<double>::min());

    const double field_norm = std::sqrt(real_values.squaredNorm() +
                                        imag_values.squaredNorm());
    const double update_norm = std::abs(alpha) * std::sqrt(
        real_result.solution.squaredNorm() + imag_result.solution.squaredNorm());
    m.relative_update = update_norm /
        std::max(field_norm, std::numeric_limits<double>::min());

    if (all_finite(corrected_real) && all_finite(corrected_imag)) {
        write_local_values(model, system, corrected_real, corrected_imag, wavefield);
    }
    return m;
}


/* ============================================================
 * Wavefield interface / output helpers
 * ============================================================ */

std::vector<Complex> make_point_source(int nx, int source_ix, float amplitude)
{
    std::vector<Complex> u(static_cast<std::size_t>(nx), Complex(0.0f, 0.0f));
    u[static_cast<std::size_t>(source_ix)] = Complex(amplitude, 0.0f);
    return u;
}

std::vector<Complex> gather_row(const se::huygens::Model2D& model,
                                const std::vector<Complex>& wavefield,
                                int iz)
{
    std::vector<Complex> row(static_cast<std::size_t>(model.nx));
    for (int ix = 0; ix < model.nx; ++ix) {
        row[static_cast<std::size_t>(ix)] = wavefield[model.index(ix, iz)];
    }
    return row;
}

void inject_row(const se::huygens::Model2D& model,
                std::vector<Complex>& wavefield,
                int iz,
                const std::vector<Complex>& row)
{
    if (iz < 0 || iz >= model.nz || static_cast<int>(row.size()) != model.nx) {
        throw std::invalid_argument("invalid interface row");
    }
    for (int ix = 0; ix < model.nx; ++ix) {
        wavefield[model.index(ix, iz)] = row[static_cast<std::size_t>(ix)];
    }
}

void initialize_block_guess(const se::huygens::Model2D& model,
                            const se::huygens::Block& block,
                            const std::vector<Complex>& global_wavefield,
                            std::vector<Complex>& local_wavefield)
{
    const int top = block.source_iz;
    const int bottom = block.target_end_iz;
    const double denom = std::max(1, bottom - top);

    for (int iz = top + 1; iz < bottom; ++iz) {
        const bool keep_existing = iz < block.target_start_iz;
        const float a = static_cast<float>((iz - top) / denom);

        for (int ix = 0; ix < model.nx; ++ix) {
            const std::size_t index = model.index(ix, iz);
            if (keep_existing) {
                local_wavefield[index] = global_wavefield[index];
            } else {
                const Complex top_value = local_wavefield[model.index(ix, top)];
                const Complex bottom_value = local_wavefield[model.index(ix, bottom)];
                local_wavefield[index] = (1.0f - a) * top_value + a * bottom_value;
            }
        }
    }
}

void commit_block_output(const se::huygens::Model2D& model,
                         const se::huygens::Block& block,
                         const std::vector<Complex>& local_wavefield,
                         std::vector<Complex>& global_wavefield)
{
    const int first = (block.id == 0) ? block.source_iz : block.target_start_iz;
    for (int iz = first; iz <= block.target_end_iz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            global_wavefield[model.index(ix, iz)] = local_wavefield[model.index(ix, iz)];
        }
    }
}

std::string trace_filename(const std::string& prefix,
                           int block_id,
                           const char* kind,
                           const char* component)
{
    std::ostringstream s;
    s << prefix << "_block_" << std::setw(3) << std::setfill('0') << block_id
      << '_' << kind << '_' << component << ".rsf";
    return s.str();
}

std::string block_filename(const std::string& prefix,
                           int block_id,
                           const char* component)
{
    std::ostringstream s;
    s << prefix << "_after_block_" << std::setw(3) << std::setfill('0') << block_id
      << '_' << component << ".rsf";
    return s.str();
}

void write_trace(const std::string& filename,
                 const se::huygens::Model2D& model,
                 const std::vector<Complex>& trace,
                 float frequency,
                 int block_id,
                 int iz,
                 const char* kind,
                 bool imaginary)
{
    se::huygens::ensure_parent_directory(filename);
    std::vector<float> values(trace.size());
    for (std::size_t i = 0; i < trace.size(); ++i)
        values[i] = imaginary ? trace[i].imag() : trace[i].real();

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
    sep_set_header_int(output, "block_id", block_id);
    sep_set_header_int(output, "iz", iz);
    sep_set_header_float(output, "depth", model.z(iz));
    sep_set_header_float(output, "frequency", frequency);
    sep_set_header(output, "trace_kind", kind);
    sep_set_header(output, "component", imaginary ? "imaginary" : "real");
    se_fsio_write_float(output->data->io, values.data(), values.size());
    sep_close(output);
}


/* ============================================================
 * Parameters
 * ============================================================ */

bool help_requested(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i] &&
            (std::strcmp(argv[i], "-h") == 0 ||
             std::strcmp(argv[i], "--help") == 0)) return true;
    }
    return se_have_par("help") && se_get_par_int("help") != 0;
}

void print_help(const char* program)
{
    std::printf(
        "Usage:\n"
        "  %s velocity=... block_file=... table_prefix=... [key=value ...]\n\n"
        "Required:\n"
        "  velocity=vmar.rsf\n"
        "  block_file=block_info.dat\n"
        "  table_prefix=huygens_tt/travel\n\n"
        "Main:\n"
        "  output_prefix=output/ray_iter\n"
        "  frequency=15\n"
        "  source_ix=-1          (-1 -> nx/2)\n"
        "  source_iz=0\n"
        "  source_amplitude=1\n\n"
        "Native ButterflyPACK (same meaning as test_bf_kir.cpp):\n"
        "  bf_tol=1e-4\n"
        "  bf_leaf=64\n"
        "  bf_sample=4\n"
        "  bf_knn=0\n"
        "  filter_dt=0.001\n"
        "  filter_length=0.025\n"
        "  filter_lookup_subsamples=64\n\n"
        "Local Helmholtz GMRES:\n"
        "  iter_cycles=3\n"
        "  iter_iterations=10\n"
        "  iter_restart=10\n"
        "  iter_tolerance=1e-4\n"
        "  iter_preconditioner=diagonal   (identity, diagonal, ilut)\n"
        "  iter_ilut_drop_tolerance=1e-3\n"
        "  iter_ilut_fill_factor=10\n"
        "  iter_relaxation=-1             (-1 -> residual minimizing)\n"
        "  iter_max_relaxation=1\n\n"
        "Output:\n"
        "  write_interfaces=1\n"
        "  write_each_block=1\n"
        "  threads=32\n",
        program);
}

} // namespace


/* ============================================================
 * main
 * ============================================================ */

int main(int argc, char** argv)
{
    se_par_init(argc, argv);

    try {
        if (help_requested(argc, argv)) {
            print_help(argv[0]);
            return 0;
        }

        if (!se_have_par("velocity"))
            throw std::invalid_argument("Need velocity= RSF model");
        if (!se_have_par("block_file"))
            throw std::invalid_argument("Need block_file= block_info.dat");
        if (!se_have_par("table_prefix"))
            throw std::invalid_argument("Need table_prefix= precomputed tau prefix");

        const std::string velocity = se_get_par_str("velocity");
        const std::string block_file = se_get_par_str("block_file");
        const std::string table_prefix = se_get_par_str("table_prefix");
        const std::string output_prefix = se_have_par("output_prefix")
            ? std::string(se_get_par_str("output_prefix"))
            : std::string("output/ray_iter");

        const float frequency = se_have_par("frequency")
            ? se_get_par_float("frequency") : 15.0f;
        const int source_ix_par = se_have_par("source_ix")
            ? se_get_par_int("source_ix") : -1;
        const int source_iz = se_have_par("source_iz")
            ? se_get_par_int("source_iz") : 0;
        const float source_amplitude = se_have_par("source_amplitude")
            ? se_get_par_float("source_amplitude") : 1.0f;

        const double bf_tol = se_have_par("bf_tol")
            ? static_cast<double>(se_get_par_float("bf_tol")) : 1.0e-4;
        const int bf_leaf = se_have_par("bf_leaf")
            ? se_get_par_int("bf_leaf") : 64;
        const double bf_sample = se_have_par("bf_sample")
            ? static_cast<double>(se_get_par_float("bf_sample")) : 4.0;
        const int bf_knn = se_have_par("bf_knn")
            ? se_get_par_int("bf_knn") : 0;

        const float filter_dt = se_have_par("filter_dt")
            ? se_get_par_float("filter_dt") : 0.001f;
        const float filter_length = se_have_par("filter_length")
            ? se_get_par_float("filter_length") : 0.025f;
        const int filter_lookup_subsamples = se_have_par("filter_lookup_subsamples")
            ? se_get_par_int("filter_lookup_subsamples") : 64;

        const int iter_cycles = se_have_par("iter_cycles")
            ? se_get_par_int("iter_cycles") : 3;
        const int iter_iterations = se_have_par("iter_iterations")
            ? se_get_par_int("iter_iterations") : 10;
        const int iter_restart = se_have_par("iter_restart")
            ? se_get_par_int("iter_restart") : 10;
        const double iter_tolerance = se_have_par("iter_tolerance")
            ? static_cast<double>(se_get_par_float("iter_tolerance")) : 1.0e-4;
        const std::string preconditioner_text = se_have_par("iter_preconditioner")
            ? std::string(se_get_par_str("iter_preconditioner"))
            : std::string("diagonal");
        const double ilut_drop = se_have_par("iter_ilut_drop_tolerance")
            ? static_cast<double>(se_get_par_float("iter_ilut_drop_tolerance")) : 1.0e-3;
        const int ilut_fill = se_have_par("iter_ilut_fill_factor")
            ? se_get_par_int("iter_ilut_fill_factor") : 10;
        const double iter_relaxation = se_have_par("iter_relaxation")
            ? static_cast<double>(se_get_par_float("iter_relaxation")) : -1.0;
        const double iter_max_relaxation = se_have_par("iter_max_relaxation")
            ? static_cast<double>(se_get_par_float("iter_max_relaxation")) : 1.0;

        const int write_interfaces = se_have_par("write_interfaces")
            ? se_get_par_int("write_interfaces") : 1;
        const int write_each_block = se_have_par("write_each_block")
            ? se_get_par_int("write_each_block") : 1;
        const int threads = se_have_par("threads")
            ? se_get_par_int("threads") : 32;

        if (!(frequency > 0.0f) || !(bf_tol > 0.0) || bf_leaf < 2 ||
            !(bf_sample > 0.0) || bf_knn < 0 || !(filter_dt > 0.0f) ||
            !(filter_length > 0.0f) || filter_lookup_subsamples < 1 ||
            iter_cycles < 1 || iter_iterations < 1 || iter_restart < 1 ||
            !(iter_tolerance > 0.0) || ilut_drop < 0.0 || ilut_fill < 1 ||
            iter_max_relaxation < 0.0 || iter_relaxation > iter_max_relaxation) {
            throw std::invalid_argument("invalid parameter value");
        }

        configure_threads(threads);

        const se::huygens::Model2D model =
            se::huygens::read_velocity_model(velocity);
        const se::huygens::BlockInfo info =
            se::huygens::read_block_info(block_file);
        se::huygens::validate_block_info(info, &model);

        if (info.blocks.empty())
            throw std::runtime_error("block file contains no blocks");

        if (source_iz != info.blocks.front().source_iz) {
            std::ostringstream message;
            message << "source_iz=" << source_iz
                    << " must equal first block source_iz="
                    << info.blocks.front().source_iz;
            throw std::runtime_error(message.str());
        }

        const int source_ix = source_ix_par >= 0 ? source_ix_par : model.nx / 2;
        if (source_ix < 0 || source_ix >= model.nx)
            throw std::invalid_argument("source_ix outside model");

        se::eigen::SolverOptions solver_options;
        solver_options.max_iterations = iter_iterations;
        solver_options.tolerance = iter_tolerance;
        solver_options.restart = iter_restart;
        solver_options.preconditioner =
            se::eigen::preconditioner_from_string(preconditioner_text);
        solver_options.ilut_drop_tolerance = ilut_drop;
        solver_options.ilut_fill_factor = ilut_fill;

        std::vector<Complex> wavefield(
            static_cast<std::size_t>(model.nx) * model.nz,
            Complex(0.0f, 0.0f));

        /* First block upper interface: point source, no Hankel initialization. */
        const std::vector<Complex> surface_source =
            make_point_source(model.nx, source_ix, source_amplitude);
        inject_row(model, wavefield, source_iz, surface_source);

        se::huygens::ensure_parent_directory(output_prefix + "_metrics.csv");
        std::ofstream metrics(output_prefix + "_metrics.csv");
        if (!metrics) throw std::runtime_error("cannot create metrics CSV");

        metrics << std::setprecision(12)
                << "block_id,cycle,source_iz,target_start_iz,target_end_iz,"
                   "bf_build_seconds,bf_apply_seconds,helmholtz_build_seconds,"
                   "correction_seconds,real_iterations,imag_iterations,"
                   "real_status,imag_status,residual_before,residual_after,"
                   "residual_ratio,relative_update,relaxation\n";

        std::cout
            << "Native ButterflyPACK interface propagation + block Helmholtz\n"
            << "frequency=" << frequency << " Hz, blocks=" << info.blocks.size()
            << ", overlap_rows=" << info.overlap_rows << '\n'
            << "ButterflyPACK: tol=" << bf_tol
            << ", leaf=" << bf_leaf
            << ", sample=" << bf_sample
            << ", knn=" << bf_knn << '\n'
            << "Local GMRES: cycles=" << iter_cycles
            << ", max_iter/cycle=" << iter_iterations
            << ", restart=" << iter_restart
            << ", tol=" << iter_tolerance
            << ", preconditioner="
            << se::eigen::to_string(solver_options.preconditioner) << '\n';

        if (info.overlap_rows == 0) {
            std::cout
                << "NOTE: overlap_rows=0. The next block uses the previous BF "
                   "lower boundary directly; local Helmholtz correction does not "
                   "modify that fixed Dirichlet row.\n";
        }

        for (const se::huygens::Block& block : info.blocks) {
            std::cout << "\n========== Block " << block.id << " ==========\n";

            /*
             * Read exactly the same tau file style as test_bf_kir.cpp:
             *   table_prefix + block id + _tau.rsf
             */
            const std::string tau_file =
                se::huygens::layer_table_filename(table_prefix, block.id, "tau");
            const auto t_read = Clock::now();
            const se::huygens::Table3D tau =
                se::huygens::read_table_rsf(tau_file);
            std::cout << "Traveltime: " << tau_file
                      << "  read=" << elapsed(t_read) << " s\n";

            const se::huygens::FrequencyKirchhoffFilter filter(
                frequency, filter_dt, filter_length,
                max_tau(tau), filter_lookup_subsamples);

            std::vector<Complex> top_trace;
            if (block.id == info.blocks.front().id) {
                top_trace = surface_source;
            } else {
                top_trace = gather_row(model, wavefield, block.source_iz);
            }

            double bf_build_seconds = 0.0;
            double bf_apply_seconds = 0.0;
            const std::vector<Complex> bottom_trace =
                propagate_interface_butterfly(
                    model, block, tau, filter, top_trace,
                    bf_tol, bf_leaf, bf_sample, bf_knn,
                    bf_build_seconds, bf_apply_seconds);

            std::cout << "ButterflyPACK: build=" << bf_build_seconds
                      << " s, apply=" << bf_apply_seconds << " s\n";

            std::vector<Complex> local_wavefield = wavefield;
            inject_row(model, local_wavefield, block.source_iz, top_trace);
            inject_row(model, local_wavefield, block.target_end_iz, bottom_trace);
            initialize_block_guess(model, block, wavefield, local_wavefield);

            const auto t_matrix = Clock::now();
            const LocalHelmholtzSystem system =
                build_local_system(model, block, frequency);
            const double matrix_seconds = elapsed(t_matrix);

            std::cout << "Helmholtz block: top=" << block.source_iz
                      << ", output=[" << block.target_start_iz
                      << ',' << block.target_end_iz << ']'
                      << ", unknowns=" << system.unknowns()
                      << ", matrix_build=" << matrix_seconds << " s\n";

            double first_residual = -1.0;
            for (int cycle = 0; cycle < iter_cycles; ++cycle) {
                const auto t_corr = Clock::now();
                const CorrectionMetrics c = correct_local_block(
                    model, system, local_wavefield, solver_options,
                    iter_relaxation, iter_max_relaxation);
                const double correction_seconds = elapsed(t_corr);

                if (cycle == 0) first_residual = c.residual_before;

                metrics << block.id << ',' << cycle << ','
                        << block.source_iz << ','
                        << block.target_start_iz << ','
                        << block.target_end_iz << ','
                        << bf_build_seconds << ','
                        << bf_apply_seconds << ','
                        << matrix_seconds << ','
                        << correction_seconds << ','
                        << c.real_iterations << ','
                        << c.imag_iterations << ','
                        << se::eigen::to_string(c.real_status) << ','
                        << se::eigen::to_string(c.imag_status) << ','
                        << c.residual_before << ','
                        << c.residual_after << ','
                        << c.residual_ratio << ','
                        << c.relative_update << ','
                        << c.relaxation << '\n';

                std::cout << "  cycle " << cycle
                          << ": residual " << c.residual_before
                          << " -> " << c.residual_after
                          << ", ratio=" << c.residual_ratio
                          << ", alpha=" << c.relaxation
                          << ", GMRES=" << c.real_iterations
                          << '/' << c.imag_iterations
                          << ", time=" << correction_seconds << " s\n";

                const double stop = iter_tolerance *
                    std::max(first_residual, std::numeric_limits<double>::min());
                if (c.residual_after <= stop) break;
            }

            commit_block_output(model, block, local_wavefield, wavefield);

            if (write_interfaces != 0) {
                write_trace(trace_filename(output_prefix, block.id, "top", "real"),
                            model, top_trace, frequency, block.id,
                            block.source_iz, "top", false);
                write_trace(trace_filename(output_prefix, block.id, "top", "imag"),
                            model, top_trace, frequency, block.id,
                            block.source_iz, "top", true);
                write_trace(trace_filename(output_prefix, block.id, "bottom_bf", "real"),
                            model, bottom_trace, frequency, block.id,
                            block.target_end_iz, "bottom_bf", false);
                write_trace(trace_filename(output_prefix, block.id, "bottom_bf", "imag"),
                            model, bottom_trace, frequency, block.id,
                            block.target_end_iz, "bottom_bf", true);
            }

            if (write_each_block != 0) {
                se::huygens::write_wavefield_component_rsf(
                    block_filename(output_prefix, block.id, "real"),
                    model, wavefield, frequency,
                    "native_butterflypack_block_helmholtz", false);
                se::huygens::write_wavefield_component_rsf(
                    block_filename(output_prefix, block.id, "imag"),
                    model, wavefield, frequency,
                    "native_butterflypack_block_helmholtz", true);
            }
        }

        metrics.close();

        se::huygens::write_wavefield_component_rsf(
            output_prefix + "_real.rsf", model, wavefield, frequency,
            "native_butterflypack_block_helmholtz", false);
        se::huygens::write_wavefield_component_rsf(
            output_prefix + "_imag.rsf", model, wavefield, frequency,
            "native_butterflypack_block_helmholtz", true);

        std::cout << "\nCompleted.\n"
                  << "  " << output_prefix << "_real.rsf\n"
                  << "  " << output_prefix << "_imag.rsf\n"
                  << "  " << output_prefix << "_metrics.csv\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::fprintf(stderr, "ray_helmholtz_iteration: %s\n", error.what());
        return 1;
    }
}
