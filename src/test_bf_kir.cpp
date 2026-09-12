#include <SERECKIRCH/include/huygens_sweep.hpp>
#include <SEFILESYSTEM/include/se_fs.h>

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
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Complex = std::complex<float>;
using NativeComplex = __complex__ float;
using Clock = std::chrono::steady_clock;


/* 返回从 start 到当前时刻的秒数。 */
double elapsed(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}


/* 设置 OpenMP 和 ButterflyPACK 线程数，BF_THREADS 可覆盖 main 中的默认线程数。 */
void configure_threads(int default_threads)
{
    const char* value = std::getenv("BF_THREADS");
    const int n = value ? std::atoi(value) : default_threads;
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


/* 将 std::complex<float> 转为 ButterflyPACK 使用的 C complex float。 */
NativeComplex to_native(const Complex& z)
{
    NativeComplex out;
    __real__ out = z.real();
    __imag__ out = z.imag();
    return out;
}


/* 将 ButterflyPACK 的 C complex float 转为 std::complex<float>。 */
Complex from_native(NativeComplex z)
{
    return {static_cast<float>(__real__ z), static_cast<float>(__imag__ z)};
}


/* 生成 1-D 点源：中间位置为 1，其余位置为 0。 */
std::vector<Complex> make_point_source(const se::huygens::Table3D& tau)
{
    std::vector<Complex> u(tau.nsource, Complex(0.0f, 0.0f));
    u[tau.nsource / 2] = Complex(1.0f, 0.0f);
    return u;
}


/* 生成 Kirchhoff 横向积分的梯形积分权重。 */
std::vector<float> make_weights(const se::huygens::Table3D& tau)
{
    std::vector<float> w(tau.nsource, tau.dx_source);
    w.front() *= 0.5f;
    w.back() *= 0.5f;
    return w;
}


/* 返回走时表中的最大走时。 */
float max_tau(const se::huygens::Table3D& tau)
{
    return *std::max_element(tau.values.begin(), tau.values.end());
}


/* 生成 ButterflyPACK 输出端 row 的二维坐标 (x,z)。 */
std::vector<double> make_row_coord(const se::huygens::Table3D& tau)
{
    const int rows = tau.nx_target * tau.nz_target;
    std::vector<double> coord(2 * rows);

    for (int row = 0; row < rows; ++row) {
        const int iz = row % tau.nz_target;
        const int ix = row / tau.nz_target;
        coord[2 * row] = tau.ox_target + ix * tau.dx_target;
        coord[2 * row + 1] = tau.oz_target + iz * tau.dz_target;
    }
    return coord;
}


/* 生成 ButterflyPACK 输入端 column 的二维坐标 (x,z)。 */
std::vector<double> make_col_coord(const se::huygens::Table3D& tau, float source_z)
{
    std::vector<double> coord(2 * tau.nsource);

    for (int src = 0; src < tau.nsource; ++src) {
        coord[2 * src] = tau.ox_source + src * tau.dx_source;
        coord[2 * src + 1] = source_z;
    }
    return coord;
}


/* 保存当前频率的 Kirchhoff/ButterflyPACK 参数以及行列重排。 */
struct BPackContext {
    const se::huygens::Table3D* tau = nullptr;
    const std::vector<float>* w = nullptr;
    const se::huygens::FrequencyKirchhoffFilter* filter = nullptr;

    float pi = 0.0f;
    float frequency = 0.0f;
    float source_z = 0.0f;

    double bf_tol = 0.0;
    double bf_sample = 0.0;
    int bf_leaf = 0;
    int bf_knn = 0;

    int rows = 0;
    int cols = 0;

    std::vector<int> row_new2old;
    std::vector<int> col_new2old;
};


/* 保存 ButterflyPACK C API 创建的对象句柄。 */
struct BPackHandles {
    F2Cptr ptree = nullptr, option = nullptr;
    F2Cptr row_stats = nullptr, col_stats = nullptr, bf_stats = nullptr;
    F2Cptr row_mat = nullptr, col_mat = nullptr;
    F2Cptr row_mesh = nullptr, col_mesh = nullptr, bf_mesh = nullptr;
    F2Cptr row_ker = nullptr, col_ker = nullptr, bf_ker = nullptr;
    F2Cptr bf = nullptr;

    int row_local = 0;
    int col_local = 0;
};


/* 计算单个 Kirchhoff 矩阵元素 K_ij。 */
Complex kirchhoff_entry(int row, int src, const BPackContext& ctx)
{
    const auto& tau = *ctx.tau;
    const int iz = row % tau.nz_target;
    const int ix = row / tau.nz_target;
    const float t = std::max(tau.values[tau.index(src, ix, iz)], 1.0e-8f);

    const float xt = tau.ox_target + ix * tau.dx_target;
    const float zt = tau.oz_target + iz * tau.dz_target;
    const float xs = tau.ox_source + src * tau.dx_source;

    const float dx = xt - xs;
    const float dz = zt - ctx.source_z;
    const float r2 = std::max(dx * dx + dz * dz, 1.0e-20f);

    // K_ij = dx_j/pi * Delta_z*tau/R^2 * H(tau,omega)
    return (*ctx.w)[src] * dz * t / (ctx.pi * r2) * ctx.filter->response(t);
}


/* ButterflyPACK 几何建树使用的占位 distance 回调。 */
void dummy_distance(int*, int*, double* value, C2Fptr)
{
    *value = 0.0;
}


/* ButterflyPACK 几何建树使用的占位 near/far 回调。 */
void dummy_near_far(int*, int*, int* value, C2Fptr)
{
    *value = 0;
}


/* 根据 main 中给出的参数设置 ButterflyPACK 压缩选项。 */
void set_bpack_options(F2Cptr* option, const BPackContext& ctx)
{
    auto D = [&](const char* key, double value) {
        c_c_bpack_set_D_option(option, key, value);
    };

    auto I = [&](const char* key, int value) {
        c_c_bpack_set_I_option(option, key, value);
    };

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


/* ButterflyPACK 单元素回调：将重排索引映射回原 Kirchhoff 矩阵索引。 */
void bpack_sample(int* a, int* b, NativeComplex* value, C2Fptr ptr)
{
    auto* ctx = static_cast<BPackContext*>(ptr);

    const int row_new = (*a > 0) ? *a : *b;
    const int col_new = (*a > 0) ? -*b : -*a;

    const int row = ctx->row_new2old[row_new - 1] - 1;
    const int col = ctx->col_new2old[col_new - 1] - 1;

    *value = to_native(kirchhoff_entry(row, col, *ctx));
}


/* ButterflyPACK 块元素回调：OpenMP 并行计算一批 Kirchhoff 矩阵元素。 */
void bpack_sample_block(int* nblock, int*, int*, std::int64_t*,
                        int* all_rows, int* all_cols, NativeComplex* values,
                        int* nrows, int* ncols, int*, int*, int*, C2Fptr ptr)
{
    auto* ctx = static_cast<BPackContext*>(ptr);
    std::int64_t roff = 0, coff = 0, voff = 0;

    for (int ib = 0; ib < *nblock; ++ib) {
        const int nr = nrows[ib];
        const int nc = ncols[ib];

        std::vector<int> rows(nr), cols(nc);

        for (int r = 0; r < nr; ++r)
            rows[r] = ctx->row_new2old[all_rows[roff + r] - 1] - 1;

        for (int c = 0; c < nc; ++c)
            cols[c] = ctx->col_new2old[all_cols[coff + c] - 1] - 1;

        const std::int64_t count = static_cast<std::int64_t>(nr) * nc;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096)
#endif
        for (std::int64_t k = 0; k < count; ++k) {
            const int c = static_cast<int>(k / nr);
            const int r = static_cast<int>(k % nr);
            values[voff + k] = to_native(kirchhoff_entry(rows[r], cols[c], *ctx));
        }

        roff += nr;
        coff += nc;
        voff += count;
    }
}


/* 释放 ButterflyPACK C API 创建的对象。 */
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


/* 完成 row/column 建树、Butterfly 初始化和矩阵压缩。 */
void build_butterfly(BPackContext& ctx, BPackHandles& h,
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

    ctx.row_new2old.resize(ctx.rows);
    ctx.col_new2old.resize(ctx.cols);

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


/* 调用 c_c_bf_mult 完成 Butterfly 矩阵乘并恢复原始 target 排列。 */
std::vector<Complex> apply_butterfly(const BPackContext& ctx,
                                     BPackHandles& h,
                                     const std::vector<Complex>& input)
{
    std::vector<NativeComplex> x(ctx.cols), y(ctx.rows);

    for (int i = 0; i < ctx.cols; ++i)
        x[i] = to_native(input[ctx.col_new2old[i] - 1]);

    char trans = 'N';
    int nin = ctx.cols;
    int nout = ctx.rows;
    int nrhs = 1;

    c_c_bf_mult(
        &trans, x.data(), y.data(),
        &nin, &nout, &nrhs,
        &h.bf, &h.option, &h.bf_stats, &h.ptree);

    std::vector<Complex> output(ctx.rows);

    for (int i = 0; i < ctx.rows; ++i)
        output[ctx.row_new2old[i] - 1] = from_native(y[i]);

    return output;
}


/* 将 float 数组写成与 target 区域一致的 2-D RSF。 */
void write_2d(const std::string& file,
              const se::huygens::Table3D& tau,
              const std::vector<float>& data)
{
    sep_t* out = sep_open(file.c_str(), SEP_WRITE, 0);

    out->headers->ndim = 2;
    out->headers->n[0] = tau.nz_target;
    out->headers->n[1] = tau.nx_target;
    out->headers->d[0] = tau.dz_target;
    out->headers->d[1] = tau.dx_target;
    out->headers->o[0] = tau.oz_target;
    out->headers->o[1] = tau.ox_target;
    out->headers->esize = 4;
    out->headers->le = 1;

    sep_set_header(out, "data_format", "native_float");
    sep_set_header_int(out, "esize", 4);
    se_fsio_write_float(out->data->io, data.data(), data.size());
    sep_close(out);
}


/* 提取 Butterfly 复波场的实部或虚部并写成 RSF。 */
void write_bf(const std::string& file,
              const se::huygens::Table3D& tau,
              const std::vector<Complex>& bf,
              bool imag)
{
    std::vector<float> data(bf.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(bf.size()); ++i)
        data[i] = imag ? bf[i].imag() : bf[i].real();

    write_2d(file, tau, data);
}


/* 主程序：所有运行参数集中在这里，只输出 Butterfly 波场实部和虚部。 */
int main()
{
    /* 基本参数。 */
    constexpr float PI = 3.14159265358979323846f;
    constexpr float FREQUENCY = 25.0f;
    constexpr float FILTER_DT = 0.001f;
    constexpr float FILTER_LENGTH = 0.025f;
    constexpr int BLOCK_ID = 1;
    constexpr int DEFAULT_THREADS = 32;

    /* ButterflyPACK 参数。 */
    constexpr double BF_TOL = 1.0e-4;
    constexpr int BF_LEAF = 64;
    constexpr double BF_SAMPLE = 0.1;
    constexpr int BF_KNN = 10;

    /* 输入输出文件。 */
    const std::string BLOCK_FILE = "block_info.dat";
    const std::string TABLE_PREFIX = "huygens_tt/travel";
    const std::string BF_REAL = "test_bf_block1_bf_real.rsf";
    const std::string BF_IMAG = "test_bf_block1_bf_imag.rsf";

    INFO(("Program started"));
    configure_threads(DEFAULT_THREADS);

    const auto t_read = Clock::now();
    const auto info = se::huygens::read_block_info(BLOCK_FILE);
    const auto& block = info.blocks[BLOCK_ID];

    const std::string tau_file =
        se::huygens::layer_table_filename(TABLE_PREFIX, BLOCK_ID, "tau");
    const auto tau = se::huygens::read_table_rsf(tau_file);
    INFO(("Traveltime loaded %f", elapsed(t_read)));

    const int rows = tau.nx_target * tau.nz_target;
    const int cols = tau.nsource;
    const float source_z = info.oz + block.source_iz * info.dz;

    const auto input = make_point_source(tau);
    const auto weights = make_weights(tau);

    se::huygens::FrequencyKirchhoffFilter filter(
        FREQUENCY, FILTER_DT, FILTER_LENGTH, max_tau(tau));

    auto row_coord = make_row_coord(tau);
    auto col_coord = make_col_coord(tau, source_z);

    BPackContext ctx;
    ctx.tau = &tau;
    ctx.w = &weights;
    ctx.filter = &filter;
    ctx.pi = PI;
    ctx.frequency = FREQUENCY;
    ctx.source_z = source_z;
    ctx.bf_tol = BF_TOL;
    ctx.bf_leaf = BF_LEAF;
    ctx.bf_sample = BF_SAMPLE;
    ctx.bf_knn = BF_KNN;
    ctx.rows = rows;
    ctx.cols = cols;

    BPackHandles h;

    const auto t_build = Clock::now();
    build_butterfly(ctx, h, row_coord, col_coord);
    INFO(("Butterfly build completed %f", elapsed(t_build)));

    const auto t_apply = Clock::now();
    const auto bf = apply_butterfly(ctx, h, input);
    INFO(("Butterfly apply completed %f", elapsed(t_apply)));

    const auto t_write = Clock::now();
    write_bf(BF_REAL, tau, bf, false);
    write_bf(BF_IMAG, tau, bf, true);
    INFO(("BF real/imag written %f", elapsed(t_write)));

    destroy_bpack(h);
    INFO(("Program finished"));

    return 0;
}
