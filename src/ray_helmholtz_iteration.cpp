/*
 * 这是使用项目 se_par_* 参数接口的最终版本。
 * 本文件不包含 Arguments 类、Parameters 结构体或自定义 key=value 解析器。
 */

#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_par_sep.h>
#include <SERECKIRCH/include/huygens_sweep.hpp>
#include <global_preconditioned_gmres.hpp>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/QR>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

/*
 * 程序功能
 * --------
 * 1. 使用项目已有的 eFMM 计算点源首波走时；
 * 2. 根据首波走时构造指定频率的射线近似波场；
 * 3. 在震源下方固定一段射线场，把它作为向下延拓的边界；
 * 4. 使用重叠深度块的向下扫掠预条件器和截断 FGMRES 做少量校正；
 * 5. 输出初始场、每次迭代后的总场以及累计校正场。
 *
 * 程序采用面向过程的组织方式。命令行参数通过项目已有的
 * se_par_init、se_have_par、se_get_par_* 函数读取。
 */

namespace gpg = global_preconditioned_gmres;

typedef gpg::Complex Complex;
typedef gpg::Sparse Sparse;
typedef gpg::Vector Vector;
typedef std::chrono::steady_clock Clock;

static const double PI = 3.141592653589793238462643383279502884;

/*
 * 与仓库中 point_ray 使用的射线相位常数保持一致。
 * 射线波场采用 exp[i(omega*T-3*pi/4)] 的形式。
 */
static const double RAY_PHASE = -0.75 * PI;


/*
 * 打印程序参数说明。
 *
 * 除了 -h 和 --help 外，其他参数都采用项目已有的 key=value 形式，
 * 例如：frequency=15 source_ix=512 max_iterations=4。
 */
static void print_help(const char* program_name)
{
    std::printf(
        "Usage:\n"
        "  %s [key=value ...]\n\n"
        "Parameters:\n"
        "  velocity=model/vmar.rsf\n"
        "  output_prefix=output/ray_helmholtz\n"
        "  frequency=15\n"
        "  source_amplitude=1\n"
        "  source_ix=-1                 (-1 means nx/2)\n"
        "  source_iz=1\n"
        "  max_iterations=4             (truncated FGMRES steps)\n"
        "  tolerance=0.15               (relative to initial defect)\n"
        "  anchor_rows=8                (fixed ray rows below source)\n"
        "  sweep_block_rows=32\n"
        "  sweep_overlap_rows=8         (must be >= 4 for FD8)\n"
        "  shift_beta=0.10\n"
        "  ilut_drop_tolerance=1e-2\n"
        "  ilut_fill_factor=10\n"
        "  scale_radius=4\n"
        "  write_iterations=1           (0 avoids iteration I/O)\n"
        "  write_correction=1\n\n"
        "Outputs:\n"
        "  PREFIX_traveltime.rsf\n"
        "  PREFIX_iter_NNN_real.rsf / PREFIX_iter_NNN_imag.rsf\n"
        "  PREFIX_correction_NNN_real.rsf / _imag.rsf\n"
        "  PREFIX_metrics.csv\n",
        program_name);
}


/*
 * -h 和 --help 不是 key=value 参数，因此只在这里单独判断。
 * 数值参数和文件名参数仍全部交给 se_par_* 系列函数读取。
 */
static int have_help_argument(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == NULL) continue;

        if (std::strcmp(argv[i], "-h") == 0 ||
            std::strcmp(argv[i], "--help") == 0) {
            return 1;
        }
    }

    if (se_have_par("help")) {
        return se_get_par_int("help") != 0;
    }

    return 0;
}


/* 返回从 started 到当前时刻经过的秒数。 */
static double elapsed_seconds(const Clock::time_point& started)
{
    return std::chrono::duration<double>(Clock::now() - started).count();
}


/*
 * 在速度模型四周增加吸收层。
 *
 * 原始模型位于：
 *
 *     iz = pml ... pml+nz-1
 *     ix = pml ... pml+nx-1
 *
 * 吸收层中的速度取距离它最近的物理网格点速度，避免人为引入
 * 额外的速度突变。
 */
static std::vector<float> pad_velocity(
    const se::huygens::Model2D& model)
{
    int pml = gpg::kPml;
    int padded_nz = model.nz + 2 * pml;
    int padded_nx = model.nx + 2 * pml;

    std::vector<float> padded_velocity(
        static_cast<std::size_t>(padded_nz) * padded_nx);

    for (int iz = 0; iz < padded_nz; ++iz) {
        int model_iz = std::clamp(iz - pml, 0, model.nz - 1);

        for (int ix = 0; ix < padded_nx; ++ix) {
            int model_ix = std::clamp(ix - pml, 0, model.nx - 1);
            int padded_index = gpg::index(iz, ix, padded_nx);

            padded_velocity[static_cast<std::size_t>(padded_index)] =
                model.velocity[model.index(model_ix, model_iz)];
        }
    }

    return padded_velocity;
}


/*
 * 根据首波走时构造单频射线近似波场。
 *
 * 波场形式为
 *
 *     u_ray(x,z) = A(T) exp[i(omega*T-3*pi/4)],
 *
 * 其中二维渐近振幅取
 *
 *     A(T) = 1 / sqrt(8*pi*omega*T).
 *
 * 震源点处 T=0 会导致振幅奇异，因此使用一个网格间距对应的
 * 最小传播时间进行正则化。随后还会通过 calculate_ray_scale()
 * 将射线场与离散点源的振幅和相位进一步匹配。
 */
static Vector build_ray_wavefield(
    const se::huygens::Model2D& model,
    const std::vector<float>& traveltime,
    float frequency,
    float source_amplitude,
    int source_ix,
    int source_iz)
{
    std::size_t ngrid =
        static_cast<std::size_t>(model.nz) * model.nx;

    if (traveltime.size() != ngrid) {
        throw std::runtime_error(
            "traveltime size does not match velocity model");
    }

    double omega = 2.0 * PI * frequency;
    double source_velocity =
        model.velocity[model.index(source_ix, source_iz)];

    /* 一个网格间距对应的传播时间，用于震源奇异性正则化。 */
    double minimum_time =
        std::max(model.dx, model.dz) / source_velocity;

    Vector ray_wavefield(static_cast<Eigen::Index>(ngrid));

    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            std::size_t index = model.index(ix, iz);
            double tau = std::max(
                0.0, static_cast<double>(traveltime[index]));

            if (!std::isfinite(tau)) {
                throw std::runtime_error(
                    "eFMM returned a non-finite traveltime");
            }

            double effective_time = std::max(tau, minimum_time);
            double amplitude = source_amplitude /
                std::sqrt(8.0 * PI * omega * effective_time);
            double phase = omega * tau + RAY_PHASE;

            ray_wavefield[static_cast<Eigen::Index>(index)] =
                std::polar(amplitude, phase);
        }
    }

    return ray_wavefield;
}


/*
 * 计算射线初场在吸收层中的余弦平方衰减系数。
 *
 * 物理区域内部返回 1；从物理边界向外逐渐减小；计算区域最外侧
 * 返回 0。这样可以减少未经处理的射线初场在外边界产生的残差。
 */
static double pml_taper(int coordinate, int physical_count)
{
    int pml = gpg::kPml;
    int physical_first = pml;
    int physical_last = pml + physical_count - 1;
    int outside_distance = 0;

    if (coordinate < physical_first) {
        outside_distance = physical_first - coordinate;
    }
    if (coordinate > physical_last) {
        outside_distance = coordinate - physical_last;
    }

    if (outside_distance == 0) return 1.0;

    double angle = 0.5 * PI *
        static_cast<double>(outside_distance) / pml;
    angle = std::min(angle, 0.5 * PI);

    double value = std::cos(angle);
    return value * value;
}


/*
 * 将物理模型中的射线场扩展到带吸收层的计算区域。
 *
 * 物理区域之外先复制最近边界处的波场，再乘以 pml_taper()，
 * 使初始波场在计算区域最外侧平滑衰减至零。
 */
static Vector pad_wavefield(
    const Vector& physical_wavefield,
    const se::huygens::Model2D& model)
{
    int pml = gpg::kPml;
    int padded_nz = model.nz + 2 * pml;
    int padded_nx = model.nx + 2 * pml;

    Vector padded_wavefield(
        static_cast<Eigen::Index>(padded_nz) * padded_nx);

    for (int iz = 0; iz < padded_nz; ++iz) {
        int model_iz = std::clamp(iz - pml, 0, model.nz - 1);
        double taper_z = pml_taper(iz, model.nz);

        for (int ix = 0; ix < padded_nx; ++ix) {
            int model_ix = std::clamp(ix - pml, 0, model.nx - 1);
            double taper_x = pml_taper(ix, model.nx);
            double taper = taper_z * taper_x;

            int padded_index = gpg::index(iz, ix, padded_nx);
            int physical_index =
                gpg::index(model_iz, model_ix, model.nx);

            padded_wavefield[padded_index] =
                taper * physical_wavefield[physical_index];
        }
    }

    return padded_wavefield;
}


/*
 * 在震源附近的小窗口内匹配射线场与离散点源。
 *
 * 设 v=A*u_ray，通过求解下面的一维复数最小二乘问题
 *
 *     min ||b-alpha*v||_2,
 *
 * 得到
 *
 *     alpha = v^H b / v^H v.
 *
 * 仅在震源附近拟合，是为了避免远处的高频渐近误差把射线场整体
 * 压缩得过小。
 */
static Complex calculate_ray_scale(
    const Vector& ray_action,
    const Vector& rhs,
    int padded_nz,
    int padded_nx,
    int source_iz,
    int source_ix,
    int scale_radius)
{
    Complex numerator(0.0, 0.0);
    double denominator = 0.0;

    int iz_begin = std::max(0, source_iz - scale_radius);
    int iz_end = std::min(padded_nz - 1, source_iz + scale_radius);
    int ix_begin = std::max(0, source_ix - scale_radius);
    int ix_end = std::min(padded_nx - 1, source_ix + scale_radius);

    for (int iz = iz_begin; iz <= iz_end; ++iz) {
        for (int ix = ix_begin; ix <= ix_end; ++ix) {
            int index = gpg::index(iz, ix, padded_nx);

            numerator += std::conj(ray_action[index]) * rhs[index];
            denominator += std::norm(ray_action[index]);
        }
    }

    if (denominator <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error(
            "cannot calculate the ray-field normalization factor");
    }

    return numerator / denominator;
}


/*
 * 构造 shifted-Helmholtz 预条件矩阵
 *
 *     M = A + i*beta*k^2.
 *
 * 附加的复数偏移增强了阻尼，可以提高 Helmholtz 矩阵进行 ILUT
 * 分解时的稳定性。M 只作为预条件器使用，真实残差始终用原始
 * Helmholtz 矩阵 A 计算。
 */
static Sparse build_shifted_helmholtz(
    const Sparse& helmholtz,
    const std::vector<float>& velocity,
    int nz,
    int nx,
    float frequency,
    float shift_beta)
{
    Sparse shifted_helmholtz = helmholtz;
    double omega = 2.0 * PI * frequency;

    for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            int index = gpg::index(iz, ix, nx);
            double velocity_value = std::max(
                static_cast<double>(
                    velocity[static_cast<std::size_t>(index)]),
                std::numeric_limits<double>::epsilon());
            double wavenumber = omega / velocity_value;
            double k2 = wavenumber * wavenumber;

            shifted_helmholtz.coeffRef(index, index) +=
                Complex(0.0, shift_beta * k2);
        }
    }

    shifted_helmholtz.makeCompressed();
    return shifted_helmholtz;
}


/*
 * 从稀疏矩阵中提取连续未知量对应的主子矩阵。
 *
 * 深度方向采用整行网格划分，因此一个深度块在一维编号中恰好是
 * 连续区间。显式提取可以避免依赖 Eigen 稀疏 block 表达式的具体
 * 求值行为，也便于后面重复构造重叠深度块。
 */
static Sparse extract_principal_block(
    const Sparse& matrix,
    int first,
    int count)
{
    if (first < 0 || count < 1 || first + count > matrix.rows()) {
        throw std::runtime_error("invalid sparse principal block");
    }

    std::vector<gpg::Triplet> entries;
    entries.reserve(static_cast<std::size_t>(
        matrix.nonZeros() * static_cast<double>(count) / matrix.rows()));

    int last = first + count;

    for (int column = first; column < last; ++column) {
        for (Sparse::InnerIterator value(matrix, column); value; ++value) {
            if (value.row() >= first && value.row() < last) {
                entries.emplace_back(
                    value.row() - first,
                    column - first,
                    value.value());
            }
        }
    }

    Sparse block(count, count);
    block.setFromTriplets(entries.begin(), entries.end());
    block.makeCompressed();
    return block;
}


/*
 * 一个重叠深度块只保存索引和已经构造好的 ILUT 因子。
 *
 * core 区间在所有块之间互不重叠，extended 区间在 core 两侧增加
 * overlap 行。局部方程在 extended 区间求解，但只把 core 部分写回，
 * 即 restricted additive Schwarz (RAS) 的限制写回方式。
 */
struct SweepBlock {
    int core_first = 0;
    int core_last = 0;
    int extended_first = 0;
    int extended_count = 0;
    std::unique_ptr<Eigen::IncompleteLUT<Complex, int>> ilut;
};


/*
 * 为向下扫掠预条件器建立所有局部 ILUT 因子。
 *
 * 与全局 ILUT 相比，每个因子只对应几十个深度网格行，因而建立
 * 更快、峰值内存更小。复数偏移只用于局部预条件矩阵；FGMRES 的
 * Arnoldi 运算和真实残差始终使用未偏移的 Helmholtz 算子。
 */
static std::vector<SweepBlock> build_downward_sweep_blocks(
    const Sparse& active_helmholtz,
    const std::vector<float>& active_velocity,
    int active_nz,
    int nx,
    float frequency,
    int block_rows,
    int overlap_rows,
    float shift_beta,
    float ilut_drop_tolerance,
    int ilut_fill_factor)
{
    Sparse shifted = build_shifted_helmholtz(
        active_helmholtz,
        active_velocity,
        active_nz,
        nx,
        frequency,
        shift_beta);

    std::vector<SweepBlock> blocks;

    for (int core_first_z = 0;
         core_first_z < active_nz;
         core_first_z += block_rows) {

        int core_last_z = std::min(active_nz, core_first_z + block_rows);
        int extended_first_z =
            std::max(0, core_first_z - overlap_rows);
        int extended_last_z =
            std::min(active_nz, core_last_z + overlap_rows);

        SweepBlock block;
        block.core_first = core_first_z * nx;
        block.core_last = core_last_z * nx;
        block.extended_first = extended_first_z * nx;
        block.extended_count =
            (extended_last_z - extended_first_z) * nx;

        Sparse local_matrix = extract_principal_block(
            shifted,
            block.extended_first,
            block.extended_count);

        block.ilut =
            std::make_unique<Eigen::IncompleteLUT<Complex, int>>();
        block.ilut->setDroptol(ilut_drop_tolerance);
        block.ilut->setFillfactor(ilut_fill_factor);
        block.ilut->compute(local_matrix);

        if (block.ilut->info() != Eigen::Success) {
            throw std::runtime_error(
                "local sweep ILUT construction failed; increase "
                "shift_beta or ilut_fill_factor");
        }

        blocks.push_back(std::move(block));
    }

    return blocks;
}


/*
 * 对一个向量应用一次自上而下的乘法 RAS 扫掠。
 *
 * work 保存尚未解释的残差。每个局部解只写回本块 core 区域，随后
 * 立即从 work 中减去真实算子 A 作用于该更新的结果。下一个深度块
 * 因而接收到上方块传来的界面残差。由于只从浅到深走一遍，不进行
 * 反向扫掠，预条件器会优先传递下行能量，符合本程序只快速补充
 * 下行多路径的目标。
 *
 * 矩阵为列主序。所有 core 列互不重叠，因此一整次扫掠对 A 的
 * 非零元总访问量约等于一次稀疏矩阵向量乘法，而不是“块数次”全局
 * 矩阵向量乘法。
 */
static Vector apply_downward_sweep(
    const Sparse& active_helmholtz,
    const std::vector<SweepBlock>& blocks,
    const Vector& rhs)
{
    Vector correction = Vector::Zero(rhs.size());
    Vector work = rhs;

    for (std::size_t iblock = 0; iblock < blocks.size(); ++iblock) {
        const SweepBlock& block = blocks[iblock];

        Vector local_rhs = work.segment(
            block.extended_first,
            block.extended_count);
        Vector local_solution = block.ilut->solve(local_rhs);

        if (block.ilut->info() != Eigen::Success) {
            throw std::runtime_error("local sweep ILUT solve failed");
        }

        for (int column = block.core_first;
             column < block.core_last;
             ++column) {

            Complex update = local_solution[
                column - block.extended_first];
            correction[column] += update;

            if (update == Complex(0.0, 0.0)) continue;

            for (Sparse::InnerIterator value(active_helmholtz, column);
                 value;
                 ++value) {
                work[value.row()] -= value.value() * update;
            }
        }
    }

    return correction;
}


/* 把活动区域中的校正量放回带 PML 的完整计算网格。 */
static Vector make_padded_correction(
    const Vector& active_correction,
    int padded_count,
    int active_offset)
{
    Vector padded = Vector::Zero(padded_count);
    padded.segment(active_offset, active_correction.size()) =
        active_correction;
    return padded;
}


/* 从带吸收层的计算区域中截取原始物理模型区域。 */
static std::vector<se::huygens::Complex> crop_wavefield(
    const Vector& padded_wavefield,
    const se::huygens::Model2D& model)
{
    int pml = gpg::kPml;
    int padded_nx = model.nx + 2 * pml;

    std::vector<se::huygens::Complex> physical_wavefield(
        static_cast<std::size_t>(model.nz) * model.nx);

    for (int iz = 0; iz < model.nz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            int padded_index =
                gpg::index(iz + pml, ix + pml, padded_nx);
            Complex value = padded_wavefield[padded_index];

            physical_wavefield[model.index(ix, iz)] =
                se::huygens::Complex(
                    static_cast<float>(value.real()),
                    static_cast<float>(value.imag()));
        }
    }

    return physical_wavefield;
}


/*
 * 输出某次迭代的复数波场。
 *
 * 例如 field_name="iter"、iteration=2 时输出：
 *
 *     PREFIX_iter_002_real.rsf
 *     PREFIX_iter_002_imag.rsf
 */
static void write_wavefield(
    const char* output_prefix,
    const char* field_name,
    int iteration,
    const se::huygens::Model2D& model,
    const Vector& padded_wavefield,
    float frequency,
    const char* method)
{
    char real_file[4096];
    char imag_file[4096];

    std::snprintf(
        real_file,
        sizeof(real_file),
        "%s_%s_%03d_real.rsf",
        output_prefix,
        field_name,
        iteration);
    std::snprintf(
        imag_file,
        sizeof(imag_file),
        "%s_%s_%03d_imag.rsf",
        output_prefix,
        field_name,
        iteration);

    std::vector<se::huygens::Complex> physical_wavefield =
        crop_wavefield(padded_wavefield, model);

    se::huygens::write_wavefield_component_rsf(
        real_file,
        model,
        physical_wavefield,
        frequency,
        method,
        false);
    se::huygens::write_wavefield_component_rsf(
        imag_file,
        model,
        physical_wavefield,
        frequency,
        method,
        true);
}


int main(int argc, char** argv)
{
    FILE* metrics_file = NULL;

    /*
     * 与 src 中其他程序保持一致：首先初始化项目自带的参数系统，
     * 后续通过 se_have_par() 和 se_get_par_*() 读取 key=value 参数。
     */
    se_par_init(argc, argv);

    try {
        if (have_help_argument(argc, argv)) {
            print_help(argv[0]);
            return 0;
        }

        /* -------------------- 1. 读取命令行参数 -------------------- */

        const char* velocity_file = se_have_par("velocity")
            ? se_get_par_str("velocity")
            : "model/vmar.rsf";
        const char* output_prefix = se_have_par("output_prefix")
            ? se_get_par_str("output_prefix")
            : "output/ray_helmholtz";
        float frequency = se_have_par("frequency")
            ? se_get_par_float("frequency") : 15.0f;
        float source_amplitude = se_have_par("source_amplitude")
            ? se_get_par_float("source_amplitude") : 1.0f;
        float tolerance = se_have_par("tolerance")
            ? se_get_par_float("tolerance") : 0.15f;
        float shift_beta = se_have_par("shift_beta")
            ? se_get_par_float("shift_beta") : 0.10f;
        float ilut_drop_tolerance = se_have_par("ilut_drop_tolerance")
            ? se_get_par_float("ilut_drop_tolerance") : 1.0e-2f;

        int source_ix = se_have_par("source_ix")
            ? se_get_par_int("source_ix") : -1;
        int source_iz = se_have_par("source_iz")
            ? se_get_par_int("source_iz") : 1;
        int max_iterations = se_have_par("max_iterations")
            ? se_get_par_int("max_iterations") : 4;
        int anchor_rows = se_have_par("anchor_rows")
            ? se_get_par_int("anchor_rows") : 8;
        int sweep_block_rows = se_have_par("sweep_block_rows")
            ? se_get_par_int("sweep_block_rows") : 32;
        int sweep_overlap_rows = se_have_par("sweep_overlap_rows")
            ? se_get_par_int("sweep_overlap_rows") : 8;
        int scale_radius = se_have_par("scale_radius")
            ? se_get_par_int("scale_radius") : 4;
        int ilut_fill_factor = se_have_par("ilut_fill_factor")
            ? se_get_par_int("ilut_fill_factor") : 10;
        int write_iterations = se_have_par("write_iterations")
            ? se_get_par_int("write_iterations") : 1;
        int write_correction = se_have_par("write_correction")
            ? se_get_par_int("write_correction") : 1;

        if (frequency <= 0.0f || source_amplitude == 0.0f ||
            tolerance <= 0.0f || tolerance >= 1.0f ||
            shift_beta < 0.0f || ilut_drop_tolerance <= 0.0f ||
            ilut_fill_factor < 1 || max_iterations < 1 ||
            anchor_rows < 5 || sweep_block_rows < 8 ||
            sweep_overlap_rows < 4 ||
            sweep_overlap_rows >= sweep_block_rows ||
            scale_radius < 0) {
            throw std::runtime_error("invalid command-line parameter");
        }

        /* -------------------- 2. 读取RSF速度模型 -------------------- */

        se::huygens::Model2D model =
            se::huygens::read_velocity_model(velocity_file);

        /* source_ix=-1 时将震源放在模型横向中心。 */
        if (source_ix < 0) source_ix = model.nx / 2;

        if (source_ix < 0 || source_ix >= model.nx ||
            source_iz < 0 || source_iz >= model.nz) {
            throw std::runtime_error(
                "source index is outside the velocity model");
        }

        if (source_iz + anchor_rows >= model.nz - 4) {
            throw std::runtime_error(
                "anchor_rows leaves no usable correction domain");
        }

        std::printf(
            "model: nz=%d nx=%d dz=%g dx=%g\n",
            model.nz,
            model.nx,
            model.dz,
            model.dx);
        std::printf(
            "source: iz=%d ix=%d frequency=%g Hz\n",
            source_iz,
            source_ix,
            frequency);
        std::printf(
            "method: ray anchor + downward RAS sweep + FGMRES(%d)\n",
            max_iterations);

        /* -------------------- 3. 计算并输出首波走时 -------------------- */

        Clock::time_point fmm_start = Clock::now();

        std::vector<float> traveltime =
            se::huygens::solve_fmm(model, source_ix, source_iz);

        double fmm_seconds = elapsed_seconds(fmm_start);

        char traveltime_file[4096];
        std::snprintf(
            traveltime_file,
            sizeof(traveltime_file),
            "%s_traveltime.rsf",
            output_prefix);

        se::huygens::write_image_rsf(
            traveltime_file,
            model,
            traveltime,
            "efmm_first_arrival_traveltime",
            {{"source_ix", static_cast<float>(source_ix)},
             {"source_iz", static_cast<float>(source_iz)}});

        /* -------------------- 4. 构造全局Helmholtz系统 -------------------- */

        int padded_nz = model.nz + 2 * gpg::kPml;
        int padded_nx = model.nx + 2 * gpg::kPml;
        int padded_source_iz = source_iz + gpg::kPml;
        int padded_source_ix = source_ix + gpg::kPml;

        std::vector<float> padded_velocity = pad_velocity(model);

        Clock::time_point matrix_start = Clock::now();

        /*
         * 复用 global_preconditioned_gmres.hpp 中的 FD8 Helmholtz
         * 离散算子。此处只复用矩阵构造，不调用其中的 GMRES。
         */
        Sparse helmholtz = gpg::build_helmholtz(
            padded_velocity,
            padded_nz,
            padded_nx,
            model.dz,
            model.dx,
            frequency,
            false);

        double matrix_seconds = elapsed_seconds(matrix_start);

        /* 离散点源右端项，震源位于扩展后计算区域中的对应位置。 */
        Vector rhs = Vector::Zero(helmholtz.rows());
        int source_index =
            gpg::index(padded_source_iz, padded_source_ix, padded_nx);
        rhs[source_index] = source_amplitude /
            (static_cast<double>(model.dx) * model.dz);

        /* -------------------- 5. 构造并归一化射线初场 -------------------- */

        Vector physical_ray_wavefield = build_ray_wavefield(
            model,
            traveltime,
            frequency,
            source_amplitude,
            source_ix,
            source_iz);

        Vector initial_wavefield =
            pad_wavefield(physical_ray_wavefield, model);

        Vector ray_action = helmholtz * initial_wavefield;

        Complex ray_scale = calculate_ray_scale(
            ray_action,
            rhs,
            padded_nz,
            padded_nx,
            padded_source_iz,
            padded_source_ix,
            scale_radius);

        initial_wavefield *= ray_scale;

        /* 第0次结果就是经过复数尺度匹配后的射线初场。 */
        Vector wavefield = initial_wavefield;

        write_wavefield(
            output_prefix,
            "iter",
            0,
            model,
            wavefield,
            frequency,
            "fmm_ray_initial_wavefield");

        /* -------------------- 6. 建立射线锚定的活动校正方程 -------------------- */

        /*
         * 震源及其下方 anchor_rows 行完全保留射线场，不参与校正。
         * 这样可以避免点源奇异性和射线振幅误差占据最前面的迭代，
         * 同时把保留下来的射线场作为下部 Helmholtz 方程的边界数据。
         */
        int active_first_model_iz = source_iz + anchor_rows;
        int active_first_padded_iz =
            active_first_model_iz + gpg::kPml;
        int active_offset = active_first_padded_iz * padded_nx;
        int active_nz = padded_nz - active_first_padded_iz;
        int active_count = active_nz * padded_nx;

        Sparse active_helmholtz = extract_principal_block(
            helmholtz,
            active_offset,
            active_count);

        Vector full_initial_residual = rhs - helmholtz * initial_wavefield;
        Vector correction_rhs = full_initial_residual.segment(
            active_offset,
            active_count);

        std::vector<float> active_velocity(
            padded_velocity.begin() + active_offset,
            padded_velocity.end());

        double initial_defect_norm = correction_rhs.norm();
        double defect_scale = std::max(
            initial_defect_norm,
            std::numeric_limits<double>::epsilon());
        double ray_active_norm = std::max(
            initial_wavefield.segment(active_offset, active_count).norm(),
            std::numeric_limits<double>::epsilon());

        /* -------------------- 7. 构造向下扫掠预条件器 -------------------- */

        Clock::time_point preconditioner_start = Clock::now();

        std::vector<SweepBlock> sweep_blocks =
            build_downward_sweep_blocks(
                active_helmholtz,
                active_velocity,
                active_nz,
                padded_nx,
                frequency,
                sweep_block_rows,
                sweep_overlap_rows,
                shift_beta,
                ilut_drop_tolerance,
                ilut_fill_factor);

        double preconditioner_seconds =
            elapsed_seconds(preconditioner_start);

        /* -------------------- 8. 建立收敛信息文件 -------------------- */

        char metrics_name[4096];
        std::snprintf(
            metrics_name,
            sizeof(metrics_name),
            "%s_metrics.csv",
            output_prefix);

        se::huygens::ensure_parent_directory(metrics_name);

        metrics_file = std::fopen(metrics_name, "w");
        if (metrics_file == NULL) {
            throw std::runtime_error("cannot create metrics CSV file");
        }

        std::fprintf(
            metrics_file,
            "iteration,relative_defect,defect_ratio,krylov_estimate,"
            "relative_correction,ray_correlation,solve_seconds,"
            "output_seconds\n");
        std::fprintf(
            metrics_file,
            "0,1.0,1.0,1.0,0.0,1.0,0.0,0.0\n");
        std::fflush(metrics_file);

        std::printf("FMM time: %.6f s\n", fmm_seconds);
        std::printf(
            "Helmholtz matrix: unknowns=%lld nnz=%lld time=%.6f s\n",
            static_cast<long long>(helmholtz.rows()),
            static_cast<long long>(helmholtz.nonZeros()),
            matrix_seconds);
        std::printf(
            "ray scale: %.6e %+.6ei\n",
            ray_scale.real(),
            ray_scale.imag());
        std::printf(
            "active correction: first model iz=%d, nz=%d, "
            "unknowns=%d\n",
            active_first_model_iz,
            active_nz,
            active_count);
        std::printf(
            "downward sweep: blocks=%lld block_rows=%d overlap=%d "
            "setup=%.6f s\n",
            static_cast<long long>(sweep_blocks.size()),
            sweep_block_rows,
            sweep_overlap_rows,
            preconditioner_seconds);
        std::printf(
            "iteration 0: active defect=%.6e (normalized=1)\n",
            initial_defect_norm);

        /* -------------------- 9. 截断 FGMRES 残差校正 -------------------- */

        /*
         * 在活动区域求解校正方程
         *
         *     A_active * delta = r_ray.
         *
         * FGMRES 保存所有已经得到的向下扫掠方向，并在这些方向张成的
         * Krylov 子空间中同时求最小残差解。与原程序每次只保留一个
         * M^{-1}r 方向相比，少量迭代也能组合出不同传播路径。
         */
        Vector active_correction = Vector::Zero(active_count);
        double relative_defect = initial_defect_norm > 0.0 ? 1.0 : 0.0;
        double previous_defect_norm = defect_scale;
        double cumulative_solve_seconds = 0.0;
        double cumulative_output_seconds = 0.0;
        int completed_iterations = 0;

        std::vector<Vector> residual_basis;
        std::vector<Vector> sweep_basis;
        residual_basis.reserve(static_cast<std::size_t>(max_iterations + 1));
        sweep_basis.reserve(static_cast<std::size_t>(max_iterations));

        Eigen::MatrixXcd hessenberg = Eigen::MatrixXcd::Zero(
            max_iterations + 1,
            max_iterations);

        if (initial_defect_norm >
            std::numeric_limits<double>::epsilon()) {
            residual_basis.push_back(correction_rhs / initial_defect_norm);
        }

        for (int column = 0;
             column < max_iterations && relative_defect > tolerance &&
             !residual_basis.empty();
             ++column) {

            Clock::time_point step_start = Clock::now();

            /* z_j=P_down^{-1}v_j：一次预条件应用完成一次向下扫掠。 */
            Vector direction = apply_downward_sweep(
                active_helmholtz,
                sweep_blocks,
                residual_basis[static_cast<std::size_t>(column)]);

            if (direction.norm() <=
                std::numeric_limits<double>::epsilon()) {
                std::printf(
                    "FGMRES stopped: sweep direction vanished\n");
                break;
            }

            sweep_basis.push_back(std::move(direction));
            Vector next = active_helmholtz * sweep_basis.back();

            /* 两次改进 Gram-Schmidt，减少少量 Krylov 向量的失正交。 */
            for (int pass = 0; pass < 2; ++pass) {
                for (int row = 0; row <= column; ++row) {
                    Complex coefficient =
                        residual_basis[static_cast<std::size_t>(row)].dot(next);
                    hessenberg(row, column) += coefficient;
                    next -= coefficient *
                        residual_basis[static_cast<std::size_t>(row)];
                }
            }

            double next_norm = next.norm();
            hessenberg(column + 1, column) = next_norm;

            int used = column + 1;
            Vector small_rhs = Vector::Zero(used + 1);
            small_rhs[0] = initial_defect_norm;

            Eigen::MatrixXcd small_hessenberg =
                hessenberg.topLeftCorner(used + 1, used);
            Vector coefficients = small_hessenberg
                .colPivHouseholderQr()
                .solve(small_rhs);

            Vector candidate_correction = Vector::Zero(active_count);
            for (int index = 0; index < used; ++index) {
                candidate_correction +=
                    coefficients[index] *
                    sweep_basis[static_cast<std::size_t>(index)];
            }

            /* 始终用真实未偏移算子重新计算残差。 */
            Vector candidate_residual =
                correction_rhs - active_helmholtz * candidate_correction;
            double candidate_defect_norm = candidate_residual.norm();
            relative_defect = candidate_defect_norm / defect_scale;
            double defect_ratio = candidate_defect_norm /
                std::max(
                    previous_defect_norm,
                    std::numeric_limits<double>::epsilon());
            double krylov_estimate =
                (small_rhs - small_hessenberg * coefficients).norm() /
                defect_scale;
            double relative_correction =
                candidate_correction.norm() / ray_active_norm;

            Vector candidate_active_wavefield =
                initial_wavefield.segment(active_offset, active_count) +
                candidate_correction;
            double candidate_active_norm = std::max(
                candidate_active_wavefield.norm(),
                std::numeric_limits<double>::epsilon());
            double ray_correlation = std::abs(
                initial_wavefield.segment(active_offset, active_count)
                    .dot(candidate_active_wavefield)) /
                (ray_active_norm * candidate_active_norm);

            active_correction = std::move(candidate_correction);
            previous_defect_norm = candidate_defect_norm;
            completed_iterations = used;

            Vector padded_correction = make_padded_correction(
                active_correction,
                helmholtz.rows(),
                active_offset);
            wavefield = initial_wavefield + padded_correction;

            cumulative_solve_seconds += elapsed_seconds(step_start);

            Clock::time_point output_start = Clock::now();
            if (write_iterations != 0) {
                write_wavefield(
                    output_prefix,
                    "iter",
                    used,
                    model,
                    wavefield,
                    frequency,
                    "ray_anchored_downward_sweep_fgmres");

                if (write_correction != 0) {
                    write_wavefield(
                        output_prefix,
                        "correction",
                        used,
                        model,
                        padded_correction,
                        frequency,
                        "downward_multipath_correction");
                }
            }
            cumulative_output_seconds += elapsed_seconds(output_start);

            std::fprintf(
                metrics_file,
                "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                used,
                relative_defect,
                defect_ratio,
                krylov_estimate,
                relative_correction,
                ray_correlation,
                cumulative_solve_seconds,
                cumulative_output_seconds);
            std::fflush(metrics_file);

            std::printf(
                "iteration %d: defect=%.6e ratio=%.6e "
                "correction/ray=%.6e ray_corr=%.6e\n",
                used,
                relative_defect,
                defect_ratio,
                relative_correction,
                ray_correlation);

            if (next_norm <=
                std::numeric_limits<double>::epsilon()) {
                std::printf("FGMRES stopped: Arnoldi breakdown\n");
                break;
            }

            residual_basis.push_back(next / next_norm);
        }

        /* write_iterations=0 时只写最终结果，避免逐步 RSF I/O。 */
        if (write_iterations == 0 && completed_iterations > 0) {
            Vector padded_correction = make_padded_correction(
                active_correction,
                helmholtz.rows(),
                active_offset);
            write_wavefield(
                output_prefix,
                "iter",
                completed_iterations,
                model,
                wavefield,
                frequency,
                "ray_anchored_downward_sweep_fgmres");
            if (write_correction != 0) {
                write_wavefield(
                    output_prefix,
                    "correction",
                    completed_iterations,
                    model,
                    padded_correction,
                    frequency,
                    "downward_multipath_correction");
            }
        }

        std::fclose(metrics_file);
        metrics_file = NULL;

        std::printf(
            "finished: FGMRES steps=%d final relative defect=%.6e\n",
            completed_iterations,
            relative_defect);
        std::printf(
            "traveltime: %s_traveltime.rsf\n",
            output_prefix);
        std::printf(
            "wavefields: %s_iter_NNN_real.rsf / imag.rsf\n",
            output_prefix);
        std::printf("metrics: %s\n", metrics_name);

        return 0;
    }
    catch (const std::exception& error) {
        if (metrics_file != NULL) {
            std::fclose(metrics_file);
        }

        std::fprintf(
            stderr,
            "ray_helmholtz_iteration: %s\n",
            error.what());

        return 1;
    }
}
