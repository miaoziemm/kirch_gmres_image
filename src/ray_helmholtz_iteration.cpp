/*
 * 这是使用项目 se_par_* 参数接口的最终版本。
 * 本文件不包含 Arguments 类、Parameters 结构体或自定义 key=value 解析器。
 */

#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_par_sep.h>
#include <SERECKIRCH/include/huygens_sweep.hpp>
#include <global_preconditioned_gmres.hpp>

#include <Eigen/IterativeLinearSolvers>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

/*
 * 程序功能
 * --------
 * 1. 使用项目已有的 eFMM 计算点源首波走时；
 * 2. 根据首波走时构造指定频率的射线近似波场；
 * 3. 将射线场作为初值，对全局 Helmholtz 方程做少量残差校正；
 * 4. 输出初始场、每次迭代后的总场以及累计校正场。
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
 * 例如：frequency=15 source_ix=512 tolerance=1e-3。
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
        "  max_iterations=8\n"
        "  tolerance=1e-3\n"
        "  preconditioner=ilut          (ilut, diagonal, identity)\n"
        "  shift_beta=0.10\n"
        "  ilut_drop_tolerance=1e-2\n"
        "  ilut_fill_factor=10\n"
        "  scale_radius=4\n"
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
        const char* preconditioner = se_have_par("preconditioner")
            ? se_get_par_str("preconditioner")
            : "ilut";

        float frequency = se_have_par("frequency")
            ? se_get_par_float("frequency") : 15.0f;
        float source_amplitude = se_have_par("source_amplitude")
            ? se_get_par_float("source_amplitude") : 1.0f;
        float tolerance = se_have_par("tolerance")
            ? se_get_par_float("tolerance") : 1.0e-3f;
        float shift_beta = se_have_par("shift_beta")
            ? se_get_par_float("shift_beta") : 0.10f;
        float ilut_drop_tolerance = se_have_par("ilut_drop_tolerance")
            ? se_get_par_float("ilut_drop_tolerance") : 1.0e-2f;

        int source_ix = se_have_par("source_ix")
            ? se_get_par_int("source_ix") : -1;
        int source_iz = se_have_par("source_iz")
            ? se_get_par_int("source_iz") : 1;
        int max_iterations = se_have_par("max_iterations")
            ? se_get_par_int("max_iterations") : 8;
        int scale_radius = se_have_par("scale_radius")
            ? se_get_par_int("scale_radius") : 4;
        int ilut_fill_factor = se_have_par("ilut_fill_factor")
            ? se_get_par_int("ilut_fill_factor") : 10;
        int write_correction = se_have_par("write_correction")
            ? se_get_par_int("write_correction") : 1;

        /* 后续判断使用整数标志，避免在循环中反复比较字符串。 */
        int use_ilut = std::strcmp(preconditioner, "ilut") == 0;
        int use_diagonal =
            std::strcmp(preconditioner, "diagonal") == 0;
        int use_identity =
            std::strcmp(preconditioner, "identity") == 0;

        if (frequency <= 0.0f || source_amplitude == 0.0f ||
            tolerance <= 0.0f || shift_beta < 0.0f ||
            ilut_drop_tolerance < 0.0f || ilut_fill_factor < 1 ||
            max_iterations < 1 || scale_radius < 0) {
            throw std::runtime_error("invalid command-line parameter");
        }

        if (!use_ilut && !use_diagonal && !use_identity) {
            throw std::runtime_error(
                "preconditioner must be ilut, diagonal, or identity");
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

        /* 使用原始Helmholtz矩阵计算真实残差。 */
        Vector residual = rhs - helmholtz * wavefield;
        double rhs_norm = std::max(
            rhs.norm(), std::numeric_limits<double>::epsilon());
        double relative_residual = residual.norm() / rhs_norm;

        /* -------------------- 6. 构造所选择的预条件器 -------------------- */

        Clock::time_point preconditioner_start = Clock::now();

        Eigen::IncompleteLUT<Complex, int> ilut;
        Vector diagonal;

        /*
         * ILUT 和对角预条件都基于 shifted-Helmholtz 矩阵。
         * identity 不需要构造 shifted 矩阵，也没有额外的准备开销。
         */
        if (use_ilut || use_diagonal) {
            Sparse shifted_helmholtz = build_shifted_helmholtz(
                helmholtz,
                padded_velocity,
                padded_nz,
                padded_nx,
                frequency,
                shift_beta);

            if (use_ilut) {
                ilut.setDroptol(ilut_drop_tolerance);
                ilut.setFillfactor(ilut_fill_factor);
                ilut.compute(shifted_helmholtz);

                if (ilut.info() != Eigen::Success) {
                    throw std::runtime_error(
                        "ILUT construction failed; try increasing "
                        "shift_beta or use preconditioner=diagonal");
                }
            }
            else {
                /* Jacobi 预条件：只保存 shifted 矩阵的主对角线。 */
                diagonal.resize(shifted_helmholtz.rows());

                for (Eigen::Index row = 0;
                     row < shifted_helmholtz.rows();
                     ++row) {
                    diagonal[row] =
                        shifted_helmholtz.coeff(row, row);

                    if (std::abs(diagonal[row]) <=
                        std::numeric_limits<double>::epsilon()) {
                        throw std::runtime_error(
                            "zero diagonal in Jacobi preconditioner");
                    }
                }
            }
        }

        double preconditioner_seconds =
            elapsed_seconds(preconditioner_start);

        /* -------------------- 7. 建立收敛信息文件 -------------------- */

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
            "iteration,relative_residual,residual_ratio,"
            "relative_correction,alpha_real,alpha_imag,seconds\n");
        std::fprintf(
            metrics_file,
            "0,%.12e,1.0,0.0,0.0,0.0,0.0\n",
            relative_residual);
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
            "preconditioner: %s, setup time: %.6f s\n",
            preconditioner,
            preconditioner_seconds);
        std::printf(
            "iteration 0: relative residual=%.6e\n",
            relative_residual);

        /* -------------------- 8. 开始残差校正迭代 -------------------- */

        /*
         * 每次迭代执行：
         *
         *     z       = M^{-1} r
         *     w       = A z
         *     alpha   = w^H r / w^H w
         *     u       = u + alpha*z
         *     r       = b - A u
         *
         * 其中 alpha 是当前校正方向上的最优复数步长，可使本次更新后
         * 的二范数残差最小。它比固定松弛因子更适合量级尚不完全一致
         * 的射线初场。
         */
        Clock::time_point iteration_start = Clock::now();

        for (int iteration = 1;
             iteration <= max_iterations &&
             relative_residual > tolerance;
             ++iteration) {

            Vector direction;

            if (use_ilut) {
                /* z=M^{-1}r：使用 shifted-Helmholtz ILUT。 */
                direction = ilut.solve(residual);

                if (ilut.info() != Eigen::Success) {
                    throw std::runtime_error(
                        "ILUT triangular solve failed");
                }
            }
            else if (use_diagonal) {
                /* Jacobi 预条件，每个网格点只进行一次复数除法。 */
                direction = residual.array() / diagonal.array();
            }
            else {
                /* identity：不使用预条件器，直接采用残差方向。 */
                direction = residual;
            }

            /* 计算校正方向经过真实Helmholtz算子后的结果。 */
            Vector action = helmholtz * direction;
            double denominator = action.squaredNorm();

            if (denominator <= std::numeric_limits<double>::epsilon()) {
                std::printf(
                    "iteration stopped: correction direction vanished\n");
                break;
            }

            /* 当前方向上的最优复数步长。 */
            Complex alpha = action.dot(residual) / denominator;

            /* 更新总波场，并重新计算真实残差，避免递推残差累积误差。 */
            wavefield += alpha * direction;
            residual = rhs - helmholtz * wavefield;

            double previous_residual = relative_residual;
            relative_residual = residual.norm() / rhs_norm;
            double residual_ratio = relative_residual /
                std::max(
                    previous_residual,
                    std::numeric_limits<double>::epsilon());

            /*
             * 累计校正量与射线初场的比值。该值仅用于诊断，不会限制
             * 或缩放实际校正量。
             */
            Vector correction = wavefield - initial_wavefield;
            double relative_correction = correction.norm() /
                std::max(
                    initial_wavefield.norm(),
                    std::numeric_limits<double>::epsilon());
            double iteration_seconds = elapsed_seconds(iteration_start);

            /* 保存本次迭代后的完整总波场。 */
            write_wavefield(
                output_prefix,
                "iter",
                iteration,
                model,
                wavefield,
                frequency,
                "ray_initialized_minimum_residual_iteration");

            /*
             * 单独保存累计校正场。多路径通常弱于首波，因此观察
             * correction 文件往往比直接观察总场更清楚。
             */
            if (write_correction != 0) {
                write_wavefield(
                    output_prefix,
                    "correction",
                    iteration,
                    model,
                    correction,
                    frequency,
                    "cumulative_helmholtz_correction");
            }

            std::fprintf(
                metrics_file,
                "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                iteration,
                relative_residual,
                residual_ratio,
                relative_correction,
                alpha.real(),
                alpha.imag(),
                iteration_seconds);
            std::fflush(metrics_file);

            std::printf(
                "iteration %d: residual=%.6e ratio=%.6e "
                "correction/ray=%.6e alpha=%.6e%+.6ei\n",
                iteration,
                relative_residual,
                residual_ratio,
                relative_correction,
                alpha.real(),
                alpha.imag());
        }

        std::fclose(metrics_file);
        metrics_file = NULL;

        std::printf(
            "finished: final relative residual=%.6e\n",
            relative_residual);
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