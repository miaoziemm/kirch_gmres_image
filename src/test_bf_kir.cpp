#include <SERECKIRCH/include/huygens_sweep.hpp>
#include <SERECKIRCH/include/se_butterfly.hpp>
#include <SEFILESYSTEM/include/se_fs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef KIRCH_HAS_BUTTERFLYPACK_FLOAT
#error "test_bf_kir requires ButterflyPACK single-complex support."
#endif

using Complex = std::complex<float>;
using Clock = std::chrono::steady_clock;

constexpr float PI = 3.14159265358979323846f;
constexpr float FREQUENCY = 25.0f;
constexpr float FILTER_DT = 0.001f;
constexpr float FILTER_LENGTH = 0.025f;
constexpr int BLOCK_ID = 1;

constexpr double BF_TOL = 1.0e-4;
constexpr int BF_LEAF = 64;
constexpr double BF_SAMPLE_PARA = 4.0;
constexpr int BF_KNN = 10;

const std::string BLOCK_FILE = "block_info.dat";
const std::string TABLE_PREFIX = "huygens_tt/travel";
const std::string INPUT_FILE = "test_bf_block1_input_point_source.rsf";
const std::string DIRECT_REAL_FILE = "test_bf_block1_direct_real.rsf";
const std::string DIRECT_IMAG_FILE = "test_bf_block1_direct_imag.rsf";
const std::string BF_REAL_FILE = "test_bf_block1_bf_real.rsf";
const std::string BF_IMAG_FILE = "test_bf_block1_bf_imag.rsf";
const std::string ERROR_FILE = "test_bf_block1_abs_error.rsf";


/* 1-D 点源：中间位置为 1，其余为 0。 */
std::vector<Complex> make_point_source(const se::huygens::Table3D& tau, int& source_index)
{
    if (tau.nsource < 2) throw std::runtime_error("tau.nsource must be at least 2");

    std::vector<Complex> input(static_cast<std::size_t>(tau.nsource), Complex(0.0f, 0.0f));
    source_index = tau.nsource / 2;
    input[static_cast<std::size_t>(source_index)] = Complex(1.0f, 0.0f);
    return input;
}


/* 梯形积分权重。 */
std::vector<float> make_weights(const se::huygens::Table3D& tau)
{
    std::vector<float> weights(static_cast<std::size_t>(tau.nsource), tau.dx_source);
    weights.front() *= 0.5f;
    weights.back() *= 0.5f;
    return weights;
}


float maximum_traveltime(const se::huygens::Table3D& tau)
{
    if (tau.values.empty()) throw std::runtime_error("traveltime table is empty");
    return *std::max_element(tau.values.begin(), tau.values.end());
}


/*
 * Kirch_datuming 单频形式：
 *
 * K_ij = dx_j/pi * Delta_z_ij * tau_ij / R_ij^2 * H(tau_ij,omega)
 *
 * 其中 H 由 FrequencyKirchhoffFilter::response(tau) 给出。
 */
Complex kirchhoff_entry(int row, int source,
                        const se::huygens::Table3D& tau,
                        const std::vector<float>& weights,
                        const se::huygens::FrequencyKirchhoffFilter& filter,
                        float source_z)
{
    const int iz = row % tau.nz_target;
    const int ix = row / tau.nz_target;
    const std::size_t index = tau.index(source, ix, iz);

    const float traveltime = std::max(tau.values[index], 1.0e-8f);
    const float target_x = tau.ox_target + ix * tau.dx_target;
    const float target_z = tau.oz_target + iz * tau.dz_target;
    const float source_x = tau.ox_source + source * tau.dx_source;

    const float dx = target_x - source_x;
    const float dz = target_z - source_z;
    const float r2 = std::max(dx * dx + dz * dz, 1.0e-20f);

    const float geometry = dz * traveltime / (PI * r2);
    return weights[static_cast<std::size_t>(source)] * geometry * filter.response(traveltime);
}


/* 直接 Kirchhoff 求和。 */
std::vector<Complex> direct_kirchhoff(const se::huygens::Table3D& tau,
                                      const std::vector<float>& weights,
                                      const std::vector<Complex>& input,
                                      const se::huygens::FrequencyKirchhoffFilter& filter,
                                      float source_z)
{
    if (input.size() != static_cast<std::size_t>(tau.nsource))
        throw std::runtime_error("input length does not match tau.nsource");

    const int rows = tau.nx_target * tau.nz_target;
    std::vector<Complex> output(static_cast<std::size_t>(rows), Complex(0.0f, 0.0f));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < rows; ++row) {
        Complex sum(0.0f, 0.0f);
        for (int source = 0; source < tau.nsource; ++source) {
            sum += kirchhoff_entry(row, source, tau, weights, filter, source_z)
                 * input[static_cast<std::size_t>(source)];
        }
        output[static_cast<std::size_t>(row)] = sum;
    }

    return output;
}


/* ButterflyPACK target 坐标：(x,z)。 */
std::vector<double> make_row_coordinates(const se::huygens::Table3D& tau)
{
    const int rows = tau.nx_target * tau.nz_target;
    std::vector<double> coord(static_cast<std::size_t>(2 * rows));

    for (int row = 0; row < rows; ++row) {
        const int iz = row % tau.nz_target;
        const int ix = row / tau.nz_target;

        coord[static_cast<std::size_t>(2 * row)] =
            static_cast<double>(tau.ox_target + ix * tau.dx_target);
        coord[static_cast<std::size_t>(2 * row + 1)] =
            static_cast<double>(tau.oz_target + iz * tau.dz_target);
    }

    return coord;
}


/* ButterflyPACK input datum 坐标：(x,source_z)。 */
std::vector<double> make_column_coordinates(const se::huygens::Table3D& tau, float source_z)
{
    std::vector<double> coord(static_cast<std::size_t>(2 * tau.nsource));

    for (int source = 0; source < tau.nsource; ++source) {
        coord[static_cast<std::size_t>(2 * source)] =
            static_cast<double>(tau.ox_source + source * tau.dx_source);
        coord[static_cast<std::size_t>(2 * source + 1)] = static_cast<double>(source_z);
    }

    return coord;
}


double relative_l2_error(const std::vector<Complex>& reference,
                         const std::vector<Complex>& candidate)
{
    if (reference.size() != candidate.size())
        throw std::runtime_error("vector sizes differ");

    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        num += std::norm(candidate[i] - reference[i]);
        den += std::norm(reference[i]);
    }

    return std::sqrt(num / std::max(den, 1.0e-30));
}


double maximum_normalized_error(const std::vector<Complex>& reference,
                                const std::vector<Complex>& candidate)
{
    float max_ref = 0.0f, max_diff = 0.0f;

    for (std::size_t i = 0; i < reference.size(); ++i) {
        max_ref = std::max(max_ref, std::abs(reference[i]));
        max_diff = std::max(max_diff, std::abs(candidate[i] - reference[i]));
    }

    return static_cast<double>(max_diff / std::max(max_ref, 1.0e-30f));
}


/* 输出 1-D 点源。 */
void write_input_source(const std::string& filename,
                        const se::huygens::Table3D& tau,
                        const std::vector<Complex>& input)
{
    std::vector<float> values(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) values[i] = input[i].real();

    sep_t* out = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (!out || !out->headers || !out->data || !out->data->io)
        throw std::runtime_error("cannot create input RSF");

    try {
        out->headers->ndim = 1;
        out->headers->n[0] = tau.nsource;
        out->headers->d[0] = tau.dx_source;
        out->headers->o[0] = tau.ox_source;
        out->headers->esize = 4;
        out->headers->le = 1;

        sep_set_header(out, "data_format", "native_float");
        sep_set_header_int(out, "esize", 4);
        sep_set_header(out, "label1", "Datum distance");
        sep_set_header(out, "title", "1-D point-source input");

        se_fsio_write_float(out->data->io, values.data(), values.size());
        sep_close(out);
    } catch (...) {
        sep_close(out);
        throw;
    }
}


/* 输出 2-D 波场，尺寸严格等于 travel_block_001_tau.rsf。 */
void write_field_component(const std::string& filename,
                           const se::huygens::Table3D& tau,
                           const std::vector<Complex>& field,
                           bool imaginary,
                           const std::string& method)
{
    const std::size_t count =
        static_cast<std::size_t>(tau.nz_target) * static_cast<std::size_t>(tau.nx_target);

    if (field.size() != count)
        throw std::runtime_error("field size does not match travel table");

    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i)
        values[i] = imaginary ? field[i].imag() : field[i].real();

    sep_t* out = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (!out || !out->headers || !out->data || !out->data->io)
        throw std::runtime_error("cannot create output: " + filename);

    try {
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
        sep_set_header(out, "label1", "Target depth");
        sep_set_header(out, "unit1", "km");
        sep_set_header(out, "label2", "Target distance");
        sep_set_header(out, "unit2", "km");
        sep_set_header(out, "method", method.c_str());
        sep_set_header(out, "component", imaginary ? "imaginary" : "real");
        sep_set_header_float(out, "frequency", FREQUENCY);

        se_fsio_write_float(out->data->io, values.data(), values.size());
        sep_close(out);
    } catch (...) {
        sep_close(out);
        throw;
    }
}


/* 输出 |Butterfly - Direct|。 */
void write_error(const std::string& filename,
                 const se::huygens::Table3D& tau,
                 const std::vector<Complex>& direct,
                 const std::vector<Complex>& butterfly)
{
    if (direct.size() != butterfly.size())
        throw std::runtime_error("error vector sizes differ");

    std::vector<float> values(direct.size());
    for (std::size_t i = 0; i < direct.size(); ++i)
        values[i] = std::abs(butterfly[i] - direct[i]);

    sep_t* out = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (!out || !out->headers || !out->data || !out->data->io)
        throw std::runtime_error("cannot create error output");

    try {
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
        sep_set_header(out, "quantity", "absolute_error");
        sep_set_header_float(out, "frequency", FREQUENCY);

        se_fsio_write_float(out->data->io, values.data(), values.size());
        sep_close(out);
    } catch (...) {
        sep_close(out);
        throw;
    }
}


int main()
{
    try {
        std::cout << std::setprecision(8);
        std::cout << "\n============================================\n"
                  << " ButterflyPACK Kirchhoff datuming test\n"
                  << " block     = " << BLOCK_ID << "\n"
                  << " frequency = " << FREQUENCY << " Hz\n"
                  << "============================================\n";

#ifdef _OPENMP
        std::cout << "OpenMP max threads = " << omp_get_max_threads() << "\n";
#endif

        const se::huygens::BlockInfo info = se::huygens::read_block_info(BLOCK_FILE);
        if (BLOCK_ID < 0 || BLOCK_ID >= static_cast<int>(info.blocks.size()))
            throw std::runtime_error("BLOCK_ID is outside block_info.dat");

        const se::huygens::Block& block = info.blocks[static_cast<std::size_t>(BLOCK_ID)];
        const std::string tau_file =
            se::huygens::layer_table_filename(TABLE_PREFIX, BLOCK_ID, "tau");

        std::cout << "\nReading traveltime:\n  " << tau_file << "\n";
        const se::huygens::Table3D tau = se::huygens::read_table_rsf(tau_file);

        const int rows = tau.nx_target * tau.nz_target;
        const int columns = tau.nsource;
        const float source_z = info.oz + block.source_iz * info.dz;
        const float max_tau = maximum_traveltime(tau);

        std::cout << "\nBlock geometry:\n"
                  << "  source_iz       = " << block.source_iz << "\n"
                  << "  source_z        = " << source_z << "\n"
                  << "  target_start_iz = " << block.target_start_iz << "\n"
                  << "  target_end_iz   = " << block.target_end_iz << "\n"
                  << "  halo_end_iz     = " << block.halo_end_iz << "\n";

        std::cout << "\nTraveltime table:\n"
                  << "  nz_target = " << tau.nz_target << "\n"
                  << "  nx_target = " << tau.nx_target << "\n"
                  << "  nsource   = " << tau.nsource << "\n"
                  << "  max tau   = " << max_tau << " s\n"
                  << "  matrix    = " << rows << " x " << columns << "\n";

        int point_source_index = 0;
        const std::vector<Complex> input = make_point_source(tau, point_source_index);
        const float point_source_x = tau.ox_source + point_source_index * tau.dx_source;

        std::cout << "\n1-D input point source:\n"
                  << "  index     = " << point_source_index << "\n"
                  << "  x         = " << point_source_x << "\n"
                  << "  z         = " << source_z << "\n"
                  << "  amplitude = 1 + 0i\n";

        write_input_source(INPUT_FILE, tau, input);

        const std::vector<float> weights = make_weights(tau);
        se::huygens::FrequencyKirchhoffFilter filter(
            FREQUENCY, FILTER_DT, FILTER_LENGTH, max_tau);

        std::cout << "\n[1] Direct Kirchhoff 1D -> 2D...\n";
        const auto direct_start = Clock::now();
        const std::vector<Complex> direct =
            direct_kirchhoff(tau, weights, input, filter, source_z);
        const double direct_seconds =
            std::chrono::duration<double>(Clock::now() - direct_start).count();

        std::vector<double> row_coordinates = make_row_coordinates(tau);
        std::vector<double> column_coordinates = make_column_coordinates(tau, source_z);

        se::butterfly::Options options;
        options.tolerance = BF_TOL;
        options.leaf_size = BF_LEAF;
        options.coordinate_dimension = 2;
        options.lr_level = 100;
        options.sample_parameter = BF_SAMPLE_PARA;
        options.forward_n15_flag = 0;
        options.nearest_neighbors = BF_KNN;
        options.verbosity = 1;

        std::cout << "\n[2] Building ButterflyPACK operator...\n";
        const auto build_start = Clock::now();

        se::butterfly::Matrix<float> butterfly(
            rows, columns,
            std::move(row_coordinates),
            std::move(column_coordinates),
            [&](int row, int source) -> Complex {
                return kirchhoff_entry(row, source, tau, weights, filter, source_z);
            },
            options);

        const double build_seconds =
            std::chrono::duration<double>(Clock::now() - build_start).count();

        std::cout << "\n[3] ButterflyPACK 1D -> 2D...\n";
        const auto apply_start = Clock::now();
        const std::vector<Complex> bf = butterfly.apply(input);
        const double apply_seconds =
            std::chrono::duration<double>(Clock::now() - apply_start).count();

        const double l2_error = relative_l2_error(direct, bf);
        const double max_error = maximum_normalized_error(direct, bf);
        const se::butterfly::Statistics& stats = butterfly.statistics();

        std::cout << "\n[4] Writing RSF outputs...\n";
        write_field_component(DIRECT_REAL_FILE, tau, direct, false, "direct_kirchhoff_datuming");
        write_field_component(DIRECT_IMAG_FILE, tau, direct, true, "direct_kirchhoff_datuming");
        write_field_component(BF_REAL_FILE, tau, bf, false, "butterfly_kirchhoff_datuming");
        write_field_component(BF_IMAG_FILE, tau, bf, true, "butterfly_kirchhoff_datuming");
        write_error(ERROR_FILE, tau, direct, bf);

        std::cout << "\n============================================\n"
                  << "                RESULT\n"
                  << "============================================\n"
                  << "frequency                = " << FREQUENCY << " Hz\n"
                  << "filter dt                = " << FILTER_DT << " s\n"
                  << "filter length            = " << FILTER_LENGTH << " s\n"
                  << "point source index       = " << point_source_index << "\n"
                  << "source datum z           = " << source_z << "\n"
                  << "matrix size              = " << rows << " x " << columns << "\n"
                  << "output size              = " << tau.nz_target << " x "
                  << tau.nx_target << "\n\n"
                  << "direct Kirchhoff time    = " << direct_seconds << " s\n"
                  << "BF build time            = " << build_seconds << " s\n"
                  << "BF apply time            = " << apply_seconds << " s\n\n"
                  << "relative L2 error        = " << l2_error << "\n"
                  << "maximum normalized error = " << max_error << "\n\n"
                  << "compressed memory        = " << stats.compressed_megabytes << " MB\n"
                  << "maximum rank             = " << stats.maximum_rank << "\n"
                  << "sampled entries          = " << stats.sampled_entries << "\n"
                  << "============================================\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "\ntest_bf_kir failed: " << error.what() << "\n";
        return 1;
    }
}
