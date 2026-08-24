#include "program_help.hpp"

#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_fs.h>
#include <SERECKIRCH/include/efmm.h>
#include <SERECKIRCH/include/se_blas.hpp>
#include <SERECKIRCH/include/se_butterfly.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Complex = std::complex<float>;
using Clock = std::chrono::steady_clock;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Model2D {
    int nz = 0;
    int nx = 0;
    float dz = 1.0f;
    float dx = 1.0f;
    float oz = 0.0f;
    float ox = 0.0f;
    std::vector<float> velocity;  // x-fast: ix + iz * nx, required by efmm.

    std::size_t index(int ix, int iz) const
    {
        return static_cast<std::size_t>(ix) +
               static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx);
    }

    float x(int ix) const { return ox + static_cast<float>(ix) * dx; }
    float z(int iz) const { return oz + static_cast<float>(iz) * dz; }
};

struct GridPoint {
    int ix = 0;
    int iz = 0;
};

struct Options {
    std::string velocity_file;
    std::string direct_output_real = "kirchhoff_direct_25hz_real.rsf";
    std::string direct_output_imag = "kirchhoff_direct_25hz_imag.rsf";
    std::string butterfly_output_real = "kirchhoff_butterfly_25hz_real.rsf";
    std::string butterfly_output_imag = "kirchhoff_butterfly_25hz_imag.rsf";
    std::string traveltime_file = "kirchhoff_traveltime.rsf";
    std::string traveltime_derivative_file = "kirchhoff_traveltime_normal_derivative.rsf";
    std::string timing_file = "kirchhoff_25hz_timing.txt";

    float frequency = 25.0f;
    float source_amplitude = 1.0f;
    float source_radius = 0.0f;
    float tolerance = 1.0e-4f;
    float max_traveltime_mb = 16384.0f;

    int source_ix = -1;
    int source_iz = 0;
    int datum_iz = 1;
    int top_layers = 3;
    int boundary_stride = 16;
    int target_x_stride = 1;
    int target_z_stride = 1;
    int leaf_size = 64;
    int verbosity = 0;
    int threads = 0;
    int help = 0;
};

struct Timings {
    double boundary_fmm_seconds = 0.0;
    double direct_seconds = 0.0;
    double butterfly_build_seconds = 0.0;
    double butterfly_apply_seconds = 0.0;
};

const char* required_string_parameter(const char* name)
{
    if (!se_have_par(name)) {
        ERROR(("Need %s=", name));
    }
    return se_get_par_str(name);
}

std::string optional_string_parameter(const char* name, const char* default_value)
{
    return se_have_par(name) ? std::string(se_get_par_str(name))
                             : std::string(default_value);
}

int optional_int_parameter(const char* name, int default_value)
{
    return se_have_par(name) ? se_get_par_int(name) : default_value;
}

float optional_float_parameter(const char* name, float default_value)
{
    return se_have_par(name) ? se_get_par_float(name) : default_value;
}

void print_usage(const char* program)
{
    std::cout
        << "Usage:\n  " << program << " velocity=mar_vel.rsf [key=value ...]\n\n"
        << "The program computes a 25-Hz-by-default 2-D single-frequency wavefield.\n"
        << "Rows iz=0,1,2 are initialized with the exact homogeneous 2-D Hankel\n"
        << "point-source Green function.  The centered derivative at iz=1 is then\n"
        << "continued through the model with the complete Cauchy-Kirchhoff integral.\n\n"
        << "Required:\n"
        << "  velocity=FILE             Input RSF velocity model, n1=z and n2=x.\n\n"
        << "Geometry and sampling:\n"
        << "  frequency=HZ              Single frequency; default 25.\n"
        << "  source_ix=N               Default nx/2.\n"
        << "  source_iz=N               Default 0.\n"
        << "  datum_iz=N                Kirchhoff datum; default 1 (second layer).\n"
        << "  top_layers=N              Must be at least 3; default 3.\n"
        << "  boundary_stride=N         Datum sampling stride; default 16.\n"
        << "  target_x_stride=N         1 writes every x sample; default 1.\n"
        << "  target_z_stride=N         1 writes every z sample; default 1.\n"
        << "  source_radius=M           Hankel singularity regularization; default 0.5*min(dx,dz).\n"
        << "  source_amplitude=A        Point-source amplitude; default 1.\n\n"
        << "ButterflyPACK and memory:\n"
        << "  tol=EPS                   Compression tolerance; default 1e-4.\n"
        << "  leaf=N                    Leaf size; default 64.\n"
        << "  verbosity=N               ButterflyPACK verbosity; default 0.\n"
        << "  max_traveltime_mb=MB      Guard for cached FMM tables; default 16384.\n"
        << "  threads=N                 OpenMP threads; 0 keeps runtime default.\n\n"
        << "Outputs:\n"
        << "  direct_real=FILE          Direct-method real wavefield, float RSF.\n"
        << "  direct_imag=FILE          Direct-method imaginary wavefield, float RSF.\n"
        << "  butterfly_real=FILE       ButterflyPACK real wavefield, float RSF.\n"
        << "  butterfly_imag=FILE       ButterflyPACK imaginary wavefield, float RSF.\n"
        << "  traveltime=FILE           Cached center traveltime table, float RSF.\n"
        << "  traveltime_derivative=FILE  Source-normal traveltime derivative table.\n"
        << "  timing=FILE               Timing and error report.\n\n"
        << "Direct and ButterflyPACK wavefields are stored in separate 2-D RSF files.\n"
        << "The timing comparison excludes Hankel initialization, FMM traveltime\n"
        << "calculation, traveltime output, and wavefield output.\n"
        << "  help=1                    Print this message.\n";
}

Options parse_options(int argc, char** argv)
{
    se_par_init(argc, argv);

    Options options;
    options.help = optional_int_parameter("help", 0);
    if (options.help) {
        print_usage(argv[0]);
        std::exit(0);
    }

    options.velocity_file = required_string_parameter("velocity");
    options.direct_output_real = optional_string_parameter(
        "direct_real", options.direct_output_real.c_str());
    options.direct_output_imag = optional_string_parameter(
        "direct_imag", options.direct_output_imag.c_str());
    options.butterfly_output_real = optional_string_parameter(
        "butterfly_real", options.butterfly_output_real.c_str());
    options.butterfly_output_imag = optional_string_parameter(
        "butterfly_imag", options.butterfly_output_imag.c_str());
    options.traveltime_file = optional_string_parameter(
        "traveltime", options.traveltime_file.c_str());
    options.traveltime_derivative_file = optional_string_parameter(
        "traveltime_derivative", options.traveltime_derivative_file.c_str());
    options.timing_file = optional_string_parameter("timing", options.timing_file.c_str());

    options.frequency = optional_float_parameter("frequency", options.frequency);
    options.source_amplitude = optional_float_parameter("source_amplitude", options.source_amplitude);
    options.source_radius = optional_float_parameter("source_radius", options.source_radius);
    options.tolerance = optional_float_parameter("tol", options.tolerance);
    options.max_traveltime_mb = optional_float_parameter(
        "max_traveltime_mb", options.max_traveltime_mb);

    options.source_ix = optional_int_parameter("source_ix", options.source_ix);
    options.source_iz = optional_int_parameter("source_iz", options.source_iz);
    options.datum_iz = optional_int_parameter("datum_iz", options.datum_iz);
    options.top_layers = optional_int_parameter("top_layers", options.top_layers);
    options.boundary_stride = optional_int_parameter(
        "boundary_stride", options.boundary_stride);
    options.target_x_stride = optional_int_parameter(
        "target_x_stride", options.target_x_stride);
    options.target_z_stride = optional_int_parameter(
        "target_z_stride", options.target_z_stride);
    options.leaf_size = optional_int_parameter("leaf", options.leaf_size);
    options.verbosity = optional_int_parameter("verbosity", options.verbosity);
    options.threads = optional_int_parameter("threads", options.threads);

    if (!(options.frequency > 0.0f) || !std::isfinite(options.frequency)) {
        throw std::invalid_argument("frequency must be positive");
    }
    if (options.top_layers < 3) {
        throw std::invalid_argument("top_layers must be at least 3 to form a centered derivative");
    }
    if (options.datum_iz < 1 || options.datum_iz + 1 >= options.top_layers) {
        throw std::invalid_argument(
            "datum_iz must have one Hankel layer above and below it");
    }
    if (options.boundary_stride < 1 || options.target_x_stride < 1 ||
        options.target_z_stride < 1 || options.leaf_size < 4) {
        throw std::invalid_argument("strides and leaf must be positive");
    }
    if (!(options.tolerance > 0.0f) || !(options.max_traveltime_mb > 0.0f)) {
        throw std::invalid_argument("tol and max_traveltime_mb must be positive");
    }
    if (options.threads < 0) {
        throw std::invalid_argument("threads must be non-negative");
    }
    return options;
}

Model2D read_velocity_model(const std::string& filename)
{
    sep_t* input = sep_open(filename.c_str(), SEP_READ, 0);
    if (input == nullptr || input->headers == nullptr || input->data == nullptr ||
        input->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for velocity model");
    }

    Model2D model;
    try {
        if (input->headers->ndim < 2) {
            throw std::runtime_error("velocity RSF must have at least two axes");
        }
        model.nz = input->headers->n[0];
        model.nx = input->headers->n[1];
        model.dz = static_cast<float>(input->headers->d[0]);
        model.dx = static_cast<float>(input->headers->d[1]);
        model.oz = static_cast<float>(input->headers->o[0]);
        model.ox = static_cast<float>(input->headers->o[1]);

        if (model.nz < 3 || model.nx < 2 || !(model.dz > 0.0f) || !(model.dx > 0.0f)) {
            throw std::runtime_error("invalid velocity RSF grid");
        }

        const std::size_t count = static_cast<std::size_t>(model.nz) * model.nx;
        std::vector<float> rsf_order(count);
        se_fsio_seek(input->data->io, 0);
        se_fsio_read_float(input->data->io, rsf_order.data(), count);

        model.velocity.resize(count);
        for (int ix = 0; ix < model.nx; ++ix) {
            for (int iz = 0; iz < model.nz; ++iz) {
                const float velocity = rsf_order[
                    static_cast<std::size_t>(iz) +
                    static_cast<std::size_t>(ix) * model.nz];
                if (!std::isfinite(velocity) || velocity <= 0.0f) {
                    throw std::runtime_error("velocity model contains invalid values");
                }
                model.velocity[model.index(ix, iz)] = velocity;
            }
        }
        sep_close(input);
        return model;
    } catch (...) {
        sep_close(input);
        throw;
    }
}

std::vector<float> solve_fmm(const Model2D& model, int source_ix, int source_iz)
{
    if (source_ix < 0 || source_ix >= model.nx ||
        source_iz < 0 || source_iz >= model.nz) {
        throw std::invalid_argument("FMM source lies outside the model");
    }

    efmm_t solver{};
    const float source_x = static_cast<float>(source_ix) * model.dx;
    const float source_z = static_cast<float>(source_iz) * model.dz;
    if (efmm_init(&solver, model.nx, model.nz, model.dx, model.dz,
                  source_x, source_z) != 0) {
        throw std::runtime_error("efmm_init failed");
    }

    try {
        if (efmm_set_vel(&solver, const_cast<float*>(model.velocity.data())) != 0) {
            throw std::runtime_error("efmm_set_vel failed");
        }
        const int status = efmm_solver(&solver);
        if (status != 0) {
            throw std::runtime_error("efmm_solver failed with status " +
                                     std::to_string(status));
        }
        const std::size_t count = static_cast<std::size_t>(model.nx) * model.nz;
        std::vector<float> result(solver.tt, solver.tt + count);
        efmm_free(&solver);
        return result;
    } catch (...) {
        efmm_free(&solver);
        throw;
    }
}

std::vector<int> regular_indices(int first, int last_inclusive, int stride)
{
    std::vector<int> indices;
    for (int value = first; value <= last_inclusive; value += stride) {
        indices.push_back(value);
    }
    if (indices.empty() || indices.back() != last_inclusive) {
        indices.push_back(last_inclusive);
    }
    return indices;
}

std::vector<GridPoint> make_boundary(const Model2D& model,
                                     int datum_iz,
                                     int stride)
{
    std::vector<GridPoint> boundary;
    for (int ix : regular_indices(0, model.nx - 1, stride)) {
        boundary.push_back({ix, datum_iz});
    }
    if (boundary.size() < 4) {
        throw std::invalid_argument("boundary_stride leaves fewer than four points");
    }
    return boundary;
}

std::vector<float> trapezoidal_weights(const Model2D& model,
                                       const std::vector<GridPoint>& boundary)
{
    std::vector<float> weights(boundary.size(), 0.0f);
    if (boundary.size() < 2) return weights;

    weights.front() = 0.5f * (model.x(boundary[1].ix) - model.x(boundary[0].ix));
    for (std::size_t i = 1; i + 1 < boundary.size(); ++i) {
        weights[i] = 0.5f *
            (model.x(boundary[i + 1].ix) - model.x(boundary[i - 1].ix));
    }
    weights.back() = 0.5f *
        (model.x(boundary.back().ix) - model.x(boundary[boundary.size() - 2].ix));
    return weights;
}

std::vector<GridPoint> make_targets(const Model2D& model,
                                    int first_iz,
                                    int x_stride,
                                    int z_stride)
{
    const std::vector<int> x_indices = regular_indices(0, model.nx - 1, x_stride);
    const std::vector<int> z_indices = regular_indices(first_iz, model.nz - 1, z_stride);
    std::vector<GridPoint> targets;
    targets.reserve(x_indices.size() * z_indices.size());
    for (int ix : x_indices) {
        for (int iz : z_indices) {
            targets.push_back({ix, iz});
        }
    }
    return targets;
}

Complex hankel_point_source(float omega,
                            float velocity,
                            float radius,
                            float source_amplitude)
{
    const double argument = static_cast<double>(omega) *
                            static_cast<double>(radius) /
                            static_cast<double>(velocity);
    const double j0 = se_bessel_j0(argument);
    const double y0 = se_bessel_y0(argument);
    // G = i/4 H_0^(1)(kr) = -Y_0(kr)/4 + i J_0(kr)/4.
    return source_amplitude * Complex(
        static_cast<float>(-0.25 * y0),
        static_cast<float>( 0.25 * j0));
}

std::vector<Complex> build_top_hankel_wavefield(const Model2D& model,
                                                const Options& options,
                                                int source_ix)
{
    const float omega = 2.0f * static_cast<float>(kPi) * options.frequency;
    const float top_velocity = model.velocity[model.index(source_ix, options.source_iz)];
    const float regularization = options.source_radius > 0.0f
        ? options.source_radius
        : 0.5f * std::min(model.dx, model.dz);

    std::vector<Complex> top(
        static_cast<std::size_t>(options.top_layers) * model.nx);
    for (int iz = 0; iz < options.top_layers; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            const float rx = model.x(ix) - model.x(source_ix);
            const float rz = model.z(iz) - model.z(options.source_iz);
            const float radius = std::max(std::sqrt(rx * rx + rz * rz), regularization);
            top[static_cast<std::size_t>(iz) * model.nx + ix] =
                hankel_point_source(omega, top_velocity, radius,
                                    options.source_amplitude);
        }
    }
    return top;
}

std::vector<Complex> make_boundary_data(const Model2D& model,
                                        const Options& options,
                                        const std::vector<GridPoint>& boundary,
                                        const std::vector<Complex>& top_wavefield)
{
    std::vector<Complex> input(2 * boundary.size());
    const int lower_iz = options.datum_iz - 1;
    const int upper_iz = options.datum_iz + 1;
    const float inverse_interval = 1.0f /
        (static_cast<float>(upper_iz - lower_iz) * model.dz);

    for (std::size_t source = 0; source < boundary.size(); ++source) {
        const int ix = boundary[source].ix;
        const Complex value = top_wavefield[
            static_cast<std::size_t>(options.datum_iz) * model.nx + ix];
        const Complex derivative =
            (top_wavefield[static_cast<std::size_t>(upper_iz) * model.nx + ix] -
             top_wavefield[static_cast<std::size_t>(lower_iz) * model.nx + ix]) *
            inverse_interval;
        input[2 * source] = value;
        input[2 * source + 1] = derivative;
    }
    return input;
}

Complex ray_green(float omega, float travel_time)
{
    const float minimum_time = std::max(1.0e-7f, 1.0f / (omega * 1.0e7f));
    const float tau = std::max(travel_time, minimum_time);
    const float amplitude = 1.0f /
        std::sqrt(8.0f * static_cast<float>(kPi) * omega * tau);
    return amplitude * std::exp(Complex(
        0.0f, omega * tau + 0.25f * static_cast<float>(kPi)));
}

Complex ray_green_normal_derivative(float omega,
                                    float travel_time,
                                    float source_normal_travel_time_derivative)
{
    const float minimum_time = std::max(1.0e-7f, 1.0f / (omega * 1.0e7f));
    const float tau = std::max(travel_time, minimum_time);
    return ray_green(omega, tau) *
           Complex(-0.5f / tau, omega) *
           source_normal_travel_time_derivative;
}

struct TraveltimeTables {
    int rows = 0;
    int sources = 0;
    std::vector<float> time;
    std::vector<float> source_normal_derivative;

    std::size_t offset(int source, int row) const
    {
        return static_cast<std::size_t>(source) * rows + row;
    }
};

TraveltimeTables build_boundary_traveltimes(
    const Model2D& model,
    const Options& options,
    const std::vector<GridPoint>& boundary,
    const std::vector<GridPoint>& targets)
{
    TraveltimeTables tables;
    tables.rows = static_cast<int>(targets.size());
    tables.sources = static_cast<int>(boundary.size());

    const double entries = static_cast<double>(tables.rows) * tables.sources;
    const double required_mb = 2.0 * entries * sizeof(float) / 1.0e6;
    if (required_mb > options.max_traveltime_mb) {
        throw std::runtime_error(
            "FMM cache requires " + std::to_string(required_mb) +
            " MB, exceeding max_traveltime_mb=" +
            std::to_string(options.max_traveltime_mb) +
            "; increase boundary/target strides or the memory limit");
    }

    tables.time.resize(static_cast<std::size_t>(tables.rows) * tables.sources);
    tables.source_normal_derivative.resize(
        static_cast<std::size_t>(tables.rows) * tables.sources);

    const int minus_iz = options.datum_iz - 1;
    const int plus_iz = options.datum_iz + 1;
    const float inverse_interval = 1.0f /
        (static_cast<float>(plus_iz - minus_iz) * model.dz);
    const int progress_step = std::max(1, tables.sources / 20);

    for (int source = 0; source < tables.sources; ++source) {
        const int source_ix = boundary[static_cast<std::size_t>(source)].ix;
        const std::vector<float> center = solve_fmm(model, source_ix, options.datum_iz);
        const std::vector<float> minus = solve_fmm(model, source_ix, minus_iz);
        const std::vector<float> plus = solve_fmm(model, source_ix, plus_iz);

        for (int row = 0; row < tables.rows; ++row) {
            const GridPoint target = targets[static_cast<std::size_t>(row)];
            const std::size_t grid = model.index(target.ix, target.iz);
            const std::size_t out = tables.offset(source, row);
            tables.time[out] = center[grid];
            tables.source_normal_derivative[out] =
                (plus[grid] - minus[grid]) * inverse_interval;
        }

        if ((source + 1) % progress_step == 0 || source + 1 == tables.sources) {
            INFO(("Boundary FMM %d/%d", source + 1, tables.sources));
        }
    }
    return tables;
}

struct KirchhoffKernel {
    int rows = 0;
    int sources = 0;
    float omega = 0.0f;
    const std::vector<float>* weights = nullptr;
    const TraveltimeTables* tables = nullptr;

    Complex entry(int row, int column) const
    {
        const int source = column / 2;
        const std::size_t offset = tables->offset(source, row);
        const float weight = (*weights)[static_cast<std::size_t>(source)];
        if ((column & 1) == 0) {
            return -weight * ray_green_normal_derivative(
                omega,
                tables->time[offset],
                tables->source_normal_derivative[offset]);
        }
        return weight * ray_green(omega, tables->time[offset]);
    }

    Complex apply_pair(int row,
                       int source,
                       const Complex& value,
                       const Complex& normal_derivative) const
    {
        const std::size_t offset = tables->offset(source, row);
        const float weight = (*weights)[static_cast<std::size_t>(source)];
        const Complex green = ray_green(omega, tables->time[offset]);
        const Complex green_normal = ray_green_normal_derivative(
            omega,
            tables->time[offset],
            tables->source_normal_derivative[offset]);
        return weight * (green * normal_derivative - value * green_normal);
    }
};

std::vector<Complex> direct_apply(const KirchhoffKernel& kernel,
                                  const std::vector<Complex>& input)
{
    if (input.size() != static_cast<std::size_t>(2 * kernel.sources)) {
        throw std::invalid_argument("direct Kirchhoff input length is invalid");
    }
    std::vector<Complex> output(static_cast<std::size_t>(kernel.rows), Complex(0.0f, 0.0f));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < kernel.rows; ++row) {
        Complex sum(0.0f, 0.0f);
        for (int source = 0; source < kernel.sources; ++source) {
            sum += kernel.apply_pair(
                row,
                source,
                input[static_cast<std::size_t>(2 * source)],
                input[static_cast<std::size_t>(2 * source + 1)]);
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
    return output;
}

void inject_top_and_targets(const Model2D& model,
                            const Options& options,
                            const std::vector<Complex>& top_wavefield,
                            const std::vector<GridPoint>& targets,
                            const std::vector<Complex>& target_values,
                            std::vector<Complex>& full_wavefield)
{
    full_wavefield.assign(
        static_cast<std::size_t>(model.nx) * model.nz, Complex(0.0f, 0.0f));
    for (int iz = 0; iz < options.top_layers; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            full_wavefield[model.index(ix, iz)] =
                top_wavefield[static_cast<std::size_t>(iz) * model.nx + ix];
        }
    }
    for (std::size_t row = 0; row < targets.size(); ++row) {
        full_wavefield[model.index(targets[row].ix, targets[row].iz)] =
            target_values[row];
    }
}

void configure_wavefield_header(sep_t* output,
                                const Model2D& model,
                                float frequency,
                                const char* method,
                                const char* component)
{
    output->headers->ndim = 2;
    output->headers->n[0] = model.nz;
    output->headers->n[1] = model.nx;
    output->headers->d[0] = model.dz;
    output->headers->d[1] = model.dx;
    output->headers->o[0] = model.oz;
    output->headers->o[1] = model.ox;
    output->headers->esize = 4;
    output->headers->le = 1;

    sep_set_header(output, "label1", "Depth");
    sep_set_header(output, "unit1", "m");
    sep_set_header(output, "label2", "Distance");
    sep_set_header(output, "unit2", "m");
    sep_set_header(output, "method", method);
    sep_set_header(output, "component", component);
    sep_set_header(output, "data_format", "native_float");
    sep_set_header(output, "title", "Single-frequency Cauchy-Kirchhoff wavefield");
    sep_set_header_float(output, "frequency", frequency);
    sep_set_header_int(output, "esize", 4);
}

void write_wavefield_component_rsf(const std::string& filename,
                                   const Model2D& model,
                                   const std::vector<Complex>& wavefield,
                                   float frequency,
                                   const char* method,
                                   bool imaginary)
{
    const std::size_t plane = static_cast<std::size_t>(model.nz) * model.nx;
    if (wavefield.size() != plane) {
        throw std::invalid_argument("output wavefield size does not match model");
    }

    std::vector<float> rsf_order(plane);
    for (int ix = 0; ix < model.nx; ++ix) {
        for (int iz = 0; iz < model.nz; ++iz) {
            const Complex value = wavefield[model.index(ix, iz)];
            rsf_order[static_cast<std::size_t>(iz) +
                      static_cast<std::size_t>(ix) * model.nz] =
                imaginary ? value.imag() : value.real();
        }
    }

    sep_t* output = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (output == nullptr || output->headers == nullptr || output->data == nullptr ||
        output->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for output " + filename);
    }
    configure_wavefield_header(
        output, model, frequency, method, imaginary ? "imaginary" : "real");
    se_fsio_write_float(output->data->io, rsf_order.data(), rsf_order.size());
    sep_close(output);
}

void write_traveltime_table_rsf(const std::string& filename,
                                const Model2D& model,
                                const Options& options,
                                const std::vector<GridPoint>& boundary,
                                const TraveltimeTables& tables,
                                bool derivative)
{
    if (filename.empty()) return;

    const std::vector<int> x_indices = regular_indices(
        0, model.nx - 1, options.target_x_stride);
    const std::vector<int> z_indices = regular_indices(
        options.top_layers, model.nz - 1, options.target_z_stride);
    const std::size_t expected_rows = x_indices.size() * z_indices.size();
    const std::size_t expected_size = expected_rows * boundary.size();
    const std::vector<float>& values = derivative
        ? tables.source_normal_derivative
        : tables.time;

    if (tables.rows != static_cast<int>(expected_rows) ||
        tables.sources != static_cast<int>(boundary.size()) ||
        values.size() != expected_size) {
        throw std::runtime_error("traveltime table dimensions are inconsistent");
    }

    sep_t* output = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (output == nullptr || output->headers == nullptr || output->data == nullptr ||
        output->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for traveltime output " + filename);
    }

    output->headers->ndim = 3;
    output->headers->n[0] = static_cast<int>(z_indices.size());
    output->headers->n[1] = static_cast<int>(x_indices.size());
    output->headers->n[2] = static_cast<int>(boundary.size());
    output->headers->d[0] = model.dz * options.target_z_stride;
    output->headers->d[1] = model.dx * options.target_x_stride;
    output->headers->d[2] = model.dx * options.boundary_stride;
    output->headers->o[0] = model.z(z_indices.front());
    output->headers->o[1] = model.x(x_indices.front());
    output->headers->o[2] = model.x(boundary.front().ix);
    output->headers->esize = 4;
    output->headers->le = 1;

    sep_set_header(output, "label1", "Target depth");
    sep_set_header(output, "unit1", "m");
    sep_set_header(output, "label2", "Target distance");
    sep_set_header(output, "unit2", "m");
    sep_set_header(output, "label3", "Datum source distance");
    sep_set_header(output, "unit3", "m");
    sep_set_header(output, "data_format", "native_float");
    sep_set_header(output, "quantity",
                   derivative ? "source_normal_traveltime_derivative"
                              : "traveltime");
    sep_set_header(output, "title",
                   derivative ? "Source-normal traveltime derivative table"
                              : "FMM traveltime table");
    sep_set_header_int(output, "datum_iz", options.datum_iz);
    sep_set_header_float(output, "datum_z", model.z(options.datum_iz));
    sep_set_header_int(output, "boundary_stride", options.boundary_stride);
    sep_set_header_int(output, "target_x_stride", options.target_x_stride);
    sep_set_header_int(output, "target_z_stride", options.target_z_stride);
    sep_set_header_int(output, "esize", 4);

    se_fsio_write_float(output->data->io, values.data(), values.size());
    sep_close(output);
}

double maximum_normalized_error(const std::vector<Complex>& reference,
                                const std::vector<Complex>& candidate)
{
    float maximum_reference = 0.0f;
    float maximum_difference = 0.0f;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        maximum_reference = std::max(maximum_reference, std::abs(reference[i]));
        maximum_difference = std::max(
            maximum_difference, std::abs(candidate[i] - reference[i]));
    }
    return static_cast<double>(maximum_difference /
        std::max(maximum_reference, 1.0e-30f));
}

int run(const Options& options, const Model2D& model)
{
#ifndef KIRCH_HAS_BUTTERFLYPACK_FLOAT
    (void)options;
    (void)model;
    throw std::runtime_error(
        "this program requires KIRCH_BPACK_ENABLE_FLOAT=ON");
#else
    const int source_ix = options.source_ix >= 0 ? options.source_ix : model.nx / 2;
    if (source_ix < 0 || source_ix >= model.nx ||
        options.source_iz < 0 || options.source_iz >= model.nz) {
        throw std::invalid_argument("source index lies outside the model");
    }
    if (options.top_layers > model.nz || options.datum_iz + 1 >= model.nz) {
        throw std::invalid_argument("top_layers/datum_iz is incompatible with model nz");
    }
    if (options.source_iz != 0) {
        INFO(("Warning: source_iz is %d; the requested workflow normally uses source_iz=0",
              options.source_iz));
    }

    const std::vector<GridPoint> boundary =
        make_boundary(model, options.datum_iz, options.boundary_stride);
    const std::vector<float> weights = trapezoidal_weights(model, boundary);
    const std::vector<GridPoint> targets = make_targets(
        model, options.top_layers, options.target_x_stride, options.target_z_stride);
    if (targets.empty()) {
        throw std::runtime_error("target sampling produced no points");
    }

    INFO(("Model n1/nz=%d n2/nx=%d d1/dz=%g d2/dx=%g",
          model.nz, model.nx, model.dz, model.dx));
    INFO(("Frequency=%g Hz, source=(ix=%d, iz=%d), datum iz=%d",
          options.frequency, source_ix, options.source_iz, options.datum_iz));
    INFO(("Boundary points=%zu, target points=%zu",
          boundary.size(), targets.size()));

    const std::vector<Complex> top_wavefield =
        build_top_hankel_wavefield(model, options, source_ix);
    const std::vector<Complex> boundary_input =
        make_boundary_data(model, options, boundary, top_wavefield);

    Timings timings;
    INFO(("Computing center/minus/plus FMM fields for every datum point"));
    auto started = Clock::now();
    const TraveltimeTables tables = build_boundary_traveltimes(
        model, options, boundary, targets);
    timings.boundary_fmm_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();

    INFO(("Writing FMM traveltime diagnostics"));
    write_traveltime_table_rsf(
        options.traveltime_file, model, options, boundary, tables, false);
    write_traveltime_table_rsf(
        options.traveltime_derivative_file, model, options, boundary, tables, true);

    KirchhoffKernel kernel;
    kernel.rows = static_cast<int>(targets.size());
    kernel.sources = static_cast<int>(boundary.size());
    kernel.omega = 2.0f * static_cast<float>(kPi) * options.frequency;
    kernel.weights = &weights;
    kernel.tables = &tables;

    INFO(("Applying matrix-free direct Cauchy-Kirchhoff integral"));
    started = Clock::now();
    const std::vector<Complex> direct_targets = direct_apply(kernel, boundary_input);
    timings.direct_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();

    std::vector<double> row_coordinates(2 * targets.size());
    for (std::size_t row = 0; row < targets.size(); ++row) {
        row_coordinates[2 * row] = model.x(targets[row].ix);
        row_coordinates[2 * row + 1] = model.z(targets[row].iz);
    }
    std::vector<double> column_coordinates(4 * boundary.size());
    for (std::size_t source = 0; source < boundary.size(); ++source) {
        const double x = model.x(boundary[source].ix);
        const double z = model.z(boundary[source].iz);
        column_coordinates[4 * source] = x;
        column_coordinates[4 * source + 1] = z;
        column_coordinates[4 * source + 2] = x;
        column_coordinates[4 * source + 3] = z + 1.0e-6 * model.dz;
    }

    se::butterfly::Options bpack_options;
    bpack_options.tolerance = options.tolerance;
    bpack_options.leaf_size = options.leaf_size;
    bpack_options.verbosity = options.verbosity;
    bpack_options.coordinate_dimension = 2;

    INFO(("Building ButterflyPACK operator with %d rows and %d columns",
          kernel.rows, 2 * kernel.sources));
    se::butterfly::Matrix<float> butterfly(
        kernel.rows,
        2 * kernel.sources,
        std::move(row_coordinates),
        std::move(column_coordinates),
        [&kernel](int row, int column) { return kernel.entry(row, column); },
        bpack_options);
    timings.butterfly_build_seconds = butterfly.statistics().build_seconds;

    const std::vector<Complex> butterfly_targets = butterfly.apply(boundary_input);
    timings.butterfly_apply_seconds = butterfly.statistics().apply_seconds;

    const double relative_error = static_cast<double>(
        se::blas::relative_l2_error(direct_targets, butterfly_targets));
    const double maximum_error = maximum_normalized_error(
        direct_targets, butterfly_targets);

    std::vector<Complex> direct_full;
    std::vector<Complex> butterfly_full;
    inject_top_and_targets(
        model, options, top_wavefield, targets, direct_targets, direct_full);
    inject_top_and_targets(
        model, options, top_wavefield, targets, butterfly_targets, butterfly_full);

    write_wavefield_component_rsf(
        options.direct_output_real, model, direct_full,
        options.frequency, "direct", false);
    write_wavefield_component_rsf(
        options.direct_output_imag, model, direct_full,
        options.frequency, "direct", true);
    write_wavefield_component_rsf(
        options.butterfly_output_real, model, butterfly_full,
        options.frequency, "butterfly", false);
    write_wavefield_component_rsf(
        options.butterfly_output_imag, model, butterfly_full,
        options.frequency, "butterfly", true);

    // Fair comparison: FMM traveltime calculation and all RSF I/O are excluded.
    const double butterfly_total =
        timings.butterfly_build_seconds + timings.butterfly_apply_seconds;
    const double speedup_total = timings.direct_seconds /
        std::max(butterfly_total, 1.0e-30);
    const double speedup_apply = timings.direct_seconds /
        std::max(timings.butterfly_apply_seconds, 1.0e-30);
    const se::butterfly::Statistics& stats = butterfly.statistics();

    std::ostringstream report;
    report << std::setprecision(12)
           << "frequency_hz=" << options.frequency << '\n'
           << "model_nz=" << model.nz << '\n'
           << "model_nx=" << model.nx << '\n'
           << "source_ix=" << source_ix << '\n'
           << "source_iz=" << options.source_iz << '\n'
           << "datum_iz=" << options.datum_iz << '\n'
           << "top_layers=" << options.top_layers << '\n'
           << "boundary_stride=" << options.boundary_stride << '\n'
           << "target_x_stride=" << options.target_x_stride << '\n'
           << "target_z_stride=" << options.target_z_stride << '\n'
           << "boundary_points=" << boundary.size() << '\n'
           << "target_points=" << targets.size() << '\n'
           << "matrix_rows=" << kernel.rows << '\n'
           << "matrix_columns=" << 2 * kernel.sources << '\n'
           << "traveltime_cache_megabytes="
           << 2.0 * static_cast<double>(kernel.rows) * kernel.sources * sizeof(float) / 1.0e6
           << '\n'
           << "traveltime_fmm_seconds_excluded_from_comparison="
           << timings.boundary_fmm_seconds << '\n'
           << "direct_kirchhoff_seconds=" << timings.direct_seconds << '\n'
           << "butterfly_build_seconds=" << timings.butterfly_build_seconds << '\n'
           << "butterfly_apply_seconds=" << timings.butterfly_apply_seconds << '\n'
           << "butterfly_build_plus_apply_seconds=" << butterfly_total << '\n'
           << "speedup_direct_over_butterfly_total=" << speedup_total << '\n'
           << "speedup_direct_over_butterfly_apply=" << speedup_apply << '\n'
           << "timing_scope=excludes_hankel_fmm_traveltime_io_wavefield_io\n"
           << "relative_l2_error=" << relative_error << '\n'
           << "maximum_normalized_error=" << maximum_error << '\n'
           << "butterfly_compressed_megabytes=" << stats.compressed_megabytes << '\n'
           << "butterfly_peak_megabytes=" << stats.peak_megabytes << '\n'
           << "butterfly_maximum_rank=" << stats.maximum_rank << '\n'
           << "direct_real_file=" << options.direct_output_real << '\n'
           << "direct_imag_file=" << options.direct_output_imag << '\n'
           << "butterfly_real_file=" << options.butterfly_output_real << '\n'
           << "butterfly_imag_file=" << options.butterfly_output_imag << '\n'
           << "traveltime_file=" << options.traveltime_file << '\n'
           << "traveltime_derivative_file="
           << options.traveltime_derivative_file << '\n';

    std::cout << "\n========== Kirchhoff 1D-to-2D comparison ==========\n"
              << report.str()
              << "====================================================\n";
    if (!options.timing_file.empty()) {
        std::ofstream timing(options.timing_file);
        if (!timing) {
            throw std::runtime_error("cannot create timing file " + options.timing_file);
        }
        timing << report.str();
    }
    return 0;
#endif
}

}  // namespace

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        const Options options = parse_options(argc, argv);
#ifdef _OPENMP
        if (options.threads > 0) omp_set_num_threads(options.threads);
#endif
        const Model2D model = read_velocity_model(options.velocity_file);
        const int status = run(options, model);
        se_par_destroy();
        return status;
    } catch (const std::exception& error) {
        std::cerr << "kirchhoff_butterfly_rsf: " << error.what() << '\n';
        std::cerr << "Use help=1 for command-line options.\n";
        if (se_par_initialized()) se_par_destroy();
        return 1;
    }
}