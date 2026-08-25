#ifndef FREQUENCY_KIRCHHOFF_IMAGING_COMMON_HPP
#define FREQUENCY_KIRCHHOFF_IMAGING_COMMON_HPP

#include <SERECKIRCH/include/huygens_cli.hpp>

#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_fs.h>
#include <SERECKIRCH/include/huygens_sweep.hpp>

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace frequency_kirchhoff_imaging {

using Complex = se::huygens::Complex;
using Clock = std::chrono::steady_clock;
constexpr float kPi = 3.14159265358979323846f;

struct CommonOptions {
    std::string velocity;
    std::string block_file;
    std::string table_prefix;
    std::string seismic_data;
    std::string image;
    std::string illumination;
    std::string timing;

    int cmp = 1;
    int shot_begin = 0;
    int shot_count = -1;
    int shot_stride = 1;
    int aperture_trace = -1;
    float aperture_distance = -1.0f;

    float fdom = 20.0f;
    float source_time = -1.0f;
    float source_amplitude = 1.0f;
    int nfft = 0;
    float fmin = 3.0f;
    float fmax = 45.0f;
    int frequency_stride = 1;

    float filter_dt = 0.0f;
    float filter_length = 0.025f;
    int filter_lookup_subsamples = 64;

    int source_stride = 1;
    int target_x_stride = 1;
    int target_z_stride = 1;
    int threads = 0;
    float max_state_mb = 8192.0f;
    int source_normalization = 0;
    float normalization_epsilon = 1.0e-6f;
};

inline CommonOptions parse_common_options(const std::string& method)
{
    CommonOptions o;
    o.velocity = huygens_cli::required_string("velocity");
    o.block_file = huygens_cli::required_string("block_file");
    o.table_prefix = huygens_cli::required_string("table_prefix");
    o.seismic_data = huygens_cli::required_string("seismic_data");
    o.image = huygens_cli::optional_string(
        "image", method + "_image.rsf");
    o.illumination = huygens_cli::optional_string("illumination", "");
    o.timing = huygens_cli::optional_string(
        "timing", method + "_timing.rsf");

    o.cmp = huygens_cli::optional_int("cmp", 1);
    o.shot_begin = huygens_cli::optional_int("shot_begin", 0);
    o.shot_count = huygens_cli::optional_int("shot_count", -1);
    o.shot_stride = huygens_cli::optional_int("shot_stride", 1);
    o.aperture_trace = huygens_cli::optional_int("aperture_trace", -1);
    o.aperture_distance = huygens_cli::optional_float(
        "aperture_distance", -1.0f);

    o.fdom = huygens_cli::optional_float("fdom", 20.0f);
    o.source_time = huygens_cli::optional_float("source_time", -1.0f);
    o.source_amplitude = huygens_cli::optional_float(
        "source_amplitude", 1.0f);
    o.nfft = huygens_cli::optional_int("nfft", 0);
    o.fmin = huygens_cli::optional_float("fmin", 3.0f);
    o.fmax = huygens_cli::optional_float("fmax", 45.0f);
    o.frequency_stride = huygens_cli::optional_int("frequency_stride", 1);

    o.filter_dt = huygens_cli::optional_float("filter_dt", 0.0f);
    o.filter_length = huygens_cli::optional_float("filter_length", 0.025f);
    o.filter_lookup_subsamples = huygens_cli::optional_int(
        "filter_lookup_subsamples", 64);

    o.source_stride = huygens_cli::optional_int("source_stride", 1);
    o.target_x_stride = huygens_cli::optional_int("target_x_stride", 1);
    o.target_z_stride = huygens_cli::optional_int("target_z_stride", 1);
    o.threads = huygens_cli::optional_int("threads", 0);
    o.max_state_mb = huygens_cli::optional_float("max_state_mb", 8192.0f);
    o.source_normalization = huygens_cli::optional_int(
        "source_normalization", 0);
    o.normalization_epsilon = huygens_cli::optional_float(
        "normalization_epsilon", 1.0e-6f);

    if (o.cmp != 0 && o.cmp != 1) {
        throw std::invalid_argument("cmp must be 0 or 1");
    }
    if (o.shot_begin < 0 || o.shot_stride < 1 || o.shot_count == 0) {
        throw std::invalid_argument(
            "shot_begin must be non-negative, shot_stride positive, and shot_count nonzero");
    }
    if (!(o.fdom > 0.0f) || !(o.source_amplitude != 0.0f) ||
        o.nfft < 0 || !(o.fmin >= 0.0f) || !(o.fmax > o.fmin) ||
        o.frequency_stride < 1) {
        throw std::invalid_argument("invalid source/frequency parameters");
    }
    if (o.filter_dt < 0.0f || !(o.filter_length > 0.0f) ||
        o.filter_lookup_subsamples < 1) {
        throw std::invalid_argument("invalid one-way filter parameters");
    }
    if (o.source_stride != 1 || o.target_x_stride != 1 ||
        o.target_z_stride != 1) {
        throw std::invalid_argument(
            "frequency-domain imaging currently requires all spatial strides to equal one");
    }
    if (o.threads < 0 || !(o.max_state_mb > 0.0f) ||
        (o.source_normalization != 0 && o.source_normalization != 1) ||
        !(o.normalization_epsilon > 0.0f)) {
        throw std::invalid_argument("invalid thread or normalization parameters");
    }
    return o;
}

class SeismicData {
public:
    explicit SeismicData(const std::string& filename, bool cmp_geometry)
        : filename_(filename), cmp_(cmp_geometry)
    {
        file_ = sep_open(filename.c_str(), SEP_READ, 0);
        if (file_ == nullptr || file_->headers == nullptr ||
            file_->data == nullptr || file_->data->io == nullptr) {
            throw std::runtime_error(
                "sep_open failed for seismic data: " + filename);
        }
        if (file_->headers->ndim < 2) {
            throw std::runtime_error(
                "seismic_data must have axes (time, offset/receiver[, shot])");
        }
        nt_ = file_->headers->n[0];
        ntrace_ = file_->headers->n[1];
        nshot_ = file_->headers->ndim >= 3 ? file_->headers->n[2] : 1;
        t0_ = static_cast<float>(file_->headers->o[0]);
        dt_ = static_cast<float>(file_->headers->d[0]);
        axis2_o_ = static_cast<float>(file_->headers->o[1]);
        axis2_d_ = ntrace_ > 1
            ? static_cast<float>(file_->headers->d[1])
            : 0.0f;
        shot_o_ = file_->headers->ndim >= 3
            ? static_cast<float>(file_->headers->o[2])
            : 0.0f;
        shot_d_ = nshot_ > 1
            ? static_cast<float>(file_->headers->d[2])
            : 0.0f;
        if (nt_ < 2 || ntrace_ < 1 || nshot_ < 1 || !(dt_ > 0.0f)) {
            throw std::runtime_error("seismic_data contains invalid axes");
        }
    }

    ~SeismicData()
    {
        if (file_ != nullptr) sep_close(file_);
    }

    SeismicData(const SeismicData&) = delete;
    SeismicData& operator=(const SeismicData&) = delete;

    int nt() const noexcept { return nt_; }
    int ntrace() const noexcept { return ntrace_; }
    int nshot() const noexcept { return nshot_; }
    float t0() const noexcept { return t0_; }
    float dt() const noexcept { return dt_; }
    float axis2_spacing() const noexcept { return axis2_d_; }

    float shot_x(int shot) const
    {
        return shot_o_ + static_cast<float>(shot) * shot_d_;
    }

    float receiver_x(int shot, int trace) const
    {
        const float second_axis =
            axis2_o_ + static_cast<float>(trace) * axis2_d_;
        return cmp_ ? shot_x(shot) + second_axis : second_axis;
    }

    std::vector<float> read_shot(int shot) const
    {
        if (shot < 0 || shot >= nshot_) {
            throw std::out_of_range("shot index is outside seismic_data");
        }
        const std::size_t samples =
            static_cast<std::size_t>(nt_) * ntrace_;
        const off_t byte_offset = static_cast<off_t>(shot) *
            static_cast<off_t>(samples) * static_cast<off_t>(sizeof(float));
        if (se_fsio_seek(file_->data->io, byte_offset) != CODE_SUCCESS) {
            throw std::runtime_error(
                "failed to seek shot " + std::to_string(shot) +
                " in seismic_data");
        }
        std::vector<float> values(samples);
        if (se_fsio_read_float(file_->data->io, values.data(), samples) !=
            CODE_SUCCESS) {
            throw std::runtime_error(
                "failed to read shot " + std::to_string(shot) +
                " from seismic_data");
        }
        return values;
    }

private:
    std::string filename_;
    bool cmp_ = true;
    mutable sep_t* file_ = nullptr;
    int nt_ = 0;
    int ntrace_ = 0;
    int nshot_ = 0;
    float t0_ = 0.0f;
    float dt_ = 0.0f;
    float axis2_o_ = 0.0f;
    float axis2_d_ = 0.0f;
    float shot_o_ = 0.0f;
    float shot_d_ = 0.0f;
};

struct FrequencyAxis {
    int nfft = 0;
    std::vector<int> bins;
    std::vector<float> frequencies;
    std::vector<float> imaging_weights;
};

inline int next_power_of_two(int value)
{
    int result = 1;
    while (result < value) {
        if (result > std::numeric_limits<int>::max() / 2) {
            throw std::overflow_error("FFT length overflow");
        }
        result *= 2;
    }
    return result;
}

inline FrequencyAxis make_frequency_axis(const SeismicData& data,
                                         const CommonOptions& options)
{
    FrequencyAxis axis;
    axis.nfft = options.nfft > 0
        ? options.nfft
        : next_power_of_two(data.nt());
    if (axis.nfft < data.nt()) {
        throw std::invalid_argument("nfft must be at least the data time length");
    }
    const float df = 1.0f / (static_cast<float>(axis.nfft) * data.dt());
    const float nyquist = 0.5f / data.dt();
    const float upper = std::min(options.fmax, nyquist);
    const int positive_bins = axis.nfft / 2 + 1;
    for (int bin = 1; bin < positive_bins; bin += options.frequency_stride) {
        const float frequency = static_cast<float>(bin) * df;
        if (frequency + 1.0e-6f < options.fmin ||
            frequency - 1.0e-6f > upper) {
            continue;
        }
        axis.bins.push_back(bin);
        axis.frequencies.push_back(frequency);
        const bool nyquist_bin =
            axis.nfft % 2 == 0 && bin == axis.nfft / 2;
        const float one_sided = nyquist_bin ? 1.0f : 2.0f;
        axis.imaging_weights.push_back(
            one_sided * static_cast<float>(options.frequency_stride) /
            (static_cast<float>(axis.nfft) * data.dt()));
    }
    if (axis.bins.empty()) {
        throw std::invalid_argument(
            "no FFT bins lie inside the requested frequency interval");
    }
    return axis;
}

inline std::vector<Complex> fft_traces(
    const std::vector<float>& traces,
    int trace_count,
    const SeismicData& data,
    const FrequencyAxis& axis)
{
    if (traces.size() !=
        static_cast<std::size_t>(trace_count) * data.nt()) {
        throw std::invalid_argument("trace buffer size is invalid");
    }
    std::vector<float> work(static_cast<std::size_t>(axis.nfft), 0.0f);
    const std::size_t spectrum_size =
        static_cast<std::size_t>(axis.nfft / 2 + 1);
    // fftwf_complex is typedef'd as float[2]. Array types cannot be used as
    // std::vector elements with Apple libc++, so keep an interleaved scalar
    // buffer and cast only at the FFTW C API boundary.
    std::vector<float> spectrum_storage(2 * spectrum_size, 0.0f);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(
        axis.nfft, work.data(),
        reinterpret_cast<fftwf_complex*>(spectrum_storage.data()),
        FFTW_ESTIMATE);
    if (plan == nullptr) {
        throw std::runtime_error("FFTW failed to create the data FFT plan");
    }

    std::vector<Complex> output(
        axis.bins.size() * static_cast<std::size_t>(trace_count));
    for (int trace = 0; trace < trace_count; ++trace) {
        std::fill(work.begin(), work.end(), 0.0f);
        const float* input = traces.data() +
            static_cast<std::size_t>(trace) * data.nt();
        std::copy(input, input + data.nt(), work.begin());
        fftwf_execute(plan);
        for (std::size_t ifrequency = 0;
             ifrequency < axis.bins.size(); ++ifrequency) {
            const int bin = axis.bins[ifrequency];
            const std::size_t offset =
                2 * static_cast<std::size_t>(bin);
            Complex value(spectrum_storage[offset],
                          spectrum_storage[offset + 1]);
            const float omega = 2.0f * kPi * axis.frequencies[ifrequency];
            value *= data.dt() *
                std::exp(Complex(0.0f, -omega * data.t0()));
            output[ifrequency * static_cast<std::size_t>(trace_count) +
                   static_cast<std::size_t>(trace)] = value;
        }
    }
    fftwf_destroy_plan(plan);
    return output;
}

inline std::vector<Complex> ricker_spectrum(
    const SeismicData& data,
    const FrequencyAxis& axis,
    float dominant_frequency,
    float source_time,
    float amplitude)
{
    if (!(source_time >= data.t0())) {
        source_time = data.t0() + 1.5f / dominant_frequency;
    }
    std::vector<float> samples(static_cast<std::size_t>(axis.nfft), 0.0f);
    const float p2 = kPi * kPi * dominant_frequency * dominant_frequency;
    for (int it = 0; it < data.nt(); ++it) {
        const float t = data.t0() + static_cast<float>(it) * data.dt();
        const float delay = t - source_time;
        const float q = p2 * delay * delay;
        samples[static_cast<std::size_t>(it)] =
            amplitude * (1.0f - 2.0f * q) * std::exp(-q);
    }
    const std::size_t spectrum_size =
        static_cast<std::size_t>(axis.nfft / 2 + 1);
    std::vector<float> spectrum_storage(2 * spectrum_size, 0.0f);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(
        axis.nfft, samples.data(),
        reinterpret_cast<fftwf_complex*>(spectrum_storage.data()),
        FFTW_ESTIMATE);
    if (plan == nullptr) {
        throw std::runtime_error("FFTW failed to create the Ricker FFT plan");
    }
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    std::vector<Complex> output(axis.bins.size());
    for (std::size_t ifrequency = 0;
         ifrequency < axis.bins.size(); ++ifrequency) {
        const int bin = axis.bins[ifrequency];
        const std::size_t offset =
            2 * static_cast<std::size_t>(bin);
        Complex value(spectrum_storage[offset],
                      spectrum_storage[offset + 1]);
        const float omega = 2.0f * kPi * axis.frequencies[ifrequency];
        output[ifrequency] = data.dt() * value *
            std::exp(Complex(0.0f, -omega * data.t0()));
    }
    return output;
}

inline std::vector<int> selected_shots(const SeismicData& data,
                                       const CommonOptions& options)
{
    if (options.shot_begin >= data.nshot()) {
        throw std::invalid_argument("shot_begin lies outside seismic_data");
    }
    const int available = 1 +
        (data.nshot() - 1 - options.shot_begin) / options.shot_stride;
    const int count = options.shot_count < 0
        ? available
        : std::min(options.shot_count, available);
    std::vector<int> shots(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        shots[static_cast<std::size_t>(i)] =
            options.shot_begin + i * options.shot_stride;
    }
    return shots;
}


inline double boundary_state_megabytes(std::size_t shot_count,
                                       std::size_t frequency_count,
                                       int nx)
{
    return 2.0 * static_cast<double>(shot_count) *
           static_cast<double>(frequency_count) * static_cast<double>(nx) *
           static_cast<double>(sizeof(Complex)) / (1024.0 * 1024.0);
}

inline void validate_boundary_state_memory(const CommonOptions& options,
                                           std::size_t shot_count,
                                           std::size_t frequency_count,
                                           int nx)
{
    const double required = boundary_state_megabytes(
        shot_count, frequency_count, nx);
    if (required > static_cast<double>(options.max_state_mb)) {
        throw std::runtime_error(
            "source/receiver boundary states require " +
            std::to_string(required) +
            " MiB, exceeding max_state_mb=" +
            std::to_string(options.max_state_mb) +
            "; reduce shot_count/frequency range or raise max_state_mb");
    }
}

inline std::vector<float> full_boundary_weights(
    const se::huygens::Model2D& model)
{
    std::vector<int> indices(static_cast<std::size_t>(model.nx));
    std::iota(indices.begin(), indices.end(), 0);
    return se::huygens::trapezoidal_weights(model, indices);
}

inline bool deposit_integral_sample(
    std::vector<Complex>& surface,
    std::size_t offset,
    const se::huygens::Model2D& model,
    const std::vector<float>& model_weights,
    float x,
    Complex integrated_value)
{
    if (offset + static_cast<std::size_t>(model.nx) > surface.size()) {
        throw std::out_of_range("surface deposition offset is invalid");
    }
    const float coordinate = (x - model.ox) / model.dx;
    if (coordinate < -1.0e-5f ||
        coordinate > static_cast<float>(model.nx - 1) + 1.0e-5f) {
        return false;
    }
    const float clipped = std::max(
        0.0f, std::min(coordinate, static_cast<float>(model.nx - 1)));
    int left = static_cast<int>(std::floor(clipped));
    int right = std::min(left + 1, model.nx - 1);
    const float fraction = clipped - static_cast<float>(left);
    const float left_fraction = right == left ? 1.0f : 1.0f - fraction;
    const float right_fraction = right == left ? 0.0f : fraction;
    surface[offset + static_cast<std::size_t>(left)] +=
        (left_fraction / model_weights[static_cast<std::size_t>(left)]) *
        integrated_value;
    if (right != left) {
        surface[offset + static_cast<std::size_t>(right)] +=
            (right_fraction / model_weights[static_cast<std::size_t>(right)]) *
            integrated_value;
    }
    return true;
}

struct ShotBoundaryState {
    int shot_index = 0;
    float shot_x = 0.0f;
    std::vector<Complex> source;
    // This is conj(U_r), not U_r itself.  It is obtained by applying the same
    // downward one-way operator to conjugated surface data.  Therefore
    // Re{U_s * receiver_conjugate} equals the conventional
    // Re{U_s * conj(U_r)} frequency-domain cross-correlation condition.
    std::vector<Complex> receiver_conjugate;
};

inline ShotBoundaryState initialize_shot_state(
    const SeismicData& data,
    int shot,
    const se::huygens::Model2D& model,
    const CommonOptions& options,
    const FrequencyAxis& axis,
    const std::vector<Complex>& source_spectrum,
    const std::vector<float>& model_weights,
    double& fft_seconds,
    int& skipped_receivers)
{
    const auto started = Clock::now();
    const std::vector<float> gather = data.read_shot(shot);
    const std::vector<Complex> spectra =
        fft_traces(gather, data.ntrace(), data, axis);
    fft_seconds += std::chrono::duration<double>(Clock::now() - started).count();

    ShotBoundaryState state;
    state.shot_index = shot;
    state.shot_x = data.shot_x(shot);
    state.source.assign(axis.bins.size() * static_cast<std::size_t>(model.nx),
                        Complex(0.0f, 0.0f));
    state.receiver_conjugate.assign(state.source.size(), Complex(0.0f, 0.0f));

    for (std::size_t ifrequency = 0;
         ifrequency < axis.bins.size(); ++ifrequency) {
        const std::size_t offset =
            ifrequency * static_cast<std::size_t>(model.nx);
        if (!deposit_integral_sample(
                state.source, offset, model, model_weights, state.shot_x,
                source_spectrum[ifrequency])) {
            throw std::runtime_error(
                "shot coordinate lies outside the velocity model");
        }
    }

    const float trace_spacing = data.ntrace() > 1
        ? std::abs(data.axis2_spacing())
        : model.dx;
    for (int trace = 0; trace < data.ntrace(); ++trace) {
        const float receiver_x = data.receiver_x(shot, trace);
        if (options.aperture_trace >= 0) {
            const float spacing = std::max(
                std::abs(data.axis2_spacing()), model.dx);
            const int trace_distance = static_cast<int>(std::llround(
                std::abs(receiver_x - state.shot_x) / spacing));
            if (trace_distance >= options.aperture_trace) continue;
        }
        if (options.aperture_distance > 0.0f &&
            std::abs(receiver_x - state.shot_x) > options.aperture_distance) {
            continue;
        }
        float receiver_weight = trace_spacing;
        if (data.ntrace() > 1 &&
            (trace == 0 || trace == data.ntrace() - 1)) {
            receiver_weight *= 0.5f;
        }
        bool inside = true;
        for (std::size_t ifrequency = 0;
             ifrequency < axis.bins.size(); ++ifrequency) {
            const Complex value = std::conj(
                spectra[ifrequency * static_cast<std::size_t>(data.ntrace()) +
                        static_cast<std::size_t>(trace)]);
            const std::size_t offset =
                ifrequency * static_cast<std::size_t>(model.nx);
            if (!deposit_integral_sample(
                    state.receiver_conjugate, offset, model, model_weights,
                    receiver_x, receiver_weight * value)) {
                inside = false;
                break;
            }
        }
        if (!inside) ++skipped_receivers;
    }
    return state;
}

inline void direct_apply_pair(
    const se::huygens::OneWayKernelData& kernel,
    const Complex* source_input,
    const Complex* receiver_conjugate_input,
    Complex* source_output,
    Complex* receiver_conjugate_output)
{
    if (source_input == nullptr || receiver_conjugate_input == nullptr ||
        source_output == nullptr || receiver_conjugate_output == nullptr) {
        throw std::invalid_argument("direct imaging input/output pointer is null");
    }
    for (int row = 0; row < kernel.rows; ++row) {
        Complex source_sum(0.0f, 0.0f);
        Complex receiver_sum(0.0f, 0.0f);
        for (int source = 0; source < kernel.sources; ++source) {
            const Complex entry = kernel.entry(row, source);
            source_sum += entry * source_input[source];
            receiver_sum += entry * receiver_conjugate_input[source];
        }
        source_output[row] = source_sum;
        receiver_conjugate_output[row] = receiver_sum;
    }
}

inline se::huygens::Block imaging_propagation_block(
    const se::huygens::BlockInfo& info,
    std::size_t block_index)
{
    if (block_index >= info.blocks.size()) {
        throw std::out_of_range("imaging propagation block index is invalid");
    }
    se::huygens::Block block = info.blocks[block_index];
    if (block_index == 0) {
        block.target_start_iz = 1;
        block.halo_end_iz = block.target_end_iz;
    }
    return block;
}

inline int find_local_depth(const std::vector<int>& depths, int global_depth)
{
    const auto iterator = std::find(depths.begin(), depths.end(), global_depth);
    return iterator == depths.end()
        ? -1
        : static_cast<int>(std::distance(depths.begin(), iterator));
}

inline void accumulate_image(
    const se::huygens::Model2D& model,
    const FrequencyAxis& axis,
    const std::vector<int>& target_depths,
    const std::vector<Complex>& source_output,
    const std::vector<Complex>& receiver_conjugate_output,
    std::vector<float>& image,
    std::vector<float>& illumination)
{
    const std::size_t per_frequency =
        static_cast<std::size_t>(target_depths.size()) * model.nx;
    if (source_output.size() != axis.bins.size() * per_frequency ||
        receiver_conjugate_output.size() != source_output.size()) {
        throw std::invalid_argument("propagated imaging wavefield size is invalid");
    }
    for (std::size_t ifrequency = 0;
         ifrequency < axis.bins.size(); ++ifrequency) {
        const float weight = axis.imaging_weights[ifrequency];
        for (std::size_t iz_local = 0;
             iz_local < target_depths.size(); ++iz_local) {
            const int iz = target_depths[iz_local];
            for (int ix = 0; ix < model.nx; ++ix) {
                const std::size_t local =
                    (ifrequency * target_depths.size() + iz_local) *
                        static_cast<std::size_t>(model.nx) + ix;
                const Complex source = source_output[local];
                const Complex receiver_conjugate =
                    receiver_conjugate_output[local];
                const std::size_t grid = model.index(ix, iz);
                image[grid] += weight *
                    (source * receiver_conjugate).real();
                illumination[grid] += weight * std::norm(source);
            }
        }
    }
}

inline void extract_next_boundary(
    const se::huygens::Model2D& model,
    const FrequencyAxis& axis,
    const std::vector<int>& target_depths,
    int next_depth,
    const std::vector<Complex>& source_output,
    const std::vector<Complex>& receiver_output,
    ShotBoundaryState& state)
{
    const int iz_local = find_local_depth(target_depths, next_depth);
    if (iz_local < 0) {
        throw std::runtime_error(
            "next propagation datum is not contained in the current target block");
    }
    for (std::size_t ifrequency = 0;
         ifrequency < axis.bins.size(); ++ifrequency) {
        for (int ix = 0; ix < model.nx; ++ix) {
            const std::size_t output_index =
                (ifrequency * target_depths.size() +
                 static_cast<std::size_t>(iz_local)) *
                    static_cast<std::size_t>(model.nx) + ix;
            const std::size_t boundary_index =
                ifrequency * static_cast<std::size_t>(model.nx) + ix;
            state.source[boundary_index] = source_output[output_index];
            state.receiver_conjugate[boundary_index] =
                receiver_output[output_index];
        }
    }
}

inline void finish_image(const CommonOptions& options,
                         const se::huygens::Model2D& model,
                         std::vector<float>& image,
                         const std::vector<float>& illumination)
{
    (void)model;
    if (!options.source_normalization) return;
    const float maximum = *std::max_element(
        illumination.begin(), illumination.end());
    const float epsilon = options.normalization_epsilon *
        std::max(maximum, std::numeric_limits<float>::min());
    for (std::size_t i = 0; i < image.size(); ++i) {
        image[i] /= illumination[i] + epsilon;
    }
}

inline void print_configuration(const CommonOptions& options,
                                const SeismicData& data,
                                const FrequencyAxis& axis,
                                const std::vector<int>& shots,
                                const se::huygens::Model2D& model,
                                const std::string& method)
{
    std::cout
        << "\n========== Frequency-domain Kirchhoff imaging ==========\n"
        << "Method              : " << method << '\n'
        << "Data geometry       : "
        << (options.cmp ? "CMP/offset (receiver=shot+offset)"
                        : "shot/absolute-receiver") << '\n'
        << "Data axes           : nt=" << data.nt()
        << ", traces=" << data.ntrace()
        << ", shots=" << data.nshot()
        << ", dt=" << data.dt() << " s\n"
        << "Selected shots      : " << shots.size() << '\n'
        << "Boundary states     : "
        << boundary_state_megabytes(shots.size(), axis.frequencies.size(), model.nx)
        << " MiB (limit " << options.max_state_mb << " MiB)\n"
        << "Model               : nx=" << model.nx
        << ", nz=" << model.nz << '\n'
        << "FFT                 : nfft=" << axis.nfft
        << ", frequencies=" << axis.frequencies.size()
        << ", range=" << axis.frequencies.front() << '-'
        << axis.frequencies.back() << " Hz\n"
        << "Imaging condition   : sum Re{Us * conj(Ur)}; "
           "implemented as Re{Us * P*conj(D)}\n"
        << "Source normalization: "
        << (options.source_normalization ? "enabled" : "disabled") << '\n';
#ifdef _OPENMP
    std::cout << "OpenMP              : enabled, max_threads="
              << omp_get_max_threads() << '\n';
#else
    std::cout << "OpenMP              : disabled\n";
#endif
    std::cout
        << "=========================================================\n";
}

} // namespace frequency_kirchhoff_imaging

#endif
