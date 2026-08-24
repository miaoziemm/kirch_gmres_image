#include "se_fftw.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <utility>

namespace se {
namespace fft {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename Real>
struct FftwApi;

template <>
struct FftwApi<float> {
    using plan_type = fftwf_plan;
    using native_complex = fftwf_complex;

    static plan_type make_plan(int n, std::complex<float>* input,
                               std::complex<float>* output, int sign, unsigned flags)
    {
        return fftwf_plan_dft_1d(
            n,
            reinterpret_cast<native_complex*>(input),
            reinterpret_cast<native_complex*>(output),
            sign,
            flags == 0U ? FFTW_ESTIMATE : flags);
    }

    static void execute(plan_type plan) { fftwf_execute(plan); }
    static void destroy(plan_type plan) { fftwf_destroy_plan(plan); }
};

template <>
struct FftwApi<double> {
    using plan_type = fftw_plan;
    using native_complex = fftw_complex;

    static plan_type make_plan(int n, std::complex<double>* input,
                               std::complex<double>* output, int sign, unsigned flags)
    {
        return fftw_plan_dft_1d(
            n,
            reinterpret_cast<native_complex*>(input),
            reinterpret_cast<native_complex*>(output),
            sign,
            flags == 0U ? FFTW_ESTIMATE : flags);
    }

    static void execute(plan_type plan) { fftw_execute(plan); }
    static void destroy(plan_type plan) { fftw_destroy_plan(plan); }
};

}  // namespace

template <typename Real>
struct ComplexFft1D<Real>::Impl {
    using Api = FftwApi<Real>;
    using plan_type = typename Api::plan_type;
    using complex_type = std::complex<Real>;

    explicit Impl(std::size_t count, unsigned flags)
        : size(count), input(count), output(count)
    {
        static_assert(sizeof(complex_type) == 2 * sizeof(Real),
                      "FFTW requires contiguous real/imaginary storage");
        if (count == 0 || count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("invalid FFT length");
        }
        forward_plan = Api::make_plan(static_cast<int>(count), input.data(), output.data(),
                                      FFTW_FORWARD, flags);
        inverse_plan = Api::make_plan(static_cast<int>(count), input.data(), output.data(),
                                      FFTW_BACKWARD, flags);
        if (forward_plan == nullptr || inverse_plan == nullptr) {
            if (forward_plan != nullptr) Api::destroy(forward_plan);
            if (inverse_plan != nullptr) Api::destroy(inverse_plan);
            throw std::runtime_error("FFTW failed to create a 1-D complex plan");
        }
    }

    ~Impl()
    {
        if (forward_plan != nullptr) Api::destroy(forward_plan);
        if (inverse_plan != nullptr) Api::destroy(inverse_plan);
    }

    std::size_t size;
    std::vector<complex_type> input;
    std::vector<complex_type> output;
    plan_type forward_plan = nullptr;
    plan_type inverse_plan = nullptr;
};

template <typename Real>
ComplexFft1D<Real>::ComplexFft1D(std::size_t size, unsigned flags)
    : impl_(std::make_unique<Impl>(size, flags))
{
}

template <typename Real>
ComplexFft1D<Real>::~ComplexFft1D() = default;

template <typename Real>
ComplexFft1D<Real>::ComplexFft1D(ComplexFft1D&&) noexcept = default;

template <typename Real>
ComplexFft1D<Real>& ComplexFft1D<Real>::operator=(ComplexFft1D&&) noexcept = default;

template <typename Real>
std::size_t ComplexFft1D<Real>::size() const noexcept
{
    return impl_->size;
}

template <typename Real>
std::vector<typename ComplexFft1D<Real>::complex_type>
ComplexFft1D<Real>::forward(const std::vector<complex_type>& values)
{
    if (values.size() != impl_->size) {
        throw std::invalid_argument("FFTW forward input has the wrong length");
    }
    impl_->input = values;
    FftwApi<Real>::execute(impl_->forward_plan);
    return impl_->output;
}

template <typename Real>
std::vector<typename ComplexFft1D<Real>::complex_type>
ComplexFft1D<Real>::inverse(const std::vector<complex_type>& values, bool normalize)
{
    if (values.size() != impl_->size) {
        throw std::invalid_argument("FFTW inverse input has the wrong length");
    }
    impl_->input = values;
    FftwApi<Real>::execute(impl_->inverse_plan);
    if (normalize) {
        const Real scale = Real(1) / static_cast<Real>(impl_->size);
        for (complex_type& value : impl_->output) value *= scale;
    }
    return impl_->output;
}

template <typename Real>
std::vector<std::complex<Real>> ricker_wavelet(std::size_t sample_count,
                                               Real sample_interval,
                                               Real dominant_frequency)
{
    if (sample_count == 0 || !(sample_interval > Real(0)) ||
        !(dominant_frequency > Real(0))) {
        throw std::invalid_argument("invalid Ricker wavelet parameters");
    }

    std::vector<std::complex<Real>> result(sample_count);
    const Real delay = Real(1.5) / dominant_frequency;
    const Real pi = static_cast<Real>(kPi);
    for (std::size_t index = 0; index < sample_count; ++index) {
        const Real time = static_cast<Real>(index) * sample_interval - delay;
        const Real argument = pi * dominant_frequency * time;
        const Real argument2 = argument * argument;
        result[index] = std::complex<Real>((Real(1) - Real(2) * argument2) *
                                               std::exp(-argument2),
                                           Real(0));
    }
    return result;
}

template <typename Real>
std::complex<Real> normalized_ricker_spectrum(std::size_t sample_count,
                                              Real sample_interval,
                                              Real dominant_frequency,
                                              Real requested_frequency,
                                              Real* actual_frequency)
{
    if (!(requested_frequency >= Real(0))) {
        throw std::invalid_argument("requested frequency must be non-negative");
    }
    ComplexFft1D<Real> transform(sample_count);
    const auto spectrum = transform.forward(
        ricker_wavelet<Real>(sample_count, sample_interval, dominant_frequency));

    const Real frequency_step = Real(1) /
        (static_cast<Real>(sample_count) * sample_interval);
    std::size_t index = static_cast<std::size_t>(
        std::llround(static_cast<double>(requested_frequency / frequency_step)));
    index = std::min(index, sample_count - 1);
    if (actual_frequency != nullptr) {
        *actual_frequency = static_cast<Real>(index) * frequency_step;
    }

    Real maximum = Real(0);
    for (const std::complex<Real>& value : spectrum) {
        maximum = std::max(maximum, static_cast<Real>(std::abs(value)));
    }
    if (!(maximum > Real(0))) {
        throw std::runtime_error("Ricker spectrum is identically zero");
    }
    return spectrum[index] / maximum;
}

template class ComplexFft1D<float>;
template class ComplexFft1D<double>;
template std::vector<std::complex<float>> ricker_wavelet<float>(
    std::size_t, float, float);
template std::vector<std::complex<double>> ricker_wavelet<double>(
    std::size_t, double, double);
template std::complex<float> normalized_ricker_spectrum<float>(
    std::size_t, float, float, float, float*);
template std::complex<double> normalized_ricker_spectrum<double>(
    std::size_t, double, double, double, double*);

}  // namespace fft
}  // namespace se
