#ifndef SE_FFTW_HPP
#define SE_FFTW_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

namespace se {
namespace fft {

template <typename Real>
class ComplexFft1D {
public:
    using real_type = Real;
    using complex_type = std::complex<Real>;

    explicit ComplexFft1D(std::size_t size, unsigned flags = 0U);
    ~ComplexFft1D();

    ComplexFft1D(const ComplexFft1D&) = delete;
    ComplexFft1D& operator=(const ComplexFft1D&) = delete;
    ComplexFft1D(ComplexFft1D&&) noexcept;
    ComplexFft1D& operator=(ComplexFft1D&&) noexcept;

    std::size_t size() const noexcept;

    std::vector<complex_type> forward(const std::vector<complex_type>& input);
    std::vector<complex_type> inverse(const std::vector<complex_type>& spectrum,
                                      bool normalize = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

template <typename Real>
std::vector<std::complex<Real>> ricker_wavelet(std::size_t sample_count,
                                               Real sample_interval,
                                               Real dominant_frequency);

template <typename Real>
std::complex<Real> normalized_ricker_spectrum(std::size_t sample_count,
                                              Real sample_interval,
                                              Real dominant_frequency,
                                              Real requested_frequency,
                                              Real* actual_frequency = nullptr);

extern template class ComplexFft1D<float>;
extern template class ComplexFft1D<double>;
extern template std::vector<std::complex<float>> ricker_wavelet<float>(
    std::size_t, float, float);
extern template std::vector<std::complex<double>> ricker_wavelet<double>(
    std::size_t, double, double);
extern template std::complex<float> normalized_ricker_spectrum<float>(
    std::size_t, float, float, float, float*);
extern template std::complex<double> normalized_ricker_spectrum<double>(
    std::size_t, double, double, double, double*);

}  // namespace fft
}  // namespace se

#endif
