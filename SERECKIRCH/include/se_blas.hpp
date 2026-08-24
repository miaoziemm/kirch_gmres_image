#ifndef SE_BLAS_HPP
#define SE_BLAS_HPP

#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace se {
namespace blas {

enum class Transpose {
    none,
    transpose,
    conjugate_transpose
};

template <typename T>
struct real_type {
    using type = T;
};

template <typename T>
struct real_type<std::complex<T>> {
    using type = T;
};

template <typename T>
using real_type_t = typename real_type<T>::type;

char to_blas_char(Transpose value);

template <typename T>
void axpy(int n, T alpha, const T* x, int incx, T* y, int incy);

template <typename T>
real_type_t<T> nrm2(int n, const T* x, int incx = 1);

template <typename T>
void gemv(Transpose trans,
          int rows,
          int cols,
          T alpha,
          const T* matrix,
          int leading_dimension,
          const T* x,
          int x_stride,
          T beta,
          T* y,
          int y_stride);

template <typename T>
void gemm(Transpose trans_a,
          Transpose trans_b,
          int rows,
          int cols,
          int inner,
          T alpha,
          const T* a,
          int leading_dimension_a,
          const T* b,
          int leading_dimension_b,
          T beta,
          T* c,
          int leading_dimension_c);


template <typename T>
void axpy(T alpha, const std::vector<T>& x, std::vector<T>& y)
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("BLAS axpy requires vectors with equal sizes");
    }
    if (x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("BLAS vector is too large");
    }
    axpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
}

template <typename T>
real_type_t<T> nrm2(const std::vector<T>& x)
{
    if (x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("BLAS vector is too large");
    }
    return nrm2(static_cast<int>(x.size()), x.data(), 1);
}

template <typename T>
std::vector<T> gemv(int rows,
                    int cols,
                    const std::vector<T>& column_major_matrix,
                    const std::vector<T>& x,
                    Transpose trans = Transpose::none,
                    T alpha = T(1),
                    T beta = T(0),
                    std::vector<T> initial = {})
{
    if (rows < 0 || cols < 0 ||
        column_major_matrix.size() !=
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {
        throw std::invalid_argument("BLAS gemv matrix size is invalid");
    }
    const int input_size = trans == Transpose::none ? cols : rows;
    const int output_size = trans == Transpose::none ? rows : cols;
    if (x.size() != static_cast<std::size_t>(input_size)) {
        throw std::invalid_argument("BLAS gemv input size is invalid");
    }
    if (initial.empty()) {
        initial.assign(static_cast<std::size_t>(output_size), T(0));
    } else if (initial.size() != static_cast<std::size_t>(output_size)) {
        throw std::invalid_argument("BLAS gemv output size is invalid");
    }
    gemv(trans, rows, cols, alpha, column_major_matrix.data(),
         rows > 0 ? rows : 1, x.data(), 1, beta, initial.data(), 1);
    return initial;
}

template <typename T>
real_type_t<T> relative_l2_error(const std::vector<T>& reference,
                                 const std::vector<T>& candidate)
{
    if (reference.size() != candidate.size() || reference.empty()) {
        throw std::invalid_argument("BLAS relative_l2_error requires equal non-empty vectors");
    }

    std::vector<T> difference(candidate);
    const T minus_one = T(-1);
    axpy(static_cast<int>(difference.size()), minus_one,
         reference.data(), 1, difference.data(), 1);

    const real_type_t<T> numerator =
        nrm2(static_cast<int>(difference.size()), difference.data(), 1);
    const real_type_t<T> denominator =
        nrm2(static_cast<int>(reference.size()), reference.data(), 1);

    const real_type_t<T> floor = static_cast<real_type_t<T>>(1.0e-30);
    return numerator / (denominator > floor ? denominator : floor);
}

extern template void axpy<float>(int, float, const float*, int, float*, int);
extern template void axpy<double>(int, double, const double*, int, double*, int);
extern template void axpy<std::complex<float>>(int, std::complex<float>,
                                               const std::complex<float>*, int,
                                               std::complex<float>*, int);
extern template void axpy<std::complex<double>>(int, std::complex<double>,
                                                const std::complex<double>*, int,
                                                std::complex<double>*, int);

extern template float nrm2<float>(int, const float*, int);
extern template double nrm2<double>(int, const double*, int);
extern template float nrm2<std::complex<float>>(int, const std::complex<float>*, int);
extern template double nrm2<std::complex<double>>(int, const std::complex<double>*, int);

extern template void gemv<float>(Transpose, int, int, float, const float*, int,
                                 const float*, int, float, float*, int);
extern template void gemv<double>(Transpose, int, int, double, const double*, int,
                                  const double*, int, double, double*, int);
extern template void gemv<std::complex<float>>(
    Transpose, int, int, std::complex<float>, const std::complex<float>*, int,
    const std::complex<float>*, int, std::complex<float>, std::complex<float>*, int);
extern template void gemv<std::complex<double>>(
    Transpose, int, int, std::complex<double>, const std::complex<double>*, int,
    const std::complex<double>*, int, std::complex<double>, std::complex<double>*, int);

extern template void gemm<float>(Transpose, Transpose, int, int, int, float,
                                 const float*, int, const float*, int,
                                 float, float*, int);
extern template void gemm<double>(Transpose, Transpose, int, int, int, double,
                                  const double*, int, const double*, int,
                                  double, double*, int);
extern template void gemm<std::complex<float>>(
    Transpose, Transpose, int, int, int, std::complex<float>,
    const std::complex<float>*, int, const std::complex<float>*, int,
    std::complex<float>, std::complex<float>*, int);
extern template void gemm<std::complex<double>>(
    Transpose, Transpose, int, int, int, std::complex<double>,
    const std::complex<double>*, int, const std::complex<double>*, int,
    std::complex<double>, std::complex<double>*, int);

}  // namespace blas
}  // namespace se

#endif
