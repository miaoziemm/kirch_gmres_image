#include "se_blas.hpp"

#include <complex>
#include <stdexcept>

namespace {

extern "C" {
void saxpy_(const int*, const float*, const float*, const int*, float*, const int*);
void daxpy_(const int*, const double*, const double*, const int*, double*, const int*);
void caxpy_(const int*, const std::complex<float>*, const std::complex<float>*,
            const int*, std::complex<float>*, const int*);
void zaxpy_(const int*, const std::complex<double>*, const std::complex<double>*,
            const int*, std::complex<double>*, const int*);

float snrm2_(const int*, const float*, const int*);
double dnrm2_(const int*, const double*, const int*);
float scnrm2_(const int*, const std::complex<float>*, const int*);
double dznrm2_(const int*, const std::complex<double>*, const int*);

void sgemv_(const char*, const int*, const int*, const float*, const float*,
            const int*, const float*, const int*, const float*, float*, const int*);
void dgemv_(const char*, const int*, const int*, const double*, const double*,
            const int*, const double*, const int*, const double*, double*, const int*);
void cgemv_(const char*, const int*, const int*, const std::complex<float>*,
            const std::complex<float>*, const int*, const std::complex<float>*,
            const int*, const std::complex<float>*, std::complex<float>*, const int*);
void zgemv_(const char*, const int*, const int*, const std::complex<double>*,
            const std::complex<double>*, const int*, const std::complex<double>*,
            const int*, const std::complex<double>*, std::complex<double>*, const int*);

void sgemm_(const char*, const char*, const int*, const int*, const int*,
            const float*, const float*, const int*, const float*, const int*,
            const float*, float*, const int*);
void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
void cgemm_(const char*, const char*, const int*, const int*, const int*,
            const std::complex<float>*, const std::complex<float>*, const int*,
            const std::complex<float>*, const int*, const std::complex<float>*,
            std::complex<float>*, const int*);
void zgemm_(const char*, const char*, const int*, const int*, const int*,
            const std::complex<double>*, const std::complex<double>*, const int*,
            const std::complex<double>*, const int*, const std::complex<double>*,
            std::complex<double>*, const int*);
}

template <typename T>
void validate_input_vector(int n, const T* x, int incx)
{
    if (n < 0 || incx <= 0) {
        throw std::invalid_argument("invalid BLAS vector dimensions or stride");
    }
    if (n > 0 && x == nullptr) {
        throw std::invalid_argument("null BLAS input vector pointer");
    }
}

template <typename T>
void validate_output_vector(int n, T* y, int incy)
{
    if (n < 0 || incy <= 0) {
        throw std::invalid_argument("invalid BLAS output vector dimensions or stride");
    }
    if (n > 0 && y == nullptr) {
        throw std::invalid_argument("null BLAS output vector pointer");
    }
}

void validate_matrix_arguments(int rows, int cols, int leading_dimension,
                               const void* matrix)
{
    if (rows < 0 || cols < 0 || leading_dimension < (rows > 0 ? rows : 1)) {
        throw std::invalid_argument("invalid column-major BLAS matrix dimensions");
    }
    if (rows > 0 && cols > 0 && matrix == nullptr) {
        throw std::invalid_argument("null BLAS matrix pointer");
    }
}

}  // namespace

namespace se {
namespace blas {

char to_blas_char(Transpose value)
{
    switch (value) {
    case Transpose::none:
        return 'N';
    case Transpose::transpose:
        return 'T';
    case Transpose::conjugate_transpose:
        return 'C';
    }
    throw std::invalid_argument("invalid BLAS transpose value");
}

template <typename T>
void axpy(int n, T alpha, const T* x, int incx, T* y, int incy)
{
    validate_input_vector(n, x, incx);
    validate_output_vector(n, y, incy);
    if constexpr (std::is_same<T, float>::value) {
        saxpy_(&n, &alpha, x, &incx, y, &incy);
    } else if constexpr (std::is_same<T, double>::value) {
        daxpy_(&n, &alpha, x, &incx, y, &incy);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        caxpy_(&n, &alpha, x, &incx, y, &incy);
    } else if constexpr (std::is_same<T, std::complex<double>>::value) {
        zaxpy_(&n, &alpha, x, &incx, y, &incy);
    }
}

template <typename T>
real_type_t<T> nrm2(int n, const T* x, int incx)
{
    validate_input_vector(n, x, incx);
    if constexpr (std::is_same<T, float>::value) {
        return snrm2_(&n, x, &incx);
    } else if constexpr (std::is_same<T, double>::value) {
        return dnrm2_(&n, x, &incx);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        return scnrm2_(&n, x, &incx);
    } else {
        return dznrm2_(&n, x, &incx);
    }
}

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
          int y_stride)
{
    validate_matrix_arguments(rows, cols, leading_dimension, matrix);
    const int x_length = trans == Transpose::none ? cols : rows;
    const int y_length = trans == Transpose::none ? rows : cols;
    validate_input_vector(x_length, x, x_stride);
    validate_output_vector(y_length, y, y_stride);
    if (y_length == 0) {
        return;
    }
    const char operation = to_blas_char(trans);
    if constexpr (std::is_same<T, float>::value) {
        sgemv_(&operation, &rows, &cols, &alpha, matrix, &leading_dimension,
               x, &x_stride, &beta, y, &y_stride);
    } else if constexpr (std::is_same<T, double>::value) {
        dgemv_(&operation, &rows, &cols, &alpha, matrix, &leading_dimension,
               x, &x_stride, &beta, y, &y_stride);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        cgemv_(&operation, &rows, &cols, &alpha, matrix, &leading_dimension,
               x, &x_stride, &beta, y, &y_stride);
    } else {
        zgemv_(&operation, &rows, &cols, &alpha, matrix, &leading_dimension,
               x, &x_stride, &beta, y, &y_stride);
    }
}

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
          int leading_dimension_c)
{
    const int a_rows = trans_a == Transpose::none ? rows : inner;
    const int a_cols = trans_a == Transpose::none ? inner : rows;
    const int b_rows = trans_b == Transpose::none ? inner : cols;
    const int b_cols = trans_b == Transpose::none ? cols : inner;
    validate_matrix_arguments(a_rows, a_cols, leading_dimension_a, a);
    validate_matrix_arguments(b_rows, b_cols, leading_dimension_b, b);
    validate_matrix_arguments(rows, cols, leading_dimension_c, c);

    const char operation_a = to_blas_char(trans_a);
    const char operation_b = to_blas_char(trans_b);
    if constexpr (std::is_same<T, float>::value) {
        sgemm_(&operation_a, &operation_b, &rows, &cols, &inner, &alpha,
               a, &leading_dimension_a, b, &leading_dimension_b,
               &beta, c, &leading_dimension_c);
    } else if constexpr (std::is_same<T, double>::value) {
        dgemm_(&operation_a, &operation_b, &rows, &cols, &inner, &alpha,
               a, &leading_dimension_a, b, &leading_dimension_b,
               &beta, c, &leading_dimension_c);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        cgemm_(&operation_a, &operation_b, &rows, &cols, &inner, &alpha,
               a, &leading_dimension_a, b, &leading_dimension_b,
               &beta, c, &leading_dimension_c);
    } else {
        zgemm_(&operation_a, &operation_b, &rows, &cols, &inner, &alpha,
               a, &leading_dimension_a, b, &leading_dimension_b,
               &beta, c, &leading_dimension_c);
    }
}

template void axpy<float>(int, float, const float*, int, float*, int);
template void axpy<double>(int, double, const double*, int, double*, int);
template void axpy<std::complex<float>>(int, std::complex<float>,
                                        const std::complex<float>*, int,
                                        std::complex<float>*, int);
template void axpy<std::complex<double>>(int, std::complex<double>,
                                         const std::complex<double>*, int,
                                         std::complex<double>*, int);

template float nrm2<float>(int, const float*, int);
template double nrm2<double>(int, const double*, int);
template float nrm2<std::complex<float>>(int, const std::complex<float>*, int);
template double nrm2<std::complex<double>>(int, const std::complex<double>*, int);

template void gemv<float>(Transpose, int, int, float, const float*, int,
                          const float*, int, float, float*, int);
template void gemv<double>(Transpose, int, int, double, const double*, int,
                           const double*, int, double, double*, int);
template void gemv<std::complex<float>>(
    Transpose, int, int, std::complex<float>, const std::complex<float>*, int,
    const std::complex<float>*, int, std::complex<float>, std::complex<float>*, int);
template void gemv<std::complex<double>>(
    Transpose, int, int, std::complex<double>, const std::complex<double>*, int,
    const std::complex<double>*, int, std::complex<double>, std::complex<double>*, int);

template void gemm<float>(Transpose, Transpose, int, int, int, float,
                          const float*, int, const float*, int,
                          float, float*, int);
template void gemm<double>(Transpose, Transpose, int, int, int, double,
                           const double*, int, const double*, int,
                           double, double*, int);
template void gemm<std::complex<float>>(
    Transpose, Transpose, int, int, int, std::complex<float>,
    const std::complex<float>*, int, const std::complex<float>*, int,
    std::complex<float>, std::complex<float>*, int);
template void gemm<std::complex<double>>(
    Transpose, Transpose, int, int, int, std::complex<double>,
    const std::complex<double>*, int, const std::complex<double>*, int,
    std::complex<double>, std::complex<double>*, int);

}  // namespace blas
}  // namespace se
