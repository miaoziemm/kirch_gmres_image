#ifndef SE_EIGEN_COMPLEX_DISPATCH_HPP
#define SE_EIGEN_COMPLEX_DISPATCH_HPP

#include "se_eigen_solver_impl.hpp"

namespace se {
namespace eigen {
namespace detail {

#define SE_DEFINE_COMPLEX_DISPATCH(ComplexScalar)                              \
    template <>                                                               \
    inline SolverResult<ComplexScalar> dispatch_iterative<ComplexScalar>(      \
        const SparseMatrix<ComplexScalar>& matrix,                             \
        const DenseVector<ComplexScalar>& rhs,                                 \
        IterativeMethod method,                                                \
        const SolverOptions& options,                                          \
        const DenseVector<ComplexScalar>* guess) {                             \
        switch (method) {                                                      \
            case IterativeMethod::conjugate_gradient:                          \
                return run_cg(matrix, rhs, options, guess);                    \
            case IterativeMethod::bicgstab:                                    \
                return run_bicgstab(matrix, rhs, options, guess);              \
            case IterativeMethod::gmres:                                       \
                return run_gmres(matrix, rhs, options, guess);                 \
            case IterativeMethod::dgmres:                                      \
                return run_dgmres(matrix, rhs, options, guess);                \
            case IterativeMethod::minres:                                      \
                throw std::invalid_argument(                                   \
                    "Eigen MINRES does not support complex scalar types");    \
            case IterativeMethod::least_squares_conjugate_gradient:            \
                return run_lscg(matrix, rhs, options, guess);                  \
        }                                                                      \
        throw std::invalid_argument("unknown iterative method");              \
    }

SE_DEFINE_COMPLEX_DISPATCH(std::complex<float>)
SE_DEFINE_COMPLEX_DISPATCH(std::complex<double>)

#undef SE_DEFINE_COMPLEX_DISPATCH

}  // namespace detail
}  // namespace eigen
}  // namespace se

#endif  // SE_EIGEN_COMPLEX_DISPATCH_HPP
