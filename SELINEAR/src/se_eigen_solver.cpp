#include "se_eigen_solver.hpp"

namespace se {
namespace eigen {

const char* to_string(SolverStatus status) noexcept {
    switch (status) {
        case SolverStatus::success:
            return "success";
        case SolverStatus::numerical_issue:
            return "numerical_issue";
        case SolverStatus::no_convergence:
            return "no_convergence";
        case SolverStatus::invalid_input:
            return "invalid_input";
    }
    return "invalid_input";
}

const char* to_string(IterativeMethod method) noexcept {
    switch (method) {
        case IterativeMethod::conjugate_gradient:
            return "conjugate_gradient";
        case IterativeMethod::bicgstab:
            return "bicgstab";
        case IterativeMethod::gmres:
            return "gmres";
        case IterativeMethod::dgmres:
            return "dgmres";
        case IterativeMethod::minres:
            return "minres";
        case IterativeMethod::least_squares_conjugate_gradient:
            return "least_squares_conjugate_gradient";
    }
    return "unknown";
}

const char* to_string(Preconditioner preconditioner) noexcept {
    switch (preconditioner) {
        case Preconditioner::identity:
            return "identity";
        case Preconditioner::diagonal:
            return "diagonal";
        case Preconditioner::incomplete_lut:
            return "incomplete_lut";
    }
    return "unknown";
}

IterativeMethod iterative_method_from_string(const std::string& value) {
    const std::string name = detail::normalized_name(value);
    if (name == "cg" || name == "conjugate_gradient") {
        return IterativeMethod::conjugate_gradient;
    }
    if (name == "bicgstab" || name == "bi_cgstab") {
        return IterativeMethod::bicgstab;
    }
    if (name == "gmres") return IterativeMethod::gmres;
    if (name == "dgmres" || name == "deflated_gmres") {
        return IterativeMethod::dgmres;
    }
    if (name == "minres") return IterativeMethod::minres;
    if (name == "lscg" || name == "least_squares_cg" ||
        name == "least_squares_conjugate_gradient") {
        return IterativeMethod::least_squares_conjugate_gradient;
    }
    throw std::invalid_argument("unknown iterative solver: " + value);
}

Preconditioner preconditioner_from_string(const std::string& value) {
    const std::string name = detail::normalized_name(value);
    if (name == "none" || name == "identity") {
        return Preconditioner::identity;
    }
    if (name == "diagonal" || name == "jacobi") {
        return Preconditioner::diagonal;
    }
    if (name == "ilut" || name == "ilu" ||
        name == "incomplete_lut") {
        return Preconditioner::incomplete_lut;
    }
    throw std::invalid_argument("unknown preconditioner: " + value);
}

SolverStatus solver_status_from_eigen(Eigen::ComputationInfo info) noexcept {
    switch (info) {
        case Eigen::Success:
            return SolverStatus::success;
        case Eigen::NumericalIssue:
            return SolverStatus::numerical_issue;
        case Eigen::NoConvergence:
            return SolverStatus::no_convergence;
        case Eigen::InvalidInput:
            return SolverStatus::invalid_input;
    }
    return SolverStatus::invalid_input;
}

}  // namespace eigen
}  // namespace se
