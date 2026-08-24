#ifndef SE_BUTTERFLY_HPP
#define SE_BUTTERFLY_HPP

#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace se {
namespace butterfly {

struct Options {
    double tolerance = 1.0e-4;
    int leaf_size = 64;
    int verbosity = 0;
    int coordinate_dimension = 2;
    double sample_parameter = 2.0;
    double rank_detection_factor = 0.1;
    int lr_level = 0;
    int forward_n15_flag = 0;
    int nearest_neighbors = 10;
    int compression_pattern = 3;
    int less_adapt = 1;
    // Reuse the ButterflyPACK row/column geometry mesh when an identical
    // matrix geometry is constructed repeatedly (for example, many
    // frequencies of the same Kirchhoff propagation block).
    int reuse_geometry = 0;
};

struct Statistics {
    double setup_seconds = 0.0;
    double row_tree_seconds = 0.0;
    double column_tree_seconds = 0.0;
    double structure_seconds = 0.0;
    double compression_seconds = 0.0;
    double build_seconds = 0.0;
    double apply_pack_seconds = 0.0;
    double apply_multiply_seconds = 0.0;
    double apply_unpack_seconds = 0.0;
    double apply_seconds = 0.0;
    double library_fill_seconds = 0.0;
    double library_entry_seconds = 0.0;
    double library_entry_traverse_seconds = 0.0;
    double library_entry_butterfly_seconds = 0.0;
    double library_entry_communication_seconds = 0.0;
    double library_multiply_seconds = 0.0;
    double compressed_megabytes = 0.0;
    double peak_megabytes = 0.0;
    double maximum_rank = 0.0;
    std::size_t sampled_entries = 0;
    std::size_t applied_vectors = 0;
    // 1 when this Matrix borrowed a row/column geometry plan built by an
    // earlier Matrix with identical coordinates/options; 0 otherwise.
    int geometry_reused = 0;
};

template <typename Real>
class Matrix {
public:
    using real_type = Real;
    using complex_type = std::complex<Real>;
    using entry_function = std::function<complex_type(int row, int column)>;

    Matrix(int rows,
           int columns,
           std::vector<double> row_coordinates,
           std::vector<double> column_coordinates,
           entry_function entry,
           Options options = Options());
    ~Matrix();

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) = delete;
    Matrix& operator=(Matrix&&) = delete;

    std::vector<complex_type> apply(const std::vector<complex_type>& input);
    std::vector<std::vector<complex_type>> apply_many(
        const std::vector<std::vector<complex_type>>& inputs);

    int rows() const noexcept;
    int columns() const noexcept;
    const Statistics& statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#ifdef KIRCH_HAS_BUTTERFLYPACK_FLOAT
extern template class Matrix<float>;
#endif
#ifdef KIRCH_HAS_BUTTERFLYPACK_DOUBLE
extern template class Matrix<double>;
#endif

}  // namespace butterfly
}  // namespace se

#endif
