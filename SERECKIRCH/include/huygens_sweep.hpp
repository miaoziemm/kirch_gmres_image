#ifndef HUYGENS_SWEEP_HPP
#define HUYGENS_SWEEP_HPP

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace se {
namespace huygens {

using Complex = std::complex<float>;

struct Model2D {
    int nz = 0;
    int nx = 0;
    float dz = 1.0f;
    float dx = 1.0f;
    float oz = 0.0f;
    float ox = 0.0f;
    std::vector<float> velocity; // x-fast: ix + iz * nx

    std::size_t index(int ix, int iz) const;
    float x(int ix) const;
    float z(int iz) const;
};

struct Block {
    int id = 0;
    int start_iz = 0;
    int source_iz = 0;
    int target_start_iz = 0;
    int target_end_iz = 0;
    int halo_end_iz = 0;
};

struct BlockInfo {
    int nx = 0;
    int nz = 0;
    float dx = 1.0f;
    float dz = 1.0f;
    float ox = 0.0f;
    float oz = 0.0f;
    int first_block_rows = 3;
    int block_rows = 64;
    // Move each later propagation datum upward into the preceding completed block.
    int overlap_rows = 0;
    std::vector<Block> blocks;
};

struct GridPoint {
    int ix = 0;
    int iz = 0;
};

struct Table3D {
    int nz_target = 0;
    int nx_target = 0;
    int nsource = 0;
    float dz_target = 1.0f;
    float dx_target = 1.0f;
    float dx_source = 1.0f;
    float oz_target = 0.0f;
    float ox_target = 0.0f;
    float ox_source = 0.0f;
    std::vector<float> values; // source, target-x, target-z; target-z fastest

    std::size_t index(int source, int ix_target, int iz_target) const;
    std::size_t row_index(int ix_target, int iz_target) const;
    int rows() const;
};

struct LayerTables {
    Table3D traveltime;
    Table3D amplitude;
    Table3D normal_traveltime_derivative;
};

// Traveltime-only tables for the one-way frequency-domain Kirchhoff operator.
struct OneWayLayerTables {
    Table3D traveltime;
};

struct LayerGeometry {
    std::vector<int> source_ix;
    std::vector<int> target_ix;
    std::vector<int> target_iz;
    std::vector<GridPoint> targets;
};

struct KernelData {
    int rows = 0;
    int sources = 0;
    float omega = 0.0f;
    const std::vector<float>* quadrature_weights = nullptr;
    const LayerTables* tables = nullptr;

    Complex green(int row, int source) const;
    Complex green_normal_derivative(int row, int source) const;
    Complex entry(int row, int column) const;
    Complex apply_pair(int row,
                       int source,
                       const Complex& value,
                       const Complex& normal_derivative) const;
};

class FrequencyKirchhoffFilter {
public:
    FrequencyKirchhoffFilter(float frequency,
                             float sample_interval,
                             float filter_length,
                             float maximum_traveltime,
                             int lookup_subsamples = 64);

    Complex response(float traveltime) const;
    // Return the slowly varying envelope after exp(-i*omega*tau) has been
    // removed.  This is tabulated directly so matrix construction does not
    // evaluate one complex exponential for every matrix entry.
    Complex phase_removed_response(float traveltime) const;
    float frequency() const noexcept { return frequency_; }
    float omega() const noexcept { return omega_; }

private:
    Complex exact_response(float traveltime) const;

    float frequency_ = 0.0f;
    float sample_interval_ = 0.001f;
    float filter_length_ = 0.025f;
    float lookup_step_ = 0.0f;
    float omega_ = 0.0f;
    std::vector<Complex> lookup_;
    std::vector<Complex> phase_removed_lookup_;
};

struct OneWayKernelData {
    int rows = 0;
    int sources = 0;
    float source_z = 0.0f;
    const std::vector<float>* quadrature_weights = nullptr;
    const OneWayLayerTables* tables = nullptr;
    const FrequencyKirchhoffFilter* filter = nullptr;

    float traveltime(int row, int source) const;
    float geometry_factor(int row, int source,
                          float traveltime) const;
    Complex entry(int row, int source) const;
    Complex phase_removed_amplitude(int row, int source) const;
    Complex phase_removed_amplitude(int row, int source,
                                    float traveltime) const;
    void fill_phase_removed_amplitude_row(int row, const float* traveltimes,
                                          float* interleaved_output) const;
    void fill_geometry_row(int row, const float* traveltimes,
                           float* geometry_output) const;
    void fill_phase_removed_amplitude_row_cached(
        const float* traveltimes, const float* geometry,
        float* interleaved_output) const;
};

Model2D read_velocity_model(const std::string& filename);
void validate_block_info(const BlockInfo& info, const Model2D* model = nullptr);
BlockInfo make_block_info(const Model2D& model,
                          int first_block_rows,
                          int block_rows,
                          int overlap_rows);
void write_block_info(const std::string& filename, const BlockInfo& info);
BlockInfo read_block_info(const std::string& filename);

std::vector<float> solve_fmm(const Model2D& model, int source_ix, int source_iz);
std::vector<int> regular_indices(int first, int last_inclusive, int stride);
LayerGeometry make_layer_geometry(const Model2D& model,
                                  const Block& block,
                                  int source_stride,
                                  int target_x_stride,
                                  int target_z_stride);
std::vector<float> trapezoidal_weights(const Model2D& model,
                                       const std::vector<int>& source_ix);

std::string layer_table_filename(const std::string& prefix,
                                 int block_id,
                                 const std::string& quantity);
void ensure_parent_directory(const std::string& filename);

void write_table_rsf(const std::string& filename,
                     const Model2D& model,
                     const Block& block,
                     const Table3D& table,
                     const std::string& quantity,
                     int source_stride,
                     int target_x_stride,
                     int target_z_stride);
Table3D read_table_rsf(const std::string& filename);
LayerTables read_layer_tables(const std::string& prefix, int block_id);
void validate_layer_tables(const LayerTables& tables,
                           const LayerGeometry& geometry,
                           const Block& block);
OneWayLayerTables read_one_way_layer_tables(const std::string& prefix,
                                            int block_id);
void validate_one_way_layer_tables(const OneWayLayerTables& tables,
                                   const LayerGeometry& geometry,
                                   const Block& block);

void write_wavefield_component_rsf(const std::string& filename,
                                   const Model2D& model,
                                   const std::vector<Complex>& wavefield,
                                   float frequency,
                                   const std::string& method,
                                   bool imaginary);
void write_image_rsf(
    const std::string& filename,
    const Model2D& model,
    const std::vector<float>& image,
    const std::string& method,
    const std::vector<std::pair<std::string, float>>& scalar_headers = {});
void write_timing_rsf(const std::string& filename,
                      const std::vector<float>& values,
                      int metrics,
                      int blocks,
                      const std::string& method,
                      const std::vector<std::string>& metric_names,
                      const std::vector<std::pair<std::string, float>>& scalar_headers = {});

Complex hankel_point_source(float omega,
                            float velocity,
                            float radius,
                            float source_amplitude);
void initialize_first_block_hankel(const Model2D& model,
                                   const BlockInfo& info,
                                   float frequency,
                                   int source_ix,
                                   int source_iz,
                                   float source_amplitude,
                                   float source_radius,
                                   std::vector<Complex>& wavefield);
// Launch field for the FFTW forward-transform convention used by the
// one-way operator, whose propagation factor is exp(-i*omega*tau).
void initialize_first_block_hankel_one_way(
    const Model2D& model,
    const BlockInfo& info,
    float frequency,
    int source_ix,
    int source_iz,
    float source_amplitude,
    float source_radius,
    std::vector<Complex>& wavefield);

std::vector<Complex> gather_boundary_cauchy_data(const Model2D& model,
                                                 const Block& block,
                                                 const std::vector<int>& source_ix,
                                                 const std::vector<Complex>& wavefield);
std::vector<Complex> gather_boundary_values(const Model2D& model,
                                            const Block& block,
                                            const std::vector<int>& source_ix,
                                            const std::vector<Complex>& wavefield);
void inject_target_values(const Model2D& model,
                          const std::vector<GridPoint>& targets,
                          const std::vector<Complex>& values,
                          std::vector<Complex>& wavefield);

std::vector<Complex> direct_apply(const KernelData& kernel,
                                  const std::vector<Complex>& input);
std::vector<Complex> direct_one_way_apply(const OneWayKernelData& kernel,
                                          const std::vector<Complex>& input);

std::vector<double> make_row_coordinates(const Model2D& model,
                                         const std::vector<GridPoint>& targets);
std::vector<double> make_column_coordinates(const Model2D& model,
                                            const Block& block,
                                            const std::vector<int>& source_ix);

float relative_l2_error(const std::vector<Complex>& reference,
                        const std::vector<Complex>& candidate);
float maximum_normalized_error(const std::vector<Complex>& reference,
                               const std::vector<Complex>& candidate);

} // namespace huygens
} // namespace se

#endif
