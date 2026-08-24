#include "se_butterfly.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#ifdef KIRCH_HAS_BUTTERFLYPACK_FLOAT
#include "cBPACK_wrapper.h"
#endif
#ifdef KIRCH_HAS_BUTTERFLYPACK_DOUBLE
#include "zBPACK_wrapper.h"
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace se {
namespace butterfly {
namespace {

template <typename Real>
struct Api;

#ifdef KIRCH_HAS_BUTTERFLYPACK_FLOAT
template <>
struct Api<float> {
    using native_complex = __complex__ float;

    static void create_process_tree(int* count, int* groups, MPI_Fint* communicator,
                                    F2Cptr* tree)
    { c_c_bpack_createptree(count, groups, communicator, tree); }
    static void create_option(F2Cptr* option) { c_c_bpack_createoption(option); }
    static void create_stats(F2Cptr* stats) { c_c_bpack_createstats(stats); }
    static void set_double(F2Cptr* option, const char* name, double value)
    { c_c_bpack_set_D_option(option, name, value); }
    static void set_integer(F2Cptr* option, const char* name, int value)
    { c_c_bpack_set_I_option(option, name, value); }
    static void construct_init(int* points, int* dimensions, double* locations,
                               int* counts, int* levels, int* tree, int* permutation,
                               int* local_points, F2Cptr* matrix, F2Cptr* option,
                               F2Cptr* stats, F2Cptr* mesh, F2Cptr* kernel,
                               F2Cptr* process_tree,
                               void (*distance)(int*, int*, double*, C2Fptr),
                               void (*near_far)(int*, int*, int*, C2Fptr),
                               C2Fptr context)
    {
        c_c_bpack_construct_init(points, dimensions, locations, counts, levels,
                                 tree, permutation, local_points, matrix, option,
                                 stats, mesh, kernel, process_tree, distance,
                                 near_far, context);
    }
    static void butterfly_init(int* rows, int* columns, int* local_rows,
                               int* local_columns, int* row_counts, int* column_counts,
                               F2Cptr* row_mesh, F2Cptr* column_mesh, F2Cptr* butterfly,
                               F2Cptr* option, F2Cptr* stats, F2Cptr* mesh,
                               F2Cptr* kernel, F2Cptr* process_tree,
                               void (*distance)(int*, int*, double*, C2Fptr),
                               void (*near_far)(int*, int*, int*, C2Fptr),
                               C2Fptr context)
    {
        c_c_bf_construct_init(rows, columns, local_rows, local_columns,
                              row_counts, column_counts, row_mesh, column_mesh,
                              butterfly, option, stats, mesh, kernel, process_tree,
                              distance, near_far, context);
    }
    static void butterfly_compute(
        F2Cptr* butterfly, F2Cptr* option, F2Cptr* stats, F2Cptr* mesh,
        F2Cptr* kernel, F2Cptr* process_tree,
        void (*sample)(int*, int*, native_complex*, C2Fptr),
        void (*sample_block)(int*, int*, int*, std::int64_t*, int*, int*,
                             native_complex*, int*, int*, int*, int*, int*, C2Fptr),
        C2Fptr context)
    {
        c_c_bf_construct_element_compute(butterfly, option, stats, mesh, kernel,
                                         process_tree, sample, sample_block, context);
    }
    static void multiply(const char* trans, const native_complex* input,
                         native_complex* output, int* input_rows, int* output_rows,
                         int* vectors, F2Cptr* butterfly, F2Cptr* option,
                         F2Cptr* stats, F2Cptr* tree)
    {
        c_c_bf_mult(trans, input, output, input_rows, output_rows, vectors,
                    butterfly, option, stats, tree);
    }
    static void get_stats(F2Cptr* stats, const char* name, double* value)
    { c_c_bpack_getstats(stats, name, value); }
    static void delete_butterfly(F2Cptr* value) { c_c_bf_deletebf(value); }
    static void delete_mesh(F2Cptr* value) { c_c_bpack_deletemesh(value); }
    static void delete_kernel(F2Cptr* value) { c_c_bpack_deletekernelquant(value); }
    static void delete_matrix(F2Cptr* value) { c_c_bpack_delete(value); }
    static void delete_stats(F2Cptr* value) { c_c_bpack_deletestats(value); }
    static void delete_option(F2Cptr* value) { c_c_bpack_deleteoption(value); }
    static void delete_tree(F2Cptr* value) { c_c_bpack_deleteproctree(value); }
};
#endif

#ifdef KIRCH_HAS_BUTTERFLYPACK_DOUBLE
template <>
struct Api<double> {
    using native_complex = __complex__ double;

    static void create_process_tree(int* count, int* groups, MPI_Fint* communicator,
                                    F2Cptr* tree)
    { z_c_bpack_createptree(count, groups, communicator, tree); }
    static void create_option(F2Cptr* option) { z_c_bpack_createoption(option); }
    static void create_stats(F2Cptr* stats) { z_c_bpack_createstats(stats); }
    static void set_double(F2Cptr* option, const char* name, double value)
    { z_c_bpack_set_D_option(option, name, value); }
    static void set_integer(F2Cptr* option, const char* name, int value)
    { z_c_bpack_set_I_option(option, name, value); }
    static void construct_init(int* points, int* dimensions, double* locations,
                               int* counts, int* levels, int* tree, int* permutation,
                               int* local_points, F2Cptr* matrix, F2Cptr* option,
                               F2Cptr* stats, F2Cptr* mesh, F2Cptr* kernel,
                               F2Cptr* process_tree,
                               void (*distance)(int*, int*, double*, C2Fptr),
                               void (*near_far)(int*, int*, int*, C2Fptr),
                               C2Fptr context)
    {
        z_c_bpack_construct_init(points, dimensions, locations, counts, levels,
                                 tree, permutation, local_points, matrix, option,
                                 stats, mesh, kernel, process_tree, distance,
                                 near_far, context);
    }
    static void butterfly_init(int* rows, int* columns, int* local_rows,
                               int* local_columns, int* row_counts, int* column_counts,
                               F2Cptr* row_mesh, F2Cptr* column_mesh, F2Cptr* butterfly,
                               F2Cptr* option, F2Cptr* stats, F2Cptr* mesh,
                               F2Cptr* kernel, F2Cptr* process_tree,
                               void (*distance)(int*, int*, double*, C2Fptr),
                               void (*near_far)(int*, int*, int*, C2Fptr),
                               C2Fptr context)
    {
        z_c_bf_construct_init(rows, columns, local_rows, local_columns,
                              row_counts, column_counts, row_mesh, column_mesh,
                              butterfly, option, stats, mesh, kernel, process_tree,
                              distance, near_far, context);
    }
    static void butterfly_compute(
        F2Cptr* butterfly, F2Cptr* option, F2Cptr* stats, F2Cptr* mesh,
        F2Cptr* kernel, F2Cptr* process_tree,
        void (*sample)(int*, int*, native_complex*, C2Fptr),
        void (*sample_block)(int*, int*, int*, std::int64_t*, int*, int*,
                             native_complex*, int*, int*, int*, int*, int*, C2Fptr),
        C2Fptr context)
    {
        z_c_bf_construct_element_compute(butterfly, option, stats, mesh, kernel,
                                         process_tree, sample, sample_block, context);
    }
    static void multiply(const char* trans, const native_complex* input,
                         native_complex* output, int* input_rows, int* output_rows,
                         int* vectors, F2Cptr* butterfly, F2Cptr* option,
                         F2Cptr* stats, F2Cptr* tree)
    {
        z_c_bf_mult(trans, input, output, input_rows, output_rows, vectors,
                    butterfly, option, stats, tree);
    }
    static void get_stats(F2Cptr* stats, const char* name, double* value)
    { z_c_bpack_getstats(stats, name, value); }
    static void delete_butterfly(F2Cptr* value) { z_c_bf_deletebf(value); }
    static void delete_mesh(F2Cptr* value) { z_c_bpack_deletemesh(value); }
    static void delete_kernel(F2Cptr* value) { z_c_bpack_deletekernelquant(value); }
    static void delete_matrix(F2Cptr* value) { z_c_bpack_delete(value); }
    static void delete_stats(F2Cptr* value) { z_c_bpack_deletestats(value); }
    static void delete_option(F2Cptr* value) { z_c_bpack_deleteoption(value); }
    static void delete_tree(F2Cptr* value) { z_c_bpack_deleteproctree(value); }
};
#endif

template <typename Real>
typename Api<Real>::native_complex to_native(const std::complex<Real>& input)
{
    typename Api<Real>::native_complex output;
    __real__ output = input.real();
    __imag__ output = input.imag();
    return output;
}

template <typename Real>
std::complex<Real> from_native(typename Api<Real>::native_complex input)
{
    return std::complex<Real>(static_cast<Real>(__real__ input),
                              static_cast<Real>(__imag__ input));
}

void dummy_distance(int*, int*, double* value, C2Fptr) { *value = 0.0; }
void dummy_near_far(int*, int*, int* value, C2Fptr) { *value = 0; }

template <typename Real>
void configure_options(F2Cptr* option, const Options& value)
{
    using ApiType = Api<Real>;
    ApiType::set_double(option, "tol_comp", value.tolerance);
    ApiType::set_double(option, "tol_rand", value.tolerance);
    ApiType::set_double(option, "tol_Rdetect",
                        value.rank_detection_factor * value.tolerance);
    ApiType::set_double(option, "sample_para", value.sample_parameter);
    // ButterflyPACK exposes a separate oversampling control for the
    // outermost adaptive levels. Keep it synchronized with sample_para so
    // less_adapt=1 does not silently retain a larger library default.
    ApiType::set_double(option, "sample_para_outer", value.sample_parameter);
    ApiType::set_integer(option, "nogeo", 0);
    ApiType::set_integer(option, "Nmin_leaf", value.leaf_size);
    ApiType::set_integer(option, "RecLR_leaf", 5);
    ApiType::set_integer(option, "xyzsort", 1);
    ApiType::set_integer(option, "ErrFillFull", 0);
    ApiType::set_integer(option, "BACA_Batch", 16);
    ApiType::set_integer(option, "LR_BLK_NUM", 1);
    ApiType::set_integer(option, "cpp", 1);
    ApiType::set_integer(option, "LRlevel", value.lr_level);
    ApiType::set_integer(option, "forwardN15flag", value.forward_n15_flag);
    ApiType::set_integer(option, "knn", value.nearest_neighbors);
    ApiType::set_integer(option, "verbosity", value.verbosity);
    ApiType::set_integer(option, "less_adapt", value.less_adapt);
    ApiType::set_integer(option, "pat_comp", value.compression_pattern);
    ApiType::set_integer(option, "itermax", 10);
    ApiType::set_integer(option, "ErrSol", 0);
    ApiType::set_integer(option, "elem_extract", 2);
    ApiType::set_integer(option, "format", 1);
}

template <typename Real>
struct ReusableGeometryState {
    using ApiType = Api<Real>;

    ReusableGeometryState(int row_count,
                          int column_count,
                          const std::vector<double>& row_locations,
                          const std::vector<double>& column_locations,
                          const Options& build_options)
        : rows(row_count),
          columns(column_count),
          coordinate_dimension(build_options.coordinate_dimension),
          leaf_size(build_options.leaf_size),
          nearest_neighbors(build_options.nearest_neighbors),
          row_coordinates(row_locations),
          column_coordinates(column_locations)
    {
        const auto setup_started = std::chrono::steady_clock::now();
        int process_count = 1;
        int process_groups[1] = {0};
        MPI_Fint communicator = static_cast<MPI_Fint>(321);
        ApiType::create_process_tree(
            &process_count, process_groups, &communicator, &process_tree);
        ApiType::create_option(&options);
        ApiType::create_stats(&row_stats);
        ApiType::create_stats(&column_stats);
        configure_options<Real>(&options, build_options);
        setup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - setup_started).count();

        row_new_to_old.resize(static_cast<std::size_t>(rows));
        column_new_to_old.resize(static_cast<std::size_t>(columns));
        int row_counts[1] = {rows};
        int column_counts[1] = {columns};
        int no_user_tree = 0;
        int row_tree[1] = {rows};
        int column_tree[1] = {columns};
        int dimensions = coordinate_dimension;
        int mutable_rows = rows;
        int mutable_columns = columns;

        const auto row_started = std::chrono::steady_clock::now();
        ApiType::construct_init(
            &mutable_rows, &dimensions, row_coordinates.data(), row_counts,
            &no_user_tree, row_tree, row_new_to_old.data(), &row_local,
            &row_dummy_matrix, &options, &row_stats, &row_mesh, &row_kernel,
            &process_tree, &dummy_distance, &dummy_near_far, this);
        row_tree_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - row_started).count();

        no_user_tree = 0;
        dimensions = coordinate_dimension;
        const auto column_started = std::chrono::steady_clock::now();
        ApiType::construct_init(
            &mutable_columns, &dimensions, column_coordinates.data(),
            column_counts, &no_user_tree, column_tree,
            column_new_to_old.data(), &column_local,
            &column_dummy_matrix, &options, &column_stats, &column_mesh,
            &column_kernel, &process_tree, &dummy_distance, &dummy_near_far,
            this);
        column_tree_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - column_started).count();

        if (row_local != rows || column_local != columns) {
            throw std::runtime_error(
                "ButterflyPACK reusable geometry returned a non-sequential partition");
        }
    }

    ~ReusableGeometryState()
    {
        if (row_dummy_matrix != nullptr)
            ApiType::delete_matrix(&row_dummy_matrix);
        if (column_dummy_matrix != nullptr)
            ApiType::delete_matrix(&column_dummy_matrix);
        if (row_mesh != nullptr) ApiType::delete_mesh(&row_mesh);
        if (column_mesh != nullptr) ApiType::delete_mesh(&column_mesh);
        if (row_kernel != nullptr) ApiType::delete_kernel(&row_kernel);
        if (column_kernel != nullptr) ApiType::delete_kernel(&column_kernel);
        if (row_stats != nullptr) ApiType::delete_stats(&row_stats);
        if (column_stats != nullptr) ApiType::delete_stats(&column_stats);
        if (options != nullptr) ApiType::delete_option(&options);
        if (process_tree != nullptr) ApiType::delete_tree(&process_tree);
    }

    bool matches(int row_count,
                 int column_count,
                 const std::vector<double>& row_locations,
                 const std::vector<double>& column_locations,
                 const Options& build_options) const
    {
        return rows == row_count &&
               columns == column_count &&
               coordinate_dimension == build_options.coordinate_dimension &&
               leaf_size == build_options.leaf_size &&
               nearest_neighbors == build_options.nearest_neighbors &&
               row_coordinates == row_locations &&
               column_coordinates == column_locations;
    }

    int rows = 0;
    int columns = 0;
    int coordinate_dimension = 0;
    int leaf_size = 0;
    int nearest_neighbors = 0;
    int row_local = 0;
    int column_local = 0;
    double setup_seconds = 0.0;
    double row_tree_seconds = 0.0;
    double column_tree_seconds = 0.0;
    std::vector<double> row_coordinates;
    std::vector<double> column_coordinates;
    std::vector<int> row_new_to_old;
    std::vector<int> column_new_to_old;

    F2Cptr process_tree = nullptr;
    F2Cptr options = nullptr;
    F2Cptr row_stats = nullptr;
    F2Cptr column_stats = nullptr;
    F2Cptr row_dummy_matrix = nullptr;
    F2Cptr column_dummy_matrix = nullptr;
    F2Cptr row_mesh = nullptr;
    F2Cptr column_mesh = nullptr;
    F2Cptr row_kernel = nullptr;
    F2Cptr column_kernel = nullptr;
};

template <typename Real>
std::shared_ptr<ReusableGeometryState<Real>> acquire_reusable_geometry(
    int rows,
    int columns,
    const std::vector<double>& row_coordinates,
    const std::vector<double>& column_coordinates,
    const Options& options,
    bool* cache_hit)
{
    static std::mutex cache_mutex;
    static std::shared_ptr<ReusableGeometryState<Real>> cache;

    std::lock_guard<std::mutex> lock(cache_mutex);
    if (cache &&
        cache->matches(rows, columns, row_coordinates,
                       column_coordinates, options)) {
        if (cache_hit) *cache_hit = true;
        return cache;
    }

    auto replacement = std::make_shared<ReusableGeometryState<Real>>(
        rows, columns, row_coordinates, column_coordinates, options);
    cache = replacement;
    if (cache_hit) *cache_hit = false;
    return replacement;
}

}  // namespace

template <typename Real>
struct Matrix<Real>::Impl {
    using ApiType = Api<Real>;
    using Native = typename ApiType::native_complex;
    using Complex = std::complex<Real>;

    Impl(int row_count,
         int column_count,
         std::vector<double> row_locations,
         std::vector<double> column_locations,
         entry_function entry_callback,
         Options build_options)
        : rows(row_count),
          columns(column_count),
          row_coordinates(std::move(row_locations)),
          column_coordinates(std::move(column_locations)),
          entry(std::move(entry_callback)),
          options_value(build_options)
    {
        build();
    }

    ~Impl()
    {
        if (butterfly != nullptr) ApiType::delete_butterfly(&butterfly);
        if (butterfly_mesh != nullptr) ApiType::delete_mesh(&butterfly_mesh);
        if (butterfly_kernel != nullptr) ApiType::delete_kernel(&butterfly_kernel);

        // In reuse mode the row/column meshes and process tree are borrowed
        // from reusable_geometry. Their lifetime is held by the shared_ptr.
        if (!reusable_geometry) {
            if (row_dummy_matrix != nullptr) ApiType::delete_matrix(&row_dummy_matrix);
            if (column_dummy_matrix != nullptr) ApiType::delete_matrix(&column_dummy_matrix);
            if (row_mesh != nullptr) ApiType::delete_mesh(&row_mesh);
            if (column_mesh != nullptr) ApiType::delete_mesh(&column_mesh);
            if (row_kernel != nullptr) ApiType::delete_kernel(&row_kernel);
            if (column_kernel != nullptr) ApiType::delete_kernel(&column_kernel);
            if (row_stats != nullptr) ApiType::delete_stats(&row_stats);
            if (column_stats != nullptr) ApiType::delete_stats(&column_stats);
            if (process_tree != nullptr) ApiType::delete_tree(&process_tree);
        }
        if (butterfly_stats != nullptr) ApiType::delete_stats(&butterfly_stats);
        if (options != nullptr) ApiType::delete_option(&options);
    }

    static void sample(int* first, int* second, Native* value, C2Fptr context)
    {
        Impl* self = static_cast<Impl*>(context);
        int row_new = 0;
        int column_new = 0;
        if (*first > 0) {
            row_new = *first;
            column_new = -*second;
        } else {
            row_new = *second;
            column_new = -*first;
        }
        if (row_new < 1 || row_new > self->rows ||
            column_new < 1 || column_new > self->columns) {
            *value = to_native<Real>(Complex(0, 0));
            return;
        }
        const int row_old = self->row_new_to_old[static_cast<std::size_t>(row_new - 1)] - 1;
        const int column_old =
            self->column_new_to_old[static_cast<std::size_t>(column_new - 1)] - 1;
        *value = to_native<Real>(self->entry(row_old, column_old));
        self->sampled_entries.fetch_add(1, std::memory_order_relaxed);
    }

    static void sample_block(int* intersection_count,
                             int* all_row_count,
                             int* all_column_count,
                             std::int64_t* local_value_count,
                             int* all_rows,
                             int* all_columns,
                             Native* local_values,
                             int* row_counts,
                             int* column_counts,
                             int* process_group_indices,
                             int* process_group_count,
                             int* process_maps,
                             C2Fptr context)
    {
        Impl* self = static_cast<Impl*>(context);
        if (self == nullptr || intersection_count == nullptr ||
            all_row_count == nullptr || all_column_count == nullptr ||
            local_value_count == nullptr || all_rows == nullptr ||
            all_columns == nullptr || local_values == nullptr ||
            row_counts == nullptr || column_counts == nullptr ||
            process_group_indices == nullptr || process_group_count == nullptr ||
            process_maps == nullptr || *intersection_count < 0 ||
            *local_value_count < 0 || *process_group_count < 1) {
            return;
        }

        const Native zero = to_native<Real>(Complex(0, 0));
        for (std::int64_t i = 0; i < *local_value_count; ++i) {
            local_values[i] = zero;
        }

        std::int64_t row_offset = 0;
        std::int64_t column_offset = 0;
        std::int64_t value_offset = 0;
        for (int block = 0; block < *intersection_count; ++block) {
            const int rows_in_block = row_counts[block];
            const int columns_in_block = column_counts[block];
            const int group = process_group_indices[block];
            if (rows_in_block < 0 || columns_in_block < 0 || group < 0 ||
                group >= *process_group_count) {
                return;
            }

            const int process_rows = process_maps[group];
            const int process_columns =
                process_maps[*process_group_count + group];
            const int first_process =
                process_maps[2 * (*process_group_count) + group];

            // This wrapper creates a one-process ButterflyPACK tree. Therefore
            // every requested extraction block must be local to process zero.
            if (process_rows * process_columns != 1 || first_process != 0) {
                return;
            }

            std::vector<int> row_old_indices(
                static_cast<std::size_t>(rows_in_block));
            std::vector<int> column_old_indices(
                static_cast<std::size_t>(columns_in_block));

            for (int row = 0; row < rows_in_block; ++row) {
                const int row_new = all_rows[row_offset + row];
                if (row_new < 1 || row_new > self->rows) return;
                row_old_indices[static_cast<std::size_t>(row)] =
                    self->row_new_to_old[
                        static_cast<std::size_t>(row_new - 1)] - 1;
            }
            for (int column = 0; column < columns_in_block; ++column) {
                const int column_new = all_columns[column_offset + column];
                if (column_new < 1 || column_new > self->columns) return;
                column_old_indices[static_cast<std::size_t>(column)] =
                    self->column_new_to_old[
                        static_cast<std::size_t>(column_new - 1)] - 1;
            }

            const std::int64_t entry_count =
                static_cast<std::int64_t>(rows_in_block) * columns_in_block;
            if (value_offset < 0 ||
                entry_count < 0 ||
                value_offset + entry_count > *local_value_count) {
                return;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(entry_count >= 16384)
#endif
            for (std::int64_t linear = 0; linear < entry_count; ++linear) {
                const int column = static_cast<int>(
                    linear / static_cast<std::int64_t>(rows_in_block));
                const int row = static_cast<int>(
                    linear - static_cast<std::int64_t>(column) *
                                 rows_in_block);
                local_values[value_offset + linear] = to_native<Real>(
                    self->entry(
                        row_old_indices[static_cast<std::size_t>(row)],
                        column_old_indices[static_cast<std::size_t>(column)]));
            }

            row_offset += rows_in_block;
            column_offset += columns_in_block;
            value_offset += entry_count;
        }
        self->sampled_entries.fetch_add(
            static_cast<std::size_t>(value_offset),
            std::memory_order_relaxed);
    }

    void set_options()
    {
        configure_options<Real>(&options, options_value);
    }

    void collect_build_statistics()
    {
        statistics_value.sampled_entries =
            sampled_entries.load(std::memory_order_relaxed);
        ApiType::get_stats(&butterfly_stats, "Time_Fill",
                           &statistics_value.library_fill_seconds);
        ApiType::get_stats(&butterfly_stats, "Time_Entry",
                           &statistics_value.library_entry_seconds);
        ApiType::get_stats(&butterfly_stats, "Time_Entry_Traverse",
                           &statistics_value.library_entry_traverse_seconds);
        ApiType::get_stats(&butterfly_stats, "Time_Entry_BF",
                           &statistics_value.library_entry_butterfly_seconds);
        ApiType::get_stats(&butterfly_stats, "Time_Entry_Comm",
                           &statistics_value.library_entry_communication_seconds);
        ApiType::get_stats(&butterfly_stats, "Mem_Fill",
                           &statistics_value.compressed_megabytes);
        ApiType::get_stats(&butterfly_stats, "Mem_Peak",
                           &statistics_value.peak_megabytes);
        ApiType::get_stats(&butterfly_stats, "Rank_max",
                           &statistics_value.maximum_rank);
    }

    void build_reusing_geometry()
    {
        const auto total_started = std::chrono::steady_clock::now();

        const auto setup_started = total_started;
        ApiType::create_option(&options);
        ApiType::create_stats(&butterfly_stats);
        set_options();
        statistics_value.setup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - setup_started).count();

        bool geometry_cache_hit = false;
        reusable_geometry = acquire_reusable_geometry<Real>(
            rows, columns, row_coordinates, column_coordinates,
            options_value, &geometry_cache_hit);
        statistics_value.geometry_reused = geometry_cache_hit ? 1 : 0;
        statistics_value.row_tree_seconds =
            geometry_cache_hit ? 0.0 : reusable_geometry->row_tree_seconds;
        statistics_value.column_tree_seconds =
            geometry_cache_hit ? 0.0 : reusable_geometry->column_tree_seconds;

        row_new_to_old = reusable_geometry->row_new_to_old;
        column_new_to_old = reusable_geometry->column_new_to_old;
        process_tree = reusable_geometry->process_tree;
        row_mesh = reusable_geometry->row_mesh;
        column_mesh = reusable_geometry->column_mesh;

        int row_counts[1] = {rows};
        int column_counts[1] = {columns};
        int row_local = reusable_geometry->row_local;
        int column_local = reusable_geometry->column_local;
        int mutable_rows = rows;
        int mutable_columns = columns;

        const auto structure_started = std::chrono::steady_clock::now();
        ApiType::butterfly_init(
            &mutable_rows, &mutable_columns, &row_local, &column_local,
            row_counts, column_counts, &row_mesh, &column_mesh, &butterfly,
            &options, &butterfly_stats, &butterfly_mesh, &butterfly_kernel,
            &process_tree, &dummy_distance, &dummy_near_far, this);
        statistics_value.structure_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - structure_started).count();

        const auto compression_started = std::chrono::steady_clock::now();
        ApiType::butterfly_compute(
            &butterfly, &options, &butterfly_stats, &butterfly_mesh,
            &butterfly_kernel, &process_tree, &Impl::sample,
            &Impl::sample_block, this);
        statistics_value.compression_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - compression_started).count();
        statistics_value.build_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_started).count();
        collect_build_statistics();
    }

    void build()
    {
        if (rows <= 0 || columns <= 0) {
            throw std::invalid_argument("ButterflyPACK matrix dimensions must be positive");
        }
        if (options_value.coordinate_dimension <= 0 || options_value.leaf_size < 4 ||
            !(options_value.tolerance > 0.0) ||
            !(options_value.sample_parameter > 0.0) ||
            !std::isfinite(options_value.sample_parameter) ||
            !(options_value.rank_detection_factor > 0.0) ||
            !std::isfinite(options_value.rank_detection_factor) ||
            options_value.lr_level < 0 ||
            options_value.forward_n15_flag < 0 ||
            options_value.forward_n15_flag > 2 ||
            options_value.nearest_neighbors < 0 ||
            options_value.compression_pattern < 1 ||
            options_value.compression_pattern > 3 ||
            (options_value.less_adapt != 0 && options_value.less_adapt != 1) ||
            (options_value.reuse_geometry != 0 &&
             options_value.reuse_geometry != 1)) {
            throw std::invalid_argument("invalid ButterflyPACK options");
        }
        const std::size_t row_coordinate_count =
            static_cast<std::size_t>(rows) * options_value.coordinate_dimension;
        const std::size_t column_coordinate_count =
            static_cast<std::size_t>(columns) * options_value.coordinate_dimension;
        if (row_coordinates.size() != row_coordinate_count ||
            column_coordinates.size() != column_coordinate_count || !entry) {
            throw std::invalid_argument("ButterflyPACK coordinates or entry callback are invalid");
        }

        if (options_value.reuse_geometry != 0) {
            build_reusing_geometry();
            return;
        }

        const auto total_started = std::chrono::steady_clock::now();
        const auto setup_started = total_started;
        int process_count = 1;
        int process_groups[1] = {0};
        MPI_Fint communicator = static_cast<MPI_Fint>(321);
        ApiType::create_process_tree(&process_count, process_groups, &communicator,
                                     &process_tree);
        ApiType::create_option(&options);
        ApiType::create_stats(&row_stats);
        ApiType::create_stats(&column_stats);
        ApiType::create_stats(&butterfly_stats);
        set_options();
        statistics_value.setup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - setup_started).count();

        row_new_to_old.resize(static_cast<std::size_t>(rows));
        column_new_to_old.resize(static_cast<std::size_t>(columns));
        int row_counts[1] = {rows};
        int column_counts[1] = {columns};
        int row_local = 0;
        int column_local = 0;
        int no_user_tree = 0;
        int row_tree[1] = {rows};
        int column_tree[1] = {columns};
        int dimensions = options_value.coordinate_dimension;
        int mutable_rows = rows;
        int mutable_columns = columns;

        const auto row_tree_started = std::chrono::steady_clock::now();
        ApiType::construct_init(
            &mutable_rows, &dimensions, row_coordinates.data(), row_counts,
            &no_user_tree, row_tree, row_new_to_old.data(), &row_local,
            &row_dummy_matrix, &options, &row_stats, &row_mesh, &row_kernel,
            &process_tree, &dummy_distance, &dummy_near_far, this);
        statistics_value.row_tree_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - row_tree_started).count();

        no_user_tree = 0;
        dimensions = options_value.coordinate_dimension;
        const auto column_tree_started = std::chrono::steady_clock::now();
        ApiType::construct_init(
            &mutable_columns, &dimensions, column_coordinates.data(), column_counts,
            &no_user_tree, column_tree, column_new_to_old.data(), &column_local,
            &column_dummy_matrix, &options, &column_stats, &column_mesh,
            &column_kernel, &process_tree, &dummy_distance, &dummy_near_far, this);
        statistics_value.column_tree_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - column_tree_started).count();

        if (row_local != rows || column_local != columns) {
            throw std::runtime_error("ButterflyPACK returned a non-sequential partition");
        }

        const auto structure_started = std::chrono::steady_clock::now();
        ApiType::butterfly_init(
            &mutable_rows, &mutable_columns, &row_local, &column_local,
            row_counts, column_counts, &row_mesh, &column_mesh, &butterfly,
            &options, &butterfly_stats, &butterfly_mesh, &butterfly_kernel,
            &process_tree, &dummy_distance, &dummy_near_far, this);
        statistics_value.structure_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - structure_started).count();

        const auto compression_started = std::chrono::steady_clock::now();
        ApiType::butterfly_compute(
            &butterfly, &options, &butterfly_stats, &butterfly_mesh,
            &butterfly_kernel, &process_tree, &Impl::sample,
            &Impl::sample_block, this);
        statistics_value.compression_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - compression_started).count();
        statistics_value.build_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_started).count();
        collect_build_statistics();
    }

    std::vector<std::vector<Complex>> apply_many(
        const std::vector<std::vector<Complex>>& inputs)
    {
        if (inputs.empty()) return {};
        if (inputs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("too many ButterflyPACK right-hand sides");
        }
        for (const auto& input : inputs) {
            if (input.size() != static_cast<std::size_t>(columns)) {
                throw std::invalid_argument("ButterflyPACK input length is invalid");
            }
        }

        std::lock_guard<std::mutex> lock(apply_mutex);
        const auto apply_started = std::chrono::steady_clock::now();

        const int vector_count = static_cast<int>(inputs.size());
        std::vector<Native> input_new(static_cast<std::size_t>(columns) * inputs.size());
        std::vector<Native> output_new(static_cast<std::size_t>(rows) * inputs.size());

        for (int rhs = 0; rhs < vector_count; ++rhs) {
            const auto& input = inputs[static_cast<std::size_t>(rhs)];
            const std::size_t offset = static_cast<std::size_t>(rhs) * columns;
            for (int new_index = 0; new_index < columns; ++new_index) {
                const int old_index =
                    column_new_to_old[static_cast<std::size_t>(new_index)] - 1;
                input_new[offset + static_cast<std::size_t>(new_index)] =
                    to_native<Real>(input[static_cast<std::size_t>(old_index)]);
            }
        }
        const auto multiply_started = std::chrono::steady_clock::now();
        statistics_value.apply_pack_seconds += std::chrono::duration<double>(
            multiply_started - apply_started).count();

        int mutable_vectors = vector_count;
        int mutable_rows = rows;
        int mutable_columns = columns;
        const char operation = 'N';
        ApiType::multiply(&operation, input_new.data(), output_new.data(),
                          &mutable_columns, &mutable_rows, &mutable_vectors,
                          &butterfly, &options, &butterfly_stats, &process_tree);
        const auto unpack_started = std::chrono::steady_clock::now();
        statistics_value.apply_multiply_seconds += std::chrono::duration<double>(
            unpack_started - multiply_started).count();
        ApiType::get_stats(&butterfly_stats, "Time_Multiply",
                           &statistics_value.library_multiply_seconds);

        std::vector<std::vector<Complex>> outputs(
            inputs.size(), std::vector<Complex>(static_cast<std::size_t>(rows)));
        for (int rhs = 0; rhs < vector_count; ++rhs) {
            const std::size_t offset = static_cast<std::size_t>(rhs) * rows;
            auto& output = outputs[static_cast<std::size_t>(rhs)];
            for (int new_index = 0; new_index < rows; ++new_index) {
                const int old_index =
                    row_new_to_old[static_cast<std::size_t>(new_index)] - 1;
                output[static_cast<std::size_t>(old_index)] =
                    from_native<Real>(output_new[offset +
                                                    static_cast<std::size_t>(new_index)]);
            }
        }
        const auto apply_finished = std::chrono::steady_clock::now();
        statistics_value.apply_unpack_seconds += std::chrono::duration<double>(
            apply_finished - unpack_started).count();
        statistics_value.apply_seconds += std::chrono::duration<double>(
            apply_finished - apply_started).count();
        statistics_value.applied_vectors += inputs.size();

        return outputs;
    }

    int rows;
    int columns;
    std::vector<double> row_coordinates;
    std::vector<double> column_coordinates;
    entry_function entry;
    Options options_value;
    Statistics statistics_value;
    std::vector<int> row_new_to_old;
    std::vector<int> column_new_to_old;
    std::atomic<std::size_t> sampled_entries{0};
    std::mutex apply_mutex;
    std::shared_ptr<ReusableGeometryState<Real>> reusable_geometry;

    F2Cptr process_tree = nullptr;
    F2Cptr options = nullptr;
    F2Cptr row_stats = nullptr;
    F2Cptr column_stats = nullptr;
    F2Cptr butterfly_stats = nullptr;
    F2Cptr row_dummy_matrix = nullptr;
    F2Cptr column_dummy_matrix = nullptr;
    F2Cptr row_mesh = nullptr;
    F2Cptr column_mesh = nullptr;
    F2Cptr butterfly_mesh = nullptr;
    F2Cptr row_kernel = nullptr;
    F2Cptr column_kernel = nullptr;
    F2Cptr butterfly_kernel = nullptr;
    F2Cptr butterfly = nullptr;
};

template <typename Real>
Matrix<Real>::Matrix(int rows,
                     int columns,
                     std::vector<double> row_coordinates,
                     std::vector<double> column_coordinates,
                     entry_function entry,
                     Options options)
    : impl_(std::make_unique<Impl>(rows, columns,
                                   std::move(row_coordinates),
                                   std::move(column_coordinates),
                                   std::move(entry), options))
{
}

template <typename Real>
Matrix<Real>::~Matrix() = default;

template <typename Real>
std::vector<typename Matrix<Real>::complex_type>
Matrix<Real>::apply(const std::vector<complex_type>& input)
{
    auto outputs = impl_->apply_many({input});
    return std::move(outputs.front());
}

template <typename Real>
std::vector<std::vector<typename Matrix<Real>::complex_type>>
Matrix<Real>::apply_many(const std::vector<std::vector<complex_type>>& inputs)
{
    return impl_->apply_many(inputs);
}

template <typename Real>
int Matrix<Real>::rows() const noexcept
{
    return impl_->rows;
}

template <typename Real>
int Matrix<Real>::columns() const noexcept
{
    return impl_->columns;
}

template <typename Real>
const Statistics& Matrix<Real>::statistics() const noexcept
{
    return impl_->statistics_value;
}

#ifdef KIRCH_HAS_BUTTERFLYPACK_FLOAT
template class Matrix<float>;
#endif
#ifdef KIRCH_HAS_BUTTERFLYPACK_DOUBLE
template class Matrix<double>;
#endif

}  // namespace butterfly
}  // namespace se
