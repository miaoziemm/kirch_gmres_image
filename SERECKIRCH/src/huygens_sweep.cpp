#include "huygens_sweep.hpp"

#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_fs.h>
#include <SERECKIRCH/include/efmm.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace se {
namespace huygens {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

double bessel_j0(double argument)
{
#if defined(__APPLE__)
    return ::j0(argument);
#else
    return std::cyl_bessel_j(0.0, argument);
#endif
}

double bessel_y0(double argument)
{
#if defined(__APPLE__)
    return ::y0(argument);
#else
    return std::cyl_neumann(0.0, argument);
#endif
}

void close_sep(sep_t* file)
{
    if (file != nullptr) sep_close(file);
}

void configure_float_header(sep_t* output)
{
    output->headers->esize = 4;
    output->headers->le = 1;
    sep_set_header(output, "data_format", "native_float");
    sep_set_header_int(output, "esize", 4);
}

std::vector<float> to_rsf_order(const Model2D& model,
                                const std::vector<Complex>& wavefield,
                                bool imaginary)
{
    const std::size_t count = static_cast<std::size_t>(model.nx) * model.nz;
    if (wavefield.size() != count) {
        throw std::invalid_argument("wavefield size does not match model");
    }
    std::vector<float> result(count);
    for (int ix = 0; ix < model.nx; ++ix) {
        for (int iz = 0; iz < model.nz; ++iz) {
            const Complex value = wavefield[model.index(ix, iz)];
            result[static_cast<std::size_t>(iz) +
                   static_cast<std::size_t>(ix) * model.nz] =
                imaginary ? value.imag() : value.real();
        }
    }
    return result;
}

void require_same_shape(const Table3D& a,
                        const Table3D& b,
                        const char* a_name,
                        const char* b_name)
{
    if (a.nz_target != b.nz_target ||
        a.nx_target != b.nx_target ||
        a.nsource != b.nsource ||
        a.values.size() != b.values.size()) {
        throw std::runtime_error(std::string(a_name) + " and " + b_name +
                                 " table shapes differ");
    }
}

} // namespace

std::size_t Model2D::index(int ix, int iz) const
{
    return static_cast<std::size_t>(ix) +
           static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx);
}

float Model2D::x(int ix) const { return ox + static_cast<float>(ix) * dx; }
float Model2D::z(int iz) const { return oz + static_cast<float>(iz) * dz; }

std::size_t Table3D::index(int source, int ix_target, int iz_target) const
{
    return (static_cast<std::size_t>(source) * nx_target + ix_target) *
               nz_target +
           iz_target;
}

std::size_t Table3D::row_index(int ix_target, int iz_target) const
{
    return static_cast<std::size_t>(ix_target) * nz_target + iz_target;
}

int Table3D::rows() const { return nx_target * nz_target; }

Model2D read_velocity_model(const std::string& filename)
{
    sep_t* input = sep_open(filename.c_str(), SEP_READ, 0);
    if (input == nullptr || input->headers == nullptr || input->data == nullptr ||
        input->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for velocity model: " + filename);
    }

    try {
        if (input->headers->ndim < 2) {
            throw std::runtime_error("velocity RSF must have at least two axes");
        }

        Model2D model;
        model.nz = input->headers->n[0];
        model.nx = input->headers->n[1];
        model.dz = static_cast<float>(input->headers->d[0]);
        model.dx = static_cast<float>(input->headers->d[1]);
        model.oz = static_cast<float>(input->headers->o[0]);
        model.ox = static_cast<float>(input->headers->o[1]);

        if (model.nz < 4 || model.nx < 2 ||
            !(model.dz > 0.0f) || !(model.dx > 0.0f)) {
            throw std::runtime_error("invalid velocity model grid");
        }

        const std::size_t count = static_cast<std::size_t>(model.nz) * model.nx;
        std::vector<float> rsf_order(count);
        se_fsio_seek(input->data->io, 0);
        se_fsio_read_float(input->data->io, rsf_order.data(), count);

        model.velocity.resize(count);
        for (int ix = 0; ix < model.nx; ++ix) {
            for (int iz = 0; iz < model.nz; ++iz) {
                const float value = rsf_order[
                    static_cast<std::size_t>(iz) +
                    static_cast<std::size_t>(ix) * model.nz];
                if (!(value > 0.0f) || !std::isfinite(value)) {
                    throw std::runtime_error("velocity model contains invalid values");
                }
                model.velocity[model.index(ix, iz)] = value;
            }
        }

        close_sep(input);
        return model;
    } catch (...) {
        close_sep(input);
        throw;
    }
}

BlockInfo make_block_info(const Model2D& model,
                          int first_block_rows,
                          int block_rows,
                          int overlap_rows)
{
    if (first_block_rows < 3) {
        throw std::invalid_argument("first_block_rows must be at least 3");
    }
    if (first_block_rows >= model.nz) {
        throw std::invalid_argument("first_block_rows must be smaller than nz");
    }
    if (block_rows < 2) {
        throw std::invalid_argument("block_rows must be at least 2");
    }
    if (overlap_rows < 0 || overlap_rows >= block_rows ||
        overlap_rows > first_block_rows - 2) {
        throw std::invalid_argument(
            "overlap_rows must satisfy 0 <= overlap_rows < block_rows and "
            "overlap_rows <= first_block_rows-2; got first_block_rows=" +
            std::to_string(first_block_rows) + ", block_rows=" +
            std::to_string(block_rows) + ", overlap_rows=" +
            std::to_string(overlap_rows) + ". For the requested overlap, "
            "first_block_rows must be at least overlap_rows+2. The overlap "
            "denotes how far the next propagation datum is moved upward into "
            "the preceding completed block.");
    }

    BlockInfo info;
    info.nx = model.nx;
    info.nz = model.nz;
    info.dx = model.dx;
    info.dz = model.dz;
    info.ox = model.ox;
    info.oz = model.oz;
    info.first_block_rows = first_block_rows;
    info.block_rows = block_rows;
    info.overlap_rows = overlap_rows;

    Block first;
    first.id = 0;
    first.start_iz = 0;
    first.source_iz = 0;
    first.target_start_iz = 0;
    first.target_end_iz = first_block_rows - 1;
    // With no overlap, retain one extra row so that the next datum derivative
    // can use a centered difference. With a positive overlap, the next datum
    // lies inside the completed target interval and no forward halo is needed.
    first.halo_end_iz = std::min(
        model.nz - 1,
        first.target_end_iz + (overlap_rows == 0 ? 1 : 0));
    info.blocks.push_back(first);

    int previous_target_end = first.target_end_iz;
    int id = 1;
    while (previous_target_end < model.nz - 1) {
        Block block;
        block.id = id++;
        // The propagation interval overlaps the preceding completed block,
        // while the stored output interval remains contiguous and non-gapped.
        block.source_iz = std::max(1, previous_target_end - overlap_rows);
        block.start_iz = block.source_iz;
        block.target_start_iz = previous_target_end + 1;
        block.target_end_iz = std::min(
            model.nz - 1, previous_target_end + block_rows);
        block.halo_end_iz = std::min(
            model.nz - 1,
            block.target_end_iz + (overlap_rows == 0 ? 1 : 0));
        info.blocks.push_back(block);
        previous_target_end = block.target_end_iz;
    }

    validate_block_info(info, &model);
    return info;
}

void validate_block_info(const BlockInfo& info, const Model2D* model)
{
    if (info.nx < 2 || info.nz < 4 || !(info.dx > 0.0f) || !(info.dz > 0.0f)) {
        throw std::runtime_error("block information contains an invalid grid");
    }
    if (info.blocks.size() < 2) {
        throw std::runtime_error("at least two blocks are required");
    }
    if (info.block_rows < 2 || info.overlap_rows < 0 ||
        info.overlap_rows >= info.block_rows ||
        info.overlap_rows > info.first_block_rows - 2) {
        throw std::runtime_error(
            "block information has invalid first_block_rows/block_rows/overlap_rows");
    }
    if (model != nullptr) {
        if (info.nx != model->nx || info.nz != model->nz ||
            std::abs(info.dx - model->dx) > 1.0e-5f * std::max(1.0f, model->dx) ||
            std::abs(info.dz - model->dz) > 1.0e-5f * std::max(1.0f, model->dz)) {
            throw std::runtime_error("block information does not match velocity model");
        }
    }

    for (std::size_t i = 0; i < info.blocks.size(); ++i) {
        const Block& block = info.blocks[i];
        if (block.id != static_cast<int>(i)) {
            throw std::runtime_error("block ids must be consecutive from zero");
        }
        if (block.start_iz < 0 || block.source_iz < 0 ||
            block.target_start_iz < 0 || block.target_end_iz < 0 ||
            block.halo_end_iz < 0 || block.halo_end_iz >= info.nz) {
            throw std::runtime_error("block index lies outside model");
        }
        if (block.target_end_iz < block.target_start_iz ||
            block.halo_end_iz < block.target_end_iz) {
            throw std::runtime_error("block target interval is invalid");
        }
        if (i == 0) {
            const int expected_halo = std::min(
                info.nz - 1,
                block.target_end_iz + (info.overlap_rows == 0 ? 1 : 0));
            if (block.start_iz != 0 || block.source_iz != 0 ||
                block.target_start_iz != 0 || block.target_end_iz < 2 ||
                block.halo_end_iz != expected_halo) {
                throw std::runtime_error(
                    "first block is inconsistent with overlap_rows");
            }
        } else {
            const Block& previous = info.blocks[i - 1];
            const int expected_source = std::max(
                1, previous.target_end_iz - info.overlap_rows);
            const int expected_target_start = previous.target_end_iz + 1;
            const int expected_target_end = std::min(
                info.nz - 1, previous.target_end_iz + info.block_rows);
            const int expected_halo = std::min(
                info.nz - 1,
                expected_target_end + (info.overlap_rows == 0 ? 1 : 0));

            if (block.start_iz != expected_source ||
                block.source_iz != expected_source ||
                block.target_start_iz != expected_target_start ||
                block.target_end_iz != expected_target_end ||
                block.halo_end_iz != expected_halo) {
                throw std::runtime_error(
                    "adjacent blocks do not have a contiguous output interval and "
                    "an overlapping propagation interval");
            }
            if (block.source_iz < 1 || block.source_iz + 1 > previous.halo_end_iz) {
                throw std::runtime_error(
                    "the preceding block does not contain both neighboring rows "
                    "required for the propagation-datum centered derivative");
            }
        }
    }
    if (info.blocks.back().target_end_iz != info.nz - 1) {
        throw std::runtime_error(
            "contiguous target intervals do not cover the bottom model row");
    }
}

void write_block_info(const std::string& filename, const BlockInfo& info)
{
    validate_block_info(info, nullptr);
    ensure_parent_directory(filename);
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("cannot create block information file: " + filename);
    }
    output << std::setprecision(12)
           << "HUYGENS_BLOCK_INFO_V2\n"
           << "nx " << info.nx << '\n'
           << "nz " << info.nz << '\n'
           << "dx " << info.dx << '\n'
           << "dz " << info.dz << '\n'
           << "ox " << info.ox << '\n'
           << "oz " << info.oz << '\n'
           << "first_block_rows " << info.first_block_rows << '\n'
           << "block_rows " << info.block_rows << '\n'
           << "overlap_rows " << info.overlap_rows << '\n'
           << "nblock " << info.blocks.size() << '\n';
    for (const Block& block : info.blocks) {
        output << "block " << block.id
               << " start_iz " << block.start_iz
               << " source_iz " << block.source_iz
               << " target_start_iz " << block.target_start_iz
               << " target_end_iz " << block.target_end_iz
               << " halo_end_iz " << block.halo_end_iz << '\n';
    }
}

BlockInfo read_block_info(const std::string& filename)
{
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("cannot open block information file: " + filename);
    }

    std::string magic;
    input >> magic;
    if (magic == "HUYGENS_BLOCK_INFO_V1") {
        throw std::runtime_error(
            "legacy block file uses the old overlap semantics; regenerate it with "
            "the current huygens_block_info program");
    }
    if (magic != "HUYGENS_BLOCK_INFO_V2") {
        throw std::runtime_error("unsupported block information format/version");
    }

    BlockInfo info;
    int expected_blocks = -1;
    std::string key;
    while (input >> key) {
        if (key == "nx") input >> info.nx;
        else if (key == "nz") input >> info.nz;
        else if (key == "dx") input >> info.dx;
        else if (key == "dz") input >> info.dz;
        else if (key == "ox") input >> info.ox;
        else if (key == "oz") input >> info.oz;
        else if (key == "first_block_rows") input >> info.first_block_rows;
        else if (key == "block_rows") input >> info.block_rows;
        else if (key == "overlap_rows") input >> info.overlap_rows;
        else if (key == "nblock") input >> expected_blocks;
        else if (key == "block") {
            Block block;
            input >> block.id;
            std::string field;
            input >> field >> block.start_iz;
            if (field != "start_iz") throw std::runtime_error("invalid block start field");
            input >> field >> block.source_iz;
            if (field != "source_iz") throw std::runtime_error("invalid block source field");
            input >> field >> block.target_start_iz;
            if (field != "target_start_iz") throw std::runtime_error("invalid block target-start field");
            input >> field >> block.target_end_iz;
            if (field != "target_end_iz") throw std::runtime_error("invalid block target-end field");
            input >> field >> block.halo_end_iz;
            if (field != "halo_end_iz") throw std::runtime_error("invalid block halo field");
            info.blocks.push_back(block);
        } else {
            throw std::runtime_error("unknown block information key: " + key);
        }
        if (!input) {
            throw std::runtime_error("malformed block information file");
        }
    }

    if (expected_blocks < 0 ||
        expected_blocks != static_cast<int>(info.blocks.size())) {
        throw std::runtime_error("nblock does not match block entries");
    }
    validate_block_info(info, nullptr);
    return info;
}

std::vector<float> solve_fmm(const Model2D& model, int source_ix, int source_iz)
{
    if (source_ix < 0 || source_ix >= model.nx ||
        source_iz < 0 || source_iz >= model.nz) {
        throw std::invalid_argument("FMM source lies outside model");
    }

    efmm_t solver{};
    const float source_x = static_cast<float>(source_ix) * model.dx;
    const float source_z = static_cast<float>(source_iz) * model.dz;
    if (efmm_init(&solver, model.nx, model.nz, model.dx, model.dz,
                  source_x, source_z) != 0) {
        throw std::runtime_error("efmm_init failed");
    }

    try {
        if (efmm_set_vel(&solver, const_cast<float*>(model.velocity.data())) != 0) {
            throw std::runtime_error("efmm_set_vel failed");
        }
        const int status = efmm_solver(&solver);
        if (status != 0) {
            throw std::runtime_error("efmm_solver failed with status " +
                                     std::to_string(status));
        }
        const std::size_t count = static_cast<std::size_t>(model.nx) * model.nz;
        std::vector<float> result(solver.tt, solver.tt + count);
        efmm_free(&solver);
        return result;
    } catch (...) {
        efmm_free(&solver);
        throw;
    }
}

std::vector<int> regular_indices(int first, int last_inclusive, int stride)
{
    if (stride < 1 || first < 0 || last_inclusive < first) {
        throw std::invalid_argument("invalid regular index interval");
    }
    std::vector<int> result;
    for (int value = first; value <= last_inclusive; value += stride) {
        result.push_back(value);
    }
    return result;
}

LayerGeometry make_layer_geometry(const Model2D& model,
                                  const Block& block,
                                  int source_stride,
                                  int target_x_stride,
                                  int target_z_stride)
{
    // The original single-frequency prototypes initialize block zero with an
    // analytic Hankel field and therefore must not build a zero-thickness
    // source-to-source table. Frequency-domain imaging, however, converts
    // block zero into a surface-to-first-block propagation interval by setting
    // target_start_iz > source_iz. Allow that valid imaging geometry.
    if (block.id == 0 && block.target_start_iz <= block.source_iz) {
        throw std::invalid_argument(
            "unmodified block zero uses Hankel initialization, not a travel-time table");
    }
    if ((model.nx - 1) % source_stride != 0) {
        throw std::invalid_argument(
            "source_stride must divide nx-1 so both datum endpoints are sampled");
    }
    if ((model.nx - 1) % target_x_stride != 0) {
        throw std::invalid_argument(
            "target_x_stride must divide nx-1");
    }
    if ((block.halo_end_iz - block.target_start_iz) % target_z_stride != 0) {
        throw std::invalid_argument(
            "target_z_stride must divide the local target interval");
    }

    LayerGeometry geometry;
    geometry.source_ix = regular_indices(0, model.nx - 1, source_stride);
    geometry.target_ix = regular_indices(0, model.nx - 1, target_x_stride);
    geometry.target_iz = regular_indices(
        block.target_start_iz, block.halo_end_iz, target_z_stride);
    geometry.targets.reserve(geometry.target_ix.size() * geometry.target_iz.size());
    for (int ix : geometry.target_ix) {
        for (int iz : geometry.target_iz) {
            geometry.targets.push_back({ix, iz});
        }
    }
    if (geometry.source_ix.size() < 2 || geometry.targets.empty()) {
        throw std::runtime_error("layer geometry has too few samples");
    }
    return geometry;
}

std::vector<float> trapezoidal_weights(const Model2D& model,
                                       const std::vector<int>& source_ix)
{
    if (source_ix.size() < 2) {
        throw std::invalid_argument("at least two source points are required");
    }
    std::vector<float> weights(source_ix.size(), 0.0f);
    weights.front() = 0.5f * (model.x(source_ix[1]) - model.x(source_ix[0]));
    for (std::size_t i = 1; i + 1 < source_ix.size(); ++i) {
        weights[i] = 0.5f *
            (model.x(source_ix[i + 1]) - model.x(source_ix[i - 1]));
    }
    weights.back() = 0.5f *
        (model.x(source_ix.back()) - model.x(source_ix[source_ix.size() - 2]));
    return weights;
}

std::string layer_table_filename(const std::string& prefix,
                                 int block_id,
                                 const std::string& quantity)
{
    std::ostringstream name;
    name << prefix << "_block_" << std::setw(3) << std::setfill('0') << block_id
         << '_' << quantity << ".rsf";
    return name.str();
}

void ensure_parent_directory(const std::string& filename)
{
    const std::filesystem::path path(filename);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

void write_table_rsf(const std::string& filename,
                     const Model2D& model,
                     const Block& block,
                     const Table3D& table,
                     const std::string& quantity,
                     int source_stride,
                     int target_x_stride,
                     int target_z_stride)
{
    const std::size_t expected = static_cast<std::size_t>(table.nz_target) *
                                 table.nx_target * table.nsource;
    if (table.values.size() != expected) {
        throw std::invalid_argument("table size does not match dimensions");
    }
    ensure_parent_directory(filename);
    sep_t* output = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (output == nullptr || output->headers == nullptr || output->data == nullptr ||
        output->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for table output: " + filename);
    }

    try {
        output->headers->ndim = 3;
        output->headers->n[0] = table.nz_target;
        output->headers->n[1] = table.nx_target;
        output->headers->n[2] = table.nsource;
        output->headers->d[0] = table.dz_target;
        output->headers->d[1] = table.dx_target;
        output->headers->d[2] = table.dx_source;
        output->headers->o[0] = table.oz_target;
        output->headers->o[1] = table.ox_target;
        output->headers->o[2] = table.ox_source;
        configure_float_header(output);

        sep_set_header(output, "label1", "Target depth");
        sep_set_header(output, "unit1", "m");
        sep_set_header(output, "label2", "Target distance");
        sep_set_header(output, "unit2", "m");
        sep_set_header(output, "label3", "Datum source distance");
        sep_set_header(output, "unit3", "m");
        sep_set_header(output, "quantity", quantity.c_str());
        sep_set_header(output, "title", "Huygens-sweep local asymptotic table");
        sep_set_header_int(output, "block_id", block.id);
        sep_set_header_int(output, "source_iz", block.source_iz);
        sep_set_header_int(output, "target_start_iz", block.target_start_iz);
        sep_set_header_int(output, "target_end_iz", block.target_end_iz);
        sep_set_header_int(output, "halo_end_iz", block.halo_end_iz);
        sep_set_header_int(output, "source_stride", source_stride);
        sep_set_header_int(output, "target_x_stride", target_x_stride);
        sep_set_header_int(output, "target_z_stride", target_z_stride);
        sep_set_header_float(output, "model_dx", model.dx);
        sep_set_header_float(output, "model_dz", model.dz);

        se_fsio_write_float(output->data->io, table.values.data(), table.values.size());
        close_sep(output);
    } catch (...) {
        close_sep(output);
        throw;
    }
}

Table3D read_table_rsf(const std::string& filename)
{
    sep_t* input = sep_open(filename.c_str(), SEP_READ, 0);
    if (input == nullptr || input->headers == nullptr || input->data == nullptr ||
        input->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for table input: " + filename);
    }
    try {
        if (input->headers->ndim < 3) {
            throw std::runtime_error("asymptotic table must have three axes");
        }
        Table3D table;
        table.nz_target = input->headers->n[0];
        table.nx_target = input->headers->n[1];
        table.nsource = input->headers->n[2];
        table.dz_target = static_cast<float>(input->headers->d[0]);
        table.dx_target = static_cast<float>(input->headers->d[1]);
        table.dx_source = static_cast<float>(input->headers->d[2]);
        table.oz_target = static_cast<float>(input->headers->o[0]);
        table.ox_target = static_cast<float>(input->headers->o[1]);
        table.ox_source = static_cast<float>(input->headers->o[2]);
        if (table.nz_target < 1 || table.nx_target < 1 || table.nsource < 2) {
            throw std::runtime_error("invalid asymptotic table dimensions");
        }
        const std::size_t count = static_cast<std::size_t>(table.nz_target) *
                                  table.nx_target * table.nsource;
        table.values.resize(count);
        se_fsio_seek(input->data->io, 0);
        se_fsio_read_float(input->data->io, table.values.data(), count);
        close_sep(input);
        return table;
    } catch (...) {
        close_sep(input);
        throw;
    }
}

LayerTables read_layer_tables(const std::string& prefix, int block_id)
{
    LayerTables tables;
    tables.traveltime = read_table_rsf(
        layer_table_filename(prefix, block_id, "tau"));
    tables.amplitude = read_table_rsf(
        layer_table_filename(prefix, block_id, "amp"));
    tables.normal_traveltime_derivative = read_table_rsf(
        layer_table_filename(prefix, block_id, "dtaun"));
    require_same_shape(tables.traveltime, tables.amplitude, "traveltime", "amplitude");
    require_same_shape(tables.traveltime, tables.normal_traveltime_derivative,
                       "traveltime", "normal derivative");
    return tables;
}

void validate_layer_tables(const LayerTables& tables,
                           const LayerGeometry& geometry,
                           const Block& block)
{
    const Table3D& table = tables.traveltime;
    if (table.nsource != static_cast<int>(geometry.source_ix.size()) ||
        table.nx_target != static_cast<int>(geometry.target_ix.size()) ||
        table.nz_target != static_cast<int>(geometry.target_iz.size())) {
        throw std::runtime_error("table shape does not match requested layer geometry");
    }
    if (geometry.target_iz.front() != block.target_start_iz ||
        geometry.target_iz.back() > block.halo_end_iz) {
        throw std::runtime_error("table target depths do not match block");
    }
}

OneWayLayerTables read_one_way_layer_tables(const std::string& prefix,
                                            int block_id)
{
    OneWayLayerTables tables;
    tables.traveltime = read_table_rsf(
        layer_table_filename(prefix, block_id, "tau"));
    return tables;
}

void validate_one_way_layer_tables(const OneWayLayerTables& tables,
                                   const LayerGeometry& geometry,
                                   const Block& block)
{
    const Table3D& table = tables.traveltime;
    if (table.nsource != static_cast<int>(geometry.source_ix.size()) ||
        table.nx_target != static_cast<int>(geometry.target_ix.size()) ||
        table.nz_target != static_cast<int>(geometry.target_iz.size())) {
        throw std::runtime_error(
            "one-way traveltime table shape does not match requested geometry");
    }
    if (geometry.target_iz.front() != block.target_start_iz ||
        geometry.target_iz.back() > block.halo_end_iz) {
        throw std::runtime_error(
            "one-way traveltime target depths do not match block");
    }
}

void write_wavefield_component_rsf(const std::string& filename,
                                   const Model2D& model,
                                   const std::vector<Complex>& wavefield,
                                   float frequency,
                                   const std::string& method,
                                   bool imaginary)
{
    ensure_parent_directory(filename);
    const std::vector<float> values = to_rsf_order(model, wavefield, imaginary);
    sep_t* output = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (output == nullptr || output->headers == nullptr || output->data == nullptr ||
        output->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for wavefield output: " + filename);
    }
    try {
        output->headers->ndim = 2;
        output->headers->n[0] = model.nz;
        output->headers->n[1] = model.nx;
        output->headers->d[0] = model.dz;
        output->headers->d[1] = model.dx;
        output->headers->o[0] = model.oz;
        output->headers->o[1] = model.ox;
        configure_float_header(output);
        sep_set_header(output, "label1", "Depth");
        sep_set_header(output, "unit1", "m");
        sep_set_header(output, "label2", "Distance");
        sep_set_header(output, "unit2", "m");
        sep_set_header(output, "method", method.c_str());
        sep_set_header(output, "component", imaginary ? "imaginary" : "real");
        sep_set_header(output, "title", "Layered Huygens-sweep frequency-domain wavefield");
        sep_set_header_float(output, "frequency", frequency);
        se_fsio_write_float(output->data->io, values.data(), values.size());
        close_sep(output);
    } catch (...) {
        close_sep(output);
        throw;
    }
}

void write_image_rsf(
    const std::string& filename,
    const Model2D& model,
    const std::vector<float>& image,
    const std::string& method,
    const std::vector<std::pair<std::string, float>>& scalar_headers)
{
    const std::size_t count = static_cast<std::size_t>(model.nx) * model.nz;
    if (image.size() != count) {
        throw std::invalid_argument("image size does not match model");
    }
    ensure_parent_directory(filename);
    std::vector<float> rsf_order(count);
    for (int ix = 0; ix < model.nx; ++ix) {
        for (int iz = 0; iz < model.nz; ++iz) {
            rsf_order[static_cast<std::size_t>(iz) +
                      static_cast<std::size_t>(ix) * model.nz] =
                image[model.index(ix, iz)];
        }
    }

    sep_t* output = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (output == nullptr || output->headers == nullptr ||
        output->data == nullptr || output->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for image output: " + filename);
    }
    try {
        output->headers->ndim = 2;
        output->headers->n[0] = model.nz;
        output->headers->n[1] = model.nx;
        output->headers->d[0] = model.dz;
        output->headers->d[1] = model.dx;
        output->headers->o[0] = model.oz;
        output->headers->o[1] = model.ox;
        configure_float_header(output);
        sep_set_header(output, "label1", "Depth");
        sep_set_header(output, "unit1", "km");
        sep_set_header(output, "label2", "Distance");
        sep_set_header(output, "unit2", "km");
        sep_set_header(output, "method", method.c_str());
        sep_set_header(output, "title", "Frequency-domain layered Kirchhoff image");
        for (const auto& item : scalar_headers) {
            sep_set_header_float(output, item.first.c_str(), item.second);
        }
        se_fsio_write_float(output->data->io, rsf_order.data(), rsf_order.size());
        close_sep(output);
    } catch (...) {
        close_sep(output);
        throw;
    }
}

void write_timing_rsf(const std::string& filename,
                      const std::vector<float>& values,
                      int metrics,
                      int blocks,
                      const std::string& method,
                      const std::vector<std::string>& metric_names,
                      const std::vector<std::pair<std::string, float>>& scalar_headers)
{
    if (metrics < 1 || blocks < 1 ||
        values.size() != static_cast<std::size_t>(metrics) * blocks ||
        metric_names.size() != static_cast<std::size_t>(metrics)) {
        throw std::invalid_argument("invalid timing table dimensions");
    }
    ensure_parent_directory(filename);
    sep_t* output = sep_open(filename.c_str(), SEP_WRITE, 0);
    if (output == nullptr || output->headers == nullptr || output->data == nullptr ||
        output->data->io == nullptr) {
        throw std::runtime_error("sep_open failed for timing output: " + filename);
    }
    try {
        output->headers->ndim = 2;
        output->headers->n[0] = metrics;
        output->headers->n[1] = blocks;
        output->headers->d[0] = 1.0;
        output->headers->d[1] = 1.0;
        output->headers->o[0] = 0.0;
        output->headers->o[1] = 1.0;
        configure_float_header(output);
        sep_set_header(output, "label1", "Metric");
        sep_set_header(output, "label2", "Propagation block id");
        sep_set_header(output, "method", method.c_str());
        sep_set_header(output, "title", "Huygens-sweep timing and compression statistics");
        for (int i = 0; i < metrics; ++i) {
            const std::string key = "metric" + std::to_string(i);
            sep_set_header(output, key.c_str(), metric_names[static_cast<std::size_t>(i)].c_str());
        }
        for (const auto& item : scalar_headers) {
            sep_set_header_float(output, item.first.c_str(), item.second);
        }
        se_fsio_write_float(output->data->io, values.data(), values.size());
        close_sep(output);
    } catch (...) {
        close_sep(output);
        throw;
    }
}

Complex hankel_point_source(float omega,
                            float velocity,
                            float radius,
                            float source_amplitude)
{
    const double argument = static_cast<double>(omega) * radius / velocity;
    const double j0 = bessel_j0(argument);
    const double y0 = bessel_y0(argument);
    return source_amplitude * Complex(
        static_cast<float>(-0.25 * y0),
        static_cast<float>(0.25 * j0));
}

namespace {

void initialize_first_block_hankel_impl(
    const Model2D& model,
    const BlockInfo& info,
    float frequency,
    int source_ix,
    int source_iz,
    float source_amplitude,
    float source_radius,
    bool negative_spatial_phase,
    std::vector<Complex>& wavefield)
{
    if (!(frequency > 0.0f) || source_ix < 0 || source_ix >= model.nx ||
        source_iz < 0 || source_iz >= model.nz) {
        throw std::invalid_argument("invalid Hankel initialization parameters");
    }
    const Block& first = info.blocks.front();
    const int last_iz = first.halo_end_iz;
    const float regularization = source_radius > 0.0f
        ? source_radius
        : 0.5f * std::min(model.dx, model.dz);
    const float omega = 2.0f * static_cast<float>(kPi) * frequency;
    const float velocity = model.velocity[model.index(source_ix, source_iz)];

    wavefield.assign(static_cast<std::size_t>(model.nx) * model.nz,
                     Complex(0.0f, 0.0f));
    for (int iz = 0; iz <= last_iz; ++iz) {
        for (int ix = 0; ix < model.nx; ++ix) {
            const float rx = model.x(ix) - model.x(source_ix);
            const float rz = model.z(iz) - model.z(source_iz);
            const float radius = std::max(
                std::sqrt(rx * rx + rz * rz), regularization);
            Complex value = hankel_point_source(
                omega, velocity, radius, source_amplitude);
            if (negative_spatial_phase) {
                // FFTW's forward transform stores delayed waves with the
                // exp(-i*omega*tau) phase convention.  The corresponding
                // outgoing 2-D launch field is the complex conjugate of
                // i/4 H_0^(1)(kr), i.e. -i/4 H_0^(2)(kr).
                value = std::conj(value);
            }
            wavefield[model.index(ix, iz)] = value;
        }
    }
}

} // namespace

void initialize_first_block_hankel(const Model2D& model,
                                   const BlockInfo& info,
                                   float frequency,
                                   int source_ix,
                                   int source_iz,
                                   float source_amplitude,
                                   float source_radius,
                                   std::vector<Complex>& wavefield)
{
    initialize_first_block_hankel_impl(
        model, info, frequency, source_ix, source_iz,
        source_amplitude, source_radius, false, wavefield);
}

void initialize_first_block_hankel_one_way(
    const Model2D& model,
    const BlockInfo& info,
    float frequency,
    int source_ix,
    int source_iz,
    float source_amplitude,
    float source_radius,
    std::vector<Complex>& wavefield)
{
    initialize_first_block_hankel_impl(
        model, info, frequency, source_ix, source_iz,
        source_amplitude, source_radius, true, wavefield);
}

std::vector<Complex> gather_boundary_cauchy_data(const Model2D& model,
                                                 const Block& block,
                                                 const std::vector<int>& source_ix,
                                                 const std::vector<Complex>& wavefield)
{
    if (wavefield.size() != static_cast<std::size_t>(model.nx) * model.nz) {
        throw std::invalid_argument("wavefield size does not match model");
    }
    const int iz = block.source_iz;
    if (iz < 1 || iz + 1 >= model.nz) {
        throw std::invalid_argument("boundary derivative requires rows above and below datum");
    }
    std::vector<Complex> input(2 * source_ix.size());
    const float inverse_interval = 1.0f / (2.0f * model.dz);
    for (std::size_t source = 0; source < source_ix.size(); ++source) {
        const int ix = source_ix[source];
        const Complex value = wavefield[model.index(ix, iz)];
        const Complex derivative_z =
            (wavefield[model.index(ix, iz + 1)] -
             wavefield[model.index(ix, iz - 1)]) * inverse_interval;
        // The outward normal of the lower half-space points upward: n = -z.
        input[2 * source] = value;
        input[2 * source + 1] = -derivative_z;
    }
    return input;
}


FrequencyKirchhoffFilter::FrequencyKirchhoffFilter(
    float frequency,
    float sample_interval,
    float filter_length,
    float maximum_traveltime,
    int lookup_subsamples)
    : frequency_(frequency),
      sample_interval_(sample_interval),
      filter_length_(filter_length),
      omega_(2.0f * static_cast<float>(kPi) * frequency)
{
    if (!(frequency_ > 0.0f) || !(sample_interval_ > 0.0f) ||
        !(filter_length_ > 0.0f) || !(maximum_traveltime > 0.0f) ||
        lookup_subsamples < 1) {
        throw std::invalid_argument(
            "invalid frequency-domain Kirchhoff filter parameters");
    }
    lookup_step_ = sample_interval_ / static_cast<float>(lookup_subsamples);
    const std::size_t count = static_cast<std::size_t>(
        std::ceil(maximum_traveltime / lookup_step_)) + 2;
    lookup_.resize(count);
    phase_removed_lookup_.resize(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < static_cast<long long>(count); ++i) {
        const float tau = std::max(
            lookup_step_ * static_cast<float>(i), 0.25f * sample_interval_);
        const Complex response = exact_response(tau);
        lookup_[static_cast<std::size_t>(i)] = response;
        phase_removed_lookup_[static_cast<std::size_t>(i)] =
            response * std::exp(Complex(0.0f, omega_ * tau));
    }
}

Complex FrequencyKirchhoffFilter::exact_response(float tau) const
{
    if (!(tau > 0.0f) || !std::isfinite(tau)) return Complex(0.0f, 0.0f);

    const int nsam = static_cast<int>(filter_length_ / sample_interval_) + 2;
    std::vector<float> coefficients(static_cast<std::size_t>(nsam - 1), 0.0f);
    float previous2 = 0.0f;
    float previous1 = std::sqrt(
        ((tau + sample_interval_) / tau) *
        ((tau + sample_interval_) / tau) - 1.0f);
    coefficients[0] = previous1 - previous2;
    for (int k = 1; k < nsam - 1; ++k) {
        const float t =
            (tau + static_cast<float>(k + 1) * sample_interval_) / tau;
        const float current = std::sqrt(t * t - 1.0f);
        coefficients[static_cast<std::size_t>(k)] =
            current - 2.0f * previous1 + previous2;
        previous2 = previous1;
        previous1 = current;
    }

    const float digital_omega = omega_ * sample_interval_;
    const Complex step = std::exp(Complex(0.0f, -digital_omega));
    Complex power(1.0f, 0.0f);
    Complex sum(0.0f, 0.0f);
    for (int k = 0; k <= nsam - 2; ++k) {
        sum += (coefficients[static_cast<std::size_t>(k)] / sample_interval_) *
               power;
        power *= step;
    }

    const float q = tau / sample_interval_;
    const int m = static_cast<int>(std::ceil(q));
    const float delta = static_cast<float>(m) - q;
    const Complex linear_delay =
        (1.0f - delta) *
            std::exp(Complex(0.0f, -digital_omega * static_cast<float>(m))) +
        delta * std::exp(Complex(
            0.0f, -digital_omega * static_cast<float>(m - 1)));
    return linear_delay * sum;
}

Complex FrequencyKirchhoffFilter::response(float tau) const
{
    if (!(tau > 0.0f) || !std::isfinite(tau)) return Complex(0.0f, 0.0f);
    const float coordinate = tau / lookup_step_;
    if (coordinate <= 0.0f) return lookup_.front();
    const std::size_t left = static_cast<std::size_t>(coordinate);
    if (left + 1 >= lookup_.size()) return lookup_.back();
    const float fraction = coordinate - static_cast<float>(left);
    return (1.0f - fraction) * lookup_[left] +
           fraction * lookup_[left + 1];
}

Complex FrequencyKirchhoffFilter::phase_removed_response(float tau) const
{
    if (!(tau > 0.0f) || !std::isfinite(tau)) return Complex(0.0f, 0.0f);
    const float coordinate = tau / lookup_step_;
    if (coordinate <= 0.0f) return phase_removed_lookup_.front();
    const std::size_t left = static_cast<std::size_t>(coordinate);
    if (left + 1 >= phase_removed_lookup_.size()) {
        return phase_removed_lookup_.back();
    }
    const float fraction = coordinate - static_cast<float>(left);
    return (1.0f - fraction) * phase_removed_lookup_[left] +
           fraction * phase_removed_lookup_[left + 1];
}

std::vector<Complex> gather_boundary_values(const Model2D& model,
                                            const Block& block,
                                            const std::vector<int>& source_ix,
                                            const std::vector<Complex>& wavefield)
{
    if (wavefield.size() != static_cast<std::size_t>(model.nx) * model.nz) {
        throw std::invalid_argument("wavefield size does not match model");
    }
    if (block.source_iz < 0 || block.source_iz >= model.nz) {
        throw std::invalid_argument("invalid propagation datum row");
    }
    std::vector<Complex> input(source_ix.size());
    for (std::size_t source = 0; source < source_ix.size(); ++source) {
        input[source] =
            wavefield[model.index(source_ix[source], block.source_iz)];
    }
    return input;
}

void inject_target_values(const Model2D& model,
                          const std::vector<GridPoint>& targets,
                          const std::vector<Complex>& values,
                          std::vector<Complex>& wavefield)
{
    if (targets.size() != values.size()) {
        throw std::invalid_argument("target/value sizes differ");
    }
    for (std::size_t i = 0; i < targets.size(); ++i) {
        wavefield[model.index(targets[i].ix, targets[i].iz)] = values[i];
    }
}

Complex KernelData::green(int row, int source) const
{
    const int iz_target = row % tables->traveltime.nz_target;
    const int ix_target = row / tables->traveltime.nz_target;
    const std::size_t offset = tables->traveltime.index(source, ix_target, iz_target);
    const float tau = std::max(tables->traveltime.values[offset], 1.0e-8f);
    const float amp = tables->amplitude.values[offset] /
                      std::sqrt(std::max(omega, 1.0e-12f));
    return amp * std::exp(Complex(
        0.0f, omega * tau + 0.25f * static_cast<float>(kPi)));
}

Complex KernelData::green_normal_derivative(int row, int source) const
{
    const int iz_target = row % tables->traveltime.nz_target;
    const int ix_target = row / tables->traveltime.nz_target;
    const std::size_t offset = tables->traveltime.index(source, ix_target, iz_target);
    const float dtaun = tables->normal_traveltime_derivative.values[offset];
    // Minimal prototype: retain the leading high-frequency derivative term.
    return green(row, source) * Complex(0.0f, omega * dtaun);
}

Complex KernelData::entry(int row, int column) const
{
    const int source = column / 2;
    const float weight = (*quadrature_weights)[static_cast<std::size_t>(source)];
    if ((column & 1) == 0) {
        return -weight * green_normal_derivative(row, source);
    }
    return weight * green(row, source);
}

Complex KernelData::apply_pair(int row,
                               int source,
                               const Complex& value,
                               const Complex& normal_derivative) const
{
    const float weight = (*quadrature_weights)[static_cast<std::size_t>(source)];
    return weight * (green(row, source) * normal_derivative -
                     value * green_normal_derivative(row, source));
}

std::vector<Complex> direct_apply(const KernelData& kernel,
                                  const std::vector<Complex>& input)
{
    if (input.size() != static_cast<std::size_t>(2 * kernel.sources)) {
        throw std::invalid_argument("Cauchy input length is invalid");
    }
    std::vector<Complex> output(static_cast<std::size_t>(kernel.rows),
                                Complex(0.0f, 0.0f));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < kernel.rows; ++row) {
        Complex sum(0.0f, 0.0f);
        for (int source = 0; source < kernel.sources; ++source) {
            sum += kernel.apply_pair(
                row, source,
                input[static_cast<std::size_t>(2 * source)],
                input[static_cast<std::size_t>(2 * source + 1)]);
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
    return output;
}


float OneWayKernelData::traveltime(int row, int source) const
{
    const int iz_target = row % tables->traveltime.nz_target;
    const int ix_target = row / tables->traveltime.nz_target;
    const std::size_t offset =
        tables->traveltime.index(source, ix_target, iz_target);
    return std::max(tables->traveltime.values[offset], 1.0e-8f);
}

float OneWayKernelData::geometry_factor(int row, int source,
                                        float tau) const
{
    if (quadrature_weights == nullptr || tables == nullptr) {
        throw std::logic_error(
            "one-way Kirchhoff geometry is not initialized");
    }
    const Table3D& table = tables->traveltime;
    const int iz_local = row % table.nz_target;
    const int ix_local = row / table.nz_target;
    const float target_x = table.ox_target + ix_local * table.dx_target;
    const float target_z = table.oz_target + iz_local * table.dz_target;
    const float source_x = table.ox_source + source * table.dx_source;
    const float vertical_distance = target_z - source_z;
    const float horizontal_distance = target_x - source_x;
    const float distance_squared =
        horizontal_distance * horizontal_distance +
        vertical_distance * vertical_distance;
    const float geometry = vertical_distance * tau /
        (static_cast<float>(kPi) * std::max(distance_squared, 1.0e-20f));
    return (*quadrature_weights)[static_cast<std::size_t>(source)] *
           geometry;
}

Complex OneWayKernelData::entry(int row, int source) const
{
    if (filter == nullptr || quadrature_weights == nullptr || tables == nullptr) {
        throw std::logic_error("one-way Kirchhoff kernel is not initialized");
    }
    const Table3D& table = tables->traveltime;
    const int iz_local = row % table.nz_target;
    const int ix_local = row / table.nz_target;
    const float target_x = table.ox_target + ix_local * table.dx_target;
    const float target_z = table.oz_target + iz_local * table.dz_target;
    const float source_x = table.ox_source + source * table.dx_source;
    const float vertical_distance = target_z - source_z;
    const float horizontal_distance = target_x - source_x;
    const float distance_squared =
        horizontal_distance * horizontal_distance +
        vertical_distance * vertical_distance;
    const float tau = traveltime(row, source);
    const float geometry = vertical_distance * tau /
        (static_cast<float>(kPi) * std::max(distance_squared, 1.0e-20f));
    return (*quadrature_weights)[static_cast<std::size_t>(source)] *
           geometry * filter->response(tau);
}

Complex OneWayKernelData::phase_removed_amplitude(int row, int source) const
{
    return phase_removed_amplitude(row, source, traveltime(row, source));
}

Complex OneWayKernelData::phase_removed_amplitude(int row, int source,
                                                  float tau) const
{
    const Table3D& table = tables->traveltime;
    const int iz_local = row % table.nz_target;
    const int ix_local = row / table.nz_target;
    const float target_x = table.ox_target + ix_local * table.dx_target;
    const float target_z = table.oz_target + iz_local * table.dz_target;
    const float source_x = table.ox_source + source * table.dx_source;
    const float vertical_distance = target_z - source_z;
    const float horizontal_distance = target_x - source_x;
    const float distance_squared = horizontal_distance * horizontal_distance +
                                   vertical_distance * vertical_distance;
    const float geometry = vertical_distance * tau /
        (static_cast<float>(kPi) * std::max(distance_squared, 1.0e-20f));
    return (*quadrature_weights)[static_cast<std::size_t>(source)] * geometry *
           filter->phase_removed_response(tau);
}

void OneWayKernelData::fill_phase_removed_amplitude_row(
    int row, const float* traveltimes, float* output) const
{
    const Table3D& table = tables->traveltime;
    const int iz_local = row % table.nz_target;
    const int ix_local = row / table.nz_target;
    const float target_x = table.ox_target + ix_local * table.dx_target;
    const float target_z = table.oz_target + iz_local * table.dz_target;
    const float vertical_distance = target_z - source_z;

    for (int source = 0; source < sources; ++source) {
        const float source_x = table.ox_source + source * table.dx_source;
        const float horizontal_distance = target_x - source_x;
        const float distance_squared = horizontal_distance * horizontal_distance +
                                       vertical_distance * vertical_distance;
        const float tau = traveltimes[source];
        const float geometry = vertical_distance * tau /
            (static_cast<float>(kPi) * std::max(distance_squared, 1.0e-20f));
        const Complex value =
            (*quadrature_weights)[static_cast<std::size_t>(source)] * geometry *
            filter->phase_removed_response(tau);
        output[2 * static_cast<std::size_t>(source)] = value.real();
        output[2 * static_cast<std::size_t>(source) + 1] = value.imag();
    }
}

void OneWayKernelData::fill_geometry_row(int row, const float* traveltimes,
                                         float* output) const
{
    const Table3D& table = tables->traveltime;
    const int iz_local = row % table.nz_target;
    const int ix_local = row / table.nz_target;
    const float target_x = table.ox_target + ix_local * table.dx_target;
    const float target_z = table.oz_target + iz_local * table.dz_target;
    const float vertical_distance = target_z - source_z;
    for (int source = 0; source < sources; ++source) {
        const float horizontal_distance = target_x -
            (table.ox_source + source * table.dx_source);
        const float distance_squared = horizontal_distance * horizontal_distance +
                                       vertical_distance * vertical_distance;
        output[source] =
            (*quadrature_weights)[static_cast<std::size_t>(source)] *
            vertical_distance * traveltimes[source] /
            (static_cast<float>(kPi) *
             std::max(distance_squared, 1.0e-20f));
    }
}

void OneWayKernelData::fill_phase_removed_amplitude_row_cached(
    const float* traveltimes, const float* geometry, float* output) const
{
    for (int source = 0; source < sources; ++source) {
        const Complex value = geometry[source] *
            filter->phase_removed_response(traveltimes[source]);
        output[2 * static_cast<std::size_t>(source)] = value.real();
        output[2 * static_cast<std::size_t>(source) + 1] = value.imag();
    }
}

std::vector<Complex> direct_one_way_apply(
    const OneWayKernelData& kernel,
    const std::vector<Complex>& input)
{
    if (input.size() != static_cast<std::size_t>(kernel.sources)) {
        throw std::invalid_argument("one-way Kirchhoff input length is invalid");
    }
    std::vector<Complex> output(static_cast<std::size_t>(kernel.rows),
                                Complex(0.0f, 0.0f));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < kernel.rows; ++row) {
        Complex sum(0.0f, 0.0f);
        for (int source = 0; source < kernel.sources; ++source) {
            sum += kernel.entry(row, source) *
                   input[static_cast<std::size_t>(source)];
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
    return output;
}

std::vector<double> make_row_coordinates(const Model2D& model,
                                         const std::vector<GridPoint>& targets)
{
    std::vector<double> coordinates(2 * targets.size());
    for (std::size_t i = 0; i < targets.size(); ++i) {
        coordinates[2 * i] = model.x(targets[i].ix);
        coordinates[2 * i + 1] = model.z(targets[i].iz);
    }
    return coordinates;
}

std::vector<double> make_column_coordinates(const Model2D& model,
                                            const Block& block,
                                            const std::vector<int>& source_ix)
{
    std::vector<double> coordinates(4 * source_ix.size());
    const double z = model.z(block.source_iz);
    for (std::size_t i = 0; i < source_ix.size(); ++i) {
        const double x = model.x(source_ix[i]);
        coordinates[4 * i] = x;
        coordinates[4 * i + 1] = z;
        coordinates[4 * i + 2] = x;
        coordinates[4 * i + 3] = z + 1.0e-6 * model.dz;
    }
    return coordinates;
}

float relative_l2_error(const std::vector<Complex>& reference,
                        const std::vector<Complex>& candidate)
{
    if (reference.size() != candidate.size()) {
        throw std::invalid_argument("error vectors differ in size");
    }
    long double numerator = 0.0;
    long double denominator = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        numerator += std::norm(candidate[i] - reference[i]);
        denominator += std::norm(reference[i]);
    }
    return static_cast<float>(std::sqrt(
        numerator / std::max(denominator, static_cast<long double>(1.0e-30))));
}

float maximum_normalized_error(const std::vector<Complex>& reference,
                               const std::vector<Complex>& candidate)
{
    if (reference.size() != candidate.size()) {
        throw std::invalid_argument("error vectors differ in size");
    }
    float maximum_reference = 0.0f;
    float maximum_difference = 0.0f;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        maximum_reference = std::max(maximum_reference, std::abs(reference[i]));
        maximum_difference = std::max(
            maximum_difference, std::abs(candidate[i] - reference[i]));
    }
    return maximum_difference / std::max(maximum_reference, 1.0e-30f);
}

} // namespace huygens
} // namespace se
