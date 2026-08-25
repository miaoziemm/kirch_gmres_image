#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 1. Paths: normally only BUILD_DIR and WORK_DIR need to be changed.
# ============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"
MODEL="${ROOT_DIR}/model/vmar.rsf"
WORK_DIR="${ROOT_DIR}/single_shot_test"

# ============================================================================
# 2. Runtime resources.
# ============================================================================
THREADS=32
MAX_TABLE_MB=8192

# ============================================================================
# 3. Single-shot forward-modeling parameters (coordinates use model units).
# ============================================================================
NT=4500
DT=0.0007
FDOM=20
SX=5.3715
SZ=0.0105
RZ=0.0105
NR=1024
R0=0.0
DR=0.0105
NBC=40
LAPLACE_TYPE=0                 # 0=FD8, 1=pseudospectral

# ============================================================================
# 4. Model-block and traveltime-table parameters.
#    The same strides are passed to traveltime calculation and imaging.
# ============================================================================
FIRST_BLOCK_ROWS=20
BLOCK_ROWS=50
OVERLAP_ROWS=10
SOURCE_STRIDE=1
TARGET_X_STRIDE=1
TARGET_Z_STRIDE=1

# Velocity smoothing used only by traveltime calculation and imaging.
SMOOTH_SIGMA=4

# Direct-wave front mute applied to the synthetic shot gather.
DIRECT_VELOCITY=1.5
DIRECT_EXTRA_TIME=0.08
DIRECT_TAPER_TIME=0.04

# ============================================================================
# 5. Frequency-domain imaging and ButterflyPACK parameters.
# ============================================================================
FMIN=1
FMAX=80
FREQUENCY_STRIDE=1
BPACK_TOL=1e-4
BPACK_LEAF=64
BPACK_DEPTH_CHUNK=20
GMRES_ITERATIONS=10
GMRES_RESTART=30
GMRES_TOLERANCE=1e-8
GMRES_ABSORBING_ROWS=25
GLOBAL_SMOOTH_SIGMA=0           # Model is already smoothed in Step 3.

# Derived output names. They normally do not need modification.
SHOT_DATA="${WORK_DIR}/single_shot.rsf"
MUTED_SHOT_DATA="${WORK_DIR}/single_shot_muted.rsf"
IMAGING_MODEL="${WORK_DIR}/vmar_smooth.rsf"
BLOCK_FILE="${WORK_DIR}/huygens_blocks.txt"
TABLE_PREFIX="${WORK_DIR}/traveltime/travel"
IMAGE="${WORK_DIR}/single_shot_image.rsf"
ILLUMINATION="${WORK_DIR}/single_shot_illumination.rsf"
TIMING="${WORK_DIR}/single_shot_timing.rsf"
IMAGE_NO_GMRES="${WORK_DIR}/single_shot_image_no_gmres.rsf"
ILLUMINATION_NO_GMRES="${WORK_DIR}/single_shot_illumination_no_gmres.rsf"
TIMING_NO_GMRES="${WORK_DIR}/single_shot_timing_no_gmres.rsf"

# Start every test with an empty output directory.
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/traveltime"
cd "${ROOT_DIR}"

# Check the four programs before starting a potentially expensive test.
for program in sewave2d_forward smooth_velocity_model mute_direct_wave \
               huygens_block_info travel_time_solver \
               frequency_kirchhoff_imaging_bf_gmres; do
    if [[ ! -x "${BIN_DIR}/${program}" ]]; then
        echo "Missing executable: ${BIN_DIR}/${program}" >&2
        echo "Configure and build the project first." >&2
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# Step 1: generate one synthetic shot gather from model/vmar.rsf.
# ----------------------------------------------------------------------------
echo "[1/7] Forward modeling one shot with the original velocity model"
OMP_NUM_THREADS="${THREADS}" "${BIN_DIR}/sewave2d_forward" \
    velocity="${MODEL}" output="${SHOT_DATA}" \
    ns=1 nt="${NT}" dt="${DT}" fdom="${FDOM}" \
    sx="${SX}" sz="${SZ}" rz="${RZ}" \
    nr="${NR}" r0="${R0}" dr="${DR}" nbc="${NBC}" \
    type_compute_Laplace="${LAPLACE_TYPE}"

# ----------------------------------------------------------------------------
# Step 2: remove the direct-wave arrival from the synthetic record.
# ----------------------------------------------------------------------------
echo "[2/7] Muting the direct wave"
"${BIN_DIR}/mute_direct_wave" input="${SHOT_DATA}" output="${MUTED_SHOT_DATA}" \
    direct_velocity="${DIRECT_VELOCITY}" source_depth="${SZ}" \
    receiver_depth="${RZ}" extra_time="${DIRECT_EXTRA_TIME}" \
    taper_time="${DIRECT_TAPER_TIME}"

# ----------------------------------------------------------------------------
# Step 3: create the smooth model used by traveltime calculation and imaging.
# The forward record above always uses the original model.
# ----------------------------------------------------------------------------
echo "[3/7] Smoothing the velocity model"
"${BIN_DIR}/smooth_velocity_model" \
    input="${MODEL}" output="${IMAGING_MODEL}" sigma="${SMOOTH_SIGMA}"

# ----------------------------------------------------------------------------
# Step 4: divide the smooth velocity model into propagation blocks.
# ----------------------------------------------------------------------------
echo "[4/7] Building the Huygens block description"
"${BIN_DIR}/huygens_block_info" \
    velocity="${IMAGING_MODEL}" output="${BLOCK_FILE}" \
    first_block_rows="${FIRST_BLOCK_ROWS}" block_rows="${BLOCK_ROWS}" \
    overlap_rows="${OVERLAP_ROWS}"

# ----------------------------------------------------------------------------
# Step 5: calculate tables for ButterflyPACK blocks. Block zero uses analytic
# phase-shift initialization and needs no table. The remaining tables include
# overlap rows used by Scheme-A blending.
# ----------------------------------------------------------------------------
echo "[5/7] Calculating traveltime tables with the smooth model"
"${BIN_DIR}/travel_time_solver" \
    velocity="${IMAGING_MODEL}" block_file="${BLOCK_FILE}" \
    output_prefix="${TABLE_PREFIX}" \
    timing="${WORK_DIR}/traveltime_timing.rsf" \
    include_propagation_overlap=1 source_stride="${SOURCE_STRIDE}" \
    target_x_stride="${TARGET_X_STRIDE}" target_z_stride="${TARGET_Z_STRIDE}" \
    threads="${THREADS}" max_table_mb="${MAX_TABLE_MB}"

# ----------------------------------------------------------------------------
# Step 6: image only shot 0. ButterflyPACK computes the receiver wavefield;
# global GMRES corrects both wavefields before cross-correlation imaging.
# ----------------------------------------------------------------------------
echo "[6/7] Imaging shot 0 with global GMRES correction"
OMP_NUM_THREADS="${THREADS}" "${BIN_DIR}/frequency_kirchhoff_imaging_bf_gmres" \
    velocity="${IMAGING_MODEL}" seismic_data="${MUTED_SHOT_DATA}" cmp=0 \
    block_file="${BLOCK_FILE}" table_prefix="${TABLE_PREFIX}" \
    shot_begin=0 shot_count=1 shot_stride=1 \
    fdom="${FDOM}" fmin="${FMIN}" fmax="${FMAX}" \
    frequency_stride="${FREQUENCY_STRIDE}" \
    source_stride="${SOURCE_STRIDE}" target_x_stride="${TARGET_X_STRIDE}" \
    target_z_stride="${TARGET_Z_STRIDE}" threads="${THREADS}" \
    bpack_tol="${BPACK_TOL}" bpack_leaf="${BPACK_LEAF}" \
    bpack_depth_chunk="${BPACK_DEPTH_CHUNK}" \
    gmres_enable=1 \
    global_gmres_iterations="${GMRES_ITERATIONS}" \
    gmres_restart="${GMRES_RESTART}" gmres_tolerance="${GMRES_TOLERANCE}" \
    gmres_nabs="${GMRES_ABSORBING_ROWS}" \
    global_smooth_sigma="${GLOBAL_SMOOTH_SIGMA}" \
    image="${IMAGE}" illumination="${ILLUMINATION}" timing="${TIMING}" \
    correction_csv="${WORK_DIR}/gmres_metrics.csv" \
    seam_csv="${WORK_DIR}/seam_metrics.csv"

# ButterflyPACK uses a Fortran STOP for internal NaNs, which may return status
# zero on some compilers.  Require the final image so such a stop cannot be
# mistaken for a successful Step 6.
if [[ ! -s "${IMAGE}" ]]; then
    echo "GMRES imaging did not create ${IMAGE}" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Step 7: run the same single-shot imaging again with GMRES disabled. All
# acquisition, frequency, ButterflyPACK, and traveltime parameters remain the
# same, so the two output images directly show the effect of global correction.
# ----------------------------------------------------------------------------
echo "[7/7] Imaging shot 0 without global GMRES correction"
OMP_NUM_THREADS="${THREADS}" "${BIN_DIR}/frequency_kirchhoff_imaging_bf_gmres" \
    velocity="${IMAGING_MODEL}" seismic_data="${MUTED_SHOT_DATA}" cmp=0 \
    block_file="${BLOCK_FILE}" table_prefix="${TABLE_PREFIX}" \
    shot_begin=0 shot_count=1 shot_stride=1 \
    fdom="${FDOM}" fmin="${FMIN}" fmax="${FMAX}" \
    frequency_stride="${FREQUENCY_STRIDE}" \
    source_stride="${SOURCE_STRIDE}" target_x_stride="${TARGET_X_STRIDE}" \
    target_z_stride="${TARGET_Z_STRIDE}" threads="${THREADS}" \
    bpack_tol="${BPACK_TOL}" bpack_leaf="${BPACK_LEAF}" \
    bpack_depth_chunk="${BPACK_DEPTH_CHUNK}" gmres_enable=0 \
    gmres_nabs="${GMRES_ABSORBING_ROWS}" \
    global_smooth_sigma="${GLOBAL_SMOOTH_SIGMA}" \
    image="${IMAGE_NO_GMRES}" illumination="${ILLUMINATION_NO_GMRES}" \
    timing="${TIMING_NO_GMRES}" \
    correction_csv="${WORK_DIR}/gmres_metrics_disabled.csv" \
    seam_csv="${WORK_DIR}/seam_metrics_no_gmres.csv"

if [[ ! -s "${IMAGE_NO_GMRES}" ]]; then
    echo "No-GMRES imaging did not create ${IMAGE_NO_GMRES}" >&2
    exit 1
fi

echo "Single-shot test completed."
echo "GMRES image:    ${IMAGE}"
echo "No-GMRES image: ${IMAGE_NO_GMRES}"
echo "All outputs: ${WORK_DIR}"
