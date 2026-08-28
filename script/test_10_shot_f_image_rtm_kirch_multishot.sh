#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Parameter-aligned test for f_image_rtm_kirch.
# Only parameter values are aligned; executable programs and workflow are unchanged.
# ============================================================================

REUSE="${REUSE:-1}"

# ============================================================================
# 1. Paths
# ============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"
MODEL="${ROOT_DIR}/model/vmar.rsf"
WORK_DIR="${ROOT_DIR}/ten_shot_f_image_rtm_kirch_test"

# ============================================================================
# 2. Runtime resources
# ============================================================================
THREADS=32
MAX_TABLE_MB=8192

# ============================================================================
# 3. Forward-modeling parameters
#    Exactly aligned with run_mar_single_shot_bf_gmres_compare_on_off_big(2).sh
# ============================================================================
NT=6000
DT=0.0007
FDOM=20

SX=7.14
SZ=0.0105
RZ=0.0

NS=1
DS=0
NR=1024
R0=0.0
DR=0.0105

NBC=40
LAPLACE_TYPE=0

# ============================================================================
# 4. Model-block / traveltime parameters
# ============================================================================
FIRST_BLOCK_ROWS=25
BLOCK_ROWS=100
OVERLAP_ROWS=20

SOURCE_STRIDE=1
TARGET_X_STRIDE=1
TARGET_Z_STRIDE=1

# Velocity smoothing parameter.
SMOOTH_SIGMA=2

# Keep the original direct-wave mute workflow.
DIRECT_VELOCITY=1.5
DIRECT_EXTRA_TIME=0.2
DIRECT_TAPER_TIME=0.04

# ============================================================================
# 5. Frequency-domain imaging / GMRES parameters
# ============================================================================
FMIN="${FMIN:-10}"
FMAX="${FMAX:-40}"
FREQUENCY_STRIDE="${FREQUENCY_STRIDE:-1}"

CORRECTION_Z0=0.105
SOURCE_Z="${SOURCE_Z:-${SZ}}"
GMRES_OUTER=30
GMRES_RESTART=30
GMRES_FREQUENCY_THREADS="${GMRES_FREQUENCY_THREADS:-4}"

# ButterflyPACK receiver-wavefield compression parameters.  Keep every
# algorithm control explicit here so a test run does not silently change when
# library defaults are adjusted.  Each value may be overridden from the
# environment for accuracy/performance experiments.
BPACK_TOL="${BPACK_TOL:-1e-4}"
BPACK_LEAF="${BPACK_LEAF:-64}"
BPACK_DEPTH_CHUNK="${BPACK_DEPTH_CHUNK:-20}"
BPACK_LRLEVEL="${BPACK_LRLEVEL:-100}"
BPACK_SAMPLE_PARA="${BPACK_SAMPLE_PARA:-1.5}"
BPACK_FORWARD_N15="${BPACK_FORWARD_N15:-0}"
BPACK_KNN="${BPACK_KNN:-10}"
BPACK_PAT_COMP="${BPACK_PAT_COMP:-1}"
BPACK_LESS_ADAPT="${BPACK_LESS_ADAPT:-1}"
BPACK_RDETECT_FACTOR="${BPACK_RDETECT_FACTOR:-0.3}"
BPACK_REUSE_TREE="${BPACK_REUSE_TREE:-1}"
BPACK_GEOMETRY_CACHE_MB="${BPACK_GEOMETRY_CACHE_MB:-1024}"
BPACK_VERBOSITY="${BPACK_VERBOSITY:--1}"
BPACK_PROGRESS="${BPACK_PROGRESS:-1}"
BPACK_PROGRESS_EVERY="${BPACK_PROGRESS_EVERY:-1}"

# ============================================================================
# 6. Derived output names
# ============================================================================
SHOT_DATA="${WORK_DIR}/ten_shot.rsf"
MUTED_SHOT_DATA="${WORK_DIR}/ten_shot_muted.rsf"

IMAGING_MODEL="${WORK_DIR}/vmar_smooth.rsf"
BLOCK_FILE="${WORK_DIR}/huygens_blocks.txt"

TABLE_DIR="${WORK_DIR}/traveltime"
TABLE_PREFIX="${TABLE_DIR}/travel"
TRAVEL_TIMING="${WORK_DIR}/traveltime_timing.rsf"

IMAGE="${WORK_DIR}/ten_shot_image_after_gmres.rsf"
ILLUMINATION="${WORK_DIR}/ten_shot_illumination.rsf"
TIMING="${WORK_DIR}/ten_shot_timing.rsf"

IMAGE_NO_GMRES="${WORK_DIR}/unused_before.rsf"
ILLUMINATION_NO_GMRES="${WORK_DIR}/unused_before_illumination.rsf"

FREQUENCY_FIELD_DIR="${WORK_DIR}/per_shot_images"
RECEIVER_STORE_DIR="${WORK_DIR}/receiver_scratch"

# ============================================================================
# 7. Helper functions
# ============================================================================
rsf_exists()
{
    local file="$1"
    [[ -s "${file}" ]]
}

traveltime_exists()
{
    find "${TABLE_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'travel*.rsf' \
        -size +0c \
        -print -quit 2>/dev/null | grep -q .
}

print_reuse()
{
    echo "      -> reuse: $1"
}

print_compute()
{
    echo "      -> compute: $1"
}

# ============================================================================
# 8. Initialize
# ============================================================================
if [[ "${REUSE}" == "0" ]]; then
    echo "============================================================"
    echo "REUSE=0: removing previous test results"
    echo "============================================================"
    rm -rf "${WORK_DIR}"
else
    echo "============================================================"
    echo "REUSE=1: existing aligned intermediate results will be reused"
    echo "NOTE: after replacing an older non-aligned script, run once with REUSE=0."
    echo "============================================================"
fi

mkdir -p "${WORK_DIR}" "${TABLE_DIR}" "${FREQUENCY_FIELD_DIR}"
cd "${ROOT_DIR}"

# ============================================================================
# 9. Check executables
# ============================================================================
for program in \
    sewave2d_forward \
    mute_direct_wave \
    smooth_velocity_model \
    huygens_block_info \
    travel_time_solver \
    f_image_rtm_kirch_multishot
do
    if [[ ! -x "${BIN_DIR}/${program}" ]]; then
        echo "Missing executable: ${BIN_DIR}/${program}" >&2
        echo "Configure and build the project first." >&2
        exit 1
    fi
done

# ============================================================================
# Step 1: Forward modeling in the original model
# ============================================================================
echo
echo "[1/6] Forward modeling ten shots with the original velocity model"

if [[ "${REUSE}" == "1" ]] && rsf_exists "${SHOT_DATA}"; then
    print_reuse "${SHOT_DATA}"
else
    print_compute "${SHOT_DATA}"

    OMP_NUM_THREADS="${THREADS}" \
    "${BIN_DIR}/sewave2d_forward" \
        velocity="${MODEL}" \
        output="${SHOT_DATA}" \
        ns="${NS}" \
        nt="${NT}" \
        dt="${DT}" \
        fdom="${FDOM}" \
        sx="${SX}" \
        sz="${SZ}" \
        rz="${RZ}" \
        ds="${DS}" \
        nr="${NR}" \
        r0="${R0}" \
        dr="${DR}" \
        nbc="${NBC}" \
        type_compute_Laplace="${LAPLACE_TYPE}"
    if ! rsf_exists "${SHOT_DATA}"; then
        echo "Forward modeling failed: ${SHOT_DATA} was not created." >&2
        exit 1
    fi
fi

# ============================================================================
# Step 2: Direct-wave mute
# ============================================================================
echo
echo "[2/6] Muting the direct wave"

if [[ "${REUSE}" == "1" ]] && rsf_exists "${MUTED_SHOT_DATA}"; then
    print_reuse "${MUTED_SHOT_DATA}"
else
    print_compute "${MUTED_SHOT_DATA}"

    "${BIN_DIR}/mute_direct_wave" \
        input="${SHOT_DATA}" \
        output="${MUTED_SHOT_DATA}" \
        direct_velocity="${DIRECT_VELOCITY}" \
        source_depth="${SZ}" \
        receiver_depth="${RZ}" \
        extra_time="${DIRECT_EXTRA_TIME}" \
        taper_time="${DIRECT_TAPER_TIME}"

    if ! rsf_exists "${MUTED_SHOT_DATA}"; then
        echo "Direct-wave mute failed: ${MUTED_SHOT_DATA} was not created." >&2
        exit 1
    fi
fi

# ============================================================================
# Step 3: Smooth imaging velocity model
# ============================================================================
echo
echo "[3/6] Smoothing the velocity model"

if [[ "${REUSE}" == "1" ]] && rsf_exists "${IMAGING_MODEL}"; then
    print_reuse "${IMAGING_MODEL}"
else
    print_compute "${IMAGING_MODEL}"

    "${BIN_DIR}/smooth_velocity_model" \
        input="${MODEL}" \
        output="${IMAGING_MODEL}" \
        sigma="${SMOOTH_SIGMA}"

    if ! rsf_exists "${IMAGING_MODEL}"; then
        echo "Velocity smoothing failed: ${IMAGING_MODEL} was not created." >&2
        exit 1
    fi
fi

# ============================================================================
# Step 4: Huygens block description
# ============================================================================
echo
echo "[4/6] Building Huygens blocks: first=25, block=100, overlap=20"

if [[ "${REUSE}" == "1" && -s "${BLOCK_FILE}" ]]; then
    print_reuse "${BLOCK_FILE}"
else
    print_compute "${BLOCK_FILE}"

    "${BIN_DIR}/huygens_block_info" \
        velocity="${IMAGING_MODEL}" \
        output="${BLOCK_FILE}" \
        first_block_rows="${FIRST_BLOCK_ROWS}" \
        block_rows="${BLOCK_ROWS}" \
        overlap_rows="${OVERLAP_ROWS}"

    if [[ ! -s "${BLOCK_FILE}" ]]; then
        echo "Block generation failed." >&2
        exit 1
    fi
fi

# ============================================================================
# Step 5: One-way traveltime calculation
# ============================================================================
echo
echo "[5/6] Calculating one-way traveltime tables"

if [[ "${REUSE}" == "1" ]] && traveltime_exists; then
    print_reuse "${TABLE_DIR}"
else
    print_compute "${TABLE_DIR}"

    rm -f "${TABLE_DIR}"/travel*.rsf
    rm -f "${TABLE_DIR}"/travel*.rsf@
    rm -f "${TRAVEL_TIMING}" "${TRAVEL_TIMING}@"

    "${BIN_DIR}/travel_time_solver" \
        velocity="${IMAGING_MODEL}" \
        block_file="${BLOCK_FILE}" \
        output_prefix="${TABLE_PREFIX}" \
        timing="${TRAVEL_TIMING}" \
        include_propagation_overlap=1 \
        source_stride="${SOURCE_STRIDE}" \
        target_x_stride="${TARGET_X_STRIDE}" \
        target_z_stride="${TARGET_Z_STRIDE}" \
        threads="${THREADS}" \
        max_table_mb="${MAX_TABLE_MB}"

    if ! traveltime_exists; then
        echo "Traveltime calculation failed: no travel*.rsf files found." >&2
        exit 1
    fi
fi

# ============================================================================
# Step 6: memory-bounded ten-shot imaging
# ============================================================================
echo
echo "[6/6] Imaging 10 shots with reusable Butterfly trees and disk-backed receiver tiles"
rm -f "${IMAGE}" "${IMAGE}@" "${ILLUMINATION}" "${ILLUMINATION}@" "${TIMING}" "${TIMING}@"
rm -rf "${FREQUENCY_FIELD_DIR}" "${RECEIVER_STORE_DIR}"
mkdir -p "${FREQUENCY_FIELD_DIR}"

START_SECONDS="${SECONDS}"
OMP_NUM_THREADS="${THREADS}" \
"${BIN_DIR}/f_image_rtm_kirch_multishot" \
    velocity="${IMAGING_MODEL}" block_file="${BLOCK_FILE}" \
    table_prefix="${TABLE_PREFIX}" seismic_data="${MUTED_SHOT_DATA}" cmp=0 \
    shot_begin=0 shot_count=10 shot_stride=1 \
    fdom="${FDOM}" fmin="${FMIN}" fmax="${FMAX}" \
    frequency_stride="${FREQUENCY_STRIDE}" \
    source_stride="${SOURCE_STRIDE}" target_x_stride="${TARGET_X_STRIDE}" \
    target_z_stride="${TARGET_Z_STRIDE}" threads="${THREADS}" \
    bpack_tol="${BPACK_TOL}" bpack_leaf="${BPACK_LEAF}" \
    bpack_depth_chunk="${BPACK_DEPTH_CHUNK}" \
    bpack_lrlevel="${BPACK_LRLEVEL}" \
    bpack_sample_para="${BPACK_SAMPLE_PARA}" \
    bpack_forward_n15="${BPACK_FORWARD_N15}" bpack_knn="${BPACK_KNN}" \
    bpack_pat_comp="${BPACK_PAT_COMP}" bpack_less_adapt="${BPACK_LESS_ADAPT}" \
    bpack_rdetect_factor="${BPACK_RDETECT_FACTOR}" \
    bpack_reuse_tree="${BPACK_REUSE_TREE}" \
    bpack_geometry_cache_mb="${BPACK_GEOMETRY_CACHE_MB}" \
    bpack_verbosity="${BPACK_VERBOSITY}" bpack_progress="${BPACK_PROGRESS}" \
    bpack_progress_every="${BPACK_PROGRESS_EVERY}" \
    gmres_enable=1 global_gmres_iterations="${GMRES_OUTER}" \
    gmres_restart="${GMRES_RESTART}" \
    gmres_frequency_threads="${GMRES_FREQUENCY_THREADS}" \
    global_source_z="${SOURCE_Z}" \
    global_source_correction_z0="${CORRECTION_Z0}" \
    receiver_store=disk receiver_store_dir="${RECEIVER_STORE_DIR}" \
    per_shot_image_dir="${FREQUENCY_FIELD_DIR}" \
    per_shot_image_max_mb=8192 \
    image="${IMAGE}" illumination="${ILLUMINATION}" timing="${TIMING}" \
    correction_csv="${WORK_DIR}/gmres_metrics.csv" \
    seam_csv="${WORK_DIR}/seam_metrics.csv"
IMAGING_SECONDS=$((SECONDS - START_SECONDS))

if [[ ! -s "${IMAGE}" || ! -s "${TIMING}" ]]; then
    echo "Ten-shot imaging did not create image and timing outputs." >&2
    exit 1
fi
for shot in $(seq 0 9); do
    for stage in before_gmres after_gmres; do
        file="${FREQUENCY_FIELD_DIR}/shot_${shot}_${stage}.rsf"
        [[ -s "${file}" ]] || { echo "Missing per-shot image: ${file}" >&2; exit 1; }
    done
done
[[ ! -e "${RECEIVER_STORE_DIR}" ]] || {
    echo "Receiver scratch directory was not cleaned: ${RECEIVER_STORE_DIR}" >&2
    exit 1
}
echo "Ten-shot imaging wall time: ${IMAGING_SECONDS} seconds"

# ============================================================================
# 11. Summary
# ============================================================================
echo
echo "======================================================================"
echo "Ten-shot f_image_rtm_kirch_multishot test completed."
echo "======================================================================"
echo "REUSE:                         ${REUSE}"
echo
echo "Forward nt / dt / fdom:        ${NT} / ${DT} / ${FDOM}"
echo "Source x / z:                  ${SX} / ${SZ}"
echo "Receiver z:                    ${RZ}"
echo "Receivers nr / r0 / dr:        ${NR} / ${R0} / ${DR}"
echo "Blocks first/block/overlap:    ${FIRST_BLOCK_ROWS}/${BLOCK_ROWS}/${OVERLAP_ROWS}"
echo "Smoothing sigma:               ${SMOOTH_SIGMA}"
echo "Frequency fmin/fmax/stride:    ${FMIN}/${FMAX}/${FREQUENCY_STRIDE}"
echo "GMRES outer/restart:           ${GMRES_OUTER}/${GMRES_RESTART}"
echo "GMRES frequency threads:       ${GMRES_FREQUENCY_THREADS}"
echo "Global source z:               ${SOURCE_Z}"
echo "Correction z0:                 ${CORRECTION_Z0}"
echo
echo "Forward record:                ${SHOT_DATA}"
echo "Muted record:                  ${MUTED_SHOT_DATA}"
echo "Smooth imaging model:          ${IMAGING_MODEL}"
echo "Block file:                    ${BLOCK_FILE}"
echo "Traveltime tables:             ${TABLE_DIR}"
echo "Per-shot before images:        ${FREQUENCY_FIELD_DIR}/shot_*_before_gmres.rsf"
echo "Image after correction:        ${IMAGE}"
echo "Per-shot image directory:      ${FREQUENCY_FIELD_DIR}"
echo "Imaging wall seconds:          ${IMAGING_SECONDS}"
echo "All outputs:                   ${WORK_DIR}"
echo "======================================================================"
