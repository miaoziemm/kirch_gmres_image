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
WORK_DIR="${ROOT_DIR}/single_shot_f_image_rtm_kirch_test"

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
DIRECT_EXTRA_TIME=0.08
DIRECT_TAPER_TIME=0.04

# ============================================================================
# 5. Frequency-domain imaging / GMRES parameters
# ============================================================================
FMIN="${FMIN:-10}"
FMAX="${FMAX:-40}"
FREQUENCY_STRIDE="${FREQUENCY_STRIDE:-1}"

CORRECTION_Z0=0.105
GMRES_OUTER=30
GMRES_RESTART=30

# ============================================================================
# 6. Derived output names
# ============================================================================
SHOT_DATA="${WORK_DIR}/single_shot.rsf"
MUTED_SHOT_DATA="${WORK_DIR}/single_shot_muted.rsf"

IMAGING_MODEL="${WORK_DIR}/vmar_smooth.rsf"
BLOCK_FILE="${WORK_DIR}/huygens_blocks.txt"

TABLE_DIR="${WORK_DIR}/traveltime"
TABLE_PREFIX="${TABLE_DIR}/travel"
TRAVEL_TIMING="${WORK_DIR}/traveltime_timing.rsf"

IMAGE="${WORK_DIR}/single_shot_image.rsf"
ILLUMINATION="${WORK_DIR}/single_shot_illumination.rsf"
TIMING="${WORK_DIR}/single_shot_timing.rsf"

IMAGE_NO_GMRES="${WORK_DIR}/single_shot_image_no_gmres.rsf"
ILLUMINATION_NO_GMRES="${WORK_DIR}/single_shot_illumination_no_gmres.rsf"

FREQUENCY_FIELD_DIR="${WORK_DIR}/frequency_fields"

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
    f_image_rtm_kirch
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
echo "[1/6] Forward modeling one shot with the original velocity model"

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
# Step 6: f_image_rtm_kirch imaging
# ============================================================================
echo
echo "[6/6] Imaging with f_image_rtm_kirch using reference-aligned parameters"
echo "      -> imaging products are recomputed every run"

rm -f "${IMAGE}" "${IMAGE}@"
rm -f "${ILLUMINATION}" "${ILLUMINATION}@"
rm -f "${TIMING}" "${TIMING}@"
rm -f "${IMAGE_NO_GMRES}" "${IMAGE_NO_GMRES}@"
rm -f "${ILLUMINATION_NO_GMRES}" "${ILLUMINATION_NO_GMRES}@"

rm -rf "${FREQUENCY_FIELD_DIR}"
mkdir -p "${FREQUENCY_FIELD_DIR}"

OMP_NUM_THREADS="${THREADS}" \
"${BIN_DIR}/f_image_rtm_kirch" \
    velocity="${IMAGING_MODEL}" \
    block_file="${BLOCK_FILE}" \
    table_prefix="${TABLE_PREFIX}" \
    seismic_data="${MUTED_SHOT_DATA}" \
    cmp=0 \
    shot_begin=0 \
    shot_count=1 \
    shot_stride=1 \
    fdom="${FDOM}" \
    fmin="${FMIN}" \
    fmax="${FMAX}" \
    frequency_stride="${FREQUENCY_STRIDE}" \
    source_stride="${SOURCE_STRIDE}" \
    target_x_stride="${TARGET_X_STRIDE}" \
    target_z_stride="${TARGET_Z_STRIDE}" \
    threads="${THREADS}" \
    global_source_z="${SZ}" \
    global_source_correction_z0="${CORRECTION_Z0}" \
    gmres_outer="${GMRES_OUTER}" \
    gmres_restart="${GMRES_RESTART}" \
    source_wavefield_output_dir="${FREQUENCY_FIELD_DIR}" \
    image_before_correction="${IMAGE_NO_GMRES}" \
    illumination_before_correction="${ILLUMINATION_NO_GMRES}" \
    image="${IMAGE}" \
    illumination="${ILLUMINATION}" \
    timing="${TIMING}"

# ============================================================================
# 10. Validate outputs
# ============================================================================
if [[ ! -s "${IMAGE}" || ! -s "${IMAGE_NO_GMRES}" ]]; then
    echo "Imaging did not create both before/after correction images." >&2
    exit 1
fi

FIELD_COUNT=0

for field in \
    source_ray \
    source_gmres \
    receiver_kirchhoff \
    receiver_gmres
do
    count=$(
        find "${FREQUENCY_FIELD_DIR}" \
            -maxdepth 1 \
            -type f \
            -name "${field}_*.rsf" | wc -l
    )

    if [[ "${count}" -eq 0 ]]; then
        echo "No ${field} frequency diagnostics were written." >&2
        exit 1
    fi

    if [[ "${FIELD_COUNT}" -ne 0 && "${count}" -ne "${FIELD_COUNT}" ]]; then
        echo "Incomplete per-frequency diagnostics for ${field}." >&2
        echo "Expected ${FIELD_COUNT}, found ${count}." >&2
        exit 1
    fi

    FIELD_COUNT="${count}"
done

# ============================================================================
# 11. Summary
# ============================================================================
echo
echo "======================================================================"
echo "Reference-parameter-aligned f_image_rtm_kirch test completed."
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
echo "NFFT:                          ${NFFT}"
echo "GMRES outer/restart:           ${GMRES_OUTER}/${GMRES_RESTART}"
echo "Correction z0:                 ${CORRECTION_Z0}"
echo
echo "Forward record:                ${SHOT_DATA}"
echo "Muted record:                  ${MUTED_SHOT_DATA}"
echo "Smooth imaging model:          ${IMAGING_MODEL}"
echo "Block file:                    ${BLOCK_FILE}"
echo "Traveltime tables:             ${TABLE_DIR}"
echo "Image before correction:       ${IMAGE_NO_GMRES}"
echo "Image after correction:        ${IMAGE}"
echo "Per-frequency fields:          ${FREQUENCY_FIELD_DIR}"
echo "Frequency field count:         ${FIELD_COUNT}"
echo "All outputs:                   ${WORK_DIR}"
echo "======================================================================"