#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
BIN="$ROOT/build/bin"
MODEL="$ROOT/model/vsalt.rsf"
OUT="$ROOT/image_test/salt_single_shot_point_source_gmres"

THREADS=32
SOURCE_X=3.84
SOURCE_Z=0.01
CORRECTION_Z0=0.1
GMRES_OUTER=40
GMRES_RESTART=30
OVERLAP_ROWS=20
FMIN=${FMIN:-10}
FMAX=${FMAX:-40}
FREQUENCY_STRIDE=${FREQUENCY_STRIDE:-2}

mkdir -p "$OUT/travel" "$OUT/source_wavefields"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
    -DKIRCH_ENABLE_BUTTERFLYPACK=ON \
    -DKIRCH_BPACK_ENABLE_FLOAT=ON \
    -DKIRCH_BPACK_ENABLE_DOUBLE=OFF
cmake --build "$ROOT/build" --parallel "$THREADS" --target \
    sewave2d_forward \
    huygens_block_info \
    travel_time_solver_oneway \
    frequency_kirchhoff_imaging_bf_global_gmres

"$BIN/sewave2d_forward" \
    velocity="$MODEL" output="$OUT/data_full.rsf" \
    nt=4000 dt=0.0007 fdom=20 \
    sx="$SOURCE_X" sz="$SOURCE_Z" rz=0 \
    ns=1 ds=0 nr=768 r0=0 dr=0.01 \
    nbc=40 L=30 alpha=1 \
    type_compute_Laplace=0 flag_smooth=0 flag_homo=0

"$BIN/sewave2d_forward" \
    velocity="$MODEL" output="$OUT/data_homogeneous.rsf" \
    nt=4000 dt=0.0007 fdom=20 \
    sx="$SOURCE_X" sz="$SOURCE_Z" rz=0 \
    ns=1 ds=0 nr=768 r0=0 dr=0.01 \
    nbc=40 L=30 alpha=1 \
    type_compute_Laplace=0 flag_smooth=0 flag_homo=1

python3 "$ROOT/model/rsf_pure_tools.py" subtract \
    "$OUT/data_full.rsf" "$OUT/data_homogeneous.rsf" "$OUT/data_scattered.rsf"

python3 "$ROOT/model/smooth_rsf_gaussian.py" \
    "$MODEL" "$OUT/vmar_smooth.rsf" 2 2

"$BIN/huygens_block_info" \
    velocity="$OUT/vmar_smooth.rsf" output="$OUT/blocks.txt" \
    first_block_rows=25 block_rows=100 overlap_rows="$OVERLAP_ROWS"

# The receiver recursion recomputes these shared rows and fuses adjacent
# Kirchhoff fields, so the one-way tables must include the same region.
"$BIN/travel_time_solver_oneway" \
    velocity="$OUT/vmar_smooth.rsf" \
    block_file="$OUT/blocks.txt" \
    output_prefix="$OUT/travel/travel" \
    timing="$OUT/travel_timing.rsf" \
    include_first_block=1 include_propagation_overlap=1 \
    source_stride=1 target_x_stride=1 target_z_stride=1 \
    threads="$THREADS"

"$BIN/frequency_kirchhoff_imaging_bf_global_gmres" \
    velocity="$OUT/vmar_smooth.rsf" \
    block_file="$OUT/blocks.txt" \
    table_prefix="$OUT/travel/travel" \
    seismic_data="$OUT/data_scattered.rsf" \
    cmp=0 shot_begin=0 shot_count=1 shot_stride=1 \
    aperture_trace=-1 aperture_distance=4 \
    fdom=20 source_time=0.075 source_amplitude=1 \
    nfft=4096 fmin="$FMIN" fmax="$FMAX" frequency_stride="$FREQUENCY_STRIDE" \
    source_stride=1 target_x_stride=1 target_z_stride=1 \
    threads="$THREADS" \
    global_source_z="$SOURCE_Z" \
    global_source_correction_z0="$CORRECTION_Z0" \
    gmres_outer="$GMRES_OUTER" \
    gmres_restart="$GMRES_RESTART" \
    source_wavefield_output_dir="$OUT/source_wavefields" \
    image_before_correction="$OUT/image_before_correction.rsf" \
    illumination_before_correction="$OUT/illumination_before_correction.rsf" \
    image="$OUT/image.rsf" \
    illumination="$OUT/illumination.rsf" \
    timing="$OUT/timing.rsf"

python3 "$ROOT/model/rsf_pure_tools.py" plot \
    "$OUT/image_before_correction.rsf" \
    "$OUT/illumination_before_correction.rsf" \
    "$OUT/image_before_correction.png"

python3 "$ROOT/model/rsf_pure_tools.py" plot \
    "$OUT/image.rsf" "$OUT/illumination.rsf" "$OUT/image.png"

printf 'image before correction: %s\n' "$OUT/image_before_correction.rsf"
printf 'image after correction: %s\n' "$OUT/image.rsf"
printf 'source ray and receiver Kirchhoff, GMRES, and update fields: %s\n' "$OUT/source_wavefields"
