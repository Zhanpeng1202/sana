#!/usr/bin/env bash
# Sana-WM yaw turn-and-return (75 & 180 deg) over ALL WBench navigation samples.
# Loads Sana-WM once and loops every nav case's image + prompt. No scoring.
# Single-GPU (the Sana-WM pipeline is not sharded); set CUDA_VISIBLE_DEVICES to pick one.
#
# Usage:
#   bash run_turn_wbench_sana.sh
# Smoke test (3 cases):
#   LIMIT=3 bash run_turn_wbench_sana.sh
# Specific cases / angles / no refiner (faster) / Pi3X intrinsics:
#   CASE_IDS=1,2,3 ANGLES=75,180 NO_REFINER=1 bash run_turn_wbench_sana.sh
set -euo pipefail

WBENCH_ROOT="${WBENCH_ROOT:-/home/builder/workspace/WBench}"
SELECTION="${SELECTION:-pure_nav}"
ANGLES="${ANGLES:-75,180}"
FRAMES="${FRAMES:-201}"               # snapped to 8k+1 by the script
DIRECTION="${DIRECTION:-right}"
MODE="${MODE:-return}"                # return = turn out then back to origin (3D-consistency); turn = one-way
FOV_DEG="${FOV_DEG:-60}"
SEED="${SEED:-42}"
STEP="${STEP:-60}"
CFG="${CFG:-5.0}"
GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${WBENCH_ROOT}/work_dirs/sana_wm_turn/videos}"
LIMIT="${LIMIT:-}"
CASE_IDS="${CASE_IDS:-}"
NO_REFINER="${NO_REFINER:-}"          # set to 1 to decode with Sana VAE (faster, lower quality)
USE_PI3X="${USE_PI3X:-}"              # set to 1 to estimate per-image intrinsics with Pi3X

cd "$(dirname "$0")"  # Sana repo root

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

ARGS=(
    --wbench_root "${WBENCH_ROOT}"
    --selection "${SELECTION}"
    --output_dir "${OUTPUT_DIR}"
    --angles "${ANGLES}"
    --frames "${FRAMES}"
    --direction "${DIRECTION}"
    --mode "${MODE}"
    --fov_deg "${FOV_DEG}"
    --seed "${SEED}"
    --step "${STEP}"
    --cfg_scale "${CFG}"
    --resume
)
[ -n "${LIMIT}" ] && ARGS+=(--limit "${LIMIT}")
[ -n "${CASE_IDS}" ] && ARGS+=(--case_ids "${CASE_IDS}")
[ -n "${NO_REFINER}" ] && ARGS+=(--no_refiner)
[ -n "${USE_PI3X}" ] && ARGS+=(--use_pi3x)

echo "=== Sana-WM yaw turn over ALL WBench ${SELECTION} samples (angles=${ANGLES}, ${FRAMES}f, mode=${MODE}) ==="
python inference_video_scripts/wm/wbench_turn_sana.py "${ARGS[@]}"

echo "Done. Per-case videos -> ${OUTPUT_DIR}/case_<id>/"
