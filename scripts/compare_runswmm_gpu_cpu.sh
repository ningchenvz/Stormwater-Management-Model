#!/usr/bin/env bash
# Compare GPU vs CPU runs of runswmm for a single INP file.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/compare_runswmm_gpu_cpu.sh <path/to/model.inp> [report_name] [output_name]

Environment variables:
  RUNSWMM    Path to runswmm executable (default: build/bin/runswmm)
  OUT_DIR    Directory to store run artifacts (default: build/gpu_cpu_compare)
EOF
    exit 1
}

die() {
    echo "Error: $*" >&2
    exit 1
}

[ $# -ge 1 ] || usage

INPUT_INP=$1
[[ -f "$INPUT_INP" ]] || die "input file not found: $INPUT_INP"
INPUT_INP=$(realpath "$INPUT_INP")

RUNSWMM_BIN=${RUNSWMM:-build/bin/runswmm}
[[ -x "$RUNSWMM_BIN" ]] || die "runswmm executable not found/executable at: $RUNSWMM_BIN"
RUNSWMM_BIN=$(realpath "$RUNSWMM_BIN")

OUT_DIR=${OUT_DIR:-build/gpu_cpu_compare}
mkdir -p "$OUT_DIR"

BASE_NAME=$(basename "${INPUT_INP}")
BASE_STEM=${BASE_NAME%.*}
RUN_STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR=${OUT_DIR}/${BASE_STEM}_${RUN_STAMP}
mkdir -p "$RUN_DIR"

GPU_RPT=${RUN_DIR}/${BASE_STEM}_gpu.rpt
GPU_OUT=${RUN_DIR}/${BASE_STEM}_gpu.out
CPU_RPT=${RUN_DIR}/${BASE_STEM}_cpu.rpt
CPU_OUT=${RUN_DIR}/${BASE_STEM}_cpu.out

echo "Runswmm binary : $RUNSWMM_BIN"
echo "Input model    : $INPUT_INP"
echo "Output folder  : $RUN_DIR"

run_mode() {
    local mode=$1
    local rpt=$2
    local out=$3
    echo
    echo "==> Running ${mode} simulation"
    SWMM_USE_CUDA=$([ "$mode" = "GPU" ] && echo 1 || echo 0) \
        "$RUNSWMM_BIN" "$INPUT_INP" "$rpt" "$out"
}

run_mode GPU "$GPU_RPT" "$GPU_OUT"
run_mode CPU "$CPU_RPT" "$CPU_OUT"

echo
echo "==> Comparing report files"
REPORT_DIFF=${RUN_DIR}/${BASE_STEM}_report.diff
if diff -u "$CPU_RPT" "$GPU_RPT" > "$REPORT_DIFF"; then
    echo "Reports match (diff stored at $REPORT_DIFF)."
else
    echo "Reports differ (see $REPORT_DIFF)."
fi

echo
echo "==> Comparing binary output files"
if cmp -s "$CPU_OUT" "$GPU_OUT"; then
    echo "Binary outputs match."
else
    echo "Binary outputs differ."
fi

echo
echo "Artifacts written to: $RUN_DIR"
