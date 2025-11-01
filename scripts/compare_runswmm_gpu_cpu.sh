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

# Parse INP file to extract model info
parse_inp_info() {
    local inp_file=$1
    local total_links=0
    local sim_duration="N/A"

    # Count links from different sections
    for section in "CONDUITS" "PUMPS" "ORIFICES" "WEIRS" "OUTLETS"; do
        local count=$(awk -v section="$section" '
            BEGIN { in_section=0; count=0 }
            /^\['"$section"'\]/ { in_section=1; next }
            /^\[/ { in_section=0 }
            in_section && /^[^;]/ && NF > 0 { count++ }
            END { print count }
        ' "$inp_file")
        total_links=$((total_links + count))
    done

    # Extract simulation duration from OPTIONS section
    sim_duration=$(awk '
        BEGIN { in_options=0; start_date=""; start_time="00:00:00"; end_date=""; end_time="00:00:00" }
        /^\[OPTIONS\]/ { in_options=1; next }
        /^\[/ { in_options=0 }
        in_options && /^START_DATE/ {
            for (i=2; i<=NF; i++) if ($i !~ /^;/) { start_date=$i; break }
        }
        in_options && /^START_TIME/ {
            for (i=2; i<=NF; i++) if ($i !~ /^;/) { start_time=$i; break }
        }
        in_options && /^END_DATE/ {
            for (i=2; i<=NF; i++) if ($i !~ /^;/) { end_date=$i; break }
        }
        in_options && /^END_TIME/ {
            for (i=2; i<=NF; i++) if ($i !~ /^;/) { end_time=$i; break }
        }
        END {
            if (start_date != "" && end_date != "") {
                # Simple duration calculation using date command (Linux)
                cmd = "date -d \"" end_date " " end_time "\" +%s 2>/dev/null"
                if ((cmd | getline end_epoch) > 0) {
                    close(cmd)
                    cmd = "date -d \"" start_date " " start_time "\" +%s 2>/dev/null"
                    if ((cmd | getline start_epoch) > 0) {
                        close(cmd)
                        duration_sec = end_epoch - start_epoch
                        duration_hrs = duration_sec / 3600.0
                        if (duration_hrs < 24) {
                            printf "%.1fh", duration_hrs
                        } else {
                            printf "%.1fd", duration_hrs / 24.0
                        }
                        exit
                    }
                }
                # Fallback: just show the date range
                print start_date " to " end_date
            } else {
                print "N/A"
            }
        }
    ' "$inp_file" 2>/dev/null || echo "N/A")

    echo "$total_links $sim_duration"
}

# Get model info
read -r TOTAL_LINKS SIM_DURATION <<< "$(parse_inp_info "$INPUT_INP")"
echo "Model info     : $TOTAL_LINKS links, $SIM_DURATION simulation"

run_mode() {
    local mode=$1
    local rpt=$2
    local out=$3
    echo
    echo "==> Running ${mode} simulation"
    local start_ns end_ns elapsed_ns elapsed_ms
    start_ns=$(date +%s%N)
    SWMM_USE_CUDA=$([ "$mode" = "GPU" ] && echo 1 || echo 0) \
        "$RUNSWMM_BIN" "$INPUT_INP" "$rpt" "$out"
    local status=$?
    end_ns=$(date +%s%N)
    elapsed_ns=$((end_ns - start_ns))
    elapsed_ms=$(awk -v ns="$elapsed_ns" 'BEGIN { printf "%.2f", ns/1000000.0 }')
    echo "==> ${mode} completed in ${elapsed_ms} ms"
    return $status
}

run_mode GPU "$GPU_RPT" "$GPU_OUT"
run_mode CPU "$CPU_RPT" "$CPU_OUT"

echo
echo "==> Comparing report files"
REPORT_DIFF=${RUN_DIR}/${BASE_STEM}_report.diff
FILTERED_REPORT_DIFF=${RUN_DIR}/${BASE_STEM}_report.filtered.diff
if diff -u "$CPU_RPT" "$GPU_RPT" > "$REPORT_DIFF"; then
    cp "$REPORT_DIFF" "$FILTERED_REPORT_DIFF"
else
    python3 - "$REPORT_DIFF" "$FILTERED_REPORT_DIFF" <<'PY'
import sys
src = sys.argv[1]
dst = sys.argv[2]
drop_substrings = ("gpu_cpu_compare", "@@", "Analysis begun on:", "Analysis ended on:", "Total elapsed time:", "No newline at end of file")
filtered = []
with open(src, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        stripped = line.strip()
        if not stripped:
            continue
        if any(token in stripped for token in drop_substrings):
            continue
        # Skip context lines (lines without +/- prefix) - these are identical in both files
        if line and line[0] not in ('+', '-'):
            continue
        filtered.append(line)
with open(dst, 'w', encoding='utf-8') as f:
    f.writelines(filtered)
PY
fi

# ANSI color codes
RED='\033[1;31m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RESET='\033[0m'

if [ ! -s "$FILTERED_REPORT_DIFF" ]; then
    echo -e "${GREEN}✓${RESET} Reports match (after filtering metadata; diff stored at $FILTERED_REPORT_DIFF)."
    echo "Reports match (after filtering metadata; diff stored at $FILTERED_REPORT_DIFF)."
else
    # Classify difference as minor or major based on number of diff lines
    DIFF_LINE_COUNT=$(wc -l < "$FILTERED_REPORT_DIFF")

    # Heuristic thresholds:
    # - Minor: 1-10 lines of diff (small numeric differences, rounding errors)
    # - Major: >10 lines of diff (significant structural or value differences)
    if [ "$DIFF_LINE_COUNT" -le 10 ]; then
        echo -e "${YELLOW}⚠${RESET} Reports differ - MINOR differences ($DIFF_LINE_COUNT lines; see $FILTERED_REPORT_DIFF)."
        echo "Reports differ - MINOR differences ($DIFF_LINE_COUNT lines; see $FILTERED_REPORT_DIFF)."
    else
        echo -e "${RED}!${RESET} Reports differ - MAJOR differences ($DIFF_LINE_COUNT lines; see $FILTERED_REPORT_DIFF)."
        echo "Reports differ - MAJOR differences ($DIFF_LINE_COUNT lines; see $FILTERED_REPORT_DIFF)."
    fi
fi

echo
echo "==> Comparing binary output files"
if cmp -s "$CPU_OUT" "$GPU_OUT"; then
    echo -e "${GREEN}✓${RESET} Binary outputs match."
    echo "Binary outputs match."
else
    echo -e "${RED}!${RESET} Binary outputs differ."
    echo "Binary outputs differ."
fi

echo
echo "Artifacts written to: $RUN_DIR"
