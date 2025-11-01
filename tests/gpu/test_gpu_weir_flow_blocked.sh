#!/bin/bash

################################################################################
# GPU Test Case 4: Weir Flow Blocked
#
# Issue: GPU weir C3 at 0.00 CFS vs CPU at 12.12 CFS
#
# Root Cause: GPU kernel missing WEIR link type handling
#
# Symptoms:
#   - Weir C3 max flow: 0.00 CFS (should be 12.12 CFS)
#   - Weir C3 max flow occurrence: 0 00:00 (should be 0 10:00)
#   - Weir C3 is never activated
#   - No overflow happens through weir (all water goes to flooding instead)
#
# Detection Method:
#   1. Run GPU simulation with model_full_features.inp
#   2. Extract weir C3 max flow from Link Flow Summary
#   3. Extract weir C3 occurrence time
#   4. Compare with CPU baseline
#
# Test Passes If:
#   - GPU weir max flow >= 11.0 CFS (allowing 2% tolerance)
#   - GPU weir carries positive flow at some point
#
# Investigation Checklist:
#   [ ] Check if GPU kernel handles WEIR link type (see isTrueConduit filter)
#   [ ] Verify weir flow calculation code exists in GPU
#   [ ] Check if weir crest elevation vs node head calculation correct
#   [ ] Verify weir geometry transferred to GPU
#   [ ] Check if weir is being skipped in GPU kernel
#   [ ] Compare GPU link type filtering with CPU dwflow_findWeirFlow()
#
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
RUNSWMM_BIN="${BUILD_DIR}/bin/runswmm"
TEST_MODEL="${REPO_ROOT}/tests/test_models/model_full_features.inp"

# Output directory
OUTPUT_DIR="${1:-.}"
mkdir -p "$OUTPUT_DIR"

TEST_NAME="gpu_weir_flow_blocked"
CPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_cpu.rpt"
GPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_gpu.rpt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "GPU Test Case 4: Weir Flow Blocked"
echo "================================================================"
echo ""

# Check prerequisites
if [ ! -f "$RUNSWMM_BIN" ]; then
    echo -e "${RED}ERROR: runswmm binary not found${NC}"
    exit 1
fi

# Run CPU baseline
echo "Running CPU baseline..."
export SWMM_USE_CUDA=0
timeout 30 "$RUNSWMM_BIN" "$TEST_MODEL" "$CPU_RPT" /tmp/test.out > /dev/null 2>&1 || {
    echo -e "${RED}CPU simulation failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ CPU simulation completed${NC}"

# Run GPU test
echo "Running GPU simulation..."
export SWMM_USE_CUDA=1
timeout 30 "$RUNSWMM_BIN" "$TEST_MODEL" "$GPU_RPT" /tmp/test.out > /dev/null 2>&1 || {
    echo -e "${RED}GPU simulation failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ GPU simulation completed${NC}"
echo ""

# Extract weir metrics
echo "Extracting weir C3 metrics..."

# Get weir max flow from Link Flow Summary
CPU_FLOW=$(awk '/Link Flow Summary/,/^$/ {if ($1 == "C3") print $4}' "$CPU_RPT" | head -1)
GPU_FLOW=$(awk '/Link Flow Summary/,/^$/ {if ($1 == "C3") print $4}' "$GPU_RPT" | head -1)

# Get occurrence time
CPU_TIME=$(awk '/Link Flow Summary/,/^$/ {if ($1 == "C3") print $6 " " $7}' "$CPU_RPT" | head -1)
GPU_TIME=$(awk '/Link Flow Summary/,/^$/ {if ($1 == "C3") print $6 " " $7}' "$GPU_RPT" | head -1)

echo ""
echo "Weir C3 Max Flow (CFS):"
echo "  CPU: $CPU_FLOW"
echo "  GPU: $GPU_FLOW"

echo ""
echo "Weir C3 Time of Max Occurrence:"
echo "  CPU: $CPU_TIME"
echo "  GPU: $GPU_TIME"

echo ""
echo "================================================================"

# Test result
if (( $(echo "${GPU_FLOW:-0} >= 11.0" | bc -l 2>/dev/null || echo "0") )); then
    echo -e "${GREEN}✓ TEST PASSED: GPU weir carrying flow correctly${NC}"
    echo "  Expected: Flow ~12.12 CFS at time 0 10:00"
    echo "  Got:      Flow $GPU_FLOW CFS at time $GPU_TIME"
    exit 0
else
    echo -e "${RED}✗ TEST FAILED: GPU weir flow blocked (0 CFS)${NC}"
    echo "  Expected: Flow ~12.12 CFS at time 0 10:00"
    echo "  Got:      Flow $GPU_FLOW CFS at time $GPU_TIME"
    echo ""
    echo "Detailed comparison:"
    diff <(grep -A 10 "Link Flow Summary" "$CPU_RPT") \
         <(grep -A 10 "Link Flow Summary" "$GPU_RPT") || true
    exit 1
fi
