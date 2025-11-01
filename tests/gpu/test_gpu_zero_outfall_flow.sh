#!/bin/bash

################################################################################
# GPU Test Case 1: Zero Outfall Flow
#
# Issue: GPU produces 0.000 M gal outfall discharge vs CPU 14.528 M gal
#
# Root Cause: Water not reaching outfall (complete flow routing failure)
#
# Symptoms:
#   - Outfall J4 frequency: 0% (should be 100%)
#   - Outfall J4 volume: 0.000 M gal (should be 14.528 M gal)
#   - System total volume: 0.000 M gal (should be 14.528 M gal)
#
# Detection Method:
#   1. Run GPU simulation with model_full_features.inp
#   2. Extract outfall frequency from report
#   3. Extract total outfall volume
#   4. Compare with CPU baseline
#
# Test Passes If:
#   - GPU outfall frequency >= 99%
#   - GPU total volume >= 14.0 M gal (allowing 2% tolerance)
#
# Investigation Checklist:
#   [ ] Verify pump (C2) is producing output flow
#   [ ] Verify weir (C3) is receiving input flow
#   [ ] Verify J4 node is receiving upstream flow
#   [ ] Check if flow classification is correct at each link
#   [ ] Verify head calculations at J4 (should be receiving water)
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

TEST_NAME="gpu_zero_outfall_flow"
CPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_cpu.rpt"
GPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_gpu.rpt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "GPU Test Case 1: Zero Outfall Flow"
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

# Extract metrics
echo "Extracting outfall metrics..."

CPU_OUTFALL=$(grep -A 5 "Outfall Loading Summary" "$CPU_RPT" | grep "J4" | awk '{print $2}')
GPU_OUTFALL=$(grep -A 5 "Outfall Loading Summary" "$GPU_RPT" | grep "J4" | awk '{print $2}')

CPU_VOLUME=$(grep -A 5 "Outfall Loading Summary" "$CPU_RPT" | grep "J4" | awk '{print $5}')
GPU_VOLUME=$(grep -A 5 "Outfall Loading Summary" "$GPU_RPT" | grep "J4" | awk '{print $5}')

echo ""
echo "Outfall Frequency (%):"
echo "  CPU: $CPU_OUTFALL"
echo "  GPU: $GPU_OUTFALL"

echo ""
echo "Outfall Total Volume (M gal):"
echo "  CPU: $CPU_VOLUME"
echo "  GPU: $GPU_VOLUME"

echo ""
echo "================================================================"

# Test result
if [ "$GPU_OUTFALL" == "100.00" ] && [ $(echo "$GPU_VOLUME >= 14.0" | bc) -eq 1 ]; then
    echo -e "${GREEN}✓ TEST PASSED: GPU outfall flow matches CPU${NC}"
    echo "  Expected: Frequency ~100%, Volume ~14.5 M gal"
    echo "  Got:      Frequency $GPU_OUTFALL%, Volume $GPU_VOLUME M gal"
    exit 0
else
    echo -e "${RED}✗ TEST FAILED: GPU zero outfall flow issue detected${NC}"
    echo "  Expected: Frequency ~100%, Volume ~14.5 M gal"
    echo "  Got:      Frequency $GPU_OUTFALL%, Volume $GPU_VOLUME M gal"
    echo ""
    echo "Detailed comparison:"
    diff <(grep -A 10 "Outfall Loading Summary" "$CPU_RPT") \
         <(grep -A 10 "Outfall Loading Summary" "$GPU_RPT") || true
    exit 1
fi
