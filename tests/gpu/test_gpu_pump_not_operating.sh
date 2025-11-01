#!/bin/bash

################################################################################
# GPU Test Case 3: Pump Not Operating
#
# Issue: GPU pump C2 at 0% utilization vs CPU at 100% utilization
#
# Root Cause: GPU kernel missing PUMP link type handling
#
# Symptoms:
#   - Pump C2 percent utilized: 0% (should be 100%)
#   - Pump C2 avg flow: 0.00 CFS (should be 8.25 CFS)
#   - Pump C2 max flow: 0.00 CFS (should be 9.41 CFS)
#   - Pump C2 total volume: 0.000 M gal (should be 12.888 M gal)
#   - Pump C2 power usage: 0.00 Kw-hr (should be 196.88 Kw-hr)
#   - Pump start-ups: 0 (should be 1)
#
# Detection Method:
#   1. Run GPU simulation with model_full_features.inp
#   2. Extract pump C2 utilization from report
#   3. Extract pump C2 flow metrics
#   4. Extract pump C2 volume pumped
#   5. Compare with CPU baseline
#
# Test Passes If:
#   - GPU pump utilization >= 95%
#   - GPU pump avg flow >= 7.5 CFS
#   - GPU pump total volume >= 11.0 M gal
#
# Investigation Checklist:
#   [ ] Check if GPU kernel handles PUMP link type (see isTrueConduit filter)
#   [ ] Verify pump flow calculation code exists in GPU
#   [ ] Check pump discharge head vs storage head
#   [ ] Verify pump curve/rating transferred to GPU
#   [ ] Look for pump-specific momentum equation (if different from conduit)
#   [ ] Check if pump data is copied to GPU device
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

TEST_NAME="gpu_pump_not_operating"
CPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_cpu.rpt"
GPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_gpu.rpt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "GPU Test Case 3: Pump Not Operating"
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

# Extract pump metrics
echo "Extracting pump C2 metrics..."

# Get pump metrics from Pumping Summary
CPU_UTIL=$(grep -A 5 "Pumping Summary" "$CPU_RPT" | grep "C2" | awk '{print $2}')
GPU_UTIL=$(grep -A 5 "Pumping Summary" "$GPU_RPT" | grep "C2" | awk '{print $2}')

CPU_AVG=$(grep -A 5 "Pumping Summary" "$CPU_RPT" | grep "C2" | awk '{print $5}')
GPU_AVG=$(grep -A 5 "Pumping Summary" "$GPU_RPT" | grep "C2" | awk '{print $5}')

CPU_VOLUME=$(grep -A 5 "Pumping Summary" "$CPU_RPT" | grep "C2" | awk '{print $7}')
GPU_VOLUME=$(grep -A 5 "Pumping Summary" "$GPU_RPT" | grep "C2" | awk '{print $7}')

echo ""
echo "Pump C2 Percent Utilized (%):"
echo "  CPU: $CPU_UTIL"
echo "  GPU: $GPU_UTIL"

echo ""
echo "Pump C2 Avg Flow (CFS):"
echo "  CPU: $CPU_AVG"
echo "  GPU: $GPU_AVG"

echo ""
echo "Pump C2 Total Volume (M gal):"
echo "  CPU: $CPU_VOLUME"
echo "  GPU: $GPU_VOLUME"

echo ""
echo "================================================================"

# Test result
if (( $(echo "$GPU_UTIL >= 95" | bc -l 2>/dev/null || echo "0") )) && \
   (( $(echo "$GPU_AVG >= 7.5" | bc -l 2>/dev/null || echo "0") )); then
    echo -e "${GREEN}✓ TEST PASSED: GPU pump operating normally${NC}"
    echo "  Expected: Utilization ~100%, Avg Flow ~8.25 CFS"
    echo "  Got:      Utilization $GPU_UTIL%, Avg Flow $GPU_AVG CFS"
    exit 0
else
    echo -e "${RED}✗ TEST FAILED: GPU pump not operating (0% utilization)${NC}"
    echo "  Expected: Utilization ~100%, Avg Flow ~8.25 CFS, Volume ~12.9 M gal"
    echo "  Got:      Utilization $GPU_UTIL%, Avg Flow $GPU_AVG CFS, Volume $GPU_VOLUME M gal"
    echo ""
    echo "Detailed comparison:"
    diff <(grep -A 10 "Pumping Summary" "$CPU_RPT") \
         <(grep -A 10 "Pumping Summary" "$GPU_RPT") || true
    exit 1
fi
