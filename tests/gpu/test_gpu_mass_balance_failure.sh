#!/bin/bash

################################################################################
# GPU Test Case 5: Mass Balance Failure
#
# Issue: GPU continuity error 9.490% vs CPU 0.001%
#
# Root Cause: Water loss due to flow routing failures (9.5% unaccounted)
#
# Symptoms:
#   - GPU continuity error: 9.490% (should be <0.1%)
#   - Water balance shows large unaccounted volume
#   - Indicates ~9.5% of inflow is lost or trapped
#   - Related to: storage not draining + outfall zero flow
#
# Detection Method:
#   1. Run GPU simulation with model_full_features.inp
#   2. Extract continuity error percentage
#   3. Examine "Highest Continuity Errors" section
#   4. Compare with CPU baseline
#
# Test Passes If:
#   - GPU continuity error < 0.1% (excellent physics)
#   - GPU continuity error < 0.5% (acceptable engineering tolerance)
#
# Investigation Checklist:
#   [ ] Verify all flow paths conserve mass at each node
#   [ ] Check node depth equation handling in GPU
#   [ ] Verify flow continuity equation implementation
#   [ ] Look for water being "lost" in pump/weir calculations
#   [ ] Check if all flow is being recorded/transferred correctly
#   [ ] Examine if some nodes have 100% error (indicates local failure)
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

TEST_NAME="gpu_mass_balance_failure"
CPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_cpu.rpt"
GPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_gpu.rpt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "GPU Test Case 5: Mass Balance Failure"
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

# Extract continuity error
echo "Extracting flow routing continuity metrics..."

CPU_ERROR=$(grep "Continuity Error" "$CPU_RPT" | tail -1 | grep -o '[0-9.]*' | head -1)
GPU_ERROR=$(grep "Continuity Error" "$GPU_RPT" | tail -1 | grep -o '[0-9.]*' | head -1)

# Check for highest continuity errors section
GPU_HIGH_ERROR=$(grep -A 5 "Highest Continuity Errors" "$GPU_RPT" | tail -1 2>/dev/null || echo "")

echo ""
echo "Flow Routing Continuity Error (%):"
echo "  CPU: $CPU_ERROR"
echo "  GPU: $GPU_ERROR"

if [ -n "$GPU_HIGH_ERROR" ]; then
    echo ""
    echo "Highest GPU Continuity Error:"
    echo "  $GPU_HIGH_ERROR"
fi

echo ""
echo "External Outflow (M gal):"
CPU_OUTFLOW=$(grep "External Outflow" "$CPU_RPT" | head -1 | awk '{print $(NF-2)}')
GPU_OUTFLOW=$(grep "External Outflow" "$GPU_RPT" | head -1 | awk '{print $(NF-2)}')
echo "  CPU: $CPU_OUTFLOW"
echo "  GPU: $GPU_OUTFLOW"

echo ""
echo "Flooding Loss (M gal):"
CPU_FLOOD=$(grep "Flooding Loss" "$CPU_RPT" | head -1 | awk '{print $(NF-2)}')
GPU_FLOOD=$(grep "Flooding Loss" "$GPU_RPT" | head -1 | awk '{print $(NF-2)}')
echo "  CPU: $CPU_FLOOD"
echo "  GPU: $GPU_FLOOD"

echo ""
echo "================================================================"

# Test result
if (( $(echo "$GPU_ERROR < 0.5" | bc -l 2>/dev/null || echo "0") )); then
    if (( $(echo "$GPU_ERROR < 0.1" | bc -l 2>/dev/null || echo "0") )); then
        echo -e "${GREEN}✓ TEST PASSED: GPU mass balance excellent${NC}"
        echo "  Expected: Continuity error <0.1%"
        echo "  Got:      Continuity error $GPU_ERROR%"
    else
        echo -e "${YELLOW}⚠ TEST PASSED: GPU mass balance acceptable${NC}"
        echo "  Expected: Continuity error <0.1% (excellent)"
        echo "  Got:      Continuity error $GPU_ERROR% (acceptable)"
    fi
    exit 0
else
    echo -e "${RED}✗ TEST FAILED: GPU mass balance failure${NC}"
    echo "  Expected: Continuity error <0.5%"
    echo "  Got:      Continuity error $GPU_ERROR%"
    echo ""
    echo "Mass balance indicates ~$GPU_ERROR% of water is unaccounted for"
    echo ""
    echo "Detailed comparison:"
    diff <(grep -B 2 -A 10 "Flow Routing Continuity" "$CPU_RPT") \
         <(grep -B 2 -A 10 "Flow Routing Continuity" "$GPU_RPT") || true
    exit 1
fi
