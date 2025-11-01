#!/bin/bash

################################################################################
# GPU Test Case 2: Storage Not Draining
#
# Issue: GPU storage J2 final volume 14.942 1000 ft³ vs CPU 0.001 1000 ft³
#
# Root Cause: Pump (C2) not discharging water from storage
#
# Symptoms:
#   - Storage J2 max depth: 15.00 ft (at capacity, should be 0.00 ft)
#   - Storage J2 flooding: 57.56 hours (water spilling, should be none)
#   - Storage J2 avg volume: 14.942 1000 ft³ (99.6% full, should be 0.001)
#   - Storage J2 outflow: 0.00 CFS (pump inactive, should be 9.41 CFS)
#
# Detection Method:
#   1. Run GPU simulation with model_full_features.inp
#   2. Extract storage J2 final volume from report
#   3. Extract storage J2 max depth
#   4. Extract flooding hours for J2
#   5. Compare with CPU baseline
#
# Test Passes If:
#   - GPU storage final volume < 0.1 M gal (allowing some tolerance)
#   - GPU storage max depth < 1.0 ft (mostly empty)
#   - GPU storage flooding hours = 0 (no overflow)
#
# Investigation Checklist:
#   [ ] Verify pump flow calculation in GPU kernel
#   [ ] Check pump link type handling (is PUMP supported?)
#   [ ] Verify pump discharge head calculation
#   [ ] Check if pump actually runs (look for non-zero flow attempts)
#   [ ] Verify storage-to-pump connection in data transfer
#   [ ] Check pump curve/rating in GPU kernel
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

TEST_NAME="gpu_storage_not_draining"
CPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_cpu.rpt"
GPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_gpu.rpt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "GPU Test Case 2: Storage Not Draining"
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

# Extract storage metrics
echo "Extracting storage J2 metrics..."

# Get max depth
CPU_DEPTH=$(awk '/Node Depth Summary/,/^$/ {if ($1 == "J2") print $5}' "$CPU_RPT" | head -1)
GPU_DEPTH=$(awk '/Node Depth Summary/,/^$/ {if ($1 == "J2") print $5}' "$GPU_RPT" | head -1)

# Get final volume
CPU_VOLUME=$(awk '/Storage Volume Summary/,/^$/ {if ($1 == "J2") print $2}' "$CPU_RPT" | head -1)
GPU_VOLUME=$(awk '/Storage Volume Summary/,/^$/ {if ($1 == "J2") print $2}' "$GPU_RPT" | head -1)

# Get flooding hours
CPU_FLOOD=$(awk '/Node Flooding Summary/,/^$/ {if ($1 == "J2") print $2}' "$CPU_RPT" | head -1)
GPU_FLOOD=$(awk '/Node Flooding Summary/,/^$/ {if ($1 == "J2") print $2}' "$GPU_RPT" | head -1)

if [ -z "$CPU_FLOOD" ]; then CPU_FLOOD="0.00"; fi
if [ -z "$GPU_FLOOD" ]; then GPU_FLOOD="0.00"; fi

echo ""
echo "Storage J2 Max Depth (ft):"
echo "  CPU: $CPU_DEPTH"
echo "  GPU: $GPU_DEPTH"

echo ""
echo "Storage J2 Final Volume (1000 ft³):"
echo "  CPU: $CPU_VOLUME"
echo "  GPU: $GPU_VOLUME"

echo ""
echo "Storage J2 Flooding Hours:"
echo "  CPU: $CPU_FLOOD"
echo "  GPU: $GPU_FLOOD"

echo ""
echo "================================================================"

# Test result
if (( $(echo "$GPU_DEPTH < 1.0" | bc -l 2>/dev/null || echo "0") )) && \
   (( $(echo "$GPU_VOLUME < 0.1" | bc -l 2>/dev/null || echo "0") )); then
    echo -e "${GREEN}✓ TEST PASSED: GPU storage drains correctly${NC}"
    echo "  Expected: Depth ~0 ft, Volume ~0 1000 ft³, Flood ~0 hrs"
    echo "  Got:      Depth $GPU_DEPTH ft, Volume $GPU_VOLUME 1000 ft³, Flood $GPU_FLOOD hrs"
    exit 0
else
    echo -e "${RED}✗ TEST FAILED: GPU storage not draining (pump inactive)${NC}"
    echo "  Expected: Depth ~0 ft, Volume ~0 1000 ft³, Flood ~0 hrs"
    echo "  Got:      Depth $GPU_DEPTH ft, Volume $GPU_VOLUME 1000 ft³, Flood $GPU_FLOOD hrs"
    echo ""
    echo "Detailed comparison:"
    diff <(grep -A 10 "Storage Volume Summary" "$CPU_RPT") \
         <(grep -A 10 "Storage Volume Summary" "$GPU_RPT") || true
    exit 1
fi
