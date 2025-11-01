#!/bin/bash

################################################################################
# GPU Test Case 6: Performance Degradation
#
# Issue: GPU simulation 147x slower (146 sec) vs CPU (1 sec)
#
# Root Cause: GPU overhead (data transfer, kernel launch) exceeds benefit
#
# Symptoms:
#   - CPU execution time: ~1 second
#   - GPU execution time: ~146 seconds
#   - GPU is 147 times SLOWER than CPU
#   - Despite same iteration count (2.00 vs 2.01 per step)
#   - Indicates GPU kernels are inefficient or data transfer is blocking
#
# Detection Method:
#   1. Run GPU simulation with time measurement
#   2. Run CPU simulation with time measurement
#   3. Compare wall clock times
#   4. Calculate speedup/slowdown ratio
#
# Test Passes If:
#   - GPU speedup >= 1.0x (at minimum, break even with CPU)
#   - Preferably GPU speedup >= 5.0x (expected on modern GPUs)
#
# Investigation Checklist:
#   [ ] Profile GPU kernel execution time vs data transfer time
#   [ ] Check if unified memory page migration is causing bottleneck
#   [ ] Verify kernel is actually executing on GPU (not CPU fallback)
#   [ ] Measure GPU memory bandwidth usage
#   [ ] Check if synchronization points are too frequent
#   [ ] Examine if small kernel < overhead makes GPU inefficient
#   [ ] Consider if problem is too small for GPU acceleration
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

TEST_NAME="gpu_performance_degradation"
CPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_cpu.rpt"
GPU_RPT="${OUTPUT_DIR}/${TEST_NAME}_gpu.rpt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "GPU Test Case 6: Performance Degradation"
echo "================================================================"
echo ""

# Check prerequisites
if [ ! -f "$RUNSWMM_BIN" ]; then
    echo -e "${RED}ERROR: runswmm binary not found${NC}"
    exit 1
fi

# Run CPU baseline with timing
echo "Running CPU baseline with timing..."
export SWMM_USE_CUDA=0
CPU_START=$(date +%s%N)
timeout 30 "$RUNSWMM_BIN" "$TEST_MODEL" "$CPU_RPT" /tmp/test.out > /dev/null 2>&1 || {
    echo -e "${RED}CPU simulation failed${NC}"
    exit 1
}
CPU_END=$(date +%s%N)
echo -e "${GREEN}✓ CPU simulation completed${NC}"

# Calculate CPU time
CPU_TIME=$(echo "scale=3; ($CPU_END - $CPU_START) / 1000000000" | bc)

# Run GPU test with timing
echo "Running GPU simulation with timing..."
export SWMM_USE_CUDA=1
GPU_START=$(date +%s%N)
timeout 300 "$RUNSWMM_BIN" "$TEST_MODEL" "$GPU_RPT" /tmp/test.out > /dev/null 2>&1 || {
    echo -e "${RED}GPU simulation failed${NC}"
    exit 1
}
GPU_END=$(date +%s%N)
echo -e "${GREEN}✓ GPU simulation completed${NC}"

# Calculate GPU time
GPU_TIME=$(echo "scale=3; ($GPU_END - $GPU_START) / 1000000000" | bc)

# Calculate speedup
SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)

echo ""
echo "Execution Time Measurement:"
echo "  CPU: $CPU_TIME seconds"
echo "  GPU: $GPU_TIME seconds"

echo ""
echo "Performance Ratio:"
echo "  Speedup: ${SPEEDUP}x"

# Extract from reports
CPU_ITER=$(grep "Average Iterations per Step" "$CPU_RPT" | awk '{print $(NF)}')
GPU_ITER=$(grep "Average Iterations per Step" "$GPU_RPT" | awk '{print $(NF)}')

echo ""
echo "Convergence (Average iterations per timestep):"
echo "  CPU: $CPU_ITER"
echo "  GPU: $GPU_ITER"

echo ""
echo "================================================================"

# Test result
if (( $(echo "$SPEEDUP >= 1.0" | bc -l 2>/dev/null || echo "0") )); then
    if (( $(echo "$SPEEDUP >= 5.0" | bc -l 2>/dev/null || echo "0") )); then
        echo -e "${GREEN}✓ TEST PASSED: GPU provides good speedup${NC}"
        echo "  Expected: Speedup >= 5.0x on realistic problem"
        echo "  Got:      Speedup ${SPEEDUP}x"
        exit 0
    else
        echo -e "${YELLOW}⚠ TEST MARGINAL: GPU slower than expected${NC}"
        echo "  Expected: Speedup >= 5.0x"
        echo "  Got:      Speedup ${SPEEDUP}x (suboptimal but not failure)"
        exit 0
    fi
else
    echo -e "${RED}✗ TEST FAILED: GPU significantly slower than CPU${NC}"
    echo "  Expected: Speedup >= 1.0x (break even)"
    echo "  Got:      Speedup ${SPEEDUP}x (GPU is $GPU_TIME sec vs CPU $CPU_TIME sec)"
    echo ""
    echo "GPU is performing worse than CPU despite acceleration effort"
    echo "Likely causes:"
    echo "  1. Unified memory page migration overhead too high"
    echo "  2. GPU kernels not optimized for this problem size"
    echo "  3. Data transfer cost exceeds computation benefit"
    echo "  4. Synchronization points between iterations too frequent"
    exit 1
fi
