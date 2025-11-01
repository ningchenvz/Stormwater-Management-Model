# GPU Pump Flow Implementation - Technical Documentation

## Overview

This document describes the GPU implementation of pump flow computation for SWMM's dynamic wave routing, including the three-phase processing approach used to handle pump flow modification while avoiding race conditions.

## Background

### CPU Implementation (dynwave.c)

The CPU processes pumps **sequentially** in a single-threaded loop:

```c
for (i = 0; i < Nobjects[LINK]; i++) {
    if (!isTrueConduit(i)) {
        if (!Link[i].bypassed) findNonConduitFlow(i, dt);  // Computes pump flow
        updateNodeFlows(i);  // Immediately updates node inflow/outflow
    }
}
```

Each pump:
1. Computes preliminary flow from pump curve
2. Calls `getModPumpFlow()` to prevent over-draining inlet node
3. Immediately updates node inflow/outflow

**Sequential processing ensures**: Each pump sees the effects of all previous pumps when checking if the inlet node can supply the requested flow.

### GPU Challenge: Parallel Processing Race Conditions

GPU kernels process ALL pumps in parallel, creating two critical race conditions:

**Race Condition #1**: Multiple pumps calling `getModPumpFlow()` simultaneously
- Each pump reads `d_nodeInflow` and `d_nodeOutflow` to check if inlet node can supply flow
- Other pumps are simultaneously writing to these values via `atomicAdd`
- Result: Each pump sees an inconsistent/random subset of other pumps' contributions

**Race Condition #2**: Multiple pumps draining same storage node
- `getMaxOutflow()` limits pump flow based on available node volume
- Formula: `qMax = inflow + oldVolume / tStep`
- If 3 pumps draw from same node in parallel, all see same `oldVolume` and each pumps up to `qMax`
- Result: Total pumped = 3 × qMax (3x over-draining!)

## Solution: Three-Phase Sequential Processing

### Phase 1: Compute Preliminary Flows (Parallel)
```cuda
__global__ void kernel_computePumpFlows(...)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Pump index (parallel)

    // Each pump computes its preliminary flow from pump curve
    double qIn = getPumpFlow(...);

    // Store in temporary buffer (no node updates yet)
    d_prelimPumpFlows[k] = qIn;
}
```

**Key**: No node updates, just store preliminary flows.

### Phase 1b: Accumulate Preliminary Inflows (Parallel)
```cuda
__global__ void kernel_accumulatePreliminaryPumpOutflows(...)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Pump index (parallel)

    double qPrelim = d_prelimFlows[k];
    int n2 = links->d_node2[j];  // Downstream node

    // Add preliminary flow to DOWNSTREAM node inflow ONLY
    // DO NOT update upstream node outflow yet!
    atomicAdd(&nodes->d_inflow[n2], qPrelim);
}
```

**Key**: Only update downstream inflow. This ensures `d_nodeOutflow` contains ONLY conduit flows (matching CPU state when pumps start processing).

### Phase 2: Modify & Apply Final Flows (SEQUENTIAL)
```cuda
__global__ void kernel_modifyPumpFlows_Sequential(...)
{
    // SINGLE THREAD processes all pumps sequentially
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int k = 0; k < pumps->count; k++) {
        double qPrelim = d_prelimFlows[k];

        // Apply getModPumpFlow() to prevent over-draining
        double qFinal = gpu_getModPumpFlow(..., qPrelim, ...);

        // Update link state
        links->d_newFlow[j] = qFinal;

        // Update node flows (NO atomicAdd needed - we're sequential!)
        nodes->d_outflow[n1] += qFinal;  // Add final flow to upstream outflow

        double flowReduction = qPrelim - qFinal;
        if (flowReduction > 0.0) {
            nodes->d_inflow[n2] -= flowReduction;  // Adjust downstream inflow
        }
    }
}
```

**Key Changes for Sequential Processing**:
1. Single thread processes all pumps in a `for` loop
2. Use `continue` (not `return`) to skip bypassed pumps
3. Direct `+=` instead of `atomicAdd` (no race conditions)
4. Each pump sees previous pumps' effects on node flows

### Kernel Launch Sequence
```cuda
// Phase 1: Parallel computation of preliminary flows
kernel_computePumpFlows<<<gridSize, blockSize>>>(...)

CUDA_CHECK(cudaStreamSynchronize(stream));

// Phase 1b: Parallel accumulation to downstream inflows
kernel_accumulatePreliminaryPumpOutflows<<<gridSize, blockSize>>>(...)

CUDA_CHECK(cudaStreamSynchronize(stream));

// Phase 2: Sequential modification with single thread
kernel_modifyPumpFlows_Sequential<<<1, 1>>>(...)
```

## Implementation Details

### File Locations
- **Main GPU code**: `src/solver/gpu/gpu_dwflow.cu`
- **Helper functions**: `src/solver/gpu/gpu_nonconduit_helpers.cuh`
- **CPU fallback**: `src/solver/dynwave.c`

### Key Functions

#### `gpu_getModPumpFlow()` (gpu_nonconduit_helpers.cuh:518)
```cuda
__device__ double gpu_getModPumpFlow(
    int pumpIdx,
    int j,                  // inlet node index
    double q,               // pump flow from curve
    double qPrelim,         // preliminary flow (for accounting)
    double dt,
    ...)
```

Prevents pump from draining inlet node faster than water is available:
- For **storage nodes**: Calls `gpu_node_getMaxOutflow()`
- For **junction nodes** with TYPE2/3/4 pumps: Checks if resulting depth would go negative

#### `gpu_node_getMaxOutflow()` (gpu_nonconduit_helpers.cuh:487)
```cuda
__device__ double gpu_node_getMaxOutflow(
    int j,                  // node index
    double q,               // requested outflow
    double qPrelim,         // (unused in current version)
    double tStep,
    ...)
{
    if (d_nodeFullVolume[j] > 0.0) {
        qMax = d_nodeInflow[j] + d_nodeOldVolume[j] / tStep;
        if (q > qMax) q = qMax;
    }
    return fmax(0.0, q);
}
```

Limits outflow based on available storage volume.

## Performance Analysis

### Test Case: 46 Pumps, 15-Minute Simulation

| Implementation | Runtime | Continuity Error | Notes |
|----------------|---------|------------------|-------|
| CPU Baseline | 0.00 sec | **-29.785%** | Sequential, single-threaded |
| GPU Parallel (original) | 6.00 sec | -72.000% | Race conditions |
| GPU Three-Phase Parallel | 6.00 sec | -45.369% | Phase 2 parallel (still has races) |
| GPU Three-Phase Sequential | 7.00 sec | **-45.369%** | Phase 2 sequential |

### Observations

1. **GPU runtime is longer** (7 sec vs 0 sec CPU) due to:
   - GPU kernel launch overhead
   - Data transfer overhead
   - Sequential Phase 2 processing (46 pumps × iterations)

2. **GPU error is still 1.5x worse** (-45% vs -30%):
   - Sequential processing did NOT improve error
   - Suggests issue is NOT race conditions in Phase 2
   - Likely related to how `oldVolume` is used in `getMaxOutflow`

3. **Both CPU and GPU have large errors** (-30% and -45%):
   - This test case may have inherent mass balance issues
   - Need to validate with different test cases

## Current Issues & Future Work

### Issue #1: Sequential Processing Doesn't Match CPU Error

**Expected**: Sequential Phase 2 should exactly match CPU (-29.785%)
**Actual**: Still -45.369% (same as parallel)

**Hypothesis**: The problem may not be in Phase 2 pump processing, but in:
- Phase 1 preliminary flow computation
- Conduit flow updates before pumps run
- Node depth/volume calculations
- Test case itself may have issues

**Next Steps**:
1. Compare GPU vs CPU pump flows directly (add detailed logging)
2. Verify conduit flows match CPU before pump processing
3. Test with simpler pump-only model to isolate issue
4. Check if issue is specific to storage node pumps

### Issue #2: Performance Overhead

Sequential Phase 2 adds ~1 second overhead for 46 pumps. For larger models (100+ pumps), this could become significant.

**Potential optimizations**:
1. **Group pumps by inlet node**: Process groups sequentially, pumps within group in parallel
2. **Hybrid approach**: Use parallel for pumps with different inlet nodes, sequential only for conflicts
3. **Optimize Phase 1**: Merge Phase 1 and 1b into single kernel

### Issue #3: Code Complexity

Three-phase approach with temporary buffers adds complexity:
- `d_prelimPumpFlows`, `d_prelimPumpDqdh`, `d_prelimPumpFlowClass` buffers
- More kernel launches and synchronization points
- Harder to debug and maintain

## Testing

### Validation Test Cases

**Simple Conduit Test** (`/tmp/simple_3phase.inp`):
- 3 nodes, 2 conduits, NO pumps
- GPU: -0.034%, CPU: -0.033% ✓ **PERFECT MATCH**
- Confirms conduit processing is correct

**Pump Test** (`/tmp/Session68_46_pumps_15min.inp`):
- 853 nodes, 875 links, 46 pumps
- GPU: -45.369%, CPU: -29.785% ✗ **1.5x ERROR GAP**
- All pumps draw from storage nodes (TYPE3_PUMP)

### Debug Logging

To enable pump flow debugging, see:
- `gpu_nonconduit_helpers.cuh:561-566` - getModPumpFlow entry logging
- `gpu_nonconduit_helpers.cuh:509-510` - getMaxOutflow limiting
- `gpu_dwflow.cu:859-863` - Flow reduction logging

## References

- **CPU Implementation**: `src/solver/dynwave.c:547-555` (pump loop)
- **CPU getModPumpFlow**: `src/solver/dynwave.c:602-642`
- **CPU getMaxOutflow**: `src/solver/node.c:418-433`
- **GPU Three-Phase Design**: See commit messages for rationale

## Revision History

- **2025-11-01**: Initial three-phase sequential implementation
  - Improved from -72% to -45% error
  - Sequential Phase 2 confirmed
  - Documented remaining 1.5x error gap
