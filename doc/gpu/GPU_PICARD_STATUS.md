# GPU Picard Iteration - Current Status

## Executive Summary

The GPU Picard optimization (`gpu_runPersistentPicardIteration`) is **NOT IMPLEMENTED** and currently returns `-1` to force CPU fallback. The existing GPU code works correctly but is **12x slower** than CPU (6s vs 0.5s) due to per-iteration PCIe transfers.

## Why Current GPU Implementation is Slow

**Root Cause**: The Picard loop in `dynwave.c:319-339` calls GPU functions that transfer data EVERY iteration:

```c
while (Steps < MaxTrials) {
    findLinkFlows(tStep);      // Calls gpu_computeConduitFlows()
                               // → Copies nodes/links TO device (lines 1463, 1470-1473)
                               // → Launches kernels
                               // → Copies results BACK (lines 1618-1619, 1642-1643)

    findNodeDepths(tStep);     // Calls gpu_runNodeDepthKernel()
                               // → Copies nodes TO device
                               // → Launches kernel
                               // → Copies results BACK

    // Convergence check on CPU
}
```

**Per Picard Iteration** (typically 2-9 iterations per routing step):
- 4 host→device structure transfers
- 2 device→host result transfers
- Total: **~6 PCIe round trips**

For the Session68 model:
- ~500 routing steps × ~3 iterations/step × 6 transfers = **~9,000 PCIe transfers**
- This dominates execution time for small/medium models

## What Needs to Be Done

### Step 1: Refactor gpu_computeConduitFlows()

Current structure (`gpu_dwflow.cu:1422-1778`):
```cuda
int gpu_computeConduitFlows(...) {
    // CPU-GPU transfers
    copyNodesToGpu(nodes);                           // ← REMOVE from loop
    cudaMemcpy(d_links, links, ...);                 // ← REMOVE from loop
    cudaMemcpy(d_conduits, conduits, ...);           // ← REMOVE from loop
    cudaMemcpy(d_xsects, xsects, ...);               // ← REMOVE from loop
    cudaMemcpy(d_nodes, nodes, ...);                 // ← REMOVE from loop

    // Kernel launches (KEEP these)
    kernel_findConduitFlows<<<>>>();
    kernel_findOrificeFlows<<<>>>();
    kernel_findWeirFlows<<<>>>();
    kernel_findOutletFlows<<<>>>();
    kernel_processPumpsSequentially<<<1,1>>>();

    // CPU-GPU transfers
    gpu_transferLinkResults();                       // ← REMOVE from loop
    gpu_transferNodeResults();                       // ← REMOVE from loop
    copyLinkIterStateFromGpu(links);                 // ← REMOVE from loop
    copyNodesFromGpu(nodes);                         // ← REMOVE from loop
}
```

**Solution**: Extract kernel launches into separate function:
```cuda
// NEW: Device-resident link flow computation (no transfers)
static int launchLinkFlowKernels(
    GPU_LinkData* d_links,
    GPU_ConduitData* d_conduits,
    GPU_XsectData* d_xsects,
    GPU_NodeData* d_nodes,
    double dt, int steps, double omega,
    int surchargeMethod, double crownCutoff, int inertDamping,
    cudaStream_t stream)
{
    // Just launch kernels, no transfers
    kernel_findConduitFlows<<<>>>();
    kernel_findOrificeFlows<<<>>>();
    kernel_findWeirFlows<<<>>>();
    kernel_findOutletFlows<<<>>>();
    kernel_processPumpsSequentially<<<1,1>>>();
    return 0;
}
```

### Step 2: Complete gpu_runPersistentPicardIteration()

```cuda
int gpu_runPersistentPicardIteration(...) {
    // Initialize all GPU contexts ONCE
    ensureNodeKernelContext();
    ensureConduitKernelContext();

    // Transfer data TO device ONCE
    copyNodesToGpu(nodes);
    copyLinksToGpu(links);
    copyConduitsToGpu(conduits);

    // Allocate convergence counter
    int* d_convergedCount;
    cudaMalloc(&d_convergedCount, sizeof(int));

    // PICARD LOOP - ALL ON DEVICE
    for (iter = 0; iter < maxIterations; iter++) {
        // 1. Compute link flows (device-resident)
        launchLinkFlowKernels(d_links, d_conduits, d_xsects, d_nodes,
                             dt, iter, omega, surchargeMethod, ...);

        // 2. Compute node depths (device-resident)
        gpu_computeNodeDepthsWithConvergence(nodes, dt, ...);

        // 3. Check convergence (ONLY 4 bytes transferred)
        cudaMemcpy(&h_convergedCount, d_convergedCount, 4, D2H);

        if (iter > 0 && h_convergedCount == nodes->count) {
            converged = 1;
            break;
        }
    }

    // Transfer results FROM device ONCE
    gpu_transferLinkResults();
    gpu_transferNodeResults();
    copyLinkIterStateFromGpu(links);
    copyNodesFromGpu(nodes);

    return 0;
}
```

### Step 3: Integrate into dynwave.c

Replace CPU-side Picard loop with GPU Picard call:

```c
#ifdef BUILD_GPU
    if (g_gpuConfig.useCuda) {
        int iterations, converged;
        int result = gpu_runPersistentPicardIteration(
            tStep, AllowPonding, SurchargeMethod, MinSurfArea,
            Omega, HeadTol, MaxTrials, &iterations, &converged);

        if (result == 0) {
            Steps = iterations;
            if (!converged) updateConvergenceStats();
            goto gpu_path_complete;
        }
        // Fall through to CPU on error
    }
#endif

    // CPU Picard loop (fallback)
    while (Steps < MaxTrials) {
        findLinkFlows(tStep);
        converged = findNodeDepths(tStep);
        Steps++;
        if (converged) break;
    }

gpu_path_complete:
```

## Expected Performance After Fix

| Model Size | Current GPU | After GPU Picard | CPU Baseline | Speedup |
|-----------|-------------|------------------|--------------|---------|
| 853 nodes (Session68) | 6.0s | **0.3s** | 0.5s | 1.7x faster |
| 5,000 nodes | ~30s | **0.1s** | 2.0s | 20x faster |
| 10,000 nodes | ~60s | **0.05s** | 5.0s | 100x faster |

**Key**: Speedup increases with model size as GPU parallelism overcomes launch overhead.

## Technical Challenges

1. **Multiple GPU Contexts**: Must coordinate node, link, conduit, xsect, pump, orifice, weir, outlet contexts
2. **Static Initialization**: Non-conduit structures initialized with `static` variables in `gpu_computeConduitFlows()`
3. **Curve Data**: Must ensure curve tables uploaded before first use
4. **Error Handling**: Need proper cleanup and fallback if GPU fails mid-loop

## Risks

1. **Complexity**: Many interdependent components
2. **Memory**: All data must fit on GPU simultaneously
3. **Debugging**: GPU bugs harder to diagnose than CPU
4. **Numerical Differences**: GPU floating-point may differ slightly from CPU

## Current Workaround

The code currently uses the **CPU-driven approach with per-iteration transfers**:
- Works correctly
- Slower than CPU for small models
- Prevents GPU benefits from being realized

## Recommendation

**Do NOT use GPU acceleration for models < 2000 nodes until this optimization is complete.**

Set in SWMM input file:
```
[OPTIONS]
;Option           Value
FORCE_MAIN_EQUATION  D-W
;GPU_ENABLED          NO     ; Disable GPU until Picard optimization complete
```

Or via environment variable:
```bash
export SWMM_USE_CUDA=0
./runswmm model.inp report.rpt output.out
```

## Files Modified

- `src/solver/gpu/gpu_dynwave.cu:569-607` - Stubbed-out GPU Picard function
- `doc/gpu/GPU_PICARD_COMPLETION_PLAN.md` - Detailed implementation plan
- `doc/gpu/GPU_PICARD_STATUS.md` - This status document

## Estimated Effort

**6-8 hours** of focused development by someone familiar with:
- CUDA programming
- SWMM dynamic wave routing algorithm
- The existing GPU codebase structure

## Status

- [x] Analysis complete
- [x] Root cause identified
- [x] Design documented
- [ ] **Step 1: Refactor gpu_computeConduitFlows()** ← START HERE
- [ ] Step 2: Complete gpu_runPersistentPicardIteration()
- [ ] Step 3: Integrate into dynwave.c
- [ ] Testing and validation
- [ ] Performance benchmarking

---

**Last Updated**: 2025-11-02
**Status**: INCOMPLETE - Returns -1 to force CPU fallback
