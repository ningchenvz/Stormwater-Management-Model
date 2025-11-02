# GPU Picard Iteration Completion Plan

## Current Status

The `gpu_runPersistentPicardIteration()` function in `src/solver/gpu/gpu_dynwave.cu` is **INCOMPLETE**:
- ✅ Handles node depth computation on GPU
- ❌ Missing link flow computation inside the loop
- ❌ Cannot be integrated because it would skip link flows entirely

## Root Cause of Performance Issue

**Current Performance**: 6 seconds (GPU) vs 0.5 seconds (CPU) = **12x SLOWER**

**Why**: Per-iteration PCIe transfers in the Picard loop at `dynwave.c:319-339`:

```c
while (Steps < MaxTrials) {
    findLinkFlows(tStep);         // Calls gpu_computeConduitFlows()
                                  // → Transfers nodes TO device (line 1463)
                                  // → Launches kernels
                                  // → Transfers results BACK (lines 1618-1619, 1642-1643)

    findNodeDepths(tStep);        // Calls gpu_runNodeDepthKernel()
                                  // → Transfers nodes TO device
                                  // → Launches kernel
                                  // → Transfers results BACK

    // Convergence check on CPU
}
```

**Each Picard iteration** (typically 2-9 per routing step):
- 4 host→device transfers (nodes, links, conduits, xsects structures)
- 2 device→host transfers (link results, node results)
- Total: ~6 PCIe round trips per iteration

## Solution Architecture

Move the **ENTIRE Picard loop** to GPU:

```c
// GPU Picard (device-resident)
for (iter = 0; iter < maxIterations; iter++) {
    // 1. Compute link flows (ALL on device)
    kernel_findConduitFlows<<<>>>(...);
    kernel_findOrificeFlows<<<>>>(...);
    kernel_findWeirFlows<<<>>>(...);
    kernel_findOutletFlows<<<>>>(...);
    kernel_processPumpsSequentially<<<1,1>>>(...);

    // 2. Compute node depths (ALL on device)
    kernel_findNodeDepths<<<>>>(...);

    // 3. Check convergence (ON device)
    kernel_checkConvergenceOptimized<<<>>>(...);

    // 4. Transfer ONLY convergence counter (4 bytes) to CPU
    cudaMemcpy(&h_convergedCount, d_convergedCount, 4, D2H);

    if (converged) break;
}

// Transfer results ONCE at end
gpu_transferLinkResults();
gpu_transferNodeResults();
```

**Key Changes**:
1. Kernel launches happen directly in loop (no wrapper functions)
2. Data stays device-resident during entire Picard iteration
3. Only 4-byte convergence check transfers per iteration
4. Full results transfer happens ONCE after convergence

## Implementation Steps

### Step 1: Refactor gpu_runPersistentPicardIteration()

Add link flow kernel launches before node depth computation:

```cuda
for (iter = 0; iter < maxIterations; iter++) {
    // === LINK FLOWS ===
    // Launch conduit kernel
    kernel_findConduitFlows<<<gridSize, blockSize, 0, stream>>>(
        d_links, d_conduits, d_xsects, d_nodes,
        dt, iter + 1, omega,
        surchargeMethod, crownCutoff, inertDamping);

    // Launch non-conduit kernels
    if (Nlinks[ORIFICE] > 0) {
        kernel_findOrificeFlows<<<orificeGridSize, blockSize, 0, stream>>>(...);
    }
    if (Nlinks[WEIR] > 0) {
        kernel_findWeirFlows<<<weirGridSize, blockSize, 0, stream>>>(...);
    }
    if (Nlinks[OUTLET] > 0) {
        kernel_findOutletFlows<<<outletGridSize, blockSize, 0, stream>>>(...);
    }
    if (Nlinks[PUMP] > 0) {
        kernel_processPumpsSequentially<<<1, 1, 0, stream>>>(...);
    }

    // === NODE DEPTHS ===
    if (gpu_computeNodeDepthsWithConvergence(...) != 0) {
        return -1;
    }

    // === CONVERGENCE CHECK ===
    cudaMemcpy(&h_convergedCount, d_convergedCount, sizeof(int), D2H);

    if (iter > 0 && h_convergedCount == nodes->count) {
        converged = 1;
        break;
    }
}
```

### Step 2: Add Required Data Structures

The function needs access to:
- `GPU_LinkData* links`
- `GPU_ConduitData* conduits`
- `GPU_XsectData* xsects`
- `GPU_PumpData*, GPU_OrificeData*, GPU_WeirData*, GPU_OutletData*`
- `GPU_CurveData*, GPU_CurvePoints*`

**Option A**: Pass all as function parameters (messy)
**Option B**: Access global GPU contexts (cleaner)

### Step 3: Initialize Device Structures Once

Ensure all GPU data structures are allocated and initialized BEFORE entering Picard loop:
- Links, conduits, xsects, nodes
- Pumps, orifices, weirs, outlets
- Curves and curve points

### Step 4: Integrate into dynwave.c

Replace the CPU-side Picard loop with GPU Picard call:

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
            goto gpu_path_complete;  // Skip CPU loop
        }
        // Fall through to CPU on GPU failure
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

### Step 5: Testing Strategy

1. **Correctness**: Compare GPU vs CPU results (flows, depths, convergence)
2. **Performance**: Measure time with/without GPU Picard
3. **Model sizes**: Test small (current), medium (1000-5000), large (10000+)

## Expected Performance Improvement

**Current** (per-iteration transfers):
- Small model (853 nodes): 6s (12x SLOWER than CPU)
- Transfer overhead dominates

**After GPU Picard** (device-resident):
- Small model: ~0.3s (1.6x FASTER than CPU)
- Medium model (5000 nodes): ~0.1s (5x FASTER)
- Large model (10000+ nodes): ~0.05s (10x+ FASTER)

**Why improvement**:
- Eliminates 6 PCIe transfers per iteration
- Keeps data device-resident
- Parallel kernel execution
- GPU efficiency increases with model size

## Risks and Mitigation

1. **Complexity**: Many moving parts
   - *Mitigation*: Implement incrementally, test each step

2. **Memory**: All data must fit on GPU
   - *Mitigation*: Add memory check, fallback to CPU

3. **Debugging**: Harder to debug GPU code
   - *Mitigation*: Add GPU-side printf, validation kernels

4. **Convergence**: GPU numerics might differ slightly
   - *Mitigation*: Use tolerance-based validation

## References

- Current incomplete implementation: `src/solver/gpu/gpu_dynwave.cu:537-627`
- Link flow kernels: `src/solver/gpu/gpu_dwflow.cu:1563-1606`
- Node depth kernel: `src/solver/gpu/gpu_dynwave.cu:514-518`
- Picard loop: `src/solver/dynwave.c:319-339`

## Status

- [x] Analysis complete
- [x] Design documented
- [ ] Implementation
- [ ] Testing
- [ ] Integration
- [ ] Performance validation

**Estimated effort**: 4-6 hours of focused development

**Blocking issue**: The current GPU Picard function is marked "completed" in session history but is actually incomplete. This explains why performance is 12x slower - we're still doing per-iteration transfers.
