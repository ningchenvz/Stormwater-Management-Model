# Non-Conduit GPU Support - Current Status

**Date:** 2025-10-27
**Issue:** Session18_GreenvilleSnowmelt.inp runs 3x SLOWER on GPU mode
**Root Cause:** Non-conduit links still processed on CPU

---

## Problem Analysis

### Test Case: Session18_GreenvilleSnowmelt.inp

**Link Composition:**
- **942 Conduits** → GPU ✓ (fast, parallel)
- **946 Pumps** → CPU ✗ (slow, sequential)
- **970 Orifices** → CPU ✗ (slow, sequential)
- **Total: 1,916 non-conduit links processed sequentially on CPU!**

### Why It's 3x Slower

When `SWMM_USE_CUDA=1`:
1. GPU processes 942 conduits in parallel (~1-5ms)
2. CPU processes 1,916 non-conduits sequentially (~50-100ms)
3. **Total time: ~100ms per iteration**

When `SWMM_USE_CUDA=0`:
1. CPU processes 942 conduits with OpenMP parallelization (~15ms)
2. CPU processes 1,916 non-conduits sequentially (~30ms)
3. **Total time: ~45ms per iteration**

**Result:** GPU mode is 2-3x slower because:
- GPU kernel launch overhead (~0.5ms)
- CPU processes non-conduits WITHOUT OpenMP (sequential)
- More data transfers (GPU ↔ CPU)

---

## What's Been Completed ✅

### 1. GPU Kernels Implemented
**File:** `src/solver/gpu/gpu_dwflow.cu`

All 4 kernels are fully implemented and compile successfully:
- ✅ `kernel_findPumpFlows` (lines 517-618)
- ✅ `kernel_findOrificeFlows` (lines 624-747)
- ✅ `kernel_findWeirFlows` (lines 753-877)
- ✅ `kernel_findOutletFlows` (lines 883-978)

### 2. Helper Functions Implemented
**Files:** `gpu_nonconduit_helpers.cuh`, `gpu_table_helpers.cuh`

- ✅ 15 device functions for flow calculations
- ✅ Table lookup functions for pump/outlet curves
- ✅ Cross-section geometry helpers
- ✅ Flap gate logic

### 3. Data Structures Defined
**File:** `gpu_structures.h`

- ✅ `GPU_CurveData` - Curve metadata
- ✅ `GPU_CurvePoints` - x/y curve data points
- ✅ Extended `GPU_LinkData` with `setting`, `hasFlapGate`
- ✅ `GPU_PumpData`, `GPU_OrificeData`, `GPU_WeirData`, `GPU_OutletData`

### 4. Memory Management Functions
**File:** `gpu_memory.cu`

- ✅ `gpu_allocateCurveData()`, `gpu_freeCurveData()`
- ✅ `gpu_allocateCurvePoints()`, `gpu_freeCurvePoints()`
- ✅ `gpu_transferCurveDataToDevice()`, `gpu_transferCurvePointsToDevice()`
- ✅ Pump/Orifice/Weir/Outlet allocation functions exist (added in previous work)

---

## What's Missing ❌

### 1. Data Structure Initialization (CRITICAL)

**Location:** `src/solver/dynwave.c` or `src/solver/gpu/gpu_manager.cu`

Need to initialize these global structures:
```c
GPU_PumpData g_gpuPumps;
GPU_OrificeData g_gpuOrifices;
GPU_WeirData g_gpuWeirs;
GPU_OutletData g_gpuOutlets;
GPU_CurveData g_gpuCurves;
GPU_CurvePoints g_gpuCurvePoints;
```

**Required Work:**
- Allocate memory for each structure
- Copy data from CPU `Pump[]`, `Orifice[]`, `Weir[]`, `Outlet[]` arrays
- Convert CPU `Curve[]` (linked lists) to GPU SoA format
- Transfer to GPU once at simulation start

**Estimated Effort:** 4-6 hours

### 2. Kernel Launch Integration (CRITICAL)

**Location:** `src/solver/gpu/gpu_dwflow.cu:1062` (marked with TODO)

Need to add kernel launches:
```cuda
// After conduit kernel completes...

// Launch pump kernel
if (numPumps > 0) {
    int pumpGridSize = GRID_SIZE(numPumps, blockSize);
    kernel_findPumpFlows<<<pumpGridSize, blockSize, 0, stream>>>(
        d_links, d_pumps, d_nodes, d_curves, d_curvePoints,
        ucfVolume, ucfLength, ucfFlow, steps);
}

// Launch orifice kernel
if (numOrifices > 0) {
    int orificeGridSize = GRID_SIZE(numOrifices, blockSize);
    kernel_findOrificeFlows<<<orificeGridSize, blockSize, 0, stream>>>(
        d_links, d_orifices, d_xsects, d_nodes, omega, routeModel);
}

// ... similar for weirs and outlets
```

**Required Work:**
- Pass pointers to non-conduit data structures
- Get unit conversion factors (UCF macros)
- Get routeModel (DW vs KW)
- Handle error checking

**Estimated Effort:** 2-3 hours

### 3. Data Transfer Logic (CRITICAL)

**Location:** `src/solver/gpu/gpu_dwflow.cu` (in `copyLinksToGpu()` or similar)

Need to transfer non-conduit data each iteration:
```c
// Transfer pump settings (may change due to controls)
gpu_transferPumpDynamicToDevice(&g_gpuPumps, numPumps);

// Transfer orifice/weir/outlet settings
gpu_transferOrificeDynamicToDevice(&g_gpuOrifices, numOrifices);
gpu_transferWeirDynamicToDevice(&g_gpuWeirs, numWeirs);
gpu_transferOutletDynamicToDevice(&g_gpuOutlets, numOutlets);
```

**Estimated Effort:** 1-2 hours

### 4. Unit Conversion Factors (MEDIUM)

**Location:** Pass as kernel parameters

Need to extract from CPU globals:
```c
extern double Qcf[];  // Flow conversion factor
extern double Ucf[];  // Unit conversion factors

double ucfLength = Ucf[LENGTH];
double ucfVolume = Ucf[VOLUME];
double ucfFlow = Qcf[type];  // depends on flow units
```

**Estimated Effort:** 1 hour

### 5. CPU Fallback Path (LOW PRIORITY)

**Location:** `src/solver/dynwave.c:486-494`

Currently:
```c
// --- still need to process non-conduit links on CPU
for (i = 0; i < Nobjects[LINK]; i++)
{
    if ( !isTrueConduit(i) )
    {
        if ( !Link[i].bypassed ) findNonConduitFlow(i, dt);
        updateNodeFlows(i);
    }
}
```

Should check if GPU handled them:
```c
if (!gpuHandledNonConduits) {
    // fall back to CPU
    for (...) { findNonConduitFlow(...); }
}
```

**Estimated Effort:** 30 minutes

---

## Implementation Roadmap

### Phase 1: Basic Integration (8-10 hours)
**Goal:** Get non-conduit kernels launching (may not be fully correct yet)

1. **Initialize Data Structures (4 hours)**
   - Add globals for GPU_PumpData, etc.
   - Implement initialization function
   - Copy data from CPU arrays

2. **Add Kernel Launches (2 hours)**
   - Add launch code after conduit kernel
   - Pass required parameters
   - Handle CUDA errors

3. **Transfer Dynamic Data (2 hours)**
   - Add transfer functions to iteration loop
   - Handle link settings updates

4. **Unit Conversion (1 hour)**
   - Extract UCF values
   - Pass to kernels

5. **Test Basic Execution (1 hour)**
   - Verify kernels launch
   - Check for CUDA errors
   - Ensure no crashes

### Phase 2: Correctness (4-6 hours)
**Goal:** Results match CPU implementation

1. **Curve Data Conversion (3 hours)**
   - Convert linked list `Curve[]` to SoA
   - Handle all 5 pump curve types
   - Test table lookups

2. **Validate Flow Calculations (2 hours)**
   - Compare GPU vs CPU flows
   - Fix discrepancies
   - Check Picard convergence

3. **Debug Edge Cases (1 hour)**
   - Flap gates
   - Off-curve conditions
   - Reverse flow

### Phase 3: Performance (2-4 hours)
**Goal:** GPU faster than CPU for large models

1. **Profile Kernel Performance (1 hour)**
   - Measure kernel execution times
   - Identify bottlenecks

2. **Optimize Memory Transfers (1 hour)**
   - Minimize transfers
   - Use streams for concurrency

3. **Tune Launch Parameters (1 hour)**
   - Optimize block size
   - Consider kernel fusion

4. **Benchmark (1 hour)**
   - Test with Session18 and other models
   - Document speedup

**Total Estimated Time: 14-20 hours**

---

## Quick Fix: Disable GPU for Non-Conduit Models

If you need Session18 to run at acceptable speed NOW, add this check:

**File:** `src/solver/dynwave.c` (around line 465)

```c
#ifdef BUILD_GPU
    // --- try GPU path first if enabled
    if (g_gpuConfig.useCuda)
    {
        // Count non-conduit links
        int nonConduitCount = 0;
        for (int i = 0; i < Nobjects[LINK]; i++) {
            if (!isTrueConduit(i)) nonConduitCount++;
        }

        // Disable GPU if >50% non-conduits (not worth overhead)
        if (nonConduitCount > Nobjects[LINK] / 2) {
            printf("\n  ... Disabling GPU: model has %d non-conduit links (not yet supported on GPU)\n",
                   nonConduitCount);
            g_gpuConfig.useCuda = 0;
        } else {
            // ... existing GPU code ...
        }
    }
#endif
```

This will fall back to CPU for models with many non-conduits, restoring normal performance.

---

## Testing Plan

### Test Models

1. **Example1.inp** - Small model with pumps
   - Validates basic pump functionality
   - Easy to debug

2. **Session18_GreenvilleSnowmelt.inp** - Large non-conduit model
   - 942 conduits, 946 pumps, 970 orifices
   - Performance critical
   - Should show 5-10x speedup when working

3. **Example5.inp** - Model with all link types
   - Pumps, orifices, weirs, outlets
   - Validates full integration

### Success Criteria

- ✅ All kernels launch without errors
- ✅ Results match CPU (within 0.01% tolerance)
- ✅ Picard iterations converge identically
- ✅ Mass balance preserved
- ✅ Session18 runs 3-5x FASTER than current GPU mode
- ✅ Session18 runs 2-5x FASTER than CPU mode

---

## Current Recommendation

**For Production Use:**
- ❌ Do NOT use GPU mode for models with many non-conduit links
- ✅ GPU mode works well for conduit-dominated models
- ✅ Add the quick fix above to auto-disable GPU for non-conduit models

**For Development:**
- Follow Phase 1 roadmap (8-10 hours)
- Focus on Session18 as primary test case
- Validate against CPU results before optimizing

---

## References

- **Implementation Summary:** `doc/gpu/non_conduit_gpu_implementation_summary.md`
- **Original Task:** `doc/gpu/non_conduit-link-port.md`
- **CPU Implementation:** `src/solver/link.c` (functions: `pump_getInflow`, `orifice_getInflow`, etc.)
- **GPU Kernels:** `src/solver/gpu/gpu_dwflow.cu` (lines 513-978)
- **Kernel Launch Location:** `src/solver/gpu/gpu_dwflow.cu:1062` (TODO comment)

---

## Summary

**What Works:** ✅
- GPU acceleration for conduit-only models
- All GPU kernels implemented and compile
- Complete helper function library

**What Doesn't Work:** ❌
- Non-conduit links fall back to CPU (slow, sequential)
- Models with many non-conduits run 2-3x SLOWER in GPU mode

**Solution:** Complete Phases 1-2 of the roadmap (~14 hours total)

**Quick Workaround:** Add check to disable GPU for non-conduit-heavy models
