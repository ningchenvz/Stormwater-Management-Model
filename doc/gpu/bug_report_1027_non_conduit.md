# Bug Report: Non-Conduit GPU Implementation Performance and Correctness Issues

**Date:** 2025-10-27
**Issue ID:** BUG-1027-NONCONDUIT
**Severity:** High
**Status:** Under Investigation
**Reporter:** Claude Code
**Component:** GPU Non-Conduit Flow Routing

---

## Summary

After implementing GPU kernel launches for non-conduit links (pumps, orifices, weirs, outlets), the Session18_GreenvilleSnowmelt.inp model shows:
- **3.3x slower performance** on GPU (47s) vs CPU (14s)
- **Major result discrepancies** (939 differences detected)
- **GPU failover triggered** at 11.4% completion (5000ms timeout exceeded)

---

## Environment

- **Model:** Session18_GreenvilleSnowmelt.inp
- **GPU:** NVIDIA GB10, Compute 12.1, 119.70 GB memory
- **Build:** SWMM 5.2.4 with CUDA support
- **Test Date:** October 27, 2025
- **Link Composition:**
  - 932 conduits (GPU-accelerated)
  - 5 pumps (GPU kernels launched)
  - 4 orifices (GPU kernels launched)
  - 2 outlets (GPU kernels launched)
  - 0 weirs
  - **Total:** 942 links (only 11 non-conduits, 1.2%)

---

## Expected Behavior

With GPU acceleration for all link types:
1. **Performance:** Similar or faster than CPU (target: 14s or less)
2. **Correctness:** Results identical to CPU mode (within numerical tolerance)
3. **Stability:** Complete full 96-hour simulation without failover

---

## Actual Behavior

### Performance Results

| Mode | Time (seconds) | Speedup | Failover |
|------|---------------|---------|----------|
| CPU (SWMM_USE_CUDA=0) | 14.00s | baseline | N/A |
| GPU (before optimization) | 51.00s | **0.27x (3.6x slower)** | Yes, at hour 13 |
| GPU (after optimization) | 47.00s | **0.30x (3.3x slower)** | Yes, at hour 11 |

### Correctness Results

```
Session18_GreenvilleSnowmelt.inp | differ | differ | ❌ major | 939 | 96.00 h | 52459.728 | 14360.274
```

- **939 discrepancies** detected between GPU and CPU outputs
- **Classification:** Major differences
- **Output size:** Both 48MB (correct size, but wrong content)

### Failover Behavior

```
... CUDA acceleration disabled after this step (cumulative kernel time 5000.0 ms exceeded 5000.0 ms)
```

- GPU disabled at **hour 11 of 96** (11.4% through simulation)
- **Threshold:** 5000ms cumulative kernel time
- **Implication:** Kernels taking >450ms per iteration on average

---

## Implementation Details

### What Was Completed (Phase 1)

#### 1. Data Structure Initialization (`gpu_manager.cu:279-475`)

**Function:** `gpu_initializeNonConduitData()`

```c
// Converts CPU Curve[] linked lists → GPU SoA arrays
// Initializes:
//   - 15 curves with 78 total points
//   - 5 pumps (GPU_PumpData)
//   - 4 orifices (GPU_OrificeData)
//   - 2 outlets (GPU_OutletData)
//   - 0 weirs (GPU_WeirData)
```

**Key Operations:**
- Traverses CPU `TTable` linked lists to count total curve points
- Flattens all curves into contiguous `h_xValues[]` and `h_yValues[]` arrays
- Stores curve metadata (`dataStart`, `dataCount`, `curveType`, `dxMin`)
- Allocates pinned host memory and device memory
- Transfers all static data to GPU once at startup

**Status:** ✅ Compiles and executes successfully

#### 2. Kernel Launch Integration (`gpu_dwflow.cu:1074-1156`)

**Location:** Inside `gpu_computeConduitFlows()` after conduit kernel launch

**Optimization Applied:**
- ❌ **Before:** `cudaMemcpy()` called **every iteration** (6 structures × ~1000 iterations = 6000 transfers)
- ✅ **After:** `cudaMemcpy()` called **once** at first iteration (6 structures × 1 time = 6 transfers)
- **Impact:** Reduced overhead by ~51s → 47s (8% improvement, not enough)

**Kernel Launches:**
```cuda
// If non-conduit data exists:
kernel_findPumpFlows<<<gridSize, blockSize, 0, stream>>>(
    d_links, d_gpuPumps, d_nodes, d_gpuCurves, d_gpuCurvePoints,
    ucfVolume, ucfLength, ucfFlow, steps);

kernel_findOrificeFlows<<<gridSize, blockSize, 0, stream>>>(
    d_links, d_gpuOrifices, d_xsects, d_nodes, omega, routeModel);

kernel_findWeirFlows<<<gridSize, blockSize, 0, stream>>>(
    d_links, d_gpuWeirs, d_xsects, d_nodes, omega, routeModel);

kernel_findOutletFlows<<<gridSize, blockSize, 0, stream>>>(
    d_links, d_gpuOutlets, d_nodes, d_gpuCurves, d_gpuCurvePoints,
    ucfLength, ucfFlow, omega, routeModel);
```

**Status:** ✅ Compiles, launches, no CUDA errors reported

#### 3. Lifecycle Integration (`dynwave.c:199-223`)

- Initialization called in `dynwave_init()`
- Cleanup called in `dynwave_close()`
- Error handling with fallback to CPU

**Status:** ✅ Working correctly

---

## Root Cause Analysis

### Theory 1: Kernel Implementation Bugs (Most Likely)

**Evidence:**
- 939 result discrepancies suggest **incorrect flow calculations**
- More Picard iterations → slower convergence → timeout
- CPU completes in 14s, GPU takes 47s despite only 11 non-conduits

**Potential Issues:**
1. **Pump flow calculations** incorrect for TYPE1-TYPE5 curves
2. **Orifice/weir flow logic** doesn't match CPU implementation
3. **Node flow accumulation** using `atomicAdd()` has race conditions
4. **Under-relaxation** (omega parameter) not applied correctly

**Files to Investigate:**
- `src/solver/gpu/gpu_dwflow.cu` (lines 531-978: kernels)
- `src/solver/gpu/gpu_nonconduit_helpers.cuh` (device helper functions)
- `src/solver/gpu/gpu_table_helpers.cuh` (curve lookup functions)

### Theory 2: Data Transfer Issues (Partially Ruled Out)

**Evidence:**
- Initial implementation: 6000 unnecessary transfers (51s)
- Optimized: 6 transfers only (47s)
- **Improvement:** 8% faster, but still 3.3x slower than CPU

**Conclusion:**
- Memory transfers contributed to slowdown but **not the root cause**
- Core problem is in kernel execution time, not transfer overhead

### Theory 3: Missing Dynamic Data Updates (Likely Secondary Issue)

**Evidence:**
- Link settings can change during simulation (pump on/off, gate positions)
- Current implementation: **No per-iteration updates** of dynamic link data
- CPU code: Updates `Link[j].setting` based on controls

**Impact:**
- Pumps may stay on when they should be off
- Orifices/weirs with dynamic settings don't update
- Results diverge from CPU over time

**Files to Check:**
- `src/solver/gpu/gpu_dwflow.cu` (need to add dynamic data transfer)
- `src/solver/controls.c` (control rules that modify link settings)

### Theory 4: Synchronization Issues (Possible)

**Evidence:**
- Conduit kernel runs first, then non-conduit kernels
- All kernels update same `d_links` and `d_nodes` arrays
- `atomicAdd()` used for node flow accumulation

**Potential Race Conditions:**
- If conduit kernel still writing when non-conduit kernel reads
- Node inflow/outflow updates interleave incorrectly
- `cudaStreamSynchronize()` happens AFTER all kernels launch

**Current Code Flow:**
```cuda
kernel_findConduitFlows<<<...>>>(d_links, d_conduits, d_xsects, d_nodes, ...);
CUDA_CHECK_LAST_ERROR();  // Does NOT synchronize!

// Immediately launch non-conduit kernels on same d_nodes
kernel_findPumpFlows<<<...>>>(d_links, d_pumps, d_nodes, ...);
kernel_findOrificeFlows<<<...>>>(d_links, d_orifices, d_xsects, d_nodes, ...);
// ...

CUDA_CHECK(cudaStreamSynchronize(stream));  // Too late?
```

---

## Diagnostic Steps Taken

### 1. Link Count Verification ✅
```bash
sed -n '/^\[PUMPS\]/,/^\[/p' Session18_GreenvilleSnowmelt.inp | grep -v "^;" | wc -l
# Result: 6 pumps (5 after parsing)

sed -n '/^\[CONDUITS\]/,/^\[/p' Session18_GreenvilleSnowmelt.inp | grep -v "^;" | wc -l
# Result: 932 conduits
```

**Confirmed:** Model is **conduit-dominated** (98.8% conduits, 1.2% non-conduits)

### 2. Build Verification ✅
```bash
cmake --build build -j
# Result: [100%] Built target runswmm (no errors)
```

### 3. Output File Comparison ✅
```bash
ls -lh /tmp/session18_*.out
# CPU:  48M Oct 27 15:25
# GPU:  48M Oct 27 15:23
```

**Confirmed:** Same file size, but different content (939 discrepancies)

### 4. GPU Console Output ✅
```
Initializing non-conduit GPU data...
  Converting 15 curves (78 total points) to GPU format...
  Initializing 5 pumps...
  Initializing 4 orifices...
  Initializing 2 outlets...
Non-conduit GPU initialization complete!
```

**Confirmed:** Initialization runs without errors

---

## Impact Assessment

### Severity: High

**Reasons:**
1. **Performance regression:** GPU 3.3x slower than CPU for conduit-dominated models
2. **Correctness issues:** 939 discrepancies indicate buggy kernel implementations
3. **Timeout failures:** GPU disabled after 11% of simulation (unusable for production)

### Affected Models

**High Impact:**
- Models with few non-conduits (<10%) but long simulations
- Example: Session18 (1.2% non-conduits, 96 hours) → GPU 3.3x slower

**Low Impact:**
- Conduit-only models → GPU still works correctly
- Models with many non-conduits → May have correct results but slow

---

## Recommended Next Steps

### Phase 2: Debugging and Validation (High Priority)

#### Step 1: Add Detailed Logging
**Goal:** Identify which kernels are causing slowdown

**Implementation:**
```cuda
// In gpu_dwflow.cu, after each kernel:
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel_findPumpFlows<<<...>>>();
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Pump kernel: %.3f ms\n", ms);
```

**Expected Output:**
- Identify which kernel(s) take >450ms
- Compare with CPU timing for equivalent link processing

#### Step 2: Validate Individual Link Flows
**Goal:** Find first discrepancy between GPU and CPU

**Implementation:**
```c
// After first iteration, copy flows back and compare
for (int i = 0; i < Nobjects[LINK]; i++) {
    double cpuFlow = Link[i].newFlow;
    double gpuFlow = gpuLinks->h_newFlow[i];
    double diff = fabs(cpuFlow - gpuFlow);
    if (diff > 0.01) {
        printf("Link %d (%s): CPU=%.6f, GPU=%.6f, diff=%.6f\n",
               i, Link[i].ID, cpuFlow, gpuFlow, diff);
    }
}
```

**Expected Output:**
- Identify which pump/orifice/weir/outlet has wrong flow
- Trace back to specific helper function in `gpu_nonconduit_helpers.cuh`

#### Step 3: Compare Curve Lookup Results
**Goal:** Verify curve conversion is correct

**Implementation:**
```c
// Test each pump curve with known head values
for (int i = 0; i < numPumps; i++) {
    double head = 10.0;  // Test value
    double cpuFlow = table_lookup(&Curve[Pump[i].pumpCurve], head);

    // Call GPU version
    double gpuFlow = gpu_test_table_lookup(i, head);

    printf("Pump %d curve: CPU=%.6f, GPU=%.6f\n", i, cpuFlow, gpuFlow);
}
```

**Expected Output:**
- Confirm curve flattening is correct
- Rule out data structure conversion bugs

#### Step 4: Add Dynamic Data Transfer
**Goal:** Update link settings each iteration

**Implementation:**
```c
// In copyLinksToGpu(), add:
for (int i = 0; i < links->count; i++) {
    links->h_setting[i] = Link[i].setting;
    links->h_targetSetting[i] = Link[i].targetSetting;
}

// Only on first iteration OR when settings change
if (steps == 0 || settingsChanged) {
    gpu_transferLinkDynamicToDevice(links, links->count);
}
```

**Expected Impact:**
- Pumps turn on/off correctly
- Results may match CPU more closely

### Phase 3: Performance Optimization (Medium Priority)

After correctness is fixed:

1. **Kernel Fusion:** Combine all non-conduit kernels into one
   - Reduces 4 kernel launches to 1
   - Saves ~200μs overhead per iteration

2. **Warp Divergence Analysis:** Profile with `nvprof` or `nsight`
   - Identify branching inefficiencies
   - Optimize pump curve type selection

3. **Memory Access Patterns:** Check for uncoalesced reads
   - Ensure SoA layout is accessed efficiently
   - Use shared memory for curve data

---

## Testing Plan

### Test Case 1: Simple Pump Model
**Model:** Example1.inp (has pumps)
**Expected:** GPU matches CPU within 0.01% tolerance
**Duration:** <1 minute

### Test Case 2: Session18 (Current Failing Case)
**Model:** Session18_GreenvilleSnowmelt.inp
**Expected:**
- GPU matches CPU (939 discrepancies → 0)
- GPU faster than CPU (47s → <14s)
- No timeout failover

**Duration:** ~14 seconds on CPU, target <10s on GPU

### Test Case 3: All Non-Conduit Types
**Model:** Example5.inp (pumps, orifices, weirs, outlets)
**Expected:** Validate all kernel types work correctly
**Duration:** <5 minutes

---

## Additional Notes

### Why Optimization Helped But Didn't Fix Issue

**Before Optimization:**
- 6 `cudaMemcpy()` calls × ~1000 iterations = **6000 transfers**
- Each transfer: ~8-10μs (device structure pointers only)
- **Total overhead:** 6000 × 10μs = **60ms** (not 51 seconds!)

**Conclusion:**
- The 51s → 47s improvement (4 seconds) suggests:
  - Transfers were cached/optimized by driver
  - Real issue is **kernel execution time**, not transfer time
  - Kernels are taking **~450ms per iteration** (way too slow)

### CPU Baseline Performance Analysis

**Session18 CPU mode:**
- 942 links × 96 hours × ~15 min/hour = ~1.4 million link-step calculations
- 14 seconds total
- **~100,000 link-steps per second** (very fast, OpenMP parallelized)

**GPU must match or exceed this to be competitive**

### Expected GPU Performance (Theoretical)

For 942 links:
- **Kernel launch overhead:** ~5-10μs
- **Kernel execution:** (942 threads × operations) / GPU throughput
- **Target:** <1ms per iteration (10x faster than CPU per iteration)
- **Current:** >450ms per iteration (450x slower than target!)

**This suggests a serious bug in kernel implementation, not just optimization needed.**

---

## References

- **Implementation Summary:** `doc/gpu/non_conduit_gpu_implementation_summary.md`
- **Status Document:** `doc/gpu/NON_CONDUIT_STATUS.md`
- **CPU Implementation:** `src/solver/link.c` (lines 1548-2100)
- **GPU Kernels:** `src/solver/gpu/gpu_dwflow.cu` (lines 531-978)
- **Helper Functions:** `src/solver/gpu/gpu_nonconduit_helpers.cuh`
- **Table Lookup:** `src/solver/gpu/gpu_table_helpers.cuh`

---

## Attachments

### Console Output (GPU Mode)
```
... CUDA acceleration enabled [SWMM_USE_CUDA=1] (device 0, compute 12.1, unified memory: yes)
  Initializing non-conduit GPU data...
    Converting 15 curves (78 total points) to GPU format...
... Allocated EXPLICIT memory for 15 curves (0.00 MB: 0.00 MB host + 0.00 MB device)
... Allocated EXPLICIT memory for 78 curve points (0.00 MB: 0.00 MB host + 0.00 MB device)
    Curves transferred to GPU successfully
    Initializing 5 pumps...
... Allocated EXPLICIT memory for 5 pumps (0.00 MB: 0.00 MB host + 0.00 MB device)
    Initializing 4 orifices...
... Allocated EXPLICIT memory for 4 orifices (0.00 MB: 0.00 MB host + 0.00 MB device)
    Initializing 2 outlets...
... Allocated EXPLICIT memory for 2 outlets (0.00 MB: 0.00 MB host + 0.00 MB device)
  Non-conduit GPU initialization complete!
    Pumps: 5, Orifices: 4, Weirs: 0, Outlets: 2
[... simulation starts ...]
 ... CUDA acceleration disabled after this step (cumulative kernel time 5000.0 ms exceeded 5000.0 ms)
[... continues on CPU ...]
... EPA SWMM completed in 47.00 seconds. There are warnings.
```

### Comparison Summary
```
Session18_GreenvilleSnowmelt.inp | differ | differ | ❌ major | 939 | 96.00 h | 52459.728 | 14360.274
```

---

## Conclusion

The non-conduit GPU infrastructure has been successfully implemented and compiles correctly, but **kernel implementations have critical bugs** causing:
1. **450ms+ kernel execution per iteration** (should be <1ms)
2. **939 result discrepancies** indicating incorrect flow calculations
3. **3.3x performance degradation** vs CPU

**Priority actions:**
1. Debug pump/orifice/outlet kernel implementations
2. Add per-link flow validation logging
3. Fix correctness issues before optimizing performance

**Estimated time to fix:** 8-12 hours of debugging and validation
