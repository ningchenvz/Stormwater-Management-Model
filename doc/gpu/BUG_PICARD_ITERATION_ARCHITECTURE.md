# GPU Picard Iteration Architecture Bug

**Date:** 2025-11-01
**Status:** IDENTIFIED - Fix in progress
**Severity:** CRITICAL - Causes massive continuity errors in GPU mode
**Test Case:** Session68_46_pumps.inp (877 links, 47 pumps)

## Summary

The GPU dynamic wave routing implementation has a fundamental architecture mismatch with the CPU implementation in the Picard iteration loop. The GPU path computes link flows **once** before entering the iteration loop, while the CPU path recomputes link flows **every iteration**. This causes severe mass balance errors because link flows depend on node heads (depths), which change during Picard iterations.

## Bug Details

### Observed Symptoms

1. **Massive continuity error**: GPU produces -117% continuity error vs CPU -29.8% (4x worse)
2. **Volume discrepancies**: Node volumes change by ~1% of expected amount
   - Example: WW-147 with outflow 4.58 cfs over 10 sec should lose 45.8 ft³
   - Actual volume loss: 0.49 ft³ (only 1.07% of expected)
3. **Results diverge significantly**: 6637+ lines of differences in report files

### Root Cause Analysis

**Location:** `src/solver/dynwave.c:307-346` (before fix)

**CPU Implementation** (CORRECT):
```c
while ( Steps < MaxTrials )
{
    initNodeStates();      // Reset node inflows/outflows
    findLinkFlows(tStep);  // ← RECOMPUTE link flows EVERY iteration
    converged = findNodeDepths(tStep);
    Steps++;
    if ( Steps > 1 ) {
        if ( converged ) break;
        findBypassedLinks();
    }
}
```

**GPU Implementation** (WRONG):
```c
initNodeStates();                    // ← Called ONCE before loop
findLinkFlows(tStep);                // ← Called ONCE before loop

// Execute entire Picard iteration loop on GPU (node depths only)
int gpuResult = gpu_runPersistentPicardIteration(
    tStep, AllowPonding, SurchargeMethod, MinSurfArea,
    Omega, HeadTol, MaxTrials, &gpuIterations, &gpuConverged);
```

**Why This Is Wrong:**

1. **Link flows depend on node heads**: Link flow equations use the difference in head between upstream and downstream nodes:
   ```
   Q = f(H_upstream - H_downstream, ...)
   ```

2. **Node depths change during Picard iteration**: Each iteration updates node depths based on continuity:
   ```
   dV/dt = Q_in - Q_out
   ```

3. **Stale link flows cause incorrect convergence**: If link flows are computed once with initial node depths, subsequent iterations use outdated flow values that don't match the updated node depths.

4. **Mass balance violation**: Volume changes computed with stale flows don't match actual node state changes, causing volume to "disappear" or "appear" in the system.

### Why The GPU Path Was Designed This Way

The GPU implementation used a "persistent Picard iteration" optimization:
- Goal: Minimize CPU-GPU data transfers by keeping the iteration loop on GPU
- Expected speedup: 2-5x for Picard iterations
- Trade-off: Only iterate on node depths (cheaper to transfer) while keeping link flows fixed

**This optimization is fundamentally incompatible with the physics of the problem.**

## Proposed Fix

### Change Summary

Remove the GPU-specific "persistent Picard iteration" and make both CPU and GPU use the **same Picard loop structure** that recomputes link flows every iteration.

### Code Changes

**File:** `src/solver/dynwave.c`

**Before:**
```c
#ifdef BUILD_GPU
    if ( g_gpuConfig.useCuda )
    {
        initNodeStates();
        findLinkFlows(tStep);
        int gpuResult = gpu_runPersistentPicardIteration(...);
        if ( gpuResult == 0 ) {
            goto gpu_path_complete;
        }
    }
#endif

    // CPU path
    while ( Steps < MaxTrials ) {
        initNodeStates();
        findLinkFlows(tStep);
        converged = findNodeDepths(tStep);
        Steps++;
        ...
    }
```

**After:**
```c
    // Unified Picard iteration loop (both CPU and GPU)
    // Link flows must be recomputed each iteration because they
    // depend on node heads, which change as node depths are updated
    while ( Steps < MaxTrials )
    {
        initNodeStates();
        findLinkFlows(tStep);      // CPU or GPU based on g_gpuConfig.useCuda
        converged = findNodeDepths(tStep);  // CPU or GPU based on g_gpuConfig.useCuda
        Steps++;
        if ( Steps > 1 ) {
            if ( converged ) break;
            findBypassedLinks();
        }
    }
```

### Functions Affected

- `dynwave_execute()` in `src/solver/dynwave.c` - Main Picard loop structure
- `gpu_runPersistentPicardIteration()` in `src/solver/gpu/gpu_dynwave.cu` - **NO LONGER USED**
- `findLinkFlows()` in `src/solver/dynwave.c` - Already has GPU path via `gpu_findLinkFlows()`
- `findNodeDepths()` in `src/solver/dynwave.c` - Already has GPU path via `gpu_runNodeDepthKernel()`

### Performance Impact

**Expected slowdown:** The fix will make GPU mode slower because:
- More kernel launches per routing step (3-8x more, depending on iterations to convergence)
- More CPU-GPU transfers per routing step (same multiplier)

**Estimated impact:**
- Previous (wrong) implementation: 2-5x speedup over CPU
- New (correct) implementation: 1.5-3x speedup over CPU (still faster, but less so)

**This is acceptable** because correctness is non-negotiable. Future optimizations can be explored after achieving correct results.

## Testing Plan

### Test Cases

**Primary Test Case:**
- File: `~/workspace/pyswmm/pyswmm/tests/Simon/Session68_46_pumps.inp`
- Short version: `/tmp/Session68_46_pumps_15min.inp` (15 minutes duration)
- Characteristics: 877 links (830 conduits, 47 pumps), 853 nodes

**Why This Test Case:**
1. Contains pumps (complex non-conduit links)
2. Large enough to show the bug clearly
3. Short enough for fast iteration (15-min version runs in ~5 seconds)

### Test Steps

#### Step 1: Verify Bug Exists (Before Fix)

```bash
# Revert to buggy version if needed
git stash  # or git checkout <commit-before-fix>

# Run GPU simulation
cd /home/ningchenspark/workspace/Stormwater-Management-Model/build
env SWMM_USE_CUDA=1 SWMM_FORCE_CUDA=1 bin/runswmm \
    /tmp/Session68_46_pumps_15min.inp \
    /tmp/gpu_before.rpt \
    /tmp/gpu_before.out

# Run CPU simulation
env SWMM_USE_CUDA=0 bin/runswmm \
    /tmp/Session68_46_pumps_15min.inp \
    /tmp/cpu_baseline.rpt \
    /tmp/cpu_baseline.out

# Compare continuity errors
grep -A20 "Routing Continuity" /tmp/gpu_before.rpt
grep -A20 "Routing Continuity" /tmp/cpu_baseline.rpt
```

**Expected Results:**
- GPU: Routing continuity error around -100% to -200%
- CPU: Routing continuity error around -20% to -30%
- GPU error is 4-8x worse than CPU

#### Step 2: Apply Fix and Verify Correctness

```bash
# Apply the fix (edit dynwave.c as described above)

# Rebuild
cmake --build build --config Debug -j8

# Run GPU simulation with fix
env SWMM_USE_CUDA=1 SWMM_FORCE_CUDA=1 bin/runswmm \
    /tmp/Session68_46_pumps_15min.inp \
    /tmp/gpu_after.rpt \
    /tmp/gpu_after.out

# Compare continuity errors
grep -A20 "Routing Continuity" /tmp/gpu_after.rpt
grep -A20 "Routing Continuity" /tmp/cpu_baseline.rpt
```

**Expected Results:**
- GPU continuity error should match CPU (within ±1%)
- Both should be around -20% to -30%

#### Step 3: Detailed Comparison

```bash
# Use comparison script
cd /home/ningchenspark/workspace/Stormwater-Management-Model/build
bash scripts/compare_runswmm_gpu_cpu.sh /tmp/Session68_46_pumps_15min.inp
```

**Expected Results:**
- Reports should match (or have minimal differences)
- Binary outputs should match
- Script should report "PASS" or minimal differences

#### Step 4: Check Mass Balance

```bash
# Look at mass balance log (if debug logging still enabled)
cat /tmp/gpu_mass_balance.txt

# Look for nodes with significant activity
# Calculate expected vs actual volume changes
# Example for node WW-147:
#   Expected dV = outflow * dt = 4.58 cfs * 10 sec = 45.8 ft³
#   Actual dV = oldVolume - newVolume
#   Should match within 1%
```

**Expected Results:**
- Volume changes should match expected values (Q * dt)
- No nodes should have volume "disappearing" or "appearing"

#### Step 5: Verify Performance (Secondary Priority)

```bash
# Run full 24-hour simulation to measure performance
cd /home/ningchenspark/workspace/Stormwater-Management-Model/build

# GPU
time env SWMM_USE_CUDA=1 SWMM_FORCE_CUDA=1 bin/runswmm \
    ~/workspace/pyswmm/pyswmm/tests/Simon/Session68_46_pumps.inp \
    /tmp/gpu_full.rpt \
    /tmp/gpu_full.out

# CPU
time env SWMM_USE_CUDA=0 bin/runswmm \
    ~/workspace/pyswmm/pyswmm/tests/Simon/Session68_46_pumps.inp \
    /tmp/cpu_full.rpt \
    /tmp/cpu_full.out

# Compare execution times
```

**Expected Results:**
- GPU should still be faster than CPU (1.5-3x speedup)
- GPU will be slower than before (that's OK - correctness first)

### Success Criteria

**MUST PASS:**
1. ✅ GPU and CPU continuity errors match within ±1%
2. ✅ Node volume changes match expected values (Q * dt) within ±1%
3. ✅ Report files match (or have only minor floating-point differences)
4. ✅ Binary output files match

**NICE TO HAVE:**
5. ⚠️  GPU is still faster than CPU (acceptable if slower than before)

## Additional Notes

### Debug Logging Added

During investigation, debug logging was added to track the issue:

1. **Mass balance logging** in `dynwave.c:293-305, 377-397`
   - Logs first 20 routing steps to `/tmp/gpu_mass_balance.txt`
   - Format: step, nodeID, oldDepth, newDepth, oldVolume, newVolume, inflow, outflow

2. **Kernel dt logging** in `gpu_dynwave.cu:68-71`
   - Prints dt value from first thread for first 3 iterations
   - Confirmed variable time-stepping is working correctly

This logging can be removed after the fix is verified.

### Related Issues

- Initially thought the problem was variable time-stepping (dt varying from 0.001 to 2+ seconds)
- That turned out to be correct SWMM behavior (Courant-based adaptive time stepping)
- The actual problem was the Picard loop structure

### Future Optimizations (Post-Fix)

After achieving correct results, consider these optimizations:

1. **Reduce kernel launch overhead**: Batch multiple node/link kernels into single launch
2. **Optimize data transfers**: Use streams and async transfers
3. **GPU-side convergence checking**: Keep convergence loop on GPU but recompute flows
4. **Persistent kernel approach**: Use CUDA dynamic parallelism to iterate on GPU

**CRITICAL:** All future optimizations must maintain correctness. Test thoroughly!

## References

- CPU implementation: `src/solver/dynwave.c:349-363` (findNodeDepths CPU path)
- GPU implementation: `src/solver/gpu/gpu_dynwave.cu` (persistent Picard iteration)
- Link flow computation: `src/solver/dynwave.c:485-517` (findLinkFlows)
- Node depth computation: `src/solver/dynwave.c:762-834` (findNodeDepths)

## Contributors

- Investigation and fix: Claude Code
- Test case: Session68_46_pumps.inp from pyswmm test suite
