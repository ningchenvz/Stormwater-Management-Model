# Session58_Interceptor GPU Error Analysis - Time-Dependent Failure

## Critical Finding

**GPU implementation shows time-dependent error accumulation for Session58_Interceptor model.**

## Test Results Summary

| Duration | CPU Error | GPU Error | GPU vs CPU | Status |
|----------|-----------|-----------|------------|--------|
| **30 minutes** | +5.365% | +3.262% | **0.61x (better)** | ✅ PASS |
| **3 hours** | +2.925% | **-75.881%** | **26x worse** | ❌ **FAIL** |
| **54 hours** (full) | Unknown | Killed (timeout) | — | ❌ **FAIL** |

## Error Growth Over Time

The GPU implementation degrades dramatically as simulation duration increases:

```
30 min:  GPU is 39% MORE accurate than CPU
3 hr:    GPU is 2600% LESS accurate than CPU
```

This indicates **accumulating numerical errors or algorithmic issues** in GPU kernels.

## Detailed 3-Hour Results

### Volume Balance (acre-feet)

| Component | CPU | GPU | Difference | % Diff |
|-----------|-----|-----|------------|--------|
| External Inflow | 1.918 | 1.920 | +0.002 | +0.1% ✓ |
| External Outflow | 0.000 | 0.000 | 0.000 | — |
| **Flooding Loss** | 0.000 | **0.598** | +0.598 | **∞** ❌ |
| Initial Storage | 0.070 | 0.070 | 0.000 | — |
| **Final Storage** | 1.930 | **2.902** | +0.972 | **+50%** ❌ |
| **Continuity Error** | 2.925% | **-75.881%** | -78.806 | **-2694%** ❌ |

### Key Observations

1. **Inflows match**: External inflow differs by only 0.1% (1.918 vs 1.920) - GPU routing is receiving correct input
2. **Massive flooding**: GPU shows 0.598 acre-feet flooding vs 0.000 on CPU - nodes overflowing incorrectly
3. **Storage accumulation**: GPU final storage is 50% higher (2.902 vs 1.930) - water not draining properly
4. **Catastrophic node errors**: Node A42BREAK1 has -780.57% error (vs CPU worst of 99.62%)

## Worst Node Errors

### CPU (3 hours)
- Node 3118: 99.62%
- Node PSA-OUT: 98.20%
- Node N23: 97.69%
- Node N22: 97.15%
- Node Q32: 85.45%

### GPU (3 hours)
- **Node A42BREAK1: -780.57%** ⚠️ **CRITICAL**
- Node S68: 100.00%
- Node SAC-IN: 100.00%
- Node N32: 99.99%
- Node S36SIPHONA: 99.95%

**Node A42BREAK1 Details:**
- Type: Junction
- Elevation: 94.93 ft
- Connected conduit: A42BREAK1A42BREAK21 (RECT_ROUND cross-section)
- Error magnitude: **-780%** (7.8x more water than mass balance allows)

## Hypothesis: Root Causes

### 1. Node Depth/Volume Update Errors (MOST LIKELY)

The 50% excess final storage suggests `kernel_findNodeDepths` may be incorrectly computing:
- Node volumes from depths
- Storage curves for complex geometries
- Surcharging conditions

**Evidence:**
- ✅ Inflows match (routing input correct)
- ❌ Storage accumulates (volumes not updating correctly)
- ❌ Flooding appears (depth thresholds wrong)

### 2. Conduit Flow Calculation for RECT_ROUND

Node A42BREAK1 has -780% error and connects to a RECT_ROUND conduit. This complex geometry may have:
- Incorrect area calculations
- Wrong hydraulic radius
- Errors in partial flow depth computation

### 3. Weir/Orifice/Outlet Flow Errors

Session58 has 9 weirs, 1 orifice, 7 outlets. If these accumulate small errors each timestep:
- 3 hours = 10,800 seconds / 5 sec timestep = **2,160 timesteps**
- 0.036% error/timestep × 2,160 = -77.8% total ✓ matches observed error!

**Evidence:**
- 30 min (360 timesteps): GPU better than CPU (errors haven't accumulated)
- 3 hr (2,160 timesteps): GPU -75.881% (errors accumulated)

### 4. Pump Sequential Processing Side Effects

The fully sequential pump processing may have unintended interactions with node state updates that only manifest over many timesteps.

## Why 30-Minute Test Passed

The 30-minute test only ran **360 timesteps** (30 min × 60 sec / 5 sec). At this duration:
- Small per-timestep errors haven't accumulated significantly
- Random floating-point variations may have actually helped GPU accuracy
- Model is in quasi-steady state (less dynamic error growth)

## Performance Impact

| Duration | CPU Runtime | GPU Runtime | GPU/CPU Ratio |
|----------|-------------|-------------|---------------|
| 30 min | 0.00 sec | ~2 sec | ∞ (worse) |
| 3 hr | 2.00 sec | 56.00 sec | 28x worse |

**Note:** GPU runtime includes debug output (printf statements), not representative of production performance.

## Recommended Actions

### Immediate (Critical)

1. **Disable GPU for Session58-type models** until fixed
   - Models with RECT_ROUND cross-sections
   - Models with complex weir/orifice configurations
   - Long-duration simulations (>1 hour)

2. **Add timestep-level validation**
   - Compare GPU vs CPU node depths every 100 timesteps
   - Detect divergence early before catastrophic failure

### Short-Term (Investigation)

1. **Compare node depth updates directly**
   ```cuda
   // Add to kernel_findNodeDepths after each timestep
   if (threadIdx.x == 0 && nodeIdx == find_node_index("A42BREAK1")) {
       printf("Step %d: A42BREAK1 depth=%.6f vol=%.6f inflow=%.6f outflow=%.6f\n",
              step, d_newDepth[nodeIdx], d_newVolume[nodeIdx],
              d_inflow[nodeIdx], d_outflow[nodeIdx]);
   }
   ```

2. **Test simple RECT_ROUND model**
   - Create minimal test case: 3 nodes, 2 RECT_ROUND conduits
   - Run for 3 hours, check if errors accumulate

3. **Disable weirs/orifices/outlets selectively**
   - Run Session58 with only conduits+pumps active
   - If error disappears, problem is in weir/orifice/outlet kernels

### Medium-Term (Fix)

1. **Review `kernel_findNodeDepths` line-by-line**
   - Compare to CPU `findNodeDepths()` in dynwave.c
   - Check volume calculations for all node types
   - Verify storage curves

2. **Review `kernel_findWeirFlows` / `kernel_findOrificeFlows` / `kernel_findOutletFlows`**
   - Check for missing `atomicAdd` operations
   - Verify flow direction logic
   - Ensure proper updating of `d_inflow` / `d_outflow`

3. **Add per-timestep conservation check**
   - After each timestep, verify: `Δstorage = inflows - outflows`
   - If violated, trigger warning or CPU fallback

## Test Files

- **30-min (PASS)**: `/tmp/Session58_Interceptor_30min.inp`
- **3-hr (FAIL)**: `/tmp/Session58_Interceptor_3hr.inp`
- **CPU 3-hr report**: `/tmp/Session58_3hr_cpu.rpt`
- **GPU 3-hr report**: `/tmp/Session58_3hr_gpu.rpt`

## Comparison to Session68 (Pump Test)

| Model | Duration | CPU Error | GPU Error | Issue |
|-------|----------|-----------|-----------|-------|
| Session68 (46 pumps) | 15 min | -29.785% | -45.369% | Pump-storage interaction |
| Session58 (mixed) | 30 min | +5.365% | +3.262% | None (GPU better!) |
| Session58 (mixed) | 3 hr | +2.925% | **-75.881%** | **Time-dependent accumulation** |

**Key Insight**: GPU errors are **MODEL-DEPENDENT and TIME-DEPENDENT**:
- Pump-heavy models: GPU 1.5x worse (constant error)
- Mixed models (short): GPU better than CPU
- **Mixed models (long): GPU catastrophically worse** ⚠️

## Conclusion

**GPU implementation is NOT production-ready for Session58-type models with >1 hour duration.**

The 30-minute test was misleadingly positive because errors hadn't accumulated yet. The 3-hour test reveals fundamental issues with:
- Node depth/volume calculations
- Weir/orifice/outlet flow accumulation
- RECT_ROUND geometry handling

**Estimated fix effort**: 1-2 weeks to identify and fix root cause

**Recommended immediate action**: Add warning in code:
```c
if (simulation_duration > 3600 && (Nlinks[WEIR] > 0 || Nlinks[ORIFICE] > 0)) {
    fprintf(stderr, "WARNING: GPU may accumulate errors for long simulations with weirs/orifices\n");
}
```

---

**Analysis Date**: 2025-11-01
**Test Status**: ❌ **CRITICAL FAILURE - GPU NOT READY FOR LONG SIMULATIONS**
