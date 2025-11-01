# GPU Pump Mass Balance Bug

**Date:** 2025-11-01
**Status:** ROOT CAUSE IDENTIFIED
**Severity:** HIGH - Causes 2.4x worse continuity error with pumps

## Summary

The GPU pump kernel skips the `getModPumpFlow()` function that prevents pumps from draining inlet nodes faster than water is available. This causes significant mass balance errors in pump-heavy models.

## Evidence

### Test Results

**Simple Test (conduits only - no pumps):**
- GPU: -0.034% continuity error
- CPU: -0.033% continuity error
- ✅ **IDENTICAL** - GPU node depth calculations are correct

**Session68_46_pumps (877 links, 47 pumps):**
- GPU: -72.391% continuity error
- CPU: -29.785% continuity error
- ❌ **2.4x WORSE** - Issue is pump-specific

### Root Cause

**File:** `src/solver/gpu/gpu_dwflow.cu:615-629`

```cuda
// TODO: Implement parallel-safe pump flow modification
// The CPU version uses Node[j].outflow which is built up sequentially,
// but this doesn't work in parallel GPU execution
// For now, we skip this check - may cause minor mass balance issues
//
// qIn = gpu_getModPumpFlow(
//     k, n1, qIn, dt, pumpType,
//     ...
```

**CPU Implementation:** `src/solver/dynwave.c:586, 603-644`

```c
qNew = link_getInflow(i);
if ( Link[i].type == PUMP ) qNew = getModPumpFlow(i, qNew, dt);
```

The `getModPumpFlow()` function:
1. **For storage nodes & TYPE1 pumps:** Calls `node_getMaxOutflow()` to ensure node volume doesn't go negative
2. **For other pump types:** Checks if pump flow would make inlet node depth negative, and if so, limits flow to inlet flow rate

**Why GPU skips it:**
The function uses `Node[j].outflow` which is accumulated during sequential link processing on CPU. On GPU, links are processed in parallel, so `outflow` values are not available when pumps execute (they're being accumulated via `atomicAdd` concurrently).

## Impact

Without `getModPumpFlow()`:
- Pumps can remove more water than exists in inlet node
- Nodes can reach negative depths (then clamped to 0)
- Water "disappears" from the system
- Mass balance errors accumulate over time

**Measured impact:**
- Architectural fix (Picard loop) improved error from -117% to -72%
- Missing `getModPumpFlow()` accounts for remaining 2.4x error vs CPU

## Proposed Fix

### Option 1: Two-Pass Pump Processing (RECOMMENDED)

Process pumps in two passes per Picard iteration:

**Pass 1:** Compute preliminary pump flows (existing kernel)
- Calculate qIn from pump curves
- Store in temporary buffer

**Pass 2:** Modify and apply pump flows
- After all links have updated node outflows
- Apply `getModPumpFlow()` logic using complete outflow values
- Update final flows and node inflows/outflows

**Pros:**
- Maintains CPU semantics exactly
- Parallel-safe
- Clean separation of concerns

**Cons:**
- Extra kernel launch per iteration
- Extra memory for temporary storage

### Option 2: Use Old Outflow Values

Modify `gpu_getModPumpFlow()` to use `d_oldOutflow` from previous timestep instead of current `d_outflow`:

```cuda
// Line 638 in dynwave.c uses current outflow:
newNetInflow = Node[j].inflow - Node[j].outflow - q;

// GPU version could use:
newNetInflow = Node[j].inflow - Node[j].oldOutflow - q;
```

**Pros:**
- Single pass
- Minimal code changes
- No extra memory

**Cons:**
- Different from CPU (uses old outflow vs current)
- May be slightly less accurate
- Could cause instability in some edge cases

### Option 3: Conservative Approximation

Use only inlet node depth and inflow (ignore outflow):

```cuda
// Simplified check - only prevent negative depths
double availableVolume = nodes->d_oldDepth[n1] * nodes->d_newSurfArea[n1];
double maxQ = (availableVolume / dt) + nodes->d_inflow[n1];
if (qIn > maxQ) qIn = maxQ;
```

**Pros:**
- Simple to implement
- Parallel-safe
- Conservative (safe side)

**Cons:**
- May be overly conservative in some cases
- Not exact match to CPU

## Recommendation

Implement **Option 1 (Two-Pass)** because:
1. Exact match to CPU behavior
2. Clean, maintainable code
3. Performance cost is acceptable (one extra kernel launch per iteration)
4. Already have two-pass architecture (conduits first, then pumps)

## Implementation Plan

### Step 1: Create intermediate pump flow buffer

```cuda
// In gpu_computeConduitFlows():
static double* d_prelimPumpFlows = NULL;
if (!d_prelimPumpFlows && Nlinks[PUMP] > 0) {
    cudaMalloc(&d_prelimPumpFlows, Nlinks[PUMP] * sizeof(double));
}
```

### Step 2: Modify pump kernel to store preliminary flows

```cuda
__global__ void kernel_findPumpFlows_Preliminary(
    GPU_PumpData* pumps,
    GPU_NodeData* nodes,
    double* d_prelimFlows,  // OUTPUT: preliminary flows
    ...)
{
    // ... compute qIn as before ...

    // Store preliminary flow instead of updating links/nodes immediately
    d_prelimFlows[k] = qIn;
}
```

### Step 3: Create pump flow modification kernel

```cuda
__global__ void kernel_modifyPumpFlows(
    GPU_LinkData* links,
    GPU_PumpData* pumps,
    GPU_NodeData* nodes,
    double* d_prelimFlows,  // INPUT: preliminary flows
    double dt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pumps->count) return;

    int j = pumps->d_linkIndex[k];
    int n1 = links->d_node1[j];
    double qIn = d_prelimFlows[k];

    // NOW we can safely use nodes->d_outflow because all links have run
    qIn = gpu_getModPumpFlow(
        k, n1, qIn, dt, pumps->d_type[k],
        nodes->d_type,
        nodes->d_inflow,
        nodes->d_outflow,  // Safe to use now!
        nodes->d_oldDepth,
        nodes->d_oldNetInflow,
        nodes->d_oldVolume,
        nodes->d_fullVolume,
        nodes->d_newSurfArea);

    // Update link flows and node inflow/outflow
    links->d_newFlow[j] = qIn;
    if (qIn > 0.0) {
        atomicAdd(&nodes->d_outflow[n1], qIn);
        atomicAdd(&nodes->d_inflow[n2], qIn);
    }
}
```

### Step 4: Update launch sequence in gpu_computeConduitFlows()

```cuda
// Current sequence (WRONG):
kernel_findConduitFlows<<<...>>>();  // Updates node outflows via atomicAdd
kernel_findPumpFlows<<<...>>>();     // PROBLEM: Can't see conduit outflows yet!

// New sequence (CORRECT):
kernel_findConduitFlows<<<...>>>();        // Updates node outflows
kernel_findPumpFlows_Preliminary<<<...>>>(); // Compute pump flows, store in buffer
cudaStreamSynchronize(stream);              // WAIT for all link outflows
kernel_modifyPumpFlows<<<...>>>();          // Modify and apply pump flows
```

## Testing Plan

### Test 1: Verify fix with Session68_46_pumps

```bash
# After implementing fix
env SWMM_USE_CUDA=1 SWMM_FORCE_CUDA=1 bin/runswmm Session68_46_pumps_15min.inp gpu.rpt gpu.out
env SWMM_USE_CUDA=0 bin/runswmm Session68_46_pumps_15min.inp cpu.rpt cpu.out

# Compare continuity errors
grep "Continuity Error" gpu.rpt
grep "Continuity Error" cpu.rpt
```

**Expected:** GPU and CPU continuity errors match within ±1%

### Test 2: Verify conduits still work

```bash
# Simple test (conduits only)
env SWMM_USE_CUDA=1 SWMM_FORCE_CUDA=1 bin/runswmm tests/test_models/simple_test.inp gpu.rpt gpu.out
```

**Expected:** Still -0.034% (no regression)

### Test 3: Full comparison suite

```bash
bash scripts/compare_runswmm_gpu_cpu.sh Session68_46_pumps.inp
```

**Expected:** PASS with minimal differences

## Performance Impact

**Before fix:**
- 1 conduit kernel + 1 pump kernel per iteration
- Total: 2 kernel launches per iteration

**After fix:**
- 1 conduit kernel + 1 preliminary pump kernel + 1 pump modify kernel per iteration
- Total: 3 kernel launches per iteration

**Estimated overhead:** +10-15% execution time for pump models
**Benefit:** Correct mass balance (worth the cost!)

## Related Files

- `src/solver/gpu/gpu_dwflow.cu:531-647` - Pump kernel
- `src/solver/dynwave.c:603-644` - CPU getModPumpFlow()
- `src/solver/gpu/gpu_nonconduit_helpers.cuh:169-204` - GPU helper (currently disabled)
- `src/solver/node.c` - node_getMaxOutflow()

## Status

- [x] Bug identified
- [x] Root cause confirmed
- [x] Isolated to pump-specific issue (conduits work correctly)
- [ ] Fix implemented
- [ ] Fix tested
- [ ] Performance validated

## Contributors

- Investigation: Claude Code
- Test isolation: Simple conduit test vs pump test comparison
