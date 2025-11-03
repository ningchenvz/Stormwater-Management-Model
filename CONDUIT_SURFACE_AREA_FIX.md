# Critical Bug Fix: GPU Conduit Surface Area Calculation

## Root Cause Analysis

**Status**: ✅ IDENTIFIED
**Impact**: CRITICAL - Causes 14x error in storage volume, -3938% continuity error

### The Problem

The GPU implementation of `gpu_computeSurfaceAreas()` in `src/solver/gpu/gpu_conduit_helpers.cuh:66` is a **grossly oversimplified** version of the CPU's `findSurfArea()` in `src/solver/dwflow.c:417`.

Even though tabular storage node base areas are calculated correctly (verified: 1200 ft² at 5ft depth), the conduit contributions to `nodes->d_newSurfArea` are **massively inflated** due to missing logic. This inflated area feeds into `gpu_setNodeDepth()`, causing the depth update `dy = dV / surfArea` to use the wrong denominator, leading to mass balance drift.

### Evidence

1. ✅ Curve data transfers correctly (byte-for-byte verified)
2. ✅ Table lookup returns correct values (1200 ft² for STOR-10 at depth=5ft)
3. ✅ Unit conversions are correct (ucfLength=1.0)
4. ✅ Storage base surface area calculation is CORRECT
5. ❌ **BUT** final stored volume: GPU=6.706 vs CPU=0.480 acre-ft (14x error!)
6. ❌ Continuity error: GPU=-3938% vs CPU=-198% (20x worse)

**Conclusion**: The bug is NOT in storage curves - it's in conduit surface area contributions overwhelming the correct base values.

---

## Missing Features in GPU Implementation

### 1. Incomplete Flow Classification

**CPU** (`getFlowClass()` at `dwflow.c:297`): Returns 6 flow classes
- `SUBCRITICAL` - normal flow
- `UP_CRITICAL` - upstream end at critical depth
- `DN_CRITICAL` - downstream end at critical depth
- `UP_DRY` - upstream end dry
- `DN_DRY` - downstream end dry
- `DRY` - both ends dry

**GPU** (`gpu_classifyFlow()` at `gpu_conduit_helpers.cuh:58`): Only 4 classes
- `GPU_SUBCRITICAL`
- `GPU_UP_DRY`
- `GPU_DN_DRY`
- `GPU_DRY`
- ❌ **MISSING**: `UP_CRITICAL`, `DN_CRITICAL`

**Impact**: Critical flow conditions are misclassified as SUBCRITICAL, causing wrong surface area calculations.

---

### 2. Missing `fasnh` Scaling Factor

**CPU** (`dwflow.c:474`):
```c
surfArea2 = (widthMid + width2) * length / 4. * fasnh;
```

Where `fasnh` = fraction interpolated between normal and critical depth (range 0.0 to 1.0):
```c
if ( ycMax - ycMin < FUDGE ) *fasnh = 0.0;
else *fasnh = (ycMax - y2) / (ycMax - ycMin);
```

**GPU**: No `fasnh` parameter anywhere in `gpu_computeSurfaceAreas()`

**Impact**: Downstream surface area not scaled correctly when flow is between normal and critical depth.

---

### 3. Wrong Surface Area Factors for Critical Flow

**CPU UP_CRITICAL** (`dwflow.c:486`):
```c
surfArea2 = (widthMid + width2) * length * 0.5;  // Half-length at downstream
surfArea1 = 0.0;                                  // No contribution upstream
```

**CPU DN_CRITICAL** (`dwflow.c:498`):
```c
surfArea1 = (width1 + widthMid) * length * 0.5;  // Half-length at upstream
surfArea2 = 0.0;                                  // No contribution downstream
```

**GPU SUBCRITICAL** (`gpu_conduit_helpers.cuh:136-137`):
```cuda
surfArea1 = (width1 + widthMid) * length * 0.25;  // Quarter-length both ends
surfArea2 = (widthMid + width2) * length * 0.25;
```

**Impact**: Surface areas use 0.25 (quarter) instead of 0.5 (half) for critical flow, AND both nodes get contributions when only one should.

---

### 4. Missing Critical/Normal Depth Calculations

**CPU** uses:
- `link_getYnorm(j, fabs(q))` - computes normal depth for given flow
- `link_getYcrit(j, fabs(q))` - computes critical depth for given flow

These are used in `getFlowClass()` to determine:
- `ycMin = MIN(yN, yC)` - minimum of normal and critical
- `ycMax = MAX(yN, yC)` - maximum of normal and critical
- Whether flow is subcritical, critical, or supercritical

**GPU**: These functions don't exist at all.

**Impact**: Cannot properly classify flow or compute `fasnh`.

---

## Implementation Plan

### Phase 1: Port Critical/Normal Depth Functions

**Files to create/modify**:
- `src/solver/gpu/gpu_link_helpers.cuh` (new file)

**Functions to port**:
1. `gpu_link_getYnorm()` - Port from `link.c:link_getYnorm()`
2. `gpu_link_getYcrit()` - Port from `link.c:link_getYcrit()`

**Dependencies**: These likely call cross-section functions which may also need porting.

---

### Phase 2: Port Flow Classification Logic

**File**: `src/solver/gpu/gpu_conduit_helpers.cuh`

**Replace `gpu_classifyFlow()`** (currently lines 58-64) with full port of `getFlowClass()` from `dwflow.c:297-413`

**New signature**:
```cuda
__device__ int gpu_getFlowClass(
    int linkIdx,
    double q,
    double h1,
    double h2,
    double y1,
    double y2,
    double offset1,
    double offset2,
    int node1Type,
    int node2Type,
    double node1NewDepth,
    double node2NewDepth,
    double node1InvertElev,
    double node2InvertElev,
    GPU_Xsect* xsect,
    double* yC_out,      // critical depth
    double* yN_out,      // normal depth
    double* fasnh_out)   // fraction between norm & crit
```

**Logic to port**:
- Lines 313-332: Get offsets, adjust for outfalls
- Lines 335-371: Both ends wet - check for critical flow
- Lines 374: Both ends dry
- Lines 377-393: Downstream wet, upstream dry - UP_DRY or UP_CRITICAL
- Lines 396-411: Upstream wet, downstream dry - DN_DRY or DN_CRITICAL

---

### Phase 3: Fix Surface Area Calculation

**File**: `src/solver/gpu/gpu_conduit_helpers.cuh`

**Replace `gpu_computeSurfaceAreas()`** (currently lines 66-143) with faithful port of `findSurfArea()` logic from `dwflow.c:452-550`

**Add to signature**:
```cuda
__device__ void gpu_computeSurfaceAreas(
    // ... existing parameters ...
    double fasnh,           // ADD THIS - from flow classification
    double criticalDepth,   // ADD THIS - from flow classification
    double normalDepth,     // ADD THIS - from flow classification
    // ... outputs ...
)
```

**Fix each flow class case**:

```cuda
case GPU_SUBCRITICAL:
    flowDepthMid = 0.5 * (flowDepth1 + flowDepth2);
    if (flowDepthMid < GPU_FUDGE) flowDepthMid = GPU_FUDGE;
    width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
    width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
    widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
    surfArea1 = (width1 + widthMid) * length / 4.0;
    surfArea2 = (widthMid + width2) * length / 4.0 * fasnh;  // ADD fasnh!
    break;

case GPU_UP_CRITICAL:
    flowDepth1 = criticalDepth;
    if (normalDepth < criticalDepth) flowDepth1 = normalDepth;
    flowDepth1 = gpu_MAX(flowDepth1, GPU_FUDGE);
    // UPDATE h1 if needed
    flowDepthMid = 0.5 * (flowDepth1 + flowDepth2);
    if (flowDepthMid < GPU_FUDGE) flowDepthMid = GPU_FUDGE;
    width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
    widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
    surfArea1 = 0.0;                                          // No upstream contribution!
    surfArea2 = (widthMid + width2) * length * 0.5;          // Half-length!
    break;

case GPU_DN_CRITICAL:
    flowDepth2 = criticalDepth;
    if (normalDepth < criticalDepth) flowDepth2 = normalDepth;
    flowDepth2 = gpu_MAX(flowDepth2, GPU_FUDGE);
    // UPDATE h2 if needed
    width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
    flowDepthMid = 0.5 * (flowDepth1 + flowDepth2);
    if (flowDepthMid < GPU_FUDGE) flowDepthMid = GPU_FUDGE;
    widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
    surfArea1 = (width1 + widthMid) * length * 0.5;          // Half-length!
    surfArea2 = 0.0;                                          // No downstream contribution!
    break;
```

---

### Phase 4: Integration

**File**: `src/solver/gpu/gpu_dwflow.cu`

**Modify link flow kernel** to:
1. Call `gpu_getFlowClass()` to get flow class, critical/normal depths, and `fasnh`
2. Pass these to `gpu_computeSurfaceAreas()`

**Current code** (approximately line 550):
```cuda
int flowClass = gpu_classifyFlow(y1, y2);
gpu_computeSurfaceAreas(xsect, length, offset1, offset2, y1, y2,
                        flowClass, surchargeMethod, crownCutoff,
                        &surfArea1, &surfArea2);
```

**New code**:
```cuda
double yC = 0.0, yN = 0.0, fasnh = 1.0;
int flowClass = gpu_getFlowClass(
    j, q, h1, h2, y1, y2,
    offset1, offset2,
    node1Type, node2Type,
    node1NewDepth, node2NewDepth,
    node1InvertElev, node2InvertElev,
    xsect, &yC, &yN, &fasnh);

gpu_computeSurfaceAreas(xsect, length, offset1, offset2, y1, y2,
                        flowClass, surchargeMethod, crownCutoff,
                        fasnh, yC, yN,
                        &surfArea1, &surfArea2);
```

---

### Phase 5: Validation & Diagnostics

**Add diagnostic kernel** (new function in `gpu_dynwave.cu`):

```cuda
void gpu_validateSurfaceAreas(GPU_NodeData* nodes, GPU_LinkData* links, int nodeCount, int linkCount)
{
    // Copy d_newSurfArea from GPU to CPU
    double* h_gpuSurfArea = (double*)malloc(nodeCount * sizeof(double));
    cudaMemcpy(h_gpuSurfArea, nodes->d_newSurfArea, nodeCount * sizeof(double), cudaMemcpyDeviceToHost);

    // Compare with CPU Xnode[].newSurfArea
    printf("\n=== SURFACE AREA VALIDATION ===\n");
    for (int i = 0; i < nodeCount; i++) {
        double cpuArea = Xnode[i].newSurfArea;
        double gpuArea = h_gpuSurfArea[i];
        double diff = fabs(gpuArea - cpuArea);
        double relErr = (cpuArea > 0.001) ? (diff / cpuArea * 100.0) : 0.0;

        if (relErr > 1.0) {  // More than 1% error
            printf("Node %d (%s): CPU=%.2f GPU=%.2f diff=%.2f (%.1f%%)\n",
                   i, Node[i].ID, cpuArea, gpuArea, diff, relErr);
        }
    }

    // Also compare link surface areas
    for (int j = 0; j < linkCount; j++) {
        if (Link[j].type == CONDUIT) {
            double cpuArea1 = Link[j].surfArea1;
            double cpuArea2 = Link[j].surfArea2;
            // Get from GPU linkData...
            printf("Link %d: CPU1=%.2f CPU2=%.2f\n", j, cpuArea1, cpuArea2);
        }
    }

    free(h_gpuSurfArea);
    printf("=== END VALIDATION ===\n\n");
}
```

**Call after `launchLinkFlowKernels()`** in first few timesteps to verify the fix.

---

## Testing Plan

### Test 1: Simple Conduit Test
Create minimal test with:
- 2 junctions
- 1 conduit with known geometry
- Manually verify surface area calculation matches CPU

### Test 2: Session18 Regression
After fix:
- Final stored volume should match CPU within 1% (0.480 acre-ft)
- Continuity error should match CPU order of magnitude (~-198%)
- All nodes should have <100% continuity error

### Test 3: Full nrtest Suite
Run all regression tests to ensure no breakage in other scenarios.

---

## Estimated Effort

- **Phase 1** (Port Ynorm/Ycrit): 4-6 hours (complex hydraulic calculations)
- **Phase 2** (Port getFlowClass): 2-3 hours
- **Phase 3** (Fix computeSurfaceAreas): 2-3 hours
- **Phase 4** (Integration): 1-2 hours
- **Phase 5** (Validation): 2-3 hours

**Total**: 11-17 hours

---

## Priority

**CRITICAL** - This blocks all GPU mass balance validation. Must be fixed before any other GPU work can be trusted.

---

## References

- CPU Implementation: `src/solver/dwflow.c:297-550`
- GPU Current (Broken): `src/solver/gpu/gpu_conduit_helpers.cuh:58-143`
- Related: `src/solver/link.c` (for Ynorm/Ycrit calculations)
