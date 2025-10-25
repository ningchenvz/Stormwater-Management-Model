# Phase 4: Complex Kernel Implementation Plan

## Overview

Phase 4 implements the most computationally intensive kernel: `findConduitFlows`. This is the core dynamic wave routing calculation that updates flow in all conduit links.

## Complexity Analysis

**Main Function:** `dwflow_findConduitFlow()` (~240 lines)
**Computational Intensity:** HIGH - called for every conduit, every Picard iteration
**Dependencies:** 15+ helper functions

## Required Helper Functions

### Critical (Must Implement First)
1. ✅ **getArea()** - Cross-sectional flow area
2. ✅ **getHydRad()** - Hydraulic radius
3. ✅ **getSlotWidth()** - Preissmann slot width for surcharged flow
4. ⏳ **findSurfArea()** - Surface area contributions to nodes

### Important (Implement Second)
5. ⏳ **link_getFroude()** - Froude number calculation
6. ⏳ **xsect_isOpen()** - Check if cross-section is open channel
7. ⏳ **findLocalLosses()** - Local head loss calculations

### Optional (Can Defer or Simplify)
8. **forcemain_getFricSlope()** - Force main friction (special case)
9. **culvert_getInflow()** - Culvert inlet control (special case)
10. **checkNormalFlow()** - Normal flow limitation
11. **link_setFlapGate()** - Flap gate logic
12. **link_getLossRate()** - Evap/seepage losses
13. **link_getFullState()** - Full/partial state determination
14. **link_getLength()** - Conduit length getter

## Implementation Strategy

### Stage 1: Simplified Version (Current Focus)
- Implement core momentum equation solver
- Support regular conduits only (no force mains, culverts)
- Skip optional features (flap gates, normal flow limits, evap/seepage)
- **Goal:** Get basic GPU kernel working and validated

### Stage 2: Full Features
- Add special conduit types (force mains, culverts)
- Implement flap gates and flow limits
- Add evap/seepage losses
- **Goal:** Feature parity with CPU implementation

### Stage 3: Optimization
- Optimize memory access patterns
- Use shared memory for frequently accessed data
- Minimize divergent branches
- **Goal:** Maximum GPU performance

## Current Implementation Status

### Completed
- ✅ `gpu_node_getVolume()` - Volume from depth
- ✅ `gpu_getFloodedDepth()` - Flooding logic
- ✅ `gpu_setNodeDepth()` - Node depth calculation
- ✅ `kernel_findNodeDepths()` - Node depth kernel

### In Progress (Phase 4 - Stage 1)
- ⏳ Cross-section helper functions (getArea, getHydRad, getSlotWidth)
- ⏳ Simplified `gpu_findConduitFlow()` device function
- ⏳ `kernel_findConduitFlows()` GPU kernel

### Pending
- Node inflow/outflow update kernel
- Convergence reduction kernel (GPU reduction)
- Full feature implementation (Stage 2)

## Code Structure

```
gpu_xsect_helpers.cuh      // Cross-section calculations (getArea, getHydRad, etc.)
gpu_conduit_helpers.cuh    // Conduit-specific helpers (findSurfArea, etc.)
gpu_dwflow_kernels.cuh     // Main conduit flow device function
gpu_dwflow.cu              // Kernel launch wrapper
```

## Performance Expectations

**Model Size:** 1000 conduits
**Iterations:** ~5-10 per timestep
**CPU Time (serial):** ~10ms per iteration
**GPU Time (target):** <1ms per iteration (10x speedup)

**DGX Spark Advantages:**
- 120GB unified memory - can handle massive models
- 48 SMs - excellent parallelism for 1000+ conduits
- Compute 12.1 - latest features (dynamic parallelism, etc.)

## Testing Strategy

1. **Unit Tests:** Test each helper function individually
2. **Single Conduit:** Validate one conduit CPU vs GPU
3. **Small Model:** 10-100 conduits, verify bit-exact results
4. **Large Model:** 1000+ conduits, verify convergence and performance
5. **Edge Cases:** Dry conduits, reverse flow, surcharging

## Timeline

- **Stage 1 (Simplified):** 2-3 days
- **Stage 2 (Full Features):** 3-4 days
- **Stage 3 (Optimization):** 2-3 days

**Total Phase 4:** 7-10 days

## Risks

| Risk | Mitigation |
|------|------------|
| Numerical divergence from CPU | Rigorous validation, bit-exact comparison |
| Complex branching hurts GPU performance | Use warp-level intrinsics, minimize divergence |
| Many helper function dependencies | Implement incrementally, test thoroughly |

## Success Criteria

✅ GPU results match CPU within floating-point tolerance
✅ Handles all flow regimes (subcritical, supercritical, mixed)
✅ Proper surcharging with Preissmann slot
✅ 5-10x speedup on models with 1000+ conduits

---

**Next Actions:**
1. Implement cross-section helpers (getArea, getHydRad, getSlotWidth)
2. Create simplified gpu_findConduitFlow() for regular conduits
3. Implement kernel_findConduitFlows() kernel
4. Add basic validation tests
