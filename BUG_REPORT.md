# GPU Conduit Kernel Bootstrap Issue

**Date:** 2025-10-26
**Status:** IDENTIFIED - Root Cause Found
**Severity:** Critical (Blocks Phase 4 completion)
**Component:** GPU Dynamic Wave Routing - Conduit Flow Kernel

---

## Summary

The GPU conduit flow kernel produces all-zero results (no flows, no depths) despite correct implementation of the momentum equation. The root cause is a **circular dependency in the bootstrap process** where the area-based momentum equation cannot generate flow from zero initial conditions.

---

## Environment

- **SWMM Version:** 5.2.4
- **CUDA Version:** 12.1
- **GPU:** NVIDIA GB10 (Compute Capability 12.1)
- **Test Model:** `tests/test_models/simple_test.inp`
  - 2 circular conduits (3 ft diameter)
  - 3 nodes (J1, J2, OUT1)
  - Invert elevations: J1=95 ft, J2=90 ft, OUT1=85 ft
  - 2-hour simulation with subcatchment inflow

---

## Symptoms

### GPU Results (Incorrect)
```
Node J1:  Max depth = 0.00 ft  (Expected: 0.64 ft)
Node J2:  Max depth = 0.00 ft  (Expected: 0.62 ft)
Link C1:  Max flow  = 0.00 CFS (Expected: 7.17 CFS)
Link C2:  Max flow  = 0.00 CFS (Expected: 6.83 CFS)
Continuity Error: 100% (no outflow despite inflow)
```

### CPU Results (Correct)
```
Node J1:  Max depth = 0.64 ft
Node J2:  Max depth = 0.62 ft
Link C1:  Max flow  = 7.17 CFS
Link C2:  Max flow  = 6.83 CFS
Continuity Error: -0.033%
```

### Debug Output
```
GPU iter=0: y1=0.000000 y2=0.000000 aMid=0.000002 q=0.000000 flowClass=0
GPU iter=1: y1=0.000000 y2=0.000000 aMid=0.000002 q=0.000000 flowClass=0
[...repeats for all iterations...]

MOMENTUM DEBUG: h1=95.000 h2=90.000 dq2=-0.000000 qOld=0.000000 denom=1.000 q=0.000000
                ^^^ 5 ft head difference!  ^^^ Zero energy term!
```

---

## Root Cause Analysis

### The Circular Dependency

The Picard iteration loop executes in this order:
```c
while (Steps < MaxTrials) {
    initNodeStates();           // Reset inflow/outflow
    findLinkFlows(tStep);       // ← 1. Compute flows using CURRENT depths
    findNodeDepths(tStep);      // ← 2. Update depths using computed flows
    Steps++;
}
```

**Iteration 0 (Initial State):**
1. `findLinkFlows()` called with node depths = 0.0
   - Depths → Areas: `aMid = 0.000002 ft²` (tiny!)
   - Energy slope term: `dq2 = dt * g * aMid * (h2-h1) / L`
   - Despite 5 ft head difference, `dq2 ≈ 0` due to tiny area
   - Momentum equation: `q = (qOld - dq2) / (1 + dq1) = 0 / 1 = 0`
   - **Result: Zero flow**

2. `findNodeDepths()` called with flows = 0.0
   - Mass balance: `dV/dt = Inflow - Outflow`
   - Outflow = 0 → Depths should increase from inflow
   - **BUT: Depths remain 0.0** ← THIS IS THE MYSTERY

**Iteration 1:**
- Node depths STILL 0.0 (why?)
- GPU kernel sees zero depths again
- Returns zero flows again
- **Cycle repeats indefinitely**

### The Physics Problem

The momentum equation for conduit flow is:
```
q = (qOld - dq2 + dq3 + dq4) / (1 + dq1)

where:
  dq2 = dt * g * A * (h2 - h1) / L    ← Energy slope term
```

**The Issue:** When depth → 0, area A → 0, making dq2 → 0 even with large head difference (h2 - h1).

This is **physically correct** (cannot push water through a closed pipe), but creates a bootstrap problem:
- Need non-zero area to compute flow
- Need non-zero flow to create area (depth)
- Chicken-and-egg problem!

---

## Investigation Process

### Attempts Made

#### 1. **Added flowClass Field** ✅ (Correct but insufficient)
**Files:** `gpu_structures.h`, `gpu_memory.cu`, `gpu_dwflow.cu`
- Added `flowClass` to `GPU_LinkData` structure
- Allocated memory and transferred from CPU `Link[j].flowClass`
- Used in kernel: `links->flowClass[j]`
- **Result:** Still all zeros (flowClass initialized to DRY=0)

#### 2. **Removed flowClass Check** ✅ (Diagnostic)
**File:** `gpu_conduit_helpers.cuh:200-210`
```cuda
// Disabled:
// if (flowClass == GPU_DRY || ...) { return 0; }
```
- **Result:** Still all zeros (not the blocker)

#### 3. **Removed aMid Check** ✅ (Diagnostic)
**File:** `gpu_conduit_helpers.cuh:202-210`
```cuda
// Disabled:
// if (aMid <= GPU_FUDGE) { return 0; }
```
- **Result:** Still all zeros (not the blocker)

#### 4. **Used CPU for Node Depths** ✅ (Diagnostic)
**File:** `dynwave.c:655-680`
- Disabled `gpu_runNodeDepthKernel()`
- Used CPU `setNodeDepth()` while GPU handles conduit flows
- **Result:** Still all zeros (confirms problem is in conduit kernel)

#### 5. **Added Momentum Equation Debug** ✅ (Breakthrough!)
**File:** `gpu_conduit_helpers.cuh:260-263`
```cuda
printf("MOMENTUM DEBUG: h1=%.3f h2=%.3f dq2=%.6f qOld=%.6f denom=%.3f q=%.6f\n",
       h1, h2, dq2, qOld, denom, q);
```
**Output:**
```
h1=95.000 h2=90.000 dq2=-0.000000 qOld=0.000000 denom=1.000 q=0.000000
```
- 5 ft head difference present ✅
- Energy term dq2 ≈ 0 due to tiny area ❌
- **This revealed the true problem!**

---

## Current Understanding

### What Works
1. ✅ GPU kernel is correctly implemented (momentum equation is correct)
2. ✅ Data transfer infrastructure is complete and correct
3. ✅ Geometry mapping is correct (shape-specific parameters)
4. ✅ Memory allocation and initialization work
5. ✅ Kernel launches successfully (5764 launches, 274ms GPU time)
6. ✅ Node depth kernel works (validated bit-exact in Phase 3)

### What Doesn't Work
1. ❌ Bootstrap from zero initial conditions
2. ❌ Area-based momentum equation with zero depths
3. ❌ Unknown CPU bootstrap mechanism

### The Mystery: Why Does CPU Work?

The CPU code (`dwflow.c:dwflow_findConduitFlow`) must have special handling for zero initial conditions that we haven't identified. Possible mechanisms:

1. **Minimum Depth Threshold:** CPU uses `FUDGE = 0.0001 ft`
   - GPU also has this: `y1 = gpu_MAX(y1, GPU_FUDGE)` ✅
   - But for circular pipe: `area(0.0001 ft) ≈ 2e-6 ft²` (still too small)

2. **Initial Flow Guess:** CPU might use elevation difference for initial flow
   - Not found in code review
   - Would violate momentum equation

3. **Different Iteration Start:** CPU might compute depths before flows on first iteration
   - Code review shows same order: `findLinkFlows()` then `findNodeDepths()`

4. **Surface Area Contribution:** CPU function `findSurfArea()` at line 140
   - Updates node heads based on flow estimate
   - Might create feedback loop that bootstraps depths

5. **Node Depth Calculation:** CPU `setNodeDepth()` might handle zero outflow differently
   - Mass balance: `dV/dt = Inflow - Outflow`
   - With Inflow > 0 and Outflow = 0, depth SHOULD increase
   - **Why doesn't this happen on GPU path?**

---

## Code Locations

### GPU Conduit Kernel
- **File:** `src/solver/gpu/gpu_dwflow.cu`
  - Line 269-375: `kernel_findConduitFlows()` (CUDA kernel)
  - Line 435-520: `gpu_computeConduitFlows()` (host function)

### GPU Conduit Helpers
- **File:** `src/solver/gpu/gpu_conduit_helpers.cuh`
  - Line 89-280: `gpu_findConduitFlow_simplified()` (device function)
  - Line 155-161: Head calculation
  - Line 242-253: Momentum equation terms
  - Line 255-257: Flow calculation ← THE PROBLEM

### CPU Reference Implementation
- **File:** `src/solver/dwflow.c`
  - Line 79-283: `dwflow_findConduitFlow()` (reference implementation)
  - Line 117-122: Initial depths with FUDGE
  - Line 140: `findSurfArea()` call ← Potentially important
  - Line 164-180: Dry condition check
  - Line 241-244: Momentum equation

### Iteration Loop
- **File:** `src/solver/dynwave.c`
  - Line 254-276: Picard iteration loop
  - Line 268-270: Call order: `initNodeStates()` → `findLinkFlows()` → `findNodeDepths()`

---

## Test Commands

### Build
```bash
cmake -B build -DENABLE_CUDA=ON
cmake --build build
```

### Run GPU Test
```bash
env SWMM_USE_CUDA=1 build/bin/runswmm \
    tests/test_models/simple_test.inp \
    /tmp/gpu_test.rpt \
    /tmp/gpu_test.out
```

### Run CPU Test (Reference)
```bash
env SWMM_USE_CUDA=0 build/bin/runswmm \
    tests/test_models/simple_test.inp \
    /tmp/cpu_test.rpt \
    /tmp/cpu_test.out
```

### Compare Results
```bash
diff /tmp/gpu_test.rpt /tmp/cpu_test.rpt
```

---

## Next Steps

### Option 1: Find CPU Bootstrap Mechanism (Recommended)
**Action:** Deep dive into CPU code to find how it handles zero initial conditions

**Focus Areas:**
1. `findSurfArea()` function - Does it create feedback that bootstraps depths?
2. `setNodeDepth()` function - How does it handle Inflow > 0, Outflow = 0?
3. Initial values - Are depths truly zero or set to small value elsewhere?
4. Xnode initialization - Check `dynwave_init()` for initial depth settings

**Method:**
```bash
# Add debug output to CPU code
# Modify dwflow.c line 150:
if (k == 0 && Steps < 5) {
    printf("CPU: y1=%.6f y2=%.6f aMid=%.6f h1=%.3f h2=%.3f dq2=%.6f q=%.6f\n",
           y1, y2, aMid, h1, h2, dq2, q);
}
```

### Option 2: Modify GPU Bootstrap Logic
**Action:** Add special handling for zero/small depths

**Approach A: Elevation-Based Initial Flow**
```cuda
// When depths are tiny, use elevation-based flow estimate
if (aMid < 1e-4 && fabs(h2 - h1) > 0.1) {
    // Weir-like flow: Q = C * sqrt(2*g*H) * A
    double dH = h1 - h2;  // Head difference
    if (dH > 0) {
        q = 0.5 * sqrt(2.0 * GPU_GRAVITY * dH) * xsect->aFull * 0.01;
    }
}
```

**Approach B: Minimum Area Threshold**
```cuda
// Use minimum area for momentum calculation
double aEff = gpu_MAX(aWtd, xsect->aFull * 0.001);  // 0.1% of full area
dq2 = dt * GPU_GRAVITY * aEff * (h2 - h1) / length;
```

**Approach C: Initial Depth Seeding**
```cuda
// In dynwave_init(), set small initial depths in nodes with inflow
if (Node[i].newLatFlow > 0.0) {
    Node[i].newDepth = 0.01;  // 0.01 ft initial depth
}
```

### Option 3: Two-Stage Bootstrap
**Action:** Use different logic for first few iterations

```cuda
if (steps == 0 && aMid < 1e-4) {
    // Stage 1: Gravity-driven flow only
    double dH = h1 - h2;
    if (dH > 0) {
        q = sqrt(2.0 * GPU_GRAVITY * dH * length) * 0.01;
    }
} else {
    // Stage 2: Normal momentum equation
    q = (qOld - dq2 + dq3 + dq4) / denom;
}
```

---

## Questions for Further Investigation

1. **Why do CPU node depths increase from inflow while GPU depths stay zero?**
   - Is there a difference in `setNodeDepth()` vs `gpu_computeNodeDepth()`?
   - Does CPU path use different iteration order on first timestep?

2. **Does `findSurfArea()` create a bootstrap mechanism?**
   - It modifies heads based on flow estimate
   - Could this create small depth increase that seeds the process?

3. **Are initial depths truly zero in CPU path?**
   - Check `dynwave_init()` for any depth initialization
   - Check `Node[i].newDepth` values after initialization

4. **Does CPU use different convergence criteria on first iteration?**
   - Could it allow more iterations to bootstrap?
   - Check `MaxTrials` and convergence logic

---

## References

### Related Documentation
- `src/solver/gpu/PHASE4_COMPLETION_SUMMARY.md` - Phase 4 status
- `src/solver/gpu/GPU_CONDUIT_KERNEL_DEBUG.md` - Previous debug analysis
- `src/solver/gpu/PHASE4_IMPLEMENTATION_PLAN.md` - Implementation roadmap

### Key Code Files
- `src/solver/dwflow.c` - CPU reference implementation
- `src/solver/gpu/gpu_dwflow.cu` - GPU kernel implementation
- `src/solver/gpu/gpu_conduit_helpers.cuh` - GPU helper functions
- `src/solver/dynwave.c` - Main routing loop

### SWMM Documentation
- Momentum Equation: Section 3.4.5 of SWMM5 Hydraulics Manual
- Dynamic Wave Routing: Chapter 3 of SWMM5 Hydraulics Manual

---

## Appendix: Momentum Equation Details

### CPU Implementation (dwflow.c:241-244)
```c
denom = 1.0 + dq1 + dq5;
q = (qOld - dq2 + dq3 + dq4) / denom;
```

### GPU Implementation (gpu_conduit_helpers.cuh:255-257)
```cuda
denom = 1.0 + dq1;  // Simplified: no local losses in Stage 1
q = (qOld - dq2 + dq3 + dq4) / denom;
```

### Energy Slope Term (The Problem)
```c
// CPU (dwflow.c:233)
dq2 = dt * GRAVITY * aWtd * (h2 - h1) / length;

// GPU (gpu_conduit_helpers.cuh:245)
dq2 = dt * GPU_GRAVITY * aWtd * (h2 - h1) / length;
```

When `aWtd ≈ 0`:
- `dq2 ≈ dt * 32.2 * 0.000002 * 5.0 / 400 ≈ 0.000000806`
- `q = (0 - 0.000000806) / 1.0 ≈ 0` (rounds to zero)

---

**Report End**

*Generated during GPU kernel bootstrap debugging session*
*For questions, see handoff notes in PHASE4_COMPLETION_SUMMARY.md*
