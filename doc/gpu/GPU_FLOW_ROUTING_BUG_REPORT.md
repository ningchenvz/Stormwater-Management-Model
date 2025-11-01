# GPU Flow Routing Bug Report & Test Case Documentation

**Report Date:** 2025-10-26
**Status:** BLOCKING - Critical issues in GPU flow routing implementation
**Severity:** CRITICAL - Complete flow routing failure on realistic network
**Reproducibility:** 100% - Consistent on model_full_features test network

---

## Executive Summary

The GPU implementation of dynamic wave flow routing produces **completely incorrect results** on a realistic 4-node stormwater network with storage, pump, and weir elements. The GPU fails to route water through downstream elements, resulting in:

- **Zero outfall discharge** (14.5 M gal disappears)
- **Storage that never drains** (fills to 100% and stays full)
- **Pump that never operates** (0% utilization)
- **Weir that never activates** (0 CFS flow)
- **9.49% continuity error** (water conservation violated)
- **147x performance penalty** (GPU slower than CPU)

---

## Test Infrastructure

Created 6 individual test cases in `/tests/gpu/` directory to isolate and verify each issue:

### Test Execution

```bash
cd ~/workspace/Stormwater-Management-Model

# Run individual tests
bash tests/gpu/test_gpu_zero_outfall_flow.sh
bash tests/gpu/test_gpu_storage_not_draining.sh
bash tests/gpu/test_gpu_pump_not_operating.sh
bash tests/gpu/test_gpu_weir_flow_blocked.sh
bash tests/gpu/test_gpu_mass_balance_failure.sh
bash tests/gpu/test_gpu_performance_degradation.sh

# Run all GPU tests
for test in tests/gpu/test_gpu_*.sh; do bash "$test" || echo "FAILED: $test"; done
```

---

## Test Cases

### Test 1: Zero Outfall Flow

**File:** `tests/gpu/test_gpu_zero_outfall_flow.sh`

**Issue:** Outfall J4 receives 0.000 M gal instead of 14.528 M gal

**Root Cause:** Water blocked at downstream elements (pump/weir not discharging)

**Metrics Tested:**
- Outfall J4 frequency (CPU: 100%, GPU: 0%) ❌
- Outfall J4 total volume (CPU: 14.528 M gal, GPU: 0.000 M gal) ❌
- System total volume (CPU: 14.528 M gal, GPU: 0.000 M gal) ❌

**Test Passes When:**
- GPU outfall frequency >= 99%
- GPU outfall volume >= 14.0 M gal

**Code Location:** `src/solver/gpu/gpu_dwflow.cu` - Kernel link processing
- Line 305: Link type filtering `isTrueConduit(j)`
- Line 325-356: Kernel calls helper for each link

**Investigation Checklist:**
- [ ] Verify pump (C2) produces output flow
- [ ] Verify weir (C3) receives input flow
- [ ] Verify J4 node receiving upstream flow
- [ ] Check flow classification at each link
- [ ] Verify head calculations propagate to J4

---

### Test 2: Storage Not Draining

**File:** `tests/gpu/test_gpu_storage_not_draining.sh`

**Issue:** Storage J2 final volume 14.942 1000 ft³ instead of 0.001 1000 ft³

**Root Cause:** Pump not discharging water (zero flow calculation)

**Metrics Tested:**
- Storage J2 max depth (CPU: 0.00 ft, GPU: 15.00 ft) ❌
- Storage J2 final volume (CPU: 0.001 1000 ft³, GPU: 14.942 1000 ft³) ❌
- Storage J2 flooding hours (CPU: 0 hrs, GPU: 57.56 hrs) ❌
- Storage J2 outflow (CPU: 9.41 CFS, GPU: 0.00 CFS) ❌

**Test Passes When:**
- GPU storage max depth < 1.0 ft
- GPU storage final volume < 0.1 1000 ft³
- GPU storage flooding hours = 0

**Code Location:** `src/solver/gpu/gpu_dwflow.cu` - Pump flow calculation
- Lines 269-375: Kernel processes all link types
- Line 305: **ISSUE:** May skip PUMP links with `isTrueConduit()` filter

**Investigation Checklist:**
- [ ] Check pump link type handling in GPU kernel
- [ ] Verify pump flow calculation code exists
- [ ] Check pump discharge head calculation
- [ ] Verify pump curve/rating transferred to GPU
- [ ] Look for pump-specific momentum equation
- [ ] Check if pump data copied to device

---

### Test 3: Pump Not Operating

**File:** `tests/gpu/test_gpu_pump_not_operating.sh`

**Issue:** Pump C2 at 0% utilization instead of 100% utilization

**Root Cause:** GPU kernel missing PUMP link type handling

**Metrics Tested:**
- Pump C2 utilization (CPU: 100%, GPU: 0%) ❌
- Pump C2 avg flow (CPU: 8.25 CFS, GPU: 0.00 CFS) ❌
- Pump C2 max flow (CPU: 9.41 CFS, GPU: 0.00 CFS) ❌
- Pump C2 total volume (CPU: 12.888 M gal, GPU: 0.000 M gal) ❌
- Pump C2 power usage (CPU: 196.88 Kw-hr, GPU: 0.00 Kw-hr) ❌
- Pump start-ups (CPU: 1, GPU: 0) ❌

**Test Passes When:**
- GPU pump utilization >= 95%
- GPU pump avg flow >= 7.5 CFS
- GPU pump total volume >= 11.0 M gal

**Code Location:** `src/solver/gpu/gpu_dwflow.cu` + `src/solver/dwflow.c`
- GPU: Line 305 link type filter (may exclude PUMP)
- CPU: `dwflow_findPumpFlow()` exists for PUMP handling
- GPU: Equivalent function missing?

**Investigation Checklist:**
- [ ] Check `isTrueConduit()` filter - does it exclude PUMP?
- [ ] Look for pump flow calculation in GPU kernel
- [ ] Compare with CPU `dwflow_findPumpFlow()` implementation
- [ ] Verify pump data structure transferred to GPU
- [ ] Check if PUMP link type constant defined in GPU
- [ ] Look for pump curve evaluation in GPU

---

### Test 4: Weir Flow Blocked

**File:** `tests/gpu/test_gpu_weir_flow_blocked.sh`

**Issue:** Weir C3 carries 0.00 CFS instead of 12.12 CFS

**Root Cause:** GPU kernel missing WEIR link type handling

**Metrics Tested:**
- Weir C3 max flow (CPU: 12.12 CFS, GPU: 0.00 CFS) ❌
- Weir C3 flow occurrence (CPU: 0 10:00, GPU: 0 00:00) ❌
- Weir activation (CPU: active, GPU: never activated) ❌

**Test Passes When:**
- GPU weir max flow >= 11.0 CFS (allowing 2% tolerance)
- GPU weir carries positive flow at some point

**Code Location:** `src/solver/gpu/gpu_dwflow.cu` + `src/solver/dwflow.c`
- GPU: Line 305 link type filter (may exclude WEIR)
- CPU: `dwflow_findWeirFlow()` exists for WEIR handling
- GPU: Equivalent function missing?

**Investigation Checklist:**
- [ ] Check `isTrueConduit()` filter - does it exclude WEIR?
- [ ] Look for weir flow calculation in GPU kernel
- [ ] Compare with CPU `dwflow_findWeirFlow()` implementation
- [ ] Verify weir geometry transferred to GPU
- [ ] Check weir crest elevation vs node head
- [ ] Look for weir-specific flow equation

---

### Test 5: Mass Balance Failure

**File:** `tests/gpu/test_gpu_mass_balance_failure.sh`

**Issue:** Continuity error 9.490% instead of 0.001%

**Root Cause:** Water loss due to flow routing failures (pump/weir inactive)

**Metrics Tested:**
- Continuity error (CPU: 0.001%, GPU: 9.490%) ❌
- External outflow (CPU: 44.585 M gal, GPU: 0.000 M gal) ❌
- Flooding loss (CPU: 8.619 M gal, GPU: 47.809 M gal) ❌
- Final stored volume (CPU: 0.002 M gal, GPU: 0.349 M gal) ❌
- Highest error node (GPU J3 shows 100.00% error) ❌

**Test Passes When:**
- GPU continuity error < 0.1% (excellent)
- GPU continuity error < 0.5% (acceptable)

**Code Location:** `src/solver/dynwave.c` + `src/solver/node.c`
- Line 655-680: Node depth calculation loop
- Related: Flow conservation equation at each node
- Issue: If pump/weir produce zero flow, node water balance breaks

**Investigation Checklist:**
- [ ] Verify node depth equation in GPU
- [ ] Check flow continuity implementation
- [ ] Look for water "lost" in pump/weir
- [ ] Examine if all flows recorded correctly
- [ ] Check if convergence check correct
- [ ] Find nodes with 100% error (local failure indicator)

---

### Test 6: Performance Degradation

**File:** `tests/gpu/test_gpu_performance_degradation.sh`

**Issue:** GPU 147x slower (146 sec vs 1 sec)

**Root Cause:** GPU overhead exceeds benefit (unified memory transfers + kernel overhead)

**Metrics Tested:**
- CPU execution time: ~1 second
- GPU execution time: ~146 seconds
- Speedup ratio: 0.0068x (147x SLOWER)
- Iterations per step: CPU 2.00, GPU 2.01 (same convergence)

**Test Passes When:**
- GPU speedup >= 1.0x (break even with CPU)
- Preferably GPU speedup >= 5.0x (expected on modern GPUs)

**Code Location:** `src/solver/gpu/gpu_dwflow.cu` - Kernel orchestration
- Lines 435-513: `gpu_computeConduitFlows()` host function
- Lines 51-107: Data transfer functions
- Lines 269-375: Kernel execution

**Investigation Checklist:**
- [ ] Profile kernel execution time vs data transfer
- [ ] Check unified memory page migration cost
- [ ] Verify kernel actually running on GPU
- [ ] Measure GPU memory bandwidth usage
- [ ] Check synchronization frequency
- [ ] Examine if problem too small for GPU
- [ ] Look for implicit synchronization points

---

## Critical Issues Summary

| Test # | Issue | CPU | GPU | Status | Severity |
|--------|-------|-----|-----|--------|----------|
| **1** | Zero Outfall Flow | 14.528 M gal | 0.000 M gal | ❌ FAIL | CRITICAL |
| **2** | Storage Not Draining | 0.001 1000ft³ | 14.942 1000ft³ | ❌ FAIL | CRITICAL |
| **3** | Pump Not Operating | 100% util | 0% util | ❌ FAIL | CRITICAL |
| **4** | Weir Flow Blocked | 12.12 CFS | 0.00 CFS | ❌ FAIL | CRITICAL |
| **5** | Mass Balance Failure | 0.001% error | 9.490% error | ❌ FAIL | CRITICAL |
| **6** | Performance Degradation | 1 sec | 146 sec | ❌ FAIL | HIGH |

---

## Root Cause Analysis

### Primary Hypothesis: Link Type Filtering Issue ⚠️ MOST LIKELY

**Location:** `src/solver/gpu/gpu_dwflow.cu` Line 305

```cuda
if (link_j < Nobjects[LINK] && isTrueConduit(j))
```

**Problem:** GPU kernel may only handle CONDUIT types and skip PUMP/WEIR

**Evidence:**
- C1:C2 (CONDUIT): Shows ~9 CFS flow ✓
- C2 (PUMP): Shows 0 CFS flow ❌
- C3 (WEIR): Shows 0 CFS flow ❌
- Pattern: Link type determines if GPU processes it

**CPU Reference:** `src/solver/dwflow.c` handles all types:
- `dwflow_findConduitFlow()` - CONDUIT
- `dwflow_findPumpFlow()` - PUMP
- `dwflow_findWeirFlow()` - WEIR
- `dwflow_findOrificeFlow()` - ORIFICE
- `dwflow_findOutletFlow()` - OUTLET

**GPU Missing:** Pump and weir implementations

### Secondary Hypothesis: Downstream Flow Dependency Loop

**Problem:** First link succeeds, but dependent downstream links fail

**Evidence:**
- Partial success (some links work, others don't)
- Suggests water can reach storage but can't leave
- Pump failure blocks all downstream

**Physics:** If pump computes zero flow, then:
- J3 receives zero inflow (should get pump outflow)
- Weir sees zero head difference (storage full, J3 empty)
- Weir calculates zero flow
- J4 receives zero flow
- Cascading failure through network

---

## Affected Code Files

### Primary Implementation

| File | Purpose | Issues |
|------|---------|--------|
| `src/solver/gpu/gpu_dwflow.cu` | Main GPU kernel | Link type filtering may skip PUMP/WEIR |
| `src/solver/gpu/gpu_conduit_helpers.cuh` | Momentum equation | Only handles CONDUIT case |
| `src/solver/dynwave.c` | Integration point | GPU kernel integration at line 410 |
| `src/solver/gpu/gpu_structures.h` | Data structures | May be missing PUMP/WEIR fields |
| `src/solver/gpu/gpu_memory.cu` | GPU memory management | May not allocate PUMP/WEIR data |

### Reference Implementation

| File | Purpose | GPU Should Match |
|------|---------|-----------------|
| `src/solver/dwflow.c` | Official CPU implementation | 5 different link types handled |
| `src/solver/pump.c` | Pump calculations | GPU needs equivalent |
| `src/solver/weir.c` | Weir calculations | GPU needs equivalent |

---

## Expected Behavior (CPU - Correct)

**Network Flow:**
1. Rainfall generates runoff (14.1 M gal total)
2. Runoff enters J1 (receives full 9.61 CFS peak) ✓
3. Flows through C1:C2 conduit to J2 storage (up to 7.22 CFS) ✓
4. **Pump (C2) activates and discharges from J2**
   - Pushes water from J2 to J3
   - Operates at ~9.41 CFS average
   - Total volume: 12.9 M gal pumped
5. **Weir (C3) spills excess from J3**
   - Spillway activated when J3 exceeds threshold
   - Carries peak 12.12 CFS
   - Routes to outfall J4
6. **Outfall J4 discharges all flow**
   - 100% frequency (always flowing)
   - Total volume: 14.528 M gal (matches input)
7. Storage J2 stays nearly empty (pump keeps draining)
8. System achieves 0.001% continuity error (excellent)

**Result: All water accounted for, system stable, physics correct** ✓

---

## Actual Behavior (GPU - Broken)

**Network Flow:**
1. Rainfall generates runoff (14.1 M gal total) ✓
2. Runoff enters J1 (receives full 9.61 CFS peak) ✓
3. Flows through C1:C2 conduit to J2 storage (similar to CPU) ✓
4. **Pump (C2) FAILS - computes zero flow**
   - GPU kernel skips or miscalculates pump
   - Water cannot leave J2
5. **Storage J2 fills completely and STAYS FULL**
   - Inflow accumulates (no outflow via pump)
   - Storage reaches 100% capacity (15.00 ft)
   - Water overflows/floods from J2
6. **Weir (C3) FAILS - receives zero input**
   - No water from J3 (pump didn't supply it)
   - Weir carries 0.00 CFS
   - Outfall not activated
7. **Outfall J4 receives ZERO flow**
   - 0% frequency (never flows)
   - Total volume: 0.000 M gal received
   - ~14.5 M gal unaccounted for
8. System shows 9.49% continuity error (water conservation violated)
9. **Water loss:** 44.6 M gal external outflow (CPU) → 0.0 M gal (GPU)

**Result: Complete routing failure, mass loss, physics violated** ❌

---

## Test Network Specifications

### Network Topology

```
Subcatchments:
  S1: 1 acre   ┐
  S2: 2 acres  ├→ J1(Junction) → C1:C2(Conduit 244.6 ft, Ø=1.0 ft)
  S3: 3 acres  ┘                     ↓
                                    J2(Storage 15 ft capacity)
                                     ↓
                                    C2(Pump, ideal)
                                     ↓
                                    J3(Junction)
                                     ↓
                                    C3(Weir)
                                     ↓
                                    J4(Outfall)
```

### Node Properties

| Node | Type | Invert Elev | Max Depth | Capacity | Purpose |
|------|------|-------------|-----------|----------|---------|
| J1 | Junction | 20.73 ft | 15.00 ft | - | Inlet from subcatchments |
| J2 | Storage | 13.39 ft | 15.00 ft | 15,000 ft³ | Detention basin |
| J3 | Junction | 6.55 ft | 15.00 ft | - | Pump discharge + weir inlet |
| J4 | Outfall | 0.00 ft | 0.00 ft | - | System outlet |

### Link Properties

| Link | Type | From | To | Properties |
|------|------|------|----|----|
| C1:C2 | Conduit | J1 | J2 | 244.6 ft long, circular 1.0 ft dia, Manning n=0.01 |
| C2 | Pump | J2 | J3 | Ideal pump (proportional to head) |
| C3 | Weir | J3 | J4 | Spillway weir |

### Simulation Configuration

- **Model:** `tests/test_models/model_full_features.inp`
- **Duration:** 58 hours (11/01/2015 14:00 to 11/04/2015 00:00)
- **Rainfall:** SCS 24-hour Type I storm (1 inch total)
- **Total Runoff:** 14.1 million gallons
- **Routing Method:** DYNWAVE
- **Timestep:** 1.0 second
- **Convergence:** Head tolerance 0.005 ft

---

## Code Locations - Detailed Reference

### GPU Implementation

**gpu_dwflow.cu:**
```cuda
// Line 305: CRITICAL - Link type filtering
if (link_j < Nobjects[LINK] && isTrueConduit(j)) {
    // This may SKIP PUMP and WEIR links!
}

// Lines 312-320: Cross-section creation (only for conduits?)
GPU_Xsect xsect = gpuXsects[k];

// Lines 325-356: Call to helper function
gpu_findConduitFlow_simplified(...);

// Lines 435-513: Host function orchestrating kernel
gpu_computeConduitFlows(...)
```

**gpu_conduit_helpers.cuh:**
```cuda
// Lines 88-284: gpu_findConduitFlow_simplified()
// Only implements CONDUIT momentum equation
// Missing PUMP and WEIR flow calculations

// Line 245: Energy slope term
dq2 = dt * GPU_GRAVITY * aWtd * (h2 - h1) / length;
```

**dynwave.c:**
```c
// Lines 410-428: GPU kernel call
if (useCUDA && gpu_dwflow_computeConduitFlows) {
    gpu_computeConduitFlows(...);
}

// Lines 435-443: CPU fallback (OpenMP)
#pragma omp parallel
{
    for (i = 0; i < Nobjects[LINK]; i++) {
        if (isTrueConduit(i)) {
            dwflow_findConduitFlow(i, ...);
        }
    }
}
```

### CPU Reference

**dwflow.c - Handles ALL link types:**
```c
// Lines 79-283: dwflow_findConduitFlow()
// Called for CONDUIT links

// Equivalent functions for other types:
// - dwflow_findPumpFlow() - PUMP
// - dwflow_findWeirFlow() - WEIR
// - dwflow_findOrificeFlow() - ORIFICE
// - dwflow_findOutletFlow() - OUTLET
```

---

## Reproduction Commands

### Quick Reproduction

```bash
cd ~/workspace/Stormwater-Management-Model/build

# CPU baseline (correct behavior)
export SWMM_USE_CUDA=0
./bin/runswmm ../tests/test_models/model_full_features.inp cpu.rpt cpu.out
grep "External Outflow" cpu.rpt        # Shows 44.585 M gal
grep "J4" cpu.rpt | grep -A 5 "Outfall"  # Shows 100% frequency

# GPU test (broken behavior)
export SWMM_USE_CUDA=1
./bin/runswmm ../tests/test_models/model_full_features.inp gpu.rpt gpu.out
grep "External Outflow" gpu.rpt        # Shows 0.000 M gal
grep "J4" gpu.rpt | grep -A 5 "Outfall"  # Shows 0% frequency
```

### Detailed Comparison

```bash
# Storage behavior
echo "=== STORAGE VOLUMES ==="
diff <(grep -A 5 "Storage Volume Summary" cpu.rpt) \
     <(grep -A 5 "Storage Volume Summary" gpu.rpt)

# Pump operation
echo "=== PUMP OPERATION ==="
diff <(grep -A 5 "Pumping Summary" cpu.rpt) \
     <(grep -A 5 "Pumping Summary" gpu.rpt)

# Outfall performance
echo "=== OUTFALL DISCHARGE ==="
diff <(grep -A 5 "Outfall Loading Summary" cpu.rpt) \
     <(grep -A 5 "Outfall Loading Summary" gpu.rpt)

# Flow classification
echo "=== FLOW CLASSIFICATION ==="
diff <(grep -A 10 "Flow Classification Summary" cpu.rpt) \
     <(grep -A 10 "Flow Classification Summary" gpu.rpt)

# Mass balance
echo "=== MASS BALANCE ==="
diff <(grep -B 2 -A 10 "Flow Routing Continuity" cpu.rpt) \
     <(grep -B 2 -A 10 "Flow Routing Continuity" gpu.rpt)
```

---

## Investigation Tasks

### Task 1: Identify Link Type Handling

**Objective:** Determine if GPU kernel processes PUMP/WEIR or only CONDUIT

```bash
# Check GPU kernel link filtering
grep -n "isTrueConduit" src/solver/gpu/gpu_dwflow.cu

# Check what isTrueConduit includes
grep -A 5 "isTrueConduit" src/solver/link.c

# Compare with CPU which handles all types
grep -n "dwflow_findPump\|dwflow_findWeir\|dwflow_findOrifice" \
    src/solver/dwflow.c
```

**Expected Finding:** GPU only handles CONDUIT, missing PUMP/WEIR

### Task 2: Locate Pump Flow Calculation

**Objective:** Find or create pump flow calculation for GPU

```bash
# Look for pump function in CPU
grep -B 5 -A 30 "dwflow_findPumpFlow" src/solver/dwflow.c

# Check if GPU has equivalent
grep -r "pump" src/solver/gpu/

# Check pump rating curve handling
grep -n "pump_getCurve\|pump_getFlow" src/solver/
```

**Expected Finding:** CPU has `dwflow_findPumpFlow()`, GPU missing equivalent

### Task 3: Locate Weir Flow Calculation

**Objective:** Find or create weir flow calculation for GPU

```bash
# Look for weir function in CPU
grep -B 5 -A 30 "dwflow_findWeirFlow" src/solver/dwflow.c

# Check if GPU has equivalent
grep -r "weir" src/solver/gpu/

# Check weir discharge formula
grep -n "weir_getFlow\|weir_getDepth" src/solver/
```

**Expected Finding:** CPU has `dwflow_findWeirFlow()`, GPU missing equivalent

### Task 4: Check GPU Data Transfer

**Objective:** Verify pump/weir properties transferred to GPU

```bash
# Check what data copied to GPU
grep -A 10 "copyLinksToGpu\|copyPumpData\|copyWeirData" \
    src/solver/gpu/gpu_dwflow.cu

# Verify link type info transferred
grep "link_type\|linkType" src/solver/gpu/gpu_structures.h
```

**Expected Finding:** Pump/weir-specific data may not be copied

### Task 5: Review Momentum Equation

**Objective:** Verify momentum equation correct for all link types

```bash
# GPU momentum equation
grep -B 5 -A 30 "dq2.*energy" \
    src/solver/gpu/gpu_conduit_helpers.cuh

# CPU momentum equation
grep -B 5 -A 30 "dq2.*energy" src/solver/dwflow.c
```

**Expected Finding:** GPU may need different equations for PUMP/WEIR

---

## Next Steps (Divide & Conquer)

### Step 1: Link Type Investigation
- [ ] Run individual tests to confirm which link types fail
- [ ] Examine GPU kernel link filtering code
- [ ] Compare with CPU implementation of all 5 link types

### Step 2: Pump Implementation
- [ ] Create `gpu_findPumpFlow()` function
- [ ] Copy pump curve/rating to GPU device
- [ ] Implement pump discharge calculation
- [ ] Test with `test_gpu_pump_not_operating.sh`

### Step 3: Weir Implementation
- [ ] Create `gpu_findWeirFlow()` function
- [ ] Copy weir geometry to GPU device
- [ ] Implement weir discharge formula
- [ ] Test with `test_gpu_weir_flow_blocked.sh`

### Step 4: Validation
- [ ] Run all 6 test cases
- [ ] Verify storage drains (Test 2)
- [ ] Verify outfall flows (Test 1)
- [ ] Verify mass balance (Test 5)
- [ ] Analyze performance (Test 6)

### Step 5: Performance Optimization
- [ ] Profile GPU kernel execution
- [ ] Measure unified memory transfer cost
- [ ] Optimize synchronization points
- [ ] Target 5-15x speedup (expected for GPU)

---

## Files for Reference

**GPU Test Cases:**
- `tests/gpu/test_gpu_zero_outfall_flow.sh` - Test 1
- `tests/gpu/test_gpu_storage_not_draining.sh` - Test 2
- `tests/gpu/test_gpu_pump_not_operating.sh` - Test 3
- `tests/gpu/test_gpu_weir_flow_blocked.sh` - Test 4
- `tests/gpu/test_gpu_mass_balance_failure.sh` - Test 5
- `tests/gpu/test_gpu_performance_degradation.sh` - Test 6

**GPU Implementation:**
- `src/solver/gpu/gpu_dwflow.cu` - Main kernel
- `src/solver/gpu/gpu_conduit_helpers.cuh` - Momentum equation
- `src/solver/dynwave.c` - Integration

**CPU Reference:**
- `src/solver/dwflow.c` - Official implementation (all 5 link types)
- `src/solver/pump.c` - Pump calculations
- `src/solver/weir.c` - Weir calculations

**Test Data:**
- `tests/test_models/model_full_features.inp` - Test network
- `/build/gpu_cpu_compare/model_full_features_20251026-173359/` - Comparison results

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Blocking Status** | YES - Critical issues prevent GPU use |
| **Test Coverage** | 6 individual test cases covering all failure modes |
| **Root Cause** | GPU kernel likely skips PUMP/WEIR link types |
| **Affected Systems** | Pump, Weir, Outfall, Storage drainage |
| **Mass Loss** | 44.6 M gal disappears (100% failure) |
| **Continuity Error** | 9.49% (physics violation) |
| **Performance** | 147x slower than CPU (no benefit) |
| **Fix Complexity** | Moderate - need to add pump/weir handling to GPU kernel |
| **Test Reproducibility** | 100% - every run shows same issues |

