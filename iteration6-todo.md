# Iteration 6 TODO: Pump/Storage Synchronization & Mass Balance Parity

## Context
- **Reference report:** `doc/gpu/SESSION68_GAP_ANALYSIS.md`
- **Current status (as of commit d3dd521):** Session68_46_pumps_2hr shows -241% continuity error (improved from -261%, CPU ≈ -15%).
- **Key insight:** Pump guards fire differently because GPU storage volumes (oldVolume) diverge after step 0, even though recent fixes aligned adaptive timesteps and conduit accumulation.

## ✅ Achievements (Iteration 5 → Iteration 6)

### 1. ✅ Adaptive Timestep Calculation Fixed (commit d3dd521)
- **Fixed Bug 1:** Incorrect under-relaxation - passed `yOld` instead of `yLast` to gpu_setNodeDepth
  - Impact: This was halving dYdT values, preventing node-based timestep limiting
- **Fixed Bug 2:** Missing node results flush - dYdT values were not transferred before getVariableStep()
  - Impact: getNodeStep() always saw dYdT=0, so timestep was stuck at 10s instead of 0.357s
- **Result:** GPU timesteps now match CPU exactly (0.2-0.4s range vs previous 10s)

### 2. ✅ Two-Stage Deterministic Accumulation (commits df5c973, 99f9554)
- **Implemented:** Replace non-deterministic atomicAdd with ordered accumulation
  - Stage 1: Each link writes contributions to its own slot (no race conditions)
  - Stage 2: kernel_accumulateNodeContributions sums in CPU-matching order
- **Extended to:** All link types (conduits, pumps, orifices, weirs, outlets)
- **Impact:** Eliminates floating-point ordering differences between CPU and GPU
  - Picard iterations improved from ~8 to ~2.6 (67% reduction in simple models)

### 3. ✅ GPU Lookup Tables for All Cross-Section Shapes (commit 13d34f9)
- **Added:** Complete lookup table support for all SWMM cross-section shapes
- **Impact:** Ensures geometry calculations match CPU across all conduit types

### 4. 📊 Current Metrics (Session68_46_pumps_2hr as of Nov 3, 2025)

| Metric | CPU | GPU | Ratio | Status |
|--------|-----|-----|-------|--------|
| Continuity Error | -14.5% | -241% | 16.6x | ❌ Improved from -261% |
| Final Stored Volume | 0.124 ac-ft | 0.266 ac-ft | 2.15x | ⚠️ Improved from 3.22x |
| External Outflow | 0.023 ac-ft | 0.012 ac-ft | 0.52x | ❌ |
| **Flooding Loss** | **0.001 ac-ft** | **0.165 ac-ft** | **165x** | 🔴 **NEW CRITICAL ISSUE** |
| Simulation Time | 1.0 sec | 50.0 sec | 50x | ❌ GPU very slow |

**Pump Performance:**

| Pump | CPU Time% | GPU Time% | CPU Starts | GPU Starts | Status |
|------|-----------|-----------|------------|------------|--------|
| PMP1-220 | 0.0% | 31.3% | 0 | 1 | ❌ Spurious (improved from 63%) |
| PMP1-1002 | 44.0% | 41.9% | 1 | 59 | ⚠️ High cycling (improved from 91) |
| PMP1-1003 | 77.5% | 71.1% | 1 | 81 | ⚠️ High cycling (improved from 116) |
| PMP1-226 | 1.3% | 1.3% | 1 | 1 | ✅ Now matches CPU |

---

## Outstanding Issues (Updated Nov 3, 2025)

### 🔴 PRIORITY 1: Massive Flooding on GPU (NEW ISSUE)
- **CPU:** 0.001 ac-ft flooding loss (3 nodes: MH_824_25, MH_825_5, MH_929_199)
- **GPU:** 0.165 ac-ft flooding loss (17+ nodes including multiple J-*-OUT nodes)
- **Impact:** 165x more flooding suggests nodes are overflowing that shouldn't
- **Hypothesis:** Storage nodes filling too fast → downstream junctions can't drain → nodes overflow
- **Key flooding nodes on GPU:**
  - J-1002-OUT, J-1003-OUT (pumping station outfalls - 0.006, 0.005 Mgal each)
  - J-147-OUT, J-230-OUT, J-232-OUT, J-246-OUT, J-247-OUT, J-250-OUT
  - Multiple JN_824_* and JN_825_* junctions (all at 1000 ft depth limit)
- **This is the smoking gun** - explains why continuity error is so bad

### ⚠️ PRIORITY 2: Storage Node Volume Accumulation (IMPROVED BUT CRITICAL)
- **GPU final storage:** 2.15× CPU (0.266 ac-ft vs 0.124 ac-ft) - improved from 3.22×
- **Root cause:** Storage nodes accumulating excess volume over time
- **Impact:** Drives spurious pump activation, flooding, and mass balance errors
- **Likely issues:**
  1. `oldVolume` / `oldNetInflow` not updating correctly between routing steps
  2. Volume integration in `gpu_setNodeDepth` accumulating errors
  3. Surface area calculations for storage curves may have subtle differences
  4. Overflow handling when storage exceeds capacity

### ⚠️ PRIORITY 3: Pump Control Issues (PARTIALLY IMPROVED)
- **PMP1-220:** Still spurious (31% runtime, should be 0%)
  - Improved from 63% but still activating incorrectly
  - Now pumps 0.016 Mgal (was 0.033 Mgal, should be 0.000 Mgal)
- **High-frequency cycling persists:**
  - PMP1-1002: GPU 59 starts vs CPU 1 (improved from 91)
  - PMP1-1003: GPU 81 starts vs CPU 1 (improved from 116)
- **Root cause:** Likely storage node depth/volume divergence causing hysteresis threshold crossings

### PRIORITY 4: Picard Iteration Convergence Issues
- **Observation:** "max iterations reached" suggests some routing steps don't converge
- **Impact:** May be using unconverged node depths in subsequent calculations
- **Need to investigate:** Which nodes fail to converge and why

---

## Investigation Plan (Iteration 6 - REVISED)

### 🔴 1. Flooding Analysis (NEW TOP PRIORITY)
- [ ] Identify which routing steps trigger flooding on GPU but not CPU
- [ ] Log node depths for flooding nodes (J-1002-OUT, J-1003-OUT, etc.) on both CPU and GPU
- [ ] Trace upstream flow paths to see if storage nodes are the source
- [ ] Check if overflow calculation in `gpu_setNodeDepth` is correct
- [ ] Verify `Node[].newVolume` doesn't exceed storage capacity
- [ ] Compare junction node inflow/outflow accumulation at flooding timesteps

### 2. Storage Node Volume Pipeline (HIGH PRIORITY)
- [ ] Instrument `gpu_setNodeDepth` for storage nodes (e.g. node 852 / STOR-10) to log per-iteration:
  - `oldVolume`, `oldNetInflow`, `dV`, `surfArea`, `newVolume`, `overflow`.
- [ ] Compare with CPU logs in the same format (use routing step 1 as baseline) to identify where the divergence starts
- [ ] Confirm `gpu_transferNodeIterationStateFromDevice` + `copyNodesFromGpu(..., copyDepthAndVolume=1)` copy the converged `newDepth/newVolume` back to `Node[]` every routing step
- [ ] Ensure `node_setOldHydState` (CPU) / host equivalent runs immediately afterward so `oldDepth/oldVolume/oldNetInflow` are updated before the next step
- [ ] Check if storage nodes that are full handle overflow correctly (should flood, not exceed capacity)
- [ ] Validate storage curve lookups (TABULAR vs FUNCTIONAL) match CPU tables exactly

### 3. Pump Control Alignment
- [ ] For pump PMP1-220, log per-step:
  - inlet node depth/volume
  - pump on/off decisions (hysteresis thresholds)
  - `qCurve`, guard-limited `qFinal`, computed `qMax`
- [ ] Compare GPU vs CPU traces to see if the inlet depth/volume is off
- [ ] Verify pump curve interpolation `gpu_pump_getType{1-5}Flow` returns the same values as CPU for the same inputs
- [ ] Check if pump kernel runs after every Picard iteration with updated node state (not stale depth)

### 4. Node Contribution Debugging
- [ ] For flooding node J-1002-OUT or J-1003-OUT, log:
  - All link contributions (inflow, outflow) per routing step
  - Compare with CPU's serial accumulation
  - Verify two-stage accumulation is producing correct totals
- [ ] Check if any link types (pumps, orifices, weirs) are double-contributing or missing contributions

### 5. Regression Harness
- [ ] Build a "minimal storage + pump + downstream junction" test case to isolate flooding issue
- [ ] Re-validate `simple_storage_test.inp` after fixes to ensure no regression
- [ ] Use `Session68_46_pumps_15min` as a quick sanity check

---

## Success Criteria (Iteration 6 - REVISED)

1. **🔴 Flooding Crisis:** GPU flooding loss within 2x of CPU (target: <0.002 ac-ft, currently 0.165 ac-ft)
2. **Session68_46_pumps_2hr:** GPU continuity error within 2x of CPU (target: <-30%, currently -241%)
3. **Storage volumes:** GPU final storage within 1.5x of CPU (target: <0.19 ac-ft, currently 0.266 ac-ft)
4. **Pump PMP1-220:** Should NOT activate (currently 31% time, target: 0%)
5. **Pump cycling:** Reduce starts to within 5x of CPU (currently 59-81x, target: <5 starts)
6. **No regression on simple tests** (simple_storage, 15 min Session68)

Nice-to-have: Match CPU pump start counts exactly (eliminate all high-frequency cycling).

---

## Next Steps

1. **Immediate:** Run Session68 with flooding node debug logging to identify when/where overflow occurs
2. **Short-term:** Fix storage node volume integration to stop excess accumulation
3. **Medium-term:** Align pump control logic to eliminate spurious activation
4. **Long-term:** Optimize GPU performance (currently 50x slower than CPU)

---

## Tracking
- This file tracks Iteration 6 progress
- Update `doc/gpu/SESSION68_GAP_ANALYSIS.md` when flooding issue is resolved
- Keep `iteration5-todo.md` for historical context
