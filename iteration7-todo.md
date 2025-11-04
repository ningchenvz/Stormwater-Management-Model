# Iteration 7 TODO: Finalize Two-Stage Accumulation & Restore Mass Balance

## Context
- **Reference docs:** `iteration6-todo.md`, `doc/gpu/SESSION68_GAP_ANALYSIS.md`
- **Latest status (post Iteration 6 experiments):**  
  *Session68_46_pumps_2hr* still shows ~−240 % continuity error (CPU ≈ −15 %), despite the deterministic conduit accumulation and adaptive timestep fixes.  
  Root cause: pumps contribute flows after node depths are solved, so the depth solver never “sees” pump inflow/outflow before computing the volume update.

---

## Key Learnings from Iteration 6 Experiments
1. **Pumps-after-depth** made continuity worse (−241 %) – node depths were computed without pump flows, breaking V(t+1) = V(t) + (Qin−Qout)·dt.
2. **Reset + re-accumulate** removed conduit state that pumps rely on to compute `qMax` (no guard parity).
3. **Direct atomic node updates** lead to hangs, confirming the value of the deterministic two-stage scheme.
4. **Zeroing contributions between passes** still hit ordering issues; we were double-counting conduits or wiping state pumping needed.

**Conclusion:** we need a single accumulation kernel that can run in two modes:
- Mode 1: sum only conduit contributions (seed node arrays so pumps see conduit state for `getModPumpFlow`)
- Pumps write their own contributions into the shared contribution array
- Mode 2: zero node arrays, then sum conduit + pump contributions together before calling the depth solver

---

## Objectives for Iteration 7
1. **Implement mode-aware accumulation** (`kernel_accumulateNodeContributions(nodes, contributions, links, numLinks, mode)`), where `mode` is:
   - `GPU_ACCUM_CONDUITS_ONLY`
   - `GPU_ACCUM_ALL`
2. **Refactor pump kernel** so it writes into its `GPU_LinkContribution` slot (no direct node updates).
3. **Update Picard iteration flow**:
   - Stage A: conduit kernels fill contributions
   - Stage B: accumulate (mode=CONDUITS_ONLY) to seed node arrays
   - Stage C: pump kernel uses node arrays for `getModPumpFlow`, writes contributions
   - Stage D: zero node arrays and re-run accumulation (mode=ALL)
   - Stage E: node-depth solver uses the final totals
4. **Add safeguards** to avoid double-counting conduits/pumps (only the final Mode=ALL pass should feed the depth solver).
5. **Restore regression stability** on Session68/Session18 (goal: bring continuity error within ×1.5 of CPU).

---

## Detailed Task List

### 1. Accumulation Kernel Refactor
- [ ] Extend `GPU_LinkContribution` (if needed) with a flag or simply rely on the entry index to distinguish contribution types.
- [ ] Modify `kernel_accumulateNodeContributions` signature to accept a `mode` enum/int.
- [ ] In “CONDUITS_ONLY” mode, ignore pump contributions (skip based on link-type or a pre-built list of pump indices).  
- [ ] In “ALL” mode, sum every contribution slot (conduits + pumps + future non-conduit link types).
- [ ] Add helper to zero node accumulators (`nodes->d_inflow`, `d_outflow`, `d_newSurfArea`, `d_sumdqdh`) before the final accumulation.

### 2. Pump Kernel Updates
- [ ] Ensure the sequential pump kernel reads the node arrays after the conduit-only accumulation so `getModPumpFlow` sees conduit flows.
- [ ] For each pump, write its inflow/outflow/surf area contributions into `GPU_LinkContribution[linkIdx]` instead of directly touching nodes.
- [ ] Maintain existing guard logic (`gpu_getModPumpFlow`) and logging hooks.

### 3. Picard Loop Integration (`gpu_runPersistentPicardIteration`)
- [ ] Stage A: flush conduit contributions from previous step (if needed) and ensure `d_contributions` is zeroed.
- [ ] Stage B: accumulate (CONDUITS_ONLY).
- [ ] Stage C: inject pump contributions via sequential kernel.
- [ ] Stage D: zero node accumulators and accumulate again (ALL).
- [ ] Stage E: run the depth solver (`gpu_computeNodeDepthsWithConvergence`).
- [ ] After convergence, copy node/link state back to host as before.

### 4. Regression Tests
- [ ] `simple_storage_test.inp` (baseline: ~−0.04 % continuity).
- [ ] `Session68_46_pumps_15min.inp` (quick sanity).
- [ ] `Session68_46_pumps_2hr.inp` (target continuity <−25 %, pump start counts within ±10 % of CPU).
- [ ] Track pump PMP1-220 (should remain inactive) and PMP1-226 (pumped volume within ±5 % of CPU).

---

## Success Criteria
1. **Continuity parity**: `Session68_46_pumps_2hr` GPU continuity error within 1.5× CPU (≈ <−25 %).
2. **Pump correctness**: pump activation (time on, starts) and volumes (PMP1-220, PMP1-226) match CPU within ±5–10 %.
3. **Storage volumes**: no >2× divergence vs CPU; worst node continuity error <−500 %.
4. **No regression on simple tests** (e.g., `simple_storage_test`, Session68 15 min).

---

## ✅ IMPLEMENTATION COMPLETE (Nov 3, 2025)

### Changes Made
1. **Added `kernel_zeroNodeAccumulatorsForFinalPass`** in gpu_dynwave.cu:216
   - Resets node accumulators (inflow, outflow, sumdqdh) to base values
   - Preserves intrinsic surface area

2. **Updated Stage 4 of Picard iteration** in gpu_dynwave.cu:1584-1616
   - OLD: Accumulate NON_CONDUITS_ONLY on top of CONDUITS (double-counting!)
   - NEW: Zero accumulators + accumulate ALL (conduits + pumps together)
   - This eliminates double-counting bug that was causing -241% continuity error

### Test Results: Session68_46_pumps_2hr.inp

| Metric | CPU | GPU Iter6 (Before) | GPU Iter7 (After) | Improvement |
|--------|-----|-------------------|-------------------|-------------|
| **Continuity Error** | -14.5% | -241% | **-90.4%** | ✅ **2.7x better** |
| **Final Stored Volume** | 0.124 ac-ft | 0.266 ac-ft (2.15x) | **0.143 ac-ft (1.15x)** | ✅ **Near parity** |
| **External Outflow** | 0.023 ac-ft | 0.012 ac-ft (0.52x) | **0.021 ac-ft (0.91x)** | ✅ **Matches CPU** |
| **Flooding Loss** | 0.001 ac-ft | 0.165 ac-ft (165x) | **0.084 ac-ft (84x)** | ✅ **2x improvement** |
| **Simulation Time** | 1.0 sec | 50.0 sec | **40.0 sec** | ✅ 20% faster |

### Pump Performance

| Pump | Metric | CPU | GPU Iter6 | GPU Iter7 | Status |
|------|--------|-----|-----------|-----------|--------|
| **PMP1-220** | Time% | 0.0% | 31.3% | **0.0%** | ✅ **SPURIOUS ACTIVATION FIXED!** |
| **PMP1-220** | Starts | 0 | 1 | **0** | ✅ **PERFECT** |
| **PMP1-220** | Mgal | 0.000 | 0.016 | **0.000** | ✅ **PERFECT** |
| PMP1-1002 | Starts | 1 | 59 | **35** | ✅ Improved (was 91 in earlier tests) |
| PMP1-1003 | Starts | 1 | 81 | **54** | ✅ Improved (was 116 in earlier tests) |
| PMP1-226 | Mgal | 0.004 | 0.004 | **0.006** | ⚠️ Slight increase (50%) |

### Success Criteria Assessment

1. **Continuity parity**: Target <-25%, Achieved: **-90.4%** ⚠️ Not quite there, but **2.7x improvement**
2. **Pump correctness**: PMP1-220 **PERFECT** ✅, PMP1-226 volume 50% high ⚠️
3. **Storage volumes**: 1.15x divergence ✅ (target <2x, achieved!)
4. **No regression**: Need to test simple_storage_test and Session68 15min

### Remaining Issues

1. **Continuity error still -90%** (target: within 1.5x of CPU = -22%)
   - Improved 2.7x but not at parity yet
   - Storage volume now close (1.15x), so this may be flooding-driven

2. **Flooding still 84x CPU** (reduced from 165x)
   - 0.084 ac-ft GPU vs 0.001 ac-ft CPU
   - This excess flooding is likely the remaining continuity error source

3. **Pump cycling still higher than CPU**
   - PMP1-1002: 35 starts vs CPU 1 (better than 59, but still high)
   - PMP1-1003: 54 starts vs CPU 1 (better than 81, but still high)
   - Likely caused by remaining storage volume oscillations

4. **PMP1-226 volume 50% high**
   - 0.006 Mgal vs 0.004 Mgal CPU
   - May be related to flooding/storage issues

### Next Steps

1. **Investigate remaining flooding** (Priority: HIGH)
   - 84x still very high, even though improved from 165x
   - Check which nodes are flooding and why

2. **Verify simple_storage_test** hasn't regressed
   - Ensure fix works for simple cases too

3. **Debug pump cycling**
   - Understand why pumps cycle 35-54 times vs CPU 1 time
   - Likely storage node depth oscillations

4. **Consider if additional fixes needed**
   - May need to look at storage volume integration
   - Or overflow handling in setNodeDepth

---

## Additional Changes Made (Later in Iteration 7)

### 3. **Added pump sequential node updates** in gpu_dwflow.cu:1190-1210
- CRITICAL FIX: Update `nodes->d_outflow[n1]` and `nodes->d_inflow[n2]` immediately after each pump
- Allows subsequent pumps to see updated in-flight balance (like CPU)
- **Result:** Verified working via debug logs, but NO CHANGE in results
- **Reason:** Each pump has its own dedicated wet well in Session68 (no shared storage nodes)
  - WW-1002 → pump 0 (node 830)
  - WW-1003 → pump 1 (node 831)
  - WW-147 → pump 2 (node 832)
- **Conclusion:** Fix is correct but doesn't apply to this specific model

### 4. **Added comprehensive storage volume debug logging** in gpu_dynwave_kernels.cuh:434-445
- Detailed logging for nodes 830/831 (WW-1002/WW-1003) volume integration
- Logs: oldNetInflow, oldVolume, inflow, outflow, dQ, dV, newVolume, surfArea
- Enables direct comparison with CPU calculations

---

## 🔍 Deep Dive Analysis: Storage Volume Integration

### GPU Volume Calculation Verification (WW-1002, Step 2)

```
Inputs:
  oldNetInflow = -0.112503342 cfs  (from previous routing step)
  oldVolume    = 235.619959394 ft³
  oldDepth     = 2.999999483 ft
  dt           = 0.357 sec

Current flows (after ALL accumulation):
  inflow  = 0.000 cfs (no lateral inflow)
  outflow = 0.203071418 cfs (pump + any conduits)
  dQ      = -0.203071418 cfs

Trapezoidal integration:
  dV = 0.5 * (oldNetInflow + dQ) * dt
  dV = 0.5 * (-0.112503342 + -0.203071418) * 0.357
  dV = -0.056330095 ft³  ✅ CORRECT FORMULA

Result:
  newVolume = 235.619959394 + (-0.056330095)
  newVolume = 235.563629299 ft³
```

**Conclusion:** The volume integration formula is **MATHEMATICALLY CORRECT** ✅

### Remaining Mystery: Why Storage Drains at Half Rate?

**Observation:**
- CPU WW-1002: avg 0.038 (4.8%), max 0.236 (30.0%) [1000 ft³]
- GPU WW-1002: avg 0.019 (2.4%), max 0.236 (30.0%) [1000 ft³]
- Same max volume (both reach 30%), but GPU average is **HALF**

**Formula is correct, so the issue must be in the INPUTS:**

1. **oldNetInflow propagation** ⚠️ PRIMARY SUSPECT
   - Is GPU's `oldNetInflow = -0.112503342` matching CPU at step 2?
   - Is it being transferred to device correctly between routing steps?
   - Code path: `Node[i].oldNetInflow` → `nodes->h_oldNetInflow[i]` → `nodes->d_oldNetInflow[i]`

2. **Pump outflow values**
   - Is GPU's `outflow = 0.203071418` matching CPU?
   - Could pump curves be returning different values?

3. **Timestep consistency**
   - Is GPU's `dt = 0.357` matching CPU at step 2?
   - Timestep fix (commit d3dd521) should have resolved this

---

## 🎯 Next Investigation Steps

### Priority 1: Verify oldNetInflow Transfer to Device
- [ ] Find `gpu_transferNodeDynamicToDevice()` implementation
- [ ] Confirm it includes `cudaMemcpy` for `d_oldNetInflow`
- [ ] Add debug logging to verify device values match host after transfer
- [ ] Compare GPU `oldNetInflow` values with CPU at steps 1-5

### Priority 2: Compare with CPU Calculations
- [ ] Add similar logging to CPU `node_setDepth()` for WW-1002/WW-1003
- [ ] Run CPU-only simulation and capture same metrics
- [ ] Side-by-side comparison of all inputs (oldNetInflow, inflow, outflow, dV)

### Priority 3: Pump Curve Verification
- [ ] Log pump curve inputs (volume/depth) for pumps 0,1,2
- [ ] Compare with CPU pump curve calls
- [ ] Verify same flow rates returned for same inputs

---

## Summary Status (End of Iteration 7 Session 1)

### ✅ Major Wins
1. **Spurious pump PMP1-220 activation completely fixed** (was 31%, now 0%) 🎉
2. **Continuity error improved 2.7x** (-241% → -90%)
3. **Storage volume near parity** (2.15x → 1.15x CPU)
4. **Flooding reduced 2x** (165x → 84x CPU)
5. **Performance improved 20%** (50sec → 40sec)

### ⚠️ Remaining Issues
1. **Continuity still -90%** (target: <-25%, within 1.5x of CPU)
2. **Storage nodes drain too fast** (WW-1002/1003 at half correct rate)
3. **Flooding still 84x CPU** (driven by storage issue)
4. **Pump cycling still high** (35-54 starts vs 1, driven by storage oscillation)

### 🔬 Root Cause Investigation
- Volume integration **formula verified correct** ✅
- Issue is in **inputs to the formula**
- Primary suspect: **oldNetInflow** not propagating correctly to device
- Next session: Deep dive into device memory transfer and CPU comparison

---

## Notes / Open Questions
- The zero+accumulate fix eliminated the spurious PMP1-220 activation completely! ✅
- Continuity improved 2.7x, storage volume near parity (1.15x), but flooding still high
- Performance improved 20% (40sec vs 50sec) despite adding extra kernel launch
- Pump sequential update fix is correct but didn't help (pumps don't share wet wells in this model)
- Volume integration math is provably correct - issue must be in data flow between routing steps
- Keep an eye on performance: the additional accumulation pass adds cost, but correctness takes priority

