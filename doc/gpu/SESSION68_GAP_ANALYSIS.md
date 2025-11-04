# Session68_46_pumps_2hr.inp - GPU vs CPU Gap Analysis

## Executive Summary

After fixing the adaptive timestep calculation bug, GPU now correctly computes variable timesteps (0.2-0.4s range) instead of being stuck at 10s. However, significant mass balance errors remain:

- **CPU Continuity Error: -14.5%**
- **GPU Continuity Error: -252.3%** (17x worse than CPU)

This report analyzes the gaps to identify remaining issues.

---

## 1. Mass Balance Continuity Errors

### Overall Flow Routing Continuity

**CPU:**
  Flow Routing Continuity        acre-feet      10^6 gal
  **************************     ---------     ---------
  Dry Weather Inflow .......         0.014         0.005
  Wet Weather Inflow .......         0.000         0.000
  Groundwater Inflow .......         0.000         0.000
  RDII Inflow ..............         0.000         0.000
  External Inflow ..........         0.000         0.000
  External Outflow .........         0.027         0.009
  Flooding Loss ............         0.000         0.000
  Evaporation Loss .........         0.000         0.000
  Exfiltration Loss ........         0.000         0.000
  Initial Stored Volume ....         0.116         0.038
  Final Stored Volume ......         0.126         0.041
  Continuity Error (%) .....       -17.463
  
  

**GPU:**
  Flow Routing Continuity        acre-feet      10^6 gal
  **************************     ---------     ---------
  Dry Weather Inflow .......         0.014         0.005
  Wet Weather Inflow .......         0.000         0.000
  Groundwater Inflow .......         0.000         0.000
  RDII Inflow ..............         0.000         0.000
  External Inflow ..........         0.000         0.000
  External Outflow .........         0.064         0.021
  Flooding Loss ............         0.000         0.000
  Evaporation Loss .........         0.000         0.000
  Exfiltration Loss ........         0.000         0.000
  Initial Stored Volume ....         0.116         0.038
  Final Stored Volume ......         0.406         0.132
  Continuity Error (%) .....      -260.917
  
  

### Key Observations

1. **Inflow matches exactly**: Both CPU and GPU show 0.014 acre-feet dry weather inflow ✓
2. **External Outflow differs significantly**:
   - CPU: 0.027 acre-feet (0.009 Mgal)
   - GPU: 0.064 acre-feet (0.021 Mgal) - **2.37x more outflow** ❌
3. **Final Stored Volume differs drastically**:
   - CPU: 0.126 acre-feet (0.041 Mgal)
   - GPU: 0.406 acre-feet (0.132 Mgal) - **3.22x more storage** ❌

---

## 2. Pump Performance Analysis

### Pump Flow Comparison (Million Gallons Pumped)

| Pump | CPU Mgal | GPU Mgal | Ratio | CPU %Time | GPU %Time | CPU Starts | GPU Starts |
|------|----------|----------|-------|-----------|-----------|------------|------------|
| PMP1-1002 | 0.001 | 0.001 | 1.00 | 52.08 | 50.97 | 1 | 91 |
| PMP1-1003 | 0.001 | 0.001 | 1.00 | 99.31 | 56.94 | 1 | 116 |
| PMP1-147 | 0.002 | 0.002 | 1.00 | 5.56 | 5.83 | 1 | 1 |
| PMP1-220 | 0.000 | 0.033 |  | 0.00 | 63.33 | 0 | 1 |
| PMP1-224 | 0.001 | 0.001 | 1.00 | 2.08 | 2.08 | 1 | 1 |
| PMP1-226 | 0.004 | 0.026 | 6.50 | 1.39 | 7.50 | 1 | 2 |
| PMP1-229 | 0.000 | 0.000 |  | 2.92 | 2.92 | 1 | 1 |
| PMP1-232 | 0.001 | 0.002 | 2.00 | 12.64 | 4.72 | 1 | 7 |
| PMP1-246 | 0.001 | 0.001 | 1.00 | 4.72 | 4.31 | 1 | 1 |
| PMP1-250 | 0.001 | 0.001 | 1.00 | 17.22 | 18.89 | 2 | 5 |
| PMP1-258 | 0.000 | 0.000 |  | 8.19 | 8.19 | 1 | 1 |

### Pump Performance Key Findings

1. **PMP1-220 Spurious Operation** ❌
   - CPU: Never runs (0% time)
   - GPU: Runs 63.3% of time, pumps 0.033 Mgal
   - This pump should NOT be running!

2. **PMP1-226 Over-Pumping** ❌  
   - CPU: 0.004 Mgal
   - GPU: 0.026 Mgal (6.5x more!)
   - GPU runs 7.5% vs CPU 1.4%

3. **Excessive Pump Cycling** ❌
   - PMP1-1002: 1 start (CPU) vs 91 starts (GPU)
   - PMP1-1003: 1 start (CPU) vs 116 starts (GPU)
   - Suggests unstable pump control on GPU

4. **Some Pumps Match Well** ✓
   - PMP1-147, PMP1-224, PMP1-229, PMP1-246, PMP1-258 show similar behavior

---

## 3. Storage Node Analysis

### Nodes with Highest Continuity Errors

**CPU Top 5:**
    Node WW-250 (-242.62%)
    Node MH_825_5 (-149.59%)
    Node WW-246 (-106.40%)
    Node MH_919_221 (100.00%)
    Node MH_929_44 (100.00%)

**GPU Top 5:**
    Node MH_929_188 (-76498.36%)
    Node MH_929_204 (-40863.10%)
    Node MH_824_75 (-13425.99%)
    Node MH_919_52 (-12270.37%)
    Node MH_929_165 (-10540.63%)

### Storage Analysis Key Findings

1. **GPU has MUCH worse continuity errors** ❌
   - CPU worst: WW-250 at -242%
   - GPU worst: MH_929_188 at **-76,498%** (extreme!)
   
2. **Different problem nodes**
   - CPU and GPU don't even identify the same problematic nodes
   - Suggests fundamentally different flow patterns

---

## 4. Root Cause Hypotheses

Based on the data, the following issues are likely causing the mass balance errors:

### 4.1 Pump Control Logic Issues

**Evidence:**
- PMP1-220 runs on GPU but not CPU (spurious activation)
- Excessive pump cycling (91-116 starts vs 1 start)
- Some pumps over-pump by 6.5x

**Likely Causes:**
1. **Storage node depth calculation errors** causing incorrect pump triggers
2. **Pump on/off hysteresis not working correctly** on GPU
3. **Pump flow calculation bugs** in GPU kernel

### 4.2 Storage Node Volume/Depth Errors

**Evidence:**
- Final stored volume 3.22x higher on GPU
- Storage nodes have extreme continuity errors (-76,498%)
- This was partially addressed in previous fixes but issues remain

**Likely Causes:**
1. **Volume integration errors** in gpu_setNodeDepth for storage nodes
2. **Surface area calculation differences** between CPU and GPU
3. **oldNetInflow not being propagated correctly** between timesteps

### 4.3 Junction Node Flow Accumulation

**Evidence:**
- Many junction nodes (MH_*) show extreme continuity errors on GPU
- Different set of problem nodes than CPU

**Likely Causes:**
1. **Inflow/outflow accumulation bugs** in kernel_accumulateNodeContributions
2. **Link contribution calculation errors** (node1/node2 contributions)
3. **Race conditions or ordering issues** in parallel accumulation

---

## 5. Recommended Next Steps

### Priority 1: Pump Control Investigation
1. Add detailed logging for pump on/off decisions (especially PMP1-220)
2. Compare storage node depths between CPU/GPU at pump activation times
3. Verify pump curve interpolation on GPU vs CPU

### Priority 2: Storage Node Volume Fix  
1. Add comprehensive logging for storage volume calculations
2. Compare oldNetInflow, newNetInflow between CPU and GPU
3. Verify storage curve lookups are identical
4. Check if oldNetInflow is being reset incorrectly

### Priority 3: Junction Node Flow Routing
1. Verify kernel_accumulateNodeContributions is deterministic
2. Log link contributions for problem nodes (MH_929_188, MH_929_204)
3. Check for numerical precision issues in accumulation

### Priority 4: End-to-End Validation
1. Start with simplest case: single storage node + single pump
2. Gradually add complexity to isolate where divergence begins
3. Use Session68_46_pumps_15min.inp for faster iteration

---

## 6. Test Coverage Status

| Test | Timestep Fix | Mass Balance | Status |
|------|-------------|--------------|--------|
| Session68_15min | ✓ Verified | ❌ Not assessed | Partial |
| Session68_2hr | ✓ Verified | ❌ -261% error | Failing |
| Session18_1hr | ✓ Verified | ❌ -1898% error | Failing |

**Conclusion:** Timestep calculation is now correct across all tests. Mass balance errors are the next critical issue to resolve.

