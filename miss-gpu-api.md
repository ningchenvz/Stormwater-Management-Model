# GPU Implementation Gap Tracking

**Last Updated**: 2025-11-02 (Commit: 63a9da6)

## Summary Status
- ✅ **6 Fixed** (Flow classification, Non-conduit surface areas, Dry-start bootstrap, Lateral inflows, Node statistics, oldNetInflow)
- ⛔ **3 Open** (Conduit extras, Losses/limits, GPU-resident iteration)

---

| Gap # | Description | Status | CPU Reference | GPU Status | Impact |
|-------|-------------|--------|---------------|------------|--------|
| **1** | Conduit extras (force mains, culverts, flap-gate/normal-flow/evap losses) still run only on CPU | ⛔ Open | `src/solver/dwflow.c:212` | Still omitted in Stage-1 helper (`src/solver/gpu/gpu_conduit_helpers.cuh:11`); no device equivalents for culvert inlet control, force-main friction, flap gates, evap/seepage, or q-limits. | Models using those features continue to diverge from CPU physics, keeping us from declaring full parity. |
| **2** | Flow classification/critical-depth logic not wired through | ✅ Fixed | `src/solver/dwflow.c:297` | `gpu_findConduitFlow_simplified` now calls the full `gpu_getFlowClass` and forwards `fasnh`, critical, and normal depths into `gpu_computeSurfaceAreas` (`src/solver/gpu/gpu_conduit_helpers.cuh:380-640`). | Conduit surface areas and dq/dh align with CPU, removing a major source of storage mass imbalance. |
| **3** | Losses, user limits, and conduit state flags skipped | ⛔ Open | `src/solver/dwflow.c:228` | Device flow update still ignores local losses, `link_getLossRate`, q-limit enforcement, and full-state flags (`src/solver/gpu/gpu_conduit_helpers.cuh:520-580`). | Head-loss, evaporation/seepage, and reporting states remain wrong on GPU, so feature completeness is incomplete. |
| **4** | Non-conduit surface areas never added on GPU | ✅ Fixed | `src/solver/dynwave.c:615` | Pump/orifice/weir/outlet kernels now call `gpu_findNonConduitSurfArea` and accumulate results (`src/solver/gpu/gpu_dwflow.cu:1018,1188,1359,1481`). | Storage nodes receive the same wetted-area contributions as CPU, eliminating the runaway surface-area drift. |
| **5** | Dry-start bootstrap missing | ✅ Fixed | `src/solver/dwflow.c:164` | Conduit kernel no longer bails when `aMid <= FUDGE` and uses the same classification logic as CPU, allowing flows to grow from zero (`src/solver/gpu/gpu_conduit_helpers.cuh:498-620`); validated with `gpu_outfall_drainage.inp`. | GPU solver escapes the zero-flow lock and can drain networks without CPU intervention. |
| **6** | Lateral inflows not transferred every timestep | ✅ Fixed (Commit: 4c4a1f0) | `src/solver/node.c:206` | `newLatFlow` moved from static to dynamic transfer in `gpu_transferNodeDynamicToDevice()` (`src/solver/gpu/gpu_memory.cu:1301`). Junction depth calculation now uses current hydrograph values instead of stale initial values. | Junctions with time-varying lateral inflows now compute correct depths. Continuity error dropped from 72% to 4.4% in test case. |
| **7** | Node statistics (inflow/outflow) not copied back | ✅ Fixed (Commit: 4c4a1f0) | `src/solver/node.c` | Added `Node[].inflow/outflow` copyback in `copyNodesFromGpu()` (`src/solver/gpu/gpu_dynwave.cu:542-543`). | Node routing statistics now report correctly in output files. |
| **8** | oldNetInflow not updated after each timestep | ✅ Fixed (Commit: 4c4a1f0) | `src/solver/node.c:339` | Added `oldNetInflow = inflow - outflow` update to mirror CPU's `node_setOldHydState()` (`src/solver/gpu/gpu_dynwave.cu:546-547`). | Junction depth calculation now uses correct previous timestep net inflow in `dV = 0.5 * (oldNetInflow + dQ) * dt`. |
| **9** | Iteration still bounces through host memory | ⛔ Open | — | Node depth kernel still copies dynamic state back to host every Picard step (`src/solver/gpu/gpu_dynwave.cu:328`), and pumps rely on a sequential CPU-order kernel (`src/solver/gpu/gpu_dwflow.cu:858`). | CPU remains in the hot path, preventing a fully GPU-resident routing loop and future multi-GPU scaling. |
