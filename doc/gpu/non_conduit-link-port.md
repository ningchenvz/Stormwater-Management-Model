# Non-Conduit Link GPU Porting Plan

**Date:** 2025-10-27  
**Owner:** Codex (GPU Acceleration Team)

Objective: Move all non-conduit link calculations (pumps, orifices, weirs, regulators, outlets, dummy links) to the GPU so the Picard loop runs entirely on device without CPU fallbacks.

---

## Task Checklist

1. [x] Inventory CPU behaviour  
   - Document required inputs/outputs for `findNonConduitFlow`, `getModPumpFlow`, `findNonConduitSurfArea`, and `updateNodeFlows` across each link type.

2. [x] Extend GPU data structures  
   - Add SoA buffers for Pump/Weir/Orifice/Outlet/Regulator properties (`gpu_structures.h`/`gpu_memory.cu`), including host/device transfer helpers.

3. [ ] Implement GPU kernels  
   - Write kernels (or kernel extensions) that replicate CPU logic for each non-conduit type and integrate them into `gpu_dwflow.cu`.

4. [ ] Update host integration  
   - Remove CPU-side non-conduit loop, ensure Picard flow updates use GPU results, and keep fallback path intact.

5. [ ] Validation & regression  
   - Run compare scripts over the non-conduit-heavy benchmark suite; resolve any diffs and log results here.

---

## Notes & Findings

### Task 1 – CPU Behaviour Inventory (Completed)

**Shared loop (`dynwave.c`)**
- `findNonConduitFlow(i, dt)` runs once per Picard iteration for every link that fails `isTrueConduit`.  
  - Reads `Link[i].newFlow`, `oldFlow`, `flowClass`, `setting`, `targetSetting`, `bypassed`, and `Omega`.  
  - Pulls upstream inflow via `link_getInflow`, which dispatches to per-type helpers.  
  - Special-cases pumps (no under-relaxation).  
  - Writes back `Link[i].newFlow`, `dqdh`, `surfArea1/2` (through `findNonConduitSurfArea`), and leaves `Link[i].newDepth` as produced by the type-specific helper.
- `findNonConduitSurfArea(i)` populates `Link[i].surfArea{1,2}` based on type and node storage status; pumps/weirs share the same surfaces.
- `updateNodeFlows(i)` consumes the updated flows to increment `Node[n].inflow/outflow`, apply conduit seepage/evap losses (if `Link.type == CONDUIT`), and accumulate `Xnode[...]` fields (`newSurfArea`, `sumdqdh`). Pumps add `dqdh` to downstream nodes conditionally.

**Type-specific inflow routines (`link.c`)**
- **Pump (`pump_getInflow`)**  
  - Needs `Pump[k]` params (curve type, coefficients, on/off depths, wet-well volume, speed setting).  
  - Reads upstream/downstream node depth, volume, inflow/outflow, overflow, invert elevations; writes `Link[j].dqdh`, `flowClass`, `newDepth` (implicitly zero), and honours flap gates through `link_setFlapGate`.  
  - Uses project-wide tables (`Curve`) and conversion factors `UCF`, `RouteStep`.
- **Orifice (`orifice_getInflow`)**  
  - Consumes `Orifice[k]` struct fields (`type`, `cDisch`, `cOrif`, `cWeir`, `hCrit`, `length`, `surfArea`).  
  - Requires upstream/downstream heads, node depths, invert elevations; respects flap gates and target setting.  
  - Updates `Link[j].flowClass`, `Link[j].dqdh`, `Link[j].newDepth`.  
  - Depends on `xsect_getAofY`, `xsect_getWofY`, `table_lookup`, and global constants (`GRAVITY`, `FUDGE`).
- **Weir (`weir_getInflow`)**  
  - Uses `Weir[k]` fields (type, coefficients, slope, length, `canSurcharge`, `cSurcharge`, `surfArea`, optional discharge curve).  
  - Evaluates crest elevation (`Link.offset1`), shape data (`Link.xsect`), upstream/downstream water surface and invert elevations.  
  - Calls helpers `weir_getFlow`, `weir_getOrificeFlow`, and `roadway_getInflow` (for roadway weirs).  
  - Outputs `Link[j].newDepth`, `flowClass`, `dqdh`; may toggle to orifice behaviour under surcharge.  
  - Needs flap gate status, routing model (kinematic vs DW), and route step size.
- **Outlet (`outlet_getInflow`)**  
  - Reads `Outlet[k]` (curve type, rating curve id, coefficients, crest offset).  
  - Considers routing model (DW vs KW), upstream/downstream heads, invert elevations, flap gates.  
  - Returns flow based on rating curve or power function; sets `Link[j].newDepth`, `flowClass`.
- **Default (DUMMY & others)**  
  - `link_getInflow` falls back to `node_getOutflow` for dummy/valve-like links so the GPU implementation must still handle the updateNodeFlows pathway even when no dedicated kernel exists.

**Additional dependencies**
- `link_setFlapGate` enforces directionality and outfall flap logic; GPU port needs equivalent checks using link/node metadata.  
- Under-relaxation parameter `Omega`, iteration counter `Steps`, and `RouteModel` influence various control paths.  
- Surface area/`dqdh` contributions from non-conduit links feed directly into node convergence criteria (`gpu_dynwave`), so GPU kernels must update these accumulators atomically like the existing conduit kernel.

### Task 2 – GPU Data & Kernel Design (Completed)

**Proposed data structures**
- `GPU_PumpData`: host/device arrays for all `TPump` fields (`type`, `pumpCurve`, `initSetting`, `yOn`, `yOff`, `xMin`, `xMax`) plus runtime scalars that today live on `Link` (`setting`, `targetSetting`, optional speed modifiers).  
- `GPU_OrificeData`: carries `type`, `shape`, `cDisch`, `orate`, `cOrif`, `hCrit`, `cWeir`, `length`, `surfArea`.  
- `GPU_WeirData`: holds `type`, `cDisch1`, `cDisch2`, `endCon`, `canSurcharge`, `roadWidth`, `roadSurface`, `cdCurve`, plus mutable `cSurcharge`, `length`, `slope`, `surfArea`.  
- `GPU_OutletData`: stores `qCoeff`, `qExpon`, `qCurve`, `curveType`.  
- Maintain per-type index lists (e.g., `pumpLinkIds`) so kernels can iterate contiguous chunks without branching on `Link.type`.
- Extend allocation/setup in `ensureConduitKernelContext()` to allocate/transfer these new structures; static transfers happen once, dynamic settings refreshed when `steps == 0` or control rules update a link.

**Kernel plan**
- Add device helpers to evaluate:
  - Pump curves (`gpu_table_lookup`, curve-type specific math, `link_setFlapGate` equivalent).  
  - Orifice/weir geometry using existing CUDA xsect helpers.  
  - Outlet rating curves/power functions.
- Launch sequence within `gpu_computeConduitFlows` (same stream):
  1. Conduit kernel (already in place).  
  2. `kernel_computePumpFlows`.  
  3. `kernel_computeOrificeFlows`.  
  4. `kernel_computeWeirFlows` (handles roadway + surcharge).  
  5. `kernel_computeOutletFlows`.  
  6. Optional fall-through kernel that copies node outflow for dummy links.
- Every kernel updates:
  - `links->d_newFlow`, `d_newDepth`, `d_dqdh`, `d_flowClass`, `d_surfArea{1,2}`.  
  - Node accumulators via atomic adds (`nodes->d_inflow/outflow`, `d_newSurfArea`, `d_sumdqdh`).
- After kernels finish, reuse the existing selective copy-back helpers to keep CPU-side state aligned.

**Host integration**
- Remove the CPU-only loop in `findLinkFlows()` whenever GPU is active; keep it for fallback.  
- Ensure control rules (`link_setTargetSetting`, `link_setSetting`) continue to run on CPU but push updated settings into the GPU dynamic buffers before each Picard iteration.  
- Preserve current safety checks (kernel timing watchdog, conduit count guard) and extend logging if a non-conduit kernel fails.

### Outstanding

- Task 3 onward still pending implementation.
- Initial groundwork complete: GPU structs, allocation helpers, and transfer stubs added for pumps/orifices/weirs/outlets; link SoA now carries `setting/targetSetting/hasFlapGate`. Kernel work still in progress.
