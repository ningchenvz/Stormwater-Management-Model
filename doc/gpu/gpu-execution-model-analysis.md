# SWMM-GPU Execution Model Analysis and Plan for One-Kernel-Per-Step

Date: 2025-10-28

Author: Cline (analysis run in repo at commit current in local workspace)

## Objective

Assess whether the current GPU implementation loads all state onto the device and runs a single kernel per timestep, and identify fragmentation points (many small kernels, CPU↔GPU switching). Propose concrete steps to keep state resident on the GPU and execute one kernel per timestep (or one persistent kernel that loops over Picard iterations on device).

---

## Summary Conclusion

- Current implementation does not run “one kernel per timestep.” It launches several kernels per Picard iteration for different link types and computes node depths on the CPU between those launches. Each iteration performs device↔host transfers of intermediate results and synchronizes the stream.
- State is partially kept on device across timesteps (allocations are persistent), but dynamic per-iteration updates are copied Host→Device before kernel and Device→Host immediately after to feed the CPU node solver.
- To achieve the goal, move node depth computation fully to the GPU (explicit-memory path), eliminate per-iteration host transfers, and fuse kernels into a single iteration kernel or adopt a persistent kernel that loops over Picard iterations entirely on the device.

---

## Evidence and Code References

- Multiple GPU kernel launches per iteration (fragmentation):
  - src/solver/gpu/gpu_dwflow.cu::gpu_computeConduitFlows launches:
    - kernel_findConduitFlows (conduits)
    - kernel_findPumpFlows (pumps if any)
    - kernel_findOrificeFlows (orifices if any)
    - kernel_findWeirFlows (weirs if any)
    - kernel_findOutletFlows (outlets if any)
  - Followed by cudaStreamSynchronize, timing events, and per-iteration Device→Host transfers:
    - gpu_transferLinkIterationResultsFromDevice(links, links->count)
    - gpu_transferNodeIterationStateFromDevice(nodes, nodes->count)
    - CPU-side copies: copyLinkIterStateFromGpu, copyNodesFromGpu

- GPU node depth path ENABLED (GPU-only, no CPU fallback):
  - src/solver/dynwave.c::findNodeDepths has GPU code path enabled at lines 750-780
  - GPU computes both link flows AND node depths when g_gpuConfig.useCuda is true
  - NO CPU fallback: If GPU kernel fails, simulation aborts with error
  - CPU path only executed when GPU is disabled (SWMM_USE_CUDA=0)

- Node kernel uses explicit memory model:
  - src/solver/gpu/gpu_dynwave.cu contains kernel_findNodeDepths and gpu_runNodeDepthKernel
  - ensureNodeKernelContext() allocates explicit device memory (d_* arrays) - NO unified memory restriction
  - Uses project's explicit memory strategy from src/solver/gpu/gpu_memory.cu (pinned host h_* and device d_* arrays)
  - Phase 1 complete: GPU node depth computation with explicit memory is operational

- Explicit memory model (good persistence, but frequent transfers):
  - src/solver/gpu/gpu_memory.cu allocates h_* (cudaMallocHost) and d_* (cudaMalloc) for all major structures and provides granular transfer helpers:
    - Static: gpu_transfer{Node,Link,Conduit,Xsect}StaticToDevice (once).
    - Dynamic pre-iteration: gpu_transferNodeDynamicToDevice, gpu_transferLinkDynamicToDevice, etc.
    - Iteration results: gpu_transferLinkIterationResultsFromDevice, gpu_transferNodeIterationStateFromDevice.
    - Full result flush later: gpu_transferLinkDynamicFromDevice, gpu_transferConduitDynamicFromDevice.

---

## Current Execution Model (Per Picard Iteration)

1) Host updates h_* dynamic arrays (links/nodes, etc.) and calls:
   - gpu_transfer*DynamicToDevice to update d_*.

2) GPU kernels:
   - kernel_findConduitFlows (conduits)
   - kernel_findPumpFlows (pumps)
   - kernel_findOrificeFlows (orifices)
   - kernel_findWeirFlows (weirs)
   - kernel_findOutletFlows (outlets)

3) Synchronize (cudaStreamSynchronize), record timing.

4) Device→Host copies of minimal iteration results:
   - gpu_transferLinkIterationResultsFromDevice (newFlow, newDepth)
   - gpu_transferNodeIterationStateFromDevice (inflow/outflow, newSurfArea, sumdqdh, converged)
   - CPU writes into SWMM structs (Link, Node, Xnode).

5) GPU computes node depths (gpu_runNodeDepthKernel) and checks convergence on device
   - Kernel returns convergence status to CPU
   - Simulation aborts with error if GPU kernel fails (no CPU fallback)

6) Repeat until convergence or max iterations.

Note: Full dynamic state flush (copy all results back) occurs later via gpu_flushConduitResults().

---

## Does It Load Once and Launch One Kernel per Step?

- No (but improving). Current state:
  - Launches multiple kernels per iteration (one per link type + one for node depths)
  - GPU computes both link flows AND node depths (Phase 1 complete)
  - Still requires per-iteration device→host transfers and synchronization
  - Not yet a single fused kernel per timestep nor a persistent kernel (Phases 2-3 remain)

---

## Hotspots and Overheads

- Multiple kernel launches per iteration (conduit + each non-conduit type + node depths), plus a stream synchronize.
- Per-iteration Device→Host transfers (link flows/depths, node inflow/outflow, surface areas, sumdqdh, flags).
- ~~Node solver on CPU introduces GPU→CPU→GPU handoffs~~ NOW FIXED: Node solver runs on GPU.
- ~~Unified-memory-only gating prevents using node GPU kernel~~ NOW FIXED: Explicit memory model is fully supported.

---

## Recommended Refactor Path

Two viable device-centric execution strategies:

1) Fused “iteration kernel” per Picard iteration
   - Single launch per iteration:
     - Zero node accumulators (inflow, outflow, newSurfArea, sumdqdh).
     - Compute flows for all link types; aggregate into node accumulators atomically (already done in each specialized kernel).
     - Compute node depths for all nodes; set converged flags.
     - Run on-device reduction (or cooperative groups) to count converged nodes.
   - Host copies back only a single small convergence scalar (e.g., int totalConverged) or uses cudaMemcpyFromSymbol, and decides to continue/stop iterations.
   - Copy final results to host only at timestep end (or when needed).

2) Persistent “timestep kernel” (preferred for “one kernel per step”)
   - Launch one cooperative kernel per timestep:
     - for (iter = 0; iter < MaxTrials; ++iter) { zero accumulators; compute link flows; compute node depths; compute convergence; if converged break; }
   - Host waits for kernel completion, then copies final results once.
   - Benefits: single launch per timestep; no host intervention mid-iteration; maximal residency.

Device-side pseudo-structure for persistent kernel:

```
__global__ void timestep_kernel(DeviceState S, Constants C) {
  for (int iter = 0; iter < C.maxIters; ++iter) {
    // 1) zero node accumulators
    for_each_node_parallel(i) { S.nodes.inflow[i]=0; S.nodes.outflow[i]=S.nodes.losses[i]; S.nodes.newSurfArea[i]=area_from_depth(...); S.nodes.sumdqdh[i]=0; }

    // 2) compute all link types in one pass (branch by type)
    for_each_link_parallel(j) {
      switch (S.links.type[j]) {
        case CONDUIT: compute_conduit(j, ... S.d_* ...); break;
        case PUMP:    compute_pump(j, ...); break;
        case ORIFICE: compute_orifice(j, ...); break;
        case WEIR:    compute_weir(j, ...); break;
        case OUTLET:  compute_outlet(j, ...); break;
      }
    }

    // 3) compute node depths
    __syncthreads_or_grid_sync(); // cooperative groups
    for_each_node_parallel(i) { compute_node_depth(i, ...); }

    // 4) convergence reduction
    int converged = device_reduce_all_nodes_converged(...);
    if (converged) break;
  }
}
```

---

## Concrete Steps (Actionable)

Phase 1: Enable GPU node depth with explicit memory ✅ COMPLETE
- ✅ Removed unified memory restriction in src/solver/gpu/gpu_dynwave.cu::ensureNodeKernelContext
- ✅ Enabled GPU node path in src/solver/dynwave.c::findNodeDepths (lines 750-771)
- ✅ gpu_runNodeDepthKernel uses explicit-memory GPU_NodeData (d_* arrays)
- ✅ Device-side convergence checking implemented with gpu_computeNodeDepths
- REMAINING: Remove per-iteration Device→Host transfers (optimization for Phase 2+)
  - TODO: Eliminate gpu_transferLinkIterationResultsFromDevice and gpu_transferNodeIterationStateFromDevice from iteration path
  - TODO: Keep Link and Node dynamic state on device across iterations
  - TODO: Only flush results at timestep end via gpu_flushConduitResults()

Phase 2: Fuse link kernels into one kernel per iteration
- Combine kernel_findConduitFlows, kernel_findPumpFlows, kernel_findOrificeFlows, kernel_findWeirFlows, kernel_findOutletFlows into a single kernel, branching by link type. This ensures:
  - One launch per iteration instead of up to five.
  - One walk over links; all atomics contribute to the same node arrays in that pass.
- Remove the cudaStreamSynchronize between different link-type kernels (no longer needed).
- Keep timing with events around the single kernel if required.

Phase 3: Persistent kernel per timestep
- Switch from “one kernel per iteration” to a single persistent kernel that loops internally over iterations.
- Use cooperative groups for grid-wide synchronization (or split into phases by separate launches as an intermediate step).
- Host only provides dt, omega, tolerances; receives a success code and kernel time. Copy back results once.

Phase 4: Minimize host-device control traffic
- For control rules or changing settings (e.g., links->setting), update only the touched fields via small cudaMemcpyAsync of those arrays or compute control logic on device if feasible.
- Avoid debug logging that requires Device→Host per iteration. Keep optional logging behind a runtime flag and throttle to end-of-timestep snapshots.

Phase 5: Validation and performance
- Add device-side asserts and optional comparisons for a small test model (tests/test_models/simple_test.inp).
- Compare GPU and CPU outputs post-step with scripts/compare_runswmm_gpu_cpu.sh.
- Profile with Nsight Systems: kernel launches per timestep should drop to 1 (persistent) or 1-2 (if keeping a small convergence-check kernel).

---

## Migration Notes and Considerations

- Memory footprint: Explicit model already allocates both h_* and d_*; persistent residency is consistent with current design. Watch total device memory for large models; consider compressing types (e.g., int32 for indices, float for some intermediates) where numerically acceptable.
- Outfalls: Currently handled on CPU. Either keep minimal host logic and pass required data to device before the iteration, or move outfall depth handling to device as a pre-phase in the iteration kernel.
- Synchronization: With fused kernel, grid-wide sync may be needed between link and node phases; cooperative groups are recommended for persistent kernel design.
- Numerical parity: Keep under-relaxation, crown cutoff, and flow class semantics identical between CPU and GPU. Preserve current physics in gpu_* helpers.
- Debug hooks: The debug log in gpu_dwflow.cu writes to /tmp. Avoid triggering per-iteration Device→Host copies when performance testing; enable only when isolating discrepancies.

---

## Acceptance Criteria

- Per timestep, either:
  - Exactly one kernel launch runs the entire Picard loop (persistent kernel), or
  - Exactly one fused “iteration kernel” is launched per Picard iteration, with no mid-iteration device→host transfers; node solver runs on device; the host checks a single small convergence flag.
- No per-iteration cudaMemcpy of large arrays; transfers only at timestep boundaries or for tiny control scalars.
- End-to-end correctness validated against CPU results within acceptable tolerances on provided test models.
- Measurable reduction in kernel launch count and memcpy volume; gpu_perf stats reflect increased kernel time share and decreased memcpy time share.

---

## Quick Checklist

- [x] Re-enable GPU node path in dynwave.c (DONE - lines 750-771)
- [x] Remove unified-memory gate; support explicit memory in gpu_dynwave.cu (DONE - ensureNodeKernelContext)
- [x] Implement device-side convergence reduction (DONE - gpu_computeNodeDepths)
- [ ] Stop per-iteration Device→Host transfers; retain results on device until timestep ends
- [ ] Fuse link-type kernels into one per iteration
- [ ] Optional: Move to persistent timestep kernel
- [ ] Validate results; profile performance
