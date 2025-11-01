## Analysis of SWMM-GPU Execution Model

(Ignoring any existing reports under doc/gpu as per feedback. This is a fresh analysis based on code inspection. Focusing on the GPU execution path per user request, with added outline of functions still relying on CPU.)

### 1. Inspection of GPU Modules: Kernel Launches and Data Transfers (GPU Path Focus)

Assuming GPU is enabled (g_gpuConfig.useCuda == 1), the flow routing in dynwave.c calls into GPU functions for link flows, but nodes remain on CPU. Inspected key files: gpu_dwflow.cu, gpu_dynwave.cu, gpu_memory.cu, gpu_manager.cu, and dynwave.c integration.

- **Kernel Launches (in GPU Path)**:
  - gpu_computeConduitFlows() in gpu_dwflow.cu is called from dynwave.c::findLinkFlows when GPU is enabled.
    - Launches multiple kernels per Picard iteration:
      - kernel_findConduitFlows<<<gridSize, blockSize>>>() for conduits.
      - kernel_findPumpFlows<<<pumpGridSize, blockSize>>>() (if pumps > 0).
      - kernel_findOrificeFlows<<<orificeGridSize, blockSize>>>() (if orifices > 0).
      - kernel_findWeirFlows<<<weirGridSize, blockSize>>>() (if weirs > 0).
      - kernel_findOutletFlows<<<outletGridSize, blockSize>>>() (if outlets > 0).
    - Followed by cudaStreamSynchronize(stream).
  - Node depths: GPU kernel kernel_findNodeDepths<<<gridSize, blockSize>>>() exists in gpu_dynwave.cu, but disabled in dynwave.c::findNodeDepths (commented out). GPU path not taken; CPU fallback used.
  - Test kernels (e.g., test_vectorAdd<<<>>>() in gpu_test_kernels.cu) not part of runtime GPU path.

- **Data Transfers (in GPU Path)**:
  - Explicit memory: Pinned h_* (cudaMallocHost) + d_* (cudaMalloc). Transfers via cudaMemcpy.
    - Static: Once at init (gpu_transfer*StaticToDevice).
    - Dynamic per iteration: gpu_transfer*DynamicToDevice before kernels.
    - Iteration results: gpu_transferLinkIterationResultsFromDevice and gpu_transferNodeIterationStateFromDevice after kernels (for CPU node solver).
    - Full flush: gpu_flushConduitResults() later via gpu_transfer*DynamicFromDevice.
  - Non-conduit (pumps/orifices/weirs/outlets/curves): Allocated/copied once if !nonConduitStructuresInitialized.
  - Unified memory mentioned but not active in GPU path (explicit model used).

- **Timing/Synchronization (in GPU Path)**:
  - cudaEventRecord/ElapsedTime per iteration.
  - cudaStreamSynchronize after kernel batch.

### 2. Identification of Kernel Launches and Transfers: Fused or Fragmented? (GPU Path Focus)

In the GPU-enabled path for links:
- **Fragmented Launches**: 1-5 kernels per iteration (by link type), not fused. Separate grid/block per type.
- **Fragmented Transfers**: Per-iteration Host→Device (dynamic state) and Device→Host (iteration results) to interleave with CPU node computation.
- **Overall GPU Path Flow** (from dynwave.c::dynwave_execute while loop):
  - GPU: gpu_computeConduitFlows (multiple kernels + transfers + sync) for links.
  - CPU: findNodeDepths (loop over nodes).
  - CPU: Convergence check.
  - Frequent switches despite GPU for links.

### 3. Review of Design Docs: Intended Architecture (GPU Path Focus)

- doc/gpu/data-transfer-summary.md & src/solver/gpu/UNIFIED_MEMORY_STRATEGY.md: Explicit for performance, unified alternative. Suggests prefetch for residency, phase 4 persistent kernel goal (not yet in GPU path).
- src/solver/gpu/gpu_config.h: Supports explicit model; tracks launches/copies for optimization.
- Intended: Persistent kernels for resident state, but current GPU path fragments with per-iteration multi-kernels and CPU handoffs.

### 4. Report: State Residency and One-Kernel-Per-Step Status (GPU Path Focus)

- **State Resident on Device?** Partially: Persistent d_* allocations, static data uploaded once. But dynamic state recopied per iteration; results transferred back for CPU nodes, breaking residency.
- **One-Kernel-Per-Time-Step?** No: Multiple kernels per iteration for links; nodes on CPU. No persistent kernel.
- **Fragmentation/Transfer Hotspots in GPU Path**:
  - Multi-kernel launches per iteration.
  - Per-iteration transfers to/from device for CPU interop.
  - Sync after kernels.
  - Disabled GPU node kernel forces CPU fallback.

### 5. Outline of Functions Still Using CPU in GPU Path

Even when GPU is enabled for link flows, several functions remain on CPU (due to disabled node GPU path and control flow). Outline:

- **dynwave.c::findNodeDepths()**: Computes node depths on CPU (loops over nodes calling setNodeDepth()). GPU path commented out; forces CPU for all node updates each iteration.
- **dynwave.c::setNodeDepth()**: Core node depth math (e.g., dV calc, surcharge handling, getFloodedDepth). Called in CPU loop; would be in kernel_findNodeDepths if GPU enabled.
- **dynwave.c::link_setOutfallDepth()**: Sets outfall depths on CPU (looped before node depths).
- **dynwave.c::findNonConduitFlow()**: Fallback for non-conduits if GPU fails, but in GPU path, non-conduits are GPU-handled; however, any bypassed links or dummies may hit CPU paths.
- **dynwave.c::updateNodeFlows()**: Aggregates link flows into node inflow/outflow on CPU (called after GPU link kernels via copy*FromGpu).
- **dynwave.c::updateConvergenceStats()**: CPU convergence check and stats update.
- **dynwave.c::getVariableStep() / getLinkStep() / getNodeStep()**: Time step calc on CPU (scans links/nodes for Courant).
- **dynwave.c::initNodeStates() / findBypassedLinks() / findLimitedLinks()**: Pre/post-iteration setup on CPU (node surf areas, bypass flags, capacity limits).

These CPU functions interleave with GPU link computation, requiring transfers and preventing full device residency.

This setup in the GPU path still relies heavily on CPU for nodes and control, causing fragmentation.
