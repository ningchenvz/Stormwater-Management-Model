# GPU Performance Improvement Roadmap

**Date:** 2025-10-27  
**Owner:** Codex (GPU Acceleration Team)

We want the GPU path to deliver order-of-magnitude speedups over the CPU (≥10×). The table
below captures the staged optimizations required to reach that goal. Each row builds on the
previous ones; the quoted speedups are approximate multipliers relative to today’s baseline.

| Milestone                         | Target Speedup | Rationale / Actions |
|----------------------------------|----------------|---------------------|
| **Baseline (current)**           | 1×             | GPU matches CPU because unified memory, per-iteration copies, and multiple kernel launches erase any raw throughput advantage. |
| **Explicit device memory**       | 5×             | Replace unified memory with explicit `cudaMalloc` + `cudaMemcpy` (or prefetch) so we control exactly when/what moves. Eliminates page faults and lets us overlap DMA with compute. |
| **Reduce transfer frequency**    | 10×            | Keep the entire Picard loop on the device. Refresh state only once per time step; copy back a small summary for reporting. Requires GPU-side convergence checks and dirty flags. |
| **Kernel fusion / persistent loop** | 15×        | Fuse conduit + node kernels (or use a persistent kernel) so a single launch performs several Picard iterations and only syncs when converged. Removes per-iteration launch latency and host scheduling. |
| **Large-network focus (≥1000 links)** | 20×     | Restrict GPU execution to large models where thousands of conduits/nodes keep the SMs busy. Small unit tests stay on CPU unless explicitly forced. |
| **Asynchronous streams + overlap** | 25×        | Run copies and kernels in parallel (multiple CUDA streams); overlap DMA with computation, batch multiple time steps, and hide remaining latency. |

## Implementation Status

| Item                                   | Status | Notes |
|---------------------------------------|--------|-------|
| Explicit device memory                 | ✓      | Pinned host buffers + device arrays are live; all kernels read/write `d_*` with explicit transfers. |
| Reduce transfer frequency              | ◑      | Per-Picard copies trimmed to only the data CPU still needs (non-conduit flows, node scalars). Full flush happens once per time step. |
| Kernel fusion / persistent loop        | ◑      | Node-flow accumulation fused into conduit kernel; node-depth kernel shares stream but still launches separately. |
| Large-network focus                    | ✓      | GPU disabled for networks with <750 conduits unless `SWMM_FORCE_CUDA=1`. |
| Async streams / overlap                | ◑      | Single non-blocking stream in use; no overlapping DMA yet. |

## Next Actions

1. **GPU Support for Non‑Conduit Links**
   - Implement GPU kernels for pumps, orifices, weirs, regulators, storage nodes, etc., and wire
     them into the Picard loop so the GPU handles every link type the CPU does.
     - Keep feature parity between CPU and GPU paths to avoid fallback or mismatched continuity.

2. **Persistent GPU Picard Loop**
   - Keep node depth convergence checks on-device and expose a single flag to the host.
   - Convert the conduit+node kernels into a persistent kernel (or a small set of fused kernels)
     that iterates until convergence without host intervention.

3. **Selective Reporting Copies**
   - After each time step, copy back only the data needed for reports (continuity totals, link/node
     summaries). Everything else stays on the GPU for the next step.

4. **Asynchronous Transfers**
   - Overlap DMA and computation via multiple CUDA streams (e.g., prefetch next iteration’s node
     data while the current iteration runs).

5. **Performance Validation**
   - Profile large benchmark models (≥10k conduits) with Nsight to confirm SM occupancy,
     memory bandwidth, and atomic contention. Tune block sizes/hardware settings based on data.

Following this roadmap should move us from “GPU ≈ CPU” to the ≥10× improvement target on
production-scale networks while keeping the code manageable for smaller models (which now stay
on the CPU by default).
