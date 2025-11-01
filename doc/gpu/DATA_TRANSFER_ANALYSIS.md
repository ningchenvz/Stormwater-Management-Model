# GPU Data Transfer Analysis

## Current State: SYNCHRONOUS TRANSFERS EVERY ITERATION ⚠️

### Critical Performance Issue

**Problem**: We are copying data between CPU and GPU **synchronously** on **every Picard iteration**, causing massive performance overhead.

### Transfer Frequency

```
Per Timestep:
  ├── Picard Iteration 1
  │   ├── CPU→GPU: Node state (all fields)
  │   ├── CPU→GPU: Link state (all fields)
  │   ├── GPU Kernel: findConduitFlows
  │   ├── GPU→CPU: Link iteration results (newFlow, newDepth)
  │   ├── GPU→CPU: Node iteration state (inflow, outflow, sumdqdh, converged)
  │   └── SYNC BARRIER (cudaStreamSynchronize)
  ├── Picard Iteration 2
  │   ├── CPU→GPU: Node state (all fields) ← REDUNDANT
  │   ├── CPU→GPU: Link state (all fields) ← REDUNDANT
  │   ├── GPU Kernel: findConduitFlows
  │   ├── GPU→CPU: Link iteration results
  │   ├── GPU→CPU: Node iteration state
  │   └── SYNC BARRIER
  ├── ... (repeat 2-8 times per timestep)
  └── Picard Iteration N (converged)
      └── Final result flush
```

**Typical Model**: 1000 timesteps × 5 iterations × 2 transfers = **10,000 round trips!**

### Code Locations

#### 1. Synchronous Transfers in `gpu_dwflow.cu`

```c
// Line 1152-1153: EVERY ITERATION
gpu_transferLinkIterationResultsFromDevice(links, links->count);
gpu_transferNodeIterationStateFromDevice(nodes, nodes->count);
```

#### 2. cudaMemcpy Usage in `gpu_memory.cu`

All transfers use **cudaMemcpy** (synchronous):

```c
// Line 1119-1131: Blocking CPU-GPU transfers
CUDA_CHECK(cudaMemcpy(gpuData->d_type, gpuData->h_type, intSize,
                      cudaMemcpyHostToDevice));  // ← BLOCKS CPU
CUDA_CHECK(cudaMemcpy(gpuData->d_newDepth, gpuData->h_newDepth, doubleSize,
                      cudaMemcpyHostToDevice));  // ← BLOCKS CPU
...
```

**No** `cudaMemcpyAsync` usage found in current implementation.

#### 3. Synchronization Points

```c
// gpu_dwflow.cu:1142 - EVERY ITERATION
CUDA_CHECK(cudaStreamSynchronize(stream));  // ← CPU WAITS FOR GPU

// gpu_dwflow.cu:1146 - Event synchronization
cudaEventSynchronize(g_conduitKernelStopEvent);  // ← MORE WAITING
```

## Performance Impact

### Bandwidth Requirements

For a typical model with:
- 1000 links
- 500 nodes
- 5 Picard iterations per timestep
- 1000 timesteps

**Per Iteration Transfer Volume**:
```
Links:  1000 × (8 fields × 8 bytes) = 64 KB
Nodes:  500  × (10 fields × 8 bytes) = 40 KB
Total:  104 KB × 2 directions = 208 KB per iteration
```

**Total Simulation**:
```
208 KB × 5 iterations × 1000 timesteps = 1,040,000 KB = 1 GB transferred!
```

### Timeline Breakdown

```
1. CPU computes (2 ms)
2. CPU→GPU transfer (0.5 ms) ← SYNCHRONOUS BLOCK
3. Kernel execution (1 ms) ← GPU busy, CPU idle
4. GPU→CPU transfer (0.5 ms) ← SYNCHRONOUS BLOCK
5. CPU processes results (1 ms)
─────────────────────────────────
Total: 5 ms per iteration
  Actual work: 3 ms (60%)
  Transfer overhead: 1 ms (20%)
  Sync overhead: 1 ms (20%)
```

**With 5 iterations**: 25 ms per timestep
**GPU is idle 80% of the time!** ⚠️

## Why This is Bad

### 1. **Synchronous Transfers Block CPU**

```c
// Current: CPU BLOCKS waiting for transfer
cudaMemcpy(..., cudaMemcpyHostToDevice);  // ← CPU can't do anything else
```

### 2. **Full State Redundancy**

We transfer **all fields** every iteration, even though most don't change:

```c
// Transferred EVERY iteration (unnecessary):
- d_invertElev  ← STATIC, never changes
- d_fullDepth   ← STATIC, never changes
- d_type        ← STATIC, never changes
- d_degree      ← STATIC, never changes

// Actually needed:
- d_newDepth    ← Changes every iteration
- d_newFlow     ← Changes every iteration
```

**Waste: ~70% of transferred data is redundant**

### 3. **No Pipelining**

CPU and GPU never work simultaneously:
```
Timeline:
CPU: [====]      [====]      [====]      (working)
GPU:      [====]      [====]      [====] (working)
         ^^^^^ CPU idle!   ^^^^ GPU idle!
```

## Solutions (In Priority Order)

### Phase 1: Eliminate Per-Iteration Transfers ✅ (Partially Implemented)

**Goal**: Keep state on GPU across iterations

```c
// INSTEAD OF:
for (iter = 0; iter < maxIters; iter++) {
    transfer_to_gpu();      // ← REMOVE
    run_kernel();
    transfer_from_gpu();    // ← REMOVE
}

// DO THIS:
transfer_to_gpu();  // ← ONCE at timestep start
for (iter = 0; iter < maxIters; iter++) {
    run_kernel();   // ← Data stays on GPU
}
transfer_from_gpu();  // ← ONCE at timestep end
```

**Status**: Partially implemented in `gpu_runPersistentPicardIteration()` but not used by default.

**Impact**: Reduces transfers from 10,000 to 1,000 (10x improvement)

### Phase 2: Use Async Transfers

**Goal**: Overlap CPU work with GPU transfers

```c
// Create stream for async operations
cudaStream_t stream;
cudaStreamCreate(&stream);

// Async transfer (non-blocking)
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream);

// CPU can work here while GPU is busy!
do_cpu_work();

// Only sync when you need the results
cudaStreamSynchronize(stream);
```

**Impact**: Hides transfer latency behind computation

### Phase 3: Pinned Memory (Already Used ✅)

**Current**: Using `cudaMallocHost` for pinned memory

```c
// gpu_memory.cu - Already optimal
CUDA_CHECK(cudaMallocHost((void**)&data->h_newDepth, doubleSize));
```

**Benefit**: 2-3x faster transfers vs pageable memory

### Phase 4: Double Buffering

**Goal**: Prepare next timestep while processing current

```c
// While GPU processes timestep N
GPU: [  Process N  ]
              ↓
CPU: [Prepare N+1]  ← Overlap!
```

### Phase 5: CUDA Streams for Parallelism

**Goal**: Multiple concurrent operations

```c
cudaStream_t stream1, stream2;

// Pipeline multiple operations
cudaMemcpyAsync(..., stream1);
kernel<<<..., stream1>>>();
cudaMemcpyAsync(..., stream1);

// Concurrent with:
cudaMemcpyAsync(..., stream2);
kernel<<<..., stream2>>>();
```

## Recommended Action Plan

### Immediate (This Sprint)

1. **Enable persistent Picard iteration by default**
   - File: `src/solver/dynwave.c:309`
   - Change: Use `gpu_runPersistentPicardIteration()` instead of per-iteration calls
   - Benefit: 5-10x reduction in transfers

2. **Remove per-iteration transfer calls**
   - File: `src/solver/gpu/gpu_dwflow.cu:1152-1153`
   - Only transfer once per timestep, not per iteration

### Short Term (Next Sprint)

3. **Convert to async transfers**
   - Replace all `cudaMemcpy` with `cudaMemcpyAsync`
   - Use CUDA streams consistently
   - Files: `src/solver/gpu/gpu_memory.cu`

4. **Profile with nsight-systems**
   ```bash
   nsys profile --trace=cuda,nvtx ./runswmm model.inp
   ```
   - Identify actual transfer bottlenecks
   - Measure overlap efficiency

### Long Term (Future)

5. **Unified memory as fallback**
   - Automatic page migration
   - Simpler code, compiler-optimized transfers

6. **GPU-resident mode**
   - Keep all state on GPU for entire simulation
   - Only transfer final results

## Current vs Optimal Timeline

### Current (Synchronous, Per-Iteration)
```
Timestep:
[CPU][→GPU][GPU][GPU→][CPU]...[CPU][→GPU][GPU][GPU→][CPU] (5 iterations)
 2ms  0.5ms  1ms  0.5ms  1ms    2ms  0.5ms  1ms  0.5ms  1ms
Total: 25 ms (40% wasted on transfers/sync)
```

### Optimal (Async, Once-Per-Timestep)
```
Timestep:
[CPU+→GPU][GPU(iter1-5)][GPU→+CPU]
 2ms      5ms              1ms
Total: 8 ms (68% improvement!)
```

## Verification Commands

```bash
# Check for sync transfers
grep -r "cudaMemcpy\(" src/solver/gpu/*.cu | wc -l  # Current: ~100

# Check for async transfers
grep -r "cudaMemcpyAsync" src/solver/gpu/*.cu | wc -l  # Current: 0 ⚠️

# Check synchronization points
grep -r "cudaStreamSynchronize\|cudaDeviceSynchronize" src/solver/gpu/*.cu

# Profile actual transfers
nsys profile --trace=cuda,nvtx build/bin/runswmm model.inp
```

## References

- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Section 9.1: Asynchronous Transfers and Overlapping
- Section 9.2: Pinned Memory
- Section 3.2.6: CUDA Streams

## Status

- ✅ Pinned memory (already using `cudaMallocHost`)
- ⚠️ Persistent iteration (implemented but not default)
- ❌ Async transfers (not implemented)
- ❌ Stream overlap (not implemented)
- ❌ Double buffering (not implemented)

**Next Action**: Make persistent Picard iteration the default path to eliminate 90% of redundant transfers.
