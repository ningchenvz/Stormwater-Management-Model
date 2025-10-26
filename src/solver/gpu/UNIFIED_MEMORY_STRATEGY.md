# Unified Memory Strategy for SWMM-GPU

## Overview

The DGX Spark (NVIDIA GB10, Compute Capability 12.1) supports **unified memory with concurrent managed access**, which significantly simplifies GPU memory management compared to traditional discrete GPU architectures.

## Verified Capabilities

- ✅ **Unified Memory (Managed Memory):** Supported
- ✅ **Concurrent Managed Access:** Enabled
- ✅ **Total Memory:** 119.70 GB
- ✅ **Compute Capability:** 12.1

## Memory Management Approach

### For DGX Spark (Unified Memory)

```c
// Allocation
float *data;
cudaMallocManaged(&data, size);

// Access from CPU
for (int i = 0; i < n; i++) {
    data[i] = initialValue;  // No explicit copy needed
}

// Launch kernel
myKernel<<<grid, block>>>(data);

// Optional: Prefetch to GPU for better performance
cudaMemPrefetchAsync(data, size, deviceId);

// Access from CPU again (concurrent access supported)
cudaDeviceSynchronize();  // Only if kernel needs to complete first
float result = data[0];
```

### For RTX 4060 (Discrete GPU - Fallback)

```c
// Host allocation
float *h_data = (float*)malloc(size);

// Device allocation
float *d_data;
cudaMalloc(&d_data, size);

// Explicit copy to device
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Launch kernel
myKernel<<<grid, block>>>(d_data);

// Explicit copy back
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
```

## Implementation Strategy

### Phase 2: Data Structures

1. **Detect unified memory at runtime:**
   ```c
   if (g_gpuConfig.unifiedMemory) {
       // Use cudaMallocManaged
   } else {
       // Use cudaMalloc + cudaMemcpy
   }
   ```

2. **Create GPU-friendly Structure of Arrays (SoA):**
   ```c
   typedef struct {
       float *depth;      // All node depths
       float *newDepth;   // Updated depths
       float *volume;     // Node volumes
       int count;
   } GPU_NodeData;

   typedef struct {
       float *flow;       // All link flows
       float *newFlow;    // Updated flows
       float *depth;      // Link depths
       int count;
   } GPU_LinkData;
   ```

3. **Allocate with managed memory:**
   ```c
   void allocate_gpu_node_data(GPU_NodeData *data, int nodeCount) {
       data->count = nodeCount;
       cudaMallocManaged(&data->depth, nodeCount * sizeof(float));
       cudaMallocManaged(&data->newDepth, nodeCount * sizeof(float));
       cudaMallocManaged(&data->volume, nodeCount * sizeof(float));
   }
   ```

### Phase 5: Memory Optimization

For unified memory systems, use prefetch hints:

```c
// Before GPU computation
cudaMemPrefetchAsync(nodeData->depth, size, deviceId);
cudaMemPrefetchAsync(linkData->flow, size, deviceId);

// Launch kernels
findLinkFlows<<<grid, block>>>(...);

// Before CPU access (if needed)
cudaMemPrefetchAsync(nodeData->depth, size, cudaCpuDeviceId);
```

## CUDA Build & Activation Checklist

1. **Enable CUDA in CMake:** Add `-DENABLE_CUDA=ON` (defaults to ON for DGX Spark presets) so `gpu_*.cu` sources are compiled into the solver.
2. **Validate GPU availability:** At startup, call `cudaGetDeviceCount`/`cudaGetDeviceProperties`; log compute capability and unified memory support, then set `g_gpuConfig.useCuda`.
3. **Runtime toggle:** Expose a CLI/env toggle (e.g., `SWMM_USE_CUDA=1`) so users can opt-in/out without recompiling.
4. **Perf guard-rails:** Capture kernel + memcpy timings via `cudaEvent` pairs; print a short summary when CUDA is enabled so we can confirm the speedup path is active.
5. **Fallback path:** If CUDA initialization fails, fall back to CPU execution immediately and emit a warning rather than crashing.

## Performance Considerations

### Advantages on DGX Spark

1. **Simplified Code:** No explicit memory copies
2. **Reduced Boilerplate:** Less error-prone
3. **Flexibility:** Can interleave CPU/GPU work easily
4. **Memory Oversubscription:** Can use more than 120GB if needed (with paging)

### Best Practices

1. **Use prefetch hints** to guide data placement:
   - Prefetch to GPU before computation-heavy kernels
   - Prefetch to CPU only if CPU needs to read results mid-computation

2. **Minimize CPU access during GPU computation:**
   - Even with concurrent access, frequent CPU reads during GPU kernels can hurt performance
   - Use `cudaDeviceSynchronize()` before bulk CPU operations

3. **Batch operations:**
   - Keep data on GPU across multiple time steps
   - Only synchronize when absolutely necessary

4. **Test on both architectures:**
   - DGX Spark: unified memory path
   - RTX 4060: explicit copy path (fallback)

## Code Pattern Template

```c
// Initialization (once per simulation)
if (g_gpuConfig.unifiedMemory) {
    allocate_managed_memory();
} else {
    allocate_discrete_memory();
}

// Time-step loop
for (int step = 0; step < numSteps; step++) {
    // CPU prepares input (e.g., boundary conditions)
    // Data is already accessible on GPU with unified memory

    if (g_gpuConfig.unifiedMemory) {
        // Optional prefetch hint
        cudaMemPrefetchAsync(data, size, deviceId);
    } else {
        // Explicit copy
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    }

    // Launch GPU kernels
    computeFlows<<<grid, block>>>(data);

    // CPU can continue other work (with concurrent access)
    // ...

    // Synchronize only when needed
    if (needResults) {
        cudaDeviceSynchronize();
        processResults(data);  // Direct access with unified memory
    }
}

// Cleanup
if (g_gpuConfig.unifiedMemory) {
    cudaFree(data);  // Same call for managed memory
} else {
    free(h_data);
    cudaFree(d_data);
}
```

## References

- [CUDA Unified Memory Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming)
- [Unified Memory on Pascal and Volta](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [Concurrent Managed Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-between-host-and-device)

## Testing

To verify unified memory support:
```bash
cd /home/ningchenspark/workspace/Stormwater-Management-Model
cmake --build build
# Check GPU info during SWMM initialization - it will report unified memory status
```
