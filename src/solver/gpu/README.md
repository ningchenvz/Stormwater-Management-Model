# SWMM-GPU: CUDA Implementation

This directory contains GPU-accelerated implementations of SWMM's dynamic wave routing algorithms using CUDA.

## Files

### Core Infrastructure
- **gpu_config.h** - GPU configuration, macros, and feature detection
- **gpu_manager.cu/.cuh** - GPU memory management (allocation, transfer, deallocation)
- **gpu_data_structures.h** - Structure-of-Arrays (SoA) data layouts for GPU

### CUDA Kernels
- **dynwave_kernels.cu/.cuh** - Main dynamic wave routing kernels
  - `findLinkFlows_kernel` - Compute flow through conduits
  - `findNodeDepths_kernel` - Compute water depth at nodes
- **dwflow_device.cu/.cuh** - Device helper functions (getArea, getHydRad, etc.)
- **reduction_kernels.cu/.cuh** - Parallel reduction for convergence checking

### Utilities
- **gpu_utils.cuh** - Utility functions and error checking macros

## Architecture

The GPU implementation follows a hybrid CPU/GPU design:

```
CPU (Host)                          GPU (Device)
-----------                         ------------
Project initialization
├─ Read input file
├─ Allocate host data
└─ Initialize GPU manager
                                    ├─ Allocate device memory
                                    └─ Copy data to GPU

Simulation loop
├─ For each time step:
│   ├─ Picard iterations:
│   │   ├─ Transfer boundary     → └─ Process on GPU:
│   │   │   conditions               ├─ findLinkFlows_kernel
│   │   ├─ Launch kernels →           └─ findNodeDepths_kernel
│   │   └─ Check convergence ←    ├─ Reduction for convergence
│   │                              └─ Return convergence flag
│   └─ Update statistics (CPU)
└─ Write results

Cleanup
├─ Copy final results from GPU
└─ Free GPU memory
```

## Data Structure Conversion

SWMM uses Array-of-Structures (AoS), which is inefficient for GPUs. We convert to Structure-of-Arrays (SoA):

### CPU (Original)
```c
struct TLink {
    int node1, node2;
    double offset1, offset2;
    // ... many fields
} Link[1000];

// Access: Link[i].node1
```

### GPU (Converted)
```c
struct LinkDataGPU {
    int* node1;        // [1000]
    int* node2;        // [1000]
    double* offset1;   // [1000]
    // ... separate arrays
};

// Access: node1[i]  ← Better memory coalescing!
```

## Performance Considerations

### When to Use GPU
- **Models with 500+ links** - Overhead becomes worthwhile
- **Long simulations** - Data transfer amortized over many time steps
- **Complex networks** - More parallel work available

### When to Use CPU
- **Small models (<500 links)** - CPU overhead lower
- **Short simulations** - GPU data transfer overhead dominates
- **No GPU available** - Automatic fallback to OpenMP

### Memory Transfer Strategy
- **Initialization:** Transfer all static data once (topology, geometry)
- **Per time step:** Only transfer boundary conditions and results
- **Keep intermediate results on GPU** across Picard iterations

## Build Configuration

Enabled with CMake option:
```bash
cmake -DBUILD_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=89
```

Runtime configuration in SWMM input file:
```
[OPTIONS]
GPU_ENABLED         YES
GPU_MIN_LINKS       500
```

## Testing

GPU kernels must produce **bit-identical** results to CPU implementation:
- Unit tests compare CPU vs GPU for each kernel
- Regression tests use existing test suite
- Tolerance: Machine epsilon for floating-point

## References

- SWMM 5 Hydraulics Manual
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- GPU Gems 3: Chapter on CFD simulations
