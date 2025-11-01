//-----------------------------------------------------------------------------
//   gpu_reduction.cuh
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/27/2025
//
//   GPU reduction kernels for convergence checking and sum/min/max operations.
//   Uses warp-level primitives and atomic operations for high performance.
//
//-----------------------------------------------------------------------------

#ifndef GPU_REDUCTION_CUH
#define GPU_REDUCTION_CUH

#include <cuda_runtime.h>
#include "gpu_config.h"

//=============================================================================
// Device-side Convergence Counter (atomic reduction)
//=============================================================================

__global__ void kernel_countConvergedNodes(
    const char* d_converged,
    int nodeCount,
    int* d_convergedCount)
//
//  Purpose: Counts how many nodes have converged using atomic operations
//  Input:   d_converged = array of convergence flags (1=converged, 0=not)
//           nodeCount = total number of nodes
//  Output:  d_convergedCount = total count of converged nodes (atomic accumulation)
//
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp-level reduction first
    int localCount = 0;
    for (int i = idx; i < nodeCount; i += blockDim.x * gridDim.x) {
        localCount += d_converged[i];
    }

    // Reduce within warp using shuffle operations
    for (int offset = 16; offset > 0; offset >>= 1) {
        localCount += __shfl_down_sync(0xffffffff, localCount, offset);
    }

    // First thread in each warp atomically adds to global counter
    if ((threadIdx.x & 31) == 0 && localCount > 0) {
        atomicAdd(d_convergedCount, localCount);
    }
}

//=============================================================================
// Fast Block-level Reduction (shared memory)
//=============================================================================

template<typename T, int BLOCK_SIZE>
__device__ void blockReduce_Sum(T* sdata, int tid)
//
//  Purpose: Reduces values within a block using shared memory
//  Input:   sdata = shared memory array (size >= BLOCK_SIZE)
//           tid = thread ID within block
//  Output:  sdata[0] contains the sum of all values
//
{
    __syncthreads();

    // Unrolled reduction for performance
    if (BLOCK_SIZE >= 1024 && tid < 512) sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (BLOCK_SIZE >= 512 && tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (BLOCK_SIZE >= 256 && tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (BLOCK_SIZE >= 128 && tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile T* smem = sdata;
        if (BLOCK_SIZE >= 64) smem[tid] += smem[tid + 32];
        if (BLOCK_SIZE >= 32) smem[tid] += smem[tid + 16];
        if (BLOCK_SIZE >= 16) smem[tid] += smem[tid + 8];
        if (BLOCK_SIZE >= 8) smem[tid] += smem[tid + 4];
        if (BLOCK_SIZE >= 4) smem[tid] += smem[tid + 2];
        if (BLOCK_SIZE >= 2) smem[tid] += smem[tid + 1];
    }
}

//=============================================================================
// Convergence Check Kernel (optimized)
//=============================================================================

__global__ void kernel_checkConvergenceOptimized(
    const double* d_newDepth,
    const double* d_oldDepth,
    const int* d_type,
    int nodeCount,
    double headTol,
    char* d_converged,
    int* d_convergedCount)
//
//  Purpose: Checks convergence and counts converged nodes in one pass
//  Input:   d_newDepth, d_oldDepth = node depths
//           d_type = node types (skip OUTFALL)
//           nodeCount = number of nodes
//           headTol = convergence tolerance (ft)
//  Output:  d_converged = convergence flags per node
//           d_convergedCount = total converged count
//
{
    __shared__ int blockConvergedCount;

    if (threadIdx.x == 0) {
        blockConvergedCount = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nodeCount) {
        // Skip outfalls
        if (d_type[idx] == GPU_OUTFALL) {
            d_converged[idx] = 1;
            atomicAdd(&blockConvergedCount, 1);
        } else {
            double depthChange = fabs(d_newDepth[idx] - d_oldDepth[idx]);
            char converged = (depthChange <= headTol) ? 1 : 0;
            d_converged[idx] = converged;

            if (converged) {
                atomicAdd(&blockConvergedCount, 1);
            }
        }
    }

    __syncthreads();

    // Block leader adds block total to global counter
    if (threadIdx.x == 0 && blockConvergedCount > 0) {
        atomicAdd(d_convergedCount, blockConvergedCount);
    }
}

#endif // GPU_REDUCTION_CUH
