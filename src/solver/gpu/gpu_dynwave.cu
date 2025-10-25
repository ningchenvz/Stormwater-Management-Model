//-----------------------------------------------------------------------------
//   gpu_dynwave.cu
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   GPU kernels for dynamic wave routing.
//   Main kernel: gpu_findNodeDepths - computes new node depths in parallel
//
//-----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <stdio.h>
#include "gpu_config.h"
#include "gpu_structures.h"
#include "gpu_dynwave_kernels.cuh"

//=============================================================================
// Kernel: Find Node Depths
//=============================================================================

__global__ void kernel_findNodeDepths(
    GPU_NodeData* nodes,
    double dt,
    int allowPonding,
    int surchargeMethod,
    double minSurfArea,
    int steps,
    double omega,
    double headTol)
//
//  Purpose: Computes new depth at all non-outfall nodes
//  Input:   nodes = GPU node data structure
//           dt = time step (sec)
//           allowPonding = TRUE if ponding allowed
//           surchargeMethod = EXTRAN or SLOT
//           minSurfArea = minimum surface area (ft2)
//           steps = current Picard iteration number
//           omega = under-relaxation parameter
//           headTol = convergence tolerance (ft)
//  Output:  Updates nodes->newDepth, nodes->newVolume, nodes->overflow
//           Sets nodes->converged flag for each node
//
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nodes->count) return;

    // Skip outfall nodes (handled separately on CPU)
    if (nodes->type[i] == GPU_OUTFALL) {
        nodes->converged[i] = 1;  // Always converged
        return;
    }

    // Store previous depth for convergence check
    double yOld = nodes->newDepth[i];

    // Call device function to compute new depth
    double newDepth, newVolume, overflow, oldSurfArea, dYdT;

    gpu_setNodeDepth(
        i, dt, allowPonding, surchargeMethod, minSurfArea, steps, omega,
        // Node data
        nodes->type[i],
        nodes->invertElev[i],
        nodes->fullDepth[i],
        nodes->surDepth[i],
        nodes->pondedArea[i],
        nodes->crownElev[i],
        nodes->oldDepth[i],
        nodes->oldNetInflow[i],
        nodes->inflow[i],
        nodes->outflow[i],
        nodes->fullVolume[i],
        nodes->degree[i]
        // Xnode data
        nodes->newSurfArea[i],
        nodes->oldSurfArea[i],
        nodes->sumdqdh[i],
        // Previous iteration value
        yOld,
        // Outputs
        &newDepth,
        &newVolume,
        &overflow,
        &oldSurfArea,
        &dYdT);

    // Update node state
    nodes->newDepth[i] = newDepth;
    nodes->newVolume[i] = newVolume;
    nodes->overflow[i] = overflow;
    nodes->oldSurfArea[i] = oldSurfArea;

    // Check convergence
    double depthChange = fabs(newDepth - yOld);
    nodes->converged[i] = (depthChange <= headTol) ? 1 : 0;
}

//=============================================================================
// Host Function: Launch Node Depths Kernel
//=============================================================================

extern "C" {

int gpu_computeNodeDepths(
    GPU_NodeData* nodes,
    double dt,
    int allowPonding,
    int surchargeMethod,
    double minSurfArea,
    int steps,
    double omega,
    double headTol,
    int* convergedCount)
//
//  Purpose: Launches GPU kernel to compute node depths
//  Returns: 0 if successful, error code otherwise
//
{
    if (nodes == NULL || nodes->count <= 0) return -1;

    // Determine launch configuration
    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = GRID_SIZE(nodes->count, blockSize);

    // Launch kernel
    kernel_findNodeDepths<<<gridSize, blockSize>>>(
        nodes, dt, allowPonding, surchargeMethod,
        minSurfArea, steps, omega, headTol);

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Count converged nodes (reduction on CPU for now)
    // TODO: Implement GPU reduction for better performance
    *convergedCount = 0;
    for (int i = 0; i < nodes->count; i++) {
        if (nodes->converged[i]) (*convergedCount)++;
    }

    return 0;
}

} // extern "C"
