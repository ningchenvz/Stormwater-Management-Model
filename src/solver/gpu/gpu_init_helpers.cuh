//-----------------------------------------------------------------------------
//   gpu_init_helpers.cuh
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    11/01/2025
//
//   GPU helper kernels for initializing node states between Picard iterations
//
//-----------------------------------------------------------------------------

#ifndef GPU_INIT_HELPERS_CUH
#define GPU_INIT_HELPERS_CUH

#include <cuda_runtime.h>
#include "gpu_structures.h"

//=============================================================================
// Simple Node State Initialization Kernel
//=============================================================================

__global__ void kernel_initNodeStates(
    GPU_NodeData* nodes,
    double* d_newLatFlow,
    double* d_losses,
    int allowPonding)
//
//  Purpose: Initialize node inflow/outflow and sumdqdh at start of each Picard iteration
//           This is the GPU equivalent of CPU initNodeStates() in dynwave.c:411
//
//  Input:   nodes = GPU node data
//           d_newLatFlow = lateral inflows (can be negative)
//           d_losses = node losses
//           allowPonding = TRUE if ponding allowed
//
//  Output:  Updates nodes->d_inflow, nodes->d_outflow, nodes->d_sumdqdh
//           Surface area is already computed by kernel_findNodeDepths
//
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodes->count) return;

    // Initialize inflow to 0, outflow to losses
    double inflow = 0.0;
    double outflow = d_losses[i];

    // Add lateral flow (can be positive inflow or negative outflow)
    double latFlow = d_newLatFlow[i];
    if (latFlow >= 0.0) {
        inflow += latFlow;
    } else {
        outflow -= latFlow;  // Negative latFlow increases outflow
    }

    nodes->d_inflow[i] = inflow;
    nodes->d_outflow[i] = outflow;

    // Reset sumdqdh (sum of dq/dh for all links connected to node)
    nodes->d_sumdqdh[i] = 0.0;

    // NOTE: Surface area (newSurfArea) is computed by kernel_findNodeDepths
    // based on current depth, so we don't need to recompute it here
}

#endif // GPU_INIT_HELPERS_CUH
