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

extern "C" {
#include "headers.h"
#include "dynwave_data.h"
}

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
        nodes->degree[i],
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
    nodes->dYdT[i] = dYdT;

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

typedef struct
{
    GPU_NodeData data;
    int nodeCapacity;
    int initialized;
} NodeKernelContext;

static NodeKernelContext g_nodeKernelCtx = {0};
static cudaEvent_t g_nodeKernelStartEvent = nullptr;
static cudaEvent_t g_nodeKernelStopEvent = nullptr;
static int g_nodeKernelEventsInitialized = 0;

static int ensureNodeKernelContext(void)
{
    if (!g_gpuConfig.unifiedMemory) {
        return -1;
    }

    int nodeCount = Nobjects[NODE];
    if (nodeCount <= 0) return -1;

    if (!g_nodeKernelCtx.initialized ||
        nodeCount != g_nodeKernelCtx.nodeCapacity)
    {
        if (g_nodeKernelCtx.initialized) {
            gpu_freeNodeData(&g_nodeKernelCtx.data);
            g_nodeKernelCtx.initialized = 0;
        }
        if (gpu_allocateNodeData(&g_nodeKernelCtx.data, nodeCount) != 0) {
            return -1;
        }
        g_nodeKernelCtx.nodeCapacity = nodeCount;
        g_nodeKernelCtx.initialized = 1;
    }

    if (!g_nodeKernelEventsInitialized) {
        cudaEventCreateWithFlags(&g_nodeKernelStartEvent, cudaEventDefault);
        cudaEventCreateWithFlags(&g_nodeKernelStopEvent, cudaEventDefault);
        g_nodeKernelEventsInitialized = 1;
    }

    return 0;
}

static void copyNodesToGpu(GPU_NodeData* nodes)
{
    int count = nodes->count;
    for (int i = 0; i < count; i++)
    {
        nodes->type[i]        = Node[i].type;
        nodes->invertElev[i]  = Node[i].invertElev;
        nodes->fullDepth[i]   = Node[i].fullDepth;
        nodes->surDepth[i]    = Node[i].surDepth;
        nodes->pondedArea[i]  = Node[i].pondedArea;
        nodes->crownElev[i]   = Node[i].crownElev;
        nodes->oldDepth[i]    = Node[i].oldDepth;
        nodes->newDepth[i]    = Node[i].newDepth;
        nodes->oldVolume[i]   = Node[i].oldVolume;
        nodes->newVolume[i]   = Node[i].newVolume;
        nodes->fullVolume[i]  = Node[i].fullVolume;
        nodes->oldNetInflow[i]= Node[i].oldNetInflow;
        nodes->inflow[i]      = Node[i].inflow;
        nodes->outflow[i]     = Node[i].outflow;
        nodes->overflow[i]    = Node[i].overflow;
        nodes->degree[i]      = Node[i].degree;
        nodes->converged[i]   = Xnode[i].converged;
        nodes->newSurfArea[i] = Xnode[i].newSurfArea;
        nodes->oldSurfArea[i] = Xnode[i].oldSurfArea;
        nodes->sumdqdh[i]     = Xnode[i].sumdqdh;
        nodes->dYdT[i]        = Xnode[i].dYdT;
    }
}

static void copyNodesFromGpu(GPU_NodeData* nodes)
{
    int count = nodes->count;
    for (int i = 0; i < count; i++)
    {
        if (Node[i].type != OUTFALL)
        {
            Node[i].newDepth  = nodes->newDepth[i];
            Node[i].newVolume = nodes->newVolume[i];
            Node[i].overflow  = nodes->overflow[i];
        }
        Xnode[i].converged = nodes->converged[i];
        Xnode[i].newSurfArea = nodes->newSurfArea[i];
        Xnode[i].oldSurfArea = nodes->oldSurfArea[i];
        Xnode[i].sumdqdh     = nodes->sumdqdh[i];
        Xnode[i].dYdT        = nodes->dYdT[i];
    }
}

extern "C" int gpu_runNodeDepthKernel(
    double dt,
    int allowPonding,
    int surchargeMethod,
    double minSurfArea,
    int steps,
    double omega,
    double headTol)
//
//  Purpose: Copies SWMM node state to GPU, runs the node depth kernel,
//           and copies results back. Returns converged node count or -1 on failure.
//
{
    if (ensureNodeKernelContext() != 0) {
        return -1;
    }

    GPU_NodeData* nodes = &g_nodeKernelCtx.data;
    copyNodesToGpu(nodes);

    int convergedCount = 0;
    cudaEventRecord(g_nodeKernelStartEvent, 0);
    if (gpu_computeNodeDepths(
            nodes,
            dt,
            allowPonding,
            surchargeMethod,
            minSurfArea,
            steps,
            omega,
            headTol,
            &convergedCount) != 0)
    {
        return -1;
    }
    cudaEventRecord(g_nodeKernelStopEvent, 0);
    cudaEventSynchronize(g_nodeKernelStopEvent);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, g_nodeKernelStartEvent, g_nodeKernelStopEvent);
    gpu_profiler_addKernelTime((double)elapsedMs);

    copyNodesFromGpu(nodes);

    return convergedCount;
}
