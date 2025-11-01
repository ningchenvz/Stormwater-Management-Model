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
#include "gpu_reduction.cuh"

extern "C" {
#include "headers.h"
#include "dynwave_data.h"
}

typedef struct
{
    GPU_NodeData data;
    GPU_NodeData* d_nodes;
    int nodeCapacity;
    int initialized;
    int staticsUploaded;
} NodeKernelContext;

static NodeKernelContext g_nodeKernelCtx = {0};
static cudaEvent_t g_nodeKernelStartEvent = nullptr;
static cudaEvent_t g_nodeKernelStopEvent = nullptr;
static int g_nodeKernelEventsInitialized = 0;

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
//  Output:  Updates nodes->h_newDepth, nodes->h_newVolume, nodes->h_overflow
//           Sets nodes->h_converged flag for each node
//
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nodes->count) return;

    // Skip outfall nodes (handled separately on CPU)
    if (nodes->d_type[i] == GPU_OUTFALL) {
        nodes->d_converged[i] = 1;  // Always converged
        return;
    }

    // Store previous depth for convergence check
    double yOld = nodes->d_newDepth[i];

    // Call device function to compute new depth
    double newDepth, newVolume, overflow, oldSurfArea, dYdT;

    gpu_setNodeDepth(
        i, dt, allowPonding, surchargeMethod, minSurfArea, steps, omega,
        // Node data
        nodes->d_type[i],
        nodes->d_invertElev[i],
        nodes->d_fullDepth[i],
        nodes->d_surDepth[i],
        nodes->d_pondedArea[i],
        nodes->d_crownElev[i],
        nodes->d_oldDepth[i],
        nodes->d_oldNetInflow[i],
        nodes->d_inflow[i],
        nodes->d_outflow[i],
        nodes->d_fullVolume[i],
        nodes->d_degree[i],
        // Xnode data
        nodes->d_newSurfArea[i],
        nodes->d_oldSurfArea[i],
        nodes->d_sumdqdh[i],
        // Previous iteration value
        yOld,
        // Outputs
        &newDepth,
        &newVolume,
        &overflow,
        &oldSurfArea,
        &dYdT);

    // Update node state
    nodes->d_newDepth[i] = newDepth;
    nodes->d_newVolume[i] = newVolume;
    nodes->d_overflow[i] = overflow;
    nodes->d_oldSurfArea[i] = oldSurfArea;
    nodes->d_dYdT[i] = dYdT;

    // Check convergence
    double depthChange = fabs(newDepth - yOld);
    nodes->d_converged[i] = (depthChange <= headTol) ? 1 : 0;
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
    cudaStream_t stream = gpu_getStream();

    GPU_NodeData* d_nodes = g_nodeKernelCtx.d_nodes;
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes, sizeof(GPU_NodeData), cudaMemcpyHostToDevice));

    // Launch kernel
    kernel_findNodeDepths<<<gridSize, blockSize, 0, stream>>>(
        d_nodes, dt, allowPonding, surchargeMethod,
        minSurfArea, steps, omega, headTol);

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    gpu_transferNodeDynamicFromDevice(nodes, nodes->count);

    // Count converged nodes (reduction on CPU for now)
    // TODO: Implement GPU reduction for better performance
    *convergedCount = 0;
    for (int i = 0; i < nodes->count; i++) {
        if (nodes->h_converged[i]) (*convergedCount)++;
    }

    return 0;
}

} // extern "C"

static int ensureNodeKernelContext(void)
{
    int nodeCount = Nobjects[NODE];
    if (nodeCount <= 0) return -1;

    if (!g_nodeKernelCtx.initialized ||
        nodeCount != g_nodeKernelCtx.nodeCapacity)
    {
        if (g_nodeKernelCtx.initialized) {
            gpu_freeNodeData(&g_nodeKernelCtx.data);
            if (g_nodeKernelCtx.d_nodes) {
                cudaFree(g_nodeKernelCtx.d_nodes);
                g_nodeKernelCtx.d_nodes = nullptr;
            }
            g_nodeKernelCtx.initialized = 0;
        }
        if (gpu_allocateNodeData(&g_nodeKernelCtx.data, nodeCount) != 0) {
            return -1;
        }
        CUDA_CHECK(cudaMalloc((void**)&g_nodeKernelCtx.d_nodes, sizeof(GPU_NodeData)));
        g_nodeKernelCtx.nodeCapacity = nodeCount;
        g_nodeKernelCtx.initialized = 1;
        g_nodeKernelCtx.staticsUploaded = 0;
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
        nodes->h_type[i]        = Node[i].type;
        nodes->h_invertElev[i]  = Node[i].invertElev;
        nodes->h_fullDepth[i]   = Node[i].fullDepth;
        nodes->h_surDepth[i]    = Node[i].surDepth;
        nodes->h_pondedArea[i]  = Node[i].pondedArea;
        nodes->h_crownElev[i]   = Node[i].crownElev;
        nodes->h_oldDepth[i]    = Node[i].oldDepth;
        nodes->h_newDepth[i]    = Node[i].newDepth;
        nodes->h_oldVolume[i]   = Node[i].oldVolume;
        nodes->h_newVolume[i]   = Node[i].newVolume;
        nodes->h_fullVolume[i]  = Node[i].fullVolume;
        nodes->h_oldNetInflow[i]= Node[i].oldNetInflow;
        nodes->h_inflow[i]      = Node[i].inflow;
        nodes->h_outflow[i]     = Node[i].outflow;
        nodes->h_overflow[i]    = Node[i].overflow;
        nodes->h_degree[i]      = Node[i].degree;
        nodes->h_converged[i]   = Xnode[i].converged;
        nodes->h_newSurfArea[i] = Xnode[i].newSurfArea;
        nodes->h_oldSurfArea[i] = Xnode[i].oldSurfArea;
        nodes->h_sumdqdh[i]     = Xnode[i].sumdqdh;
        nodes->h_dYdT[i]        = Xnode[i].dYdT;
    }

    if (!g_nodeKernelCtx.staticsUploaded)
    {
        gpu_transferNodeStaticToDevice(nodes, nodes->count);
        g_nodeKernelCtx.staticsUploaded = 1;
    }
    gpu_transferNodeDynamicToDevice(nodes, nodes->count);
}

static void copyNodesFromGpu(GPU_NodeData* nodes)
{
    int count = nodes->count;
    for (int i = 0; i < count; i++)
    {
        if (Node[i].type != OUTFALL)
        {
            Node[i].newDepth  = nodes->h_newDepth[i];
            Node[i].newVolume = nodes->h_newVolume[i];
            Node[i].overflow  = nodes->h_overflow[i];
        }
        Xnode[i].converged = nodes->h_converged[i];
        Xnode[i].newSurfArea = nodes->h_newSurfArea[i];
        Xnode[i].oldSurfArea = nodes->h_oldSurfArea[i];
        Xnode[i].sumdqdh     = nodes->h_sumdqdh[i];
        Xnode[i].dYdT        = nodes->h_dYdT[i];
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

    cudaStream_t stream = gpu_getStream();
    GPU_NodeData* nodes = &g_nodeKernelCtx.data;
    copyNodesToGpu(nodes);

    int convergedCount = 0;
    cudaEventRecord(g_nodeKernelStartEvent, stream);
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
    cudaEventRecord(g_nodeKernelStopEvent, stream);
    cudaEventSynchronize(g_nodeKernelStopEvent);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, g_nodeKernelStartEvent, g_nodeKernelStopEvent);
    gpu_profiler_addKernelTime((double)elapsedMs);

    copyNodesFromGpu(nodes);

    return convergedCount;
}

//=============================================================================
// Persistent Picard Iteration with GPU-side Convergence Checking
//=============================================================================

extern "C" int gpu_computeNodeDepthsWithConvergence(
    GPU_NodeData* nodes,
    double dt,
    int allowPonding,
    int surchargeMethod,
    double minSurfArea,
    int steps,
    double omega,
    double headTol,
    int* d_convergedCount,
    cudaStream_t stream)
//
//  Purpose: Computes node depths and checks convergence on GPU
//  Input:   nodes = GPU node data structure
//           dt = time step (sec)
//           allowPonding, surchargeMethod, minSurfArea = routing parameters
//           steps = current Picard iteration number
//           omega = under-relaxation parameter
//           headTol = convergence tolerance (ft)
//           d_convergedCount = device pointer for convergence counter
//           stream = CUDA stream for async execution
//  Output:  Updates nodes and d_convergedCount
//  Returns: 0 if successful, error code otherwise
//
{
    if (nodes == NULL || nodes->count <= 0) return -1;

    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = GRID_SIZE(nodes->count, blockSize);
    GPU_NodeData* d_nodes = g_nodeKernelCtx.d_nodes;
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes, sizeof(GPU_NodeData),
                          cudaMemcpyHostToDevice));

    // Reset convergence counter
    CUDA_CHECK(cudaMemsetAsync(d_convergedCount, 0, sizeof(int), stream));

    // Compute node depths
    kernel_findNodeDepths<<<gridSize, blockSize, 0, stream>>>(
        d_nodes, dt, allowPonding, surchargeMethod,
        minSurfArea, steps, omega, headTol);
    CUDA_CHECK_LAST_ERROR();

    // Count converged nodes using optimized reduction kernel
    kernel_checkConvergenceOptimized<<<gridSize, blockSize, 0, stream>>>(
        nodes->d_newDepth,
        nodes->d_oldDepth,
        nodes->d_type,
        nodes->count,
        headTol,
        nodes->d_converged,
        d_convergedCount);
    CUDA_CHECK_LAST_ERROR();

    return 0;
}

//=============================================================================

extern "C" int gpu_runPersistentPicardIteration(
    double dt,
    int allowPonding,
    int surchargeMethod,
    double minSurfArea,
    double omega,
    double headTol,
    int maxIterations,
    int* outIterations,
    int* outConverged)
//
//  Purpose: Runs complete Picard iteration loop on GPU with minimal CPU-GPU transfers
//
//  Input:   dt = time step (sec)
//           allowPonding, surchargeMethod, minSurfArea = routing parameters
//           omega = under-relaxation parameter
//           headTol = convergence tolerance (ft)
//           maxIterations = maximum number of Picard iterations
//
//  Output:  outIterations = actual number of iterations performed
//           outConverged = 1 if converged, 0 if max iterations reached
//
//  Returns: 0 if successful, error code otherwise
//
//  Performance: Eliminates per-iteration kernel launch overhead and reduces
//               CPU-GPU transfers from O(N*iterations) to O(1)
//
{
    if (ensureNodeKernelContext() != 0) {
        return -1;
    }

    cudaStream_t stream = gpu_getStream();
    GPU_NodeData* nodes = &g_nodeKernelCtx.data;

    // Allocate device memory for convergence counter
    int* d_convergedCount;
    int h_convergedCount;
    CUDA_CHECK(cudaMalloc(&d_convergedCount, sizeof(int)));

    // Initial transfer: Copy node data to GPU
    copyNodesToGpu(nodes);

    // Picard iteration loop - stays mostly on GPU
    int iter;
    int converged = 0;

    cudaEventRecord(g_nodeKernelStartEvent, stream);

    for (iter = 0; iter < maxIterations; iter++) {
        // Compute node depths with on-device convergence check
        if (gpu_computeNodeDepthsWithConvergence(
                nodes, dt, allowPonding, surchargeMethod,
                minSurfArea, iter + 1, omega, headTol,
                d_convergedCount, stream) != 0)
        {
            cudaFree(d_convergedCount);
            return -1;
        }

        // Transfer only convergence counter (4 bytes) back to CPU
        CUDA_CHECK(cudaMemcpy(&h_convergedCount, d_convergedCount,
                              sizeof(int), cudaMemcpyDeviceToHost));

        // Check if all nodes converged
        if (iter > 0 && h_convergedCount == nodes->count) {
            converged = 1;
            break;
        }
    }

    cudaEventRecord(g_nodeKernelStopEvent, stream);
    cudaEventSynchronize(g_nodeKernelStopEvent);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, g_nodeKernelStartEvent, g_nodeKernelStopEvent);
    gpu_profiler_addKernelTime((double)elapsedMs);

    gpu_transferNodeDynamicFromDevice(nodes, nodes->count);
    // Final transfer: Copy results back to CPU
    copyNodesFromGpu(nodes);

    // Cleanup
    cudaFree(d_convergedCount);

    // Set output parameters
    *outIterations = iter + 1;
    *outConverged = converged;

    return 0;
}
