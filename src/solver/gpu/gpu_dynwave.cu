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
#include <stdlib.h>
#include "gpu_config.h"
#include "gpu_structures.h"
#include "gpu_dynwave_kernels.cuh"
#include "gpu_reduction.cuh"

extern "C" {
#include "headers.h"
#include "dynwave_data.h"
}

// Crown cutoff constants (same as in dynwave.c)
static const double EXTRAN_CROWN_CUTOFF = 0.96;      // crown cutoff for EXTRAN
static const double SLOT_CROWN_CUTOFF   = 0.985257;  // crown cutoff for SLOT

extern "C" void setNodeDepth_hostWrapper(int i, double dt);

extern GPU_CurveData* g_gpuDeviceCurves;
extern GPU_CurvePoints* g_gpuDeviceCurvePoints;
extern GPU_CurveData g_gpuCurves;
extern GPU_CurvePoints g_gpuCurvePoints;

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
static double* g_prevNodeNewDepth = NULL;
static int g_prevNodeCapacity = 0;
static double* g_prevNodeNewVolume = NULL;

static int ensureDeviceCurvePointers(void)
{
    if (g_gpuCurves.count > 0 && g_gpuDeviceCurves == NULL) {
        CUDA_CHECK(cudaMalloc((void**)&g_gpuDeviceCurves, sizeof(GPU_CurveData)));
        CUDA_CHECK(cudaMemcpy(g_gpuDeviceCurves, &g_gpuCurves,
                              sizeof(GPU_CurveData), cudaMemcpyHostToDevice));
    }

    if (g_gpuCurvePoints.totalPoints > 0 && g_gpuDeviceCurvePoints == NULL) {
        CUDA_CHECK(cudaMalloc((void**)&g_gpuDeviceCurvePoints, sizeof(GPU_CurvePoints)));
        CUDA_CHECK(cudaMemcpy(g_gpuDeviceCurvePoints, &g_gpuCurvePoints,
                              sizeof(GPU_CurvePoints), cudaMemcpyHostToDevice));
    }

    return 0;
}

//=============================================================================
// Kernel: Reset Node Accumulators
//=============================================================================

__global__ void kernel_resetNodeAccumulators(
    GPU_NodeData* nodes,
    int allowPonding,
    int debugPrint,
    double ucfLength,
    GPU_CurveData* curves,
    GPU_CurvePoints* curvePoints)
//
//  Purpose: Resets node accumulators before each Picard iteration
//           Mirrors CPU initNodeStates() in dynwave.c
//  Input:   nodes = GPU node data structure
//           allowPonding = TRUE if ponding is allowed
//           debugPrint = iteration number for debug (0 = no print)
//           ucfLength = unit conversion factor for length
//           curves, curvePoints = storage curve data
//
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodes->count) return;

    // Reset inflow/outflow (start with lateral flows and losses)
    double latFlow = nodes->d_newLatFlow[i];

    // DEBUG: Print J1 (node 0) lateral flow in first 3 calls
    if (i == 0 && debugPrint > 0 && debugPrint <= 3) {
        printf("RESET_J1[iter=%d]: latFlow=%.6f losses=%.6f\n",
               debugPrint, latFlow, nodes->d_losses[i]);
    }

    if (latFlow >= 0.0) {
        nodes->d_inflow[i] = latFlow;
        nodes->d_outflow[i] = nodes->d_losses[i];
    } else {
        nodes->d_inflow[i] = 0.0;
        nodes->d_outflow[i] = nodes->d_losses[i] - latFlow;
    }

    // Reset sumdqdh
    nodes->d_sumdqdh[i] = 0.0;

    // Reset surface area to base value (intrinsic node area before conduit contributions)
    // Conduit contributions will be atomicAdd'ed on top of this base
    double depth = nodes->d_newDepth[i];
    double baseArea;

    // For ponded nodes above full depth, use ponded area
    if (allowPonding && depth > nodes->d_fullDepth[i]) {
        baseArea = nodes->d_pondedArea[i];
    }
    // For storage nodes, compute intrinsic surface area from storage curve/shape
    else if (nodes->d_type[i] == GPU_STORAGE) {
        baseArea = gpu_node_getSurfArea(
            nodes->d_type[i],
            depth,
            nodes->d_storageA0[i],
            nodes->d_storageA1[i],
            nodes->d_storageA2[i],
            nodes->d_storageShape[i],
            nodes->d_storageCurve[i],
            ucfLength,
            curves,
            curvePoints);
    }
    // For other nodes (junctions, outfalls, dividers), base area is zero
    else {
        baseArea = 0.0;
    }

    nodes->d_newSurfArea[i] = baseArea;
}

//=============================================================================
// Kernel: Set Outfall Depths
//=============================================================================

__global__ void kernel_setOutfallDepths(
    GPU_NodeData* nodes,
    GPU_LinkData* links)
//
//  Purpose: Sets water depth at outfall nodes based on connecting link flow
//           Simplified version: uses link depth as proxy for outfall depth
//           Mirrors CPU link_setOutfallDepth() in link.c (FREE_OUTFALL behavior)
//  Input:   nodes = GPU node data structure
//           links = GPU link data structure
//
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= links->count) return;

    // Find which end node of link is an outfall
    int n = -1;
    int upstreamNode = -1;

    if (nodes->d_type[links->d_node2[j]] == GPU_OUTFALL) {
        n = links->d_node2[j];
        upstreamNode = links->d_node1[j];
    } else if (nodes->d_type[links->d_node1[j]] == GPU_OUTFALL) {
        n = links->d_node1[j];
        upstreamNode = links->d_node2[j];
    } else {
        return;  // No outfall on this link
    }

    // Simplified approach: Set outfall depth to link depth
    // This allows water to flow out without requiring complex normal/critical depth calculations
    // For FREE_OUTFALL, use the link's flow depth as the outfall depth
    double linkDepth = links->d_newDepth[j];

    // If link has no depth but has flow, use a small depth to allow drainage
    if (linkDepth < 0.001 && fabs(links->d_newFlow[j]) > 0.001) {
        linkDepth = 0.001;
    }

    nodes->d_newDepth[n] = linkDepth;

    // Debug: Print first few outfall updates
    static __device__ int debug_count = 0;
    if (atomicAdd(&debug_count, 1) < 10) {
        printf("GPU: Setting outfall node %d to depth %.6f (link %d depth=%.6f flow=%.3f)\n",
               n, linkDepth, j, links->d_newDepth[j], links->d_newFlow[j]);
    }
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
    double headTol,
    double ucfLength,
    double ucfVolume,
    GPU_CurveData* curves,
    GPU_CurvePoints* curvePoints)
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

    // DEBUG: Print dt value from first thread, first few iterations
    if (i == 0 && steps < 3) {
        printf("GPU kernel_findNodeDepths: dt = %.6f, steps = %d\n", dt, steps);
    }

    // DEBUG: Print J1 geometry (node index 0, first 5 calls)
    if (i == 0 && steps <= 5) {
        printf("J1_GEOM[step=%d]: fullDepth=%.6f fullVol=%.6f surDepth=%.6f invertElev=%.6f crownElev=%.6f minSurfArea=%.6f\n",
               steps, nodes->d_fullDepth[0], nodes->d_fullVolume[0], nodes->d_surDepth[0],
               nodes->d_invertElev[0], nodes->d_crownElev[0], minSurfArea);
    }

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
        nodes->d_oldVolume[i],
        nodes->d_oldNetInflow[i],
        nodes->d_inflow[i],
        nodes->d_outflow[i],
        nodes->d_fullVolume[i],
        nodes->d_degree[i],
        nodes->d_storageA0[i],
        nodes->d_storageA1[i],
        nodes->d_storageA2[i],
        nodes->d_storageShape[i],
        nodes->d_storageCurve[i],
        // Xnode data
        nodes->d_newSurfArea[i],
        nodes->d_oldSurfArea[i],
        nodes->d_sumdqdh[i],
        // Units and lookup tables
        ucfLength,
        ucfVolume,
        curves,
        curvePoints,
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

    ensureDeviceCurvePointers();

    double ucfLength = UCF(LENGTH);
    double ucfVolume = UCF(VOLUME);

    // DEBUG: Print unit conversion factors once
    static int ucf_printed = 0;
    if (!ucf_printed) {
        printf("DEBUG UCF (findNodeDepths): ucfLength=%.6f ucfVolume=%.6f\n", ucfLength, ucfVolume);
        ucf_printed = 1;
    }

    GPU_CurveData* d_curves = g_gpuDeviceCurves;
    GPU_CurvePoints* d_curvePoints = g_gpuDeviceCurvePoints;

    // Launch kernel
    kernel_findNodeDepths<<<gridSize, blockSize, 0, stream>>>(
        d_nodes, dt, allowPonding, surchargeMethod,
        minSurfArea, steps, omega, headTol,
        ucfLength, ucfVolume,
        d_curves, d_curvePoints);

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
        nodes->h_newLatFlow[i]  = Node[i].newLatFlow;
        nodes->h_losses[i]      = Node[i].losses;

        nodes->h_storageA0[i]    = 0.0;
        nodes->h_storageA1[i]    = 0.0;
        nodes->h_storageA2[i]    = 0.0;
        nodes->h_storageShape[i] = 0;
        nodes->h_storageCurve[i] = -1;

        if (Node[i].type == STORAGE) {
            int sIdx = Node[i].subIndex;
            nodes->h_storageShape[i] = Storage[sIdx].shape;
            nodes->h_storageCurve[i] = Storage[sIdx].aCurve;
            nodes->h_storageA0[i]    = Storage[sIdx].a0;
            nodes->h_storageA1[i]    = Storage[sIdx].a1;
            nodes->h_storageA2[i]    = Storage[sIdx].a2;

            // DEBUG: Print storage node configuration
            static int debug_printed = 0;
            if (!debug_printed) {
                printf("DEBUG: Storage node %d (ID=%s) sIdx=%d shape=%d curve=%d a0=%.3f a1=%.3f a2=%.3f\n",
                       i, Node[i].ID, sIdx,
                       nodes->h_storageShape[i], nodes->h_storageCurve[i],
                       nodes->h_storageA0[i], nodes->h_storageA1[i], nodes->h_storageA2[i]);
            }
        }
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
    static int call_count = 0;
    call_count++;

    for (int i = 0; i < count; i++)
    {
        // Copy depth for all nodes (including outfalls, which are set by GPU outfall kernel)
        double newDepth = nodes->h_newDepth[i];

        if (Node[i].type != OUTFALL)
        {
            double newVolume_GPU = nodes->h_newVolume[i];
            double newVolume = newVolume_GPU;

            // DISABLED CPU fallback - use GPU volume directly
            if (0 && Node[i].type == STORAGE) {
                double newVolume_CPU = node_getVolume(i, newDepth);
                newVolume = newVolume_CPU;
                nodes->h_newVolume[i] = newVolume_CPU;

                // Log STOR-10 and TUNNEL_STORAGE volume comparison
                // Only log first 50 calls to avoid spam
                if (call_count <= 50) {
                    const char* nodeName = Node[i].ID;
                    if (strcmp(nodeName, "STOR-10") == 0 ||
                        strcmp(nodeName, "TUNNEL_STORAGE") == 0) {
                        double diff = newVolume_CPU - newVolume_GPU;
                        double pct_diff = (newVolume_GPU > 0.001) ?
                            (diff / newVolume_GPU * 100.0) : 0.0;
                        printf("STORAGE_VOL[%s call=%d]: depth=%.6f GPU_vol=%.6f CPU_vol=%.6f diff=%.6f (%.2f%%)\n",
                               nodeName, call_count, newDepth, newVolume_GPU, newVolume_CPU,
                               diff, pct_diff);
                    }
                }
            }

            // Log GPU vs CPU storage comparison (first 50 calls)
            if (Node[i].type == STORAGE && call_count <= 50) {
                const char* nodeName = Node[i].ID;
                if (strcmp(nodeName, "STOR-10") == 0 ||
                    strcmp(nodeName, "TUNNEL_STORAGE") == 0) {
                    // Compare GPU vs CPU volumes AT THE SAME DEPTH
                    double volume_CPU = node_getVolume(i, newDepth);
                    double volume_GPU = newVolume_GPU;
                    double vol_diff = volume_GPU - volume_CPU;
                    double vol_pct = (volume_CPU > 0.001) ? (vol_diff / volume_CPU * 100.0) : 0.0;

                    // Compare surface areas (must use Xnode[].newSurfArea which includes conduit contributions)
                    double surfArea_CPU = Xnode[i].newSurfArea;  // With conduit contributions
                    double surfArea_GPU = nodes->h_newSurfArea[i];
                    double surf_diff = surfArea_GPU - surfArea_CPU;
                    double surf_pct = (surfArea_CPU > 0.001) ? (surf_diff / surfArea_CPU * 100.0) : 0.0;

                    // Get flows
                    double inflow_gpu = nodes->h_inflow[i];
                    double outflow_gpu = nodes->h_outflow[i];

                    printf("STOR[%s c=%d]: depth=%.6f | VOL: GPU=%.6f CPU=%.6f diff=%.6f (%.2f%%) | SURF: GPU=%.1f CPU=%.1f diff=%.1f (%.2f%%) | in=%.3f out=%.3f\n",
                           nodeName, call_count, newDepth,
                           volume_GPU, volume_CPU, vol_diff, vol_pct,
                           surfArea_GPU, surfArea_CPU, surf_diff, surf_pct,
                           inflow_gpu, outflow_gpu);
                }
            }

            Node[i].newVolume = newVolume;
            Node[i].overflow  = nodes->h_overflow[i];
        }

        // Always copy depth (including for outfalls, which are set by GPU outfall kernel)
        Node[i].newDepth = newDepth;

        // Copy inflow/outflow for statistics tracking
        Node[i].inflow = nodes->h_inflow[i];
        Node[i].outflow = nodes->h_outflow[i];

        // Update oldNetInflow for next timestep (mirrors node_setOldHydState at node.c:339)
        Node[i].oldNetInflow = nodes->h_inflow[i] - nodes->h_outflow[i];
        nodes->h_oldNetInflow[i] = Node[i].oldNetInflow;

        // DEBUG: Print STOR1 and J1 flow/depth (first 10 calls)
        if (call_count <= 10) {
            const char* nodeName = Node[i].ID;
            if (strcmp(nodeName, "STOR1") == 0 || strcmp(nodeName, "J1") == 0) {
                printf("%s[call=%d]: type=%d inflow=%.3f outflow=%.3f latFlow=%.3f depth=%.6f vol=%.6f surfArea=%.3f\n",
                       nodeName, call_count, Node[i].type,
                       nodes->h_inflow[i], nodes->h_outflow[i], nodes->h_newLatFlow[i],
                       nodes->h_newDepth[i], nodes->h_newVolume[i], nodes->h_newSurfArea[i]);
            }
        }

        if (i == 852 && nodes->h_newVolume[i] < 0.05) {
            printf("copyNodesFromGpu: node 852 newVolume=%.6f newDepth=%.6f overflow=%.6f\n",
                   nodes->h_newVolume[i], nodes->h_newDepth[i], nodes->h_overflow[i]);
        }
        Xnode[i].converged = nodes->h_converged[i];
        Xnode[i].newSurfArea = nodes->h_newSurfArea[i];
        Xnode[i].oldSurfArea = nodes->h_oldSurfArea[i];
        Xnode[i].sumdqdh     = nodes->h_sumdqdh[i];
        Xnode[i].dYdT        = nodes->h_dYdT[i];
    }
}

//=============================================================================
// Diagnostic: Compare CPU vs GPU Surface Areas
//=============================================================================
static void diagnosticSurfaceAreas(
    GPU_NodeData* nodes,
    GPU_LinkData* links,
    int timestep,
    int iteration)
//
//  Purpose: Logs CPU vs GPU surface area comparison for debugging
//  Input:   nodes = GPU node data (after copyback from device)
//           links = GPU link data (after copyback from device)
//           timestep = current routing timestep
//           iteration = current Picard iteration
//
{
    extern TNode* Node;
    extern TLink* Link;
    extern TXnode* Xnode;

    static int call_count = 0;
    call_count++;

    // Only log first few timesteps/iterations to avoid spam
    if (call_count > 20) return;

    printf("\n=== SURFACE AREA DIAGNOSTIC (Step=%d Iter=%d) ===\n", timestep, iteration);

    // Node surface areas
    printf("NODE SURFACE AREAS:\n");
    printf("%-15s %-10s %12s %12s %12s %8s\n",
           "Node", "Type", "CPU(ft²)", "GPU(ft²)", "Diff(ft²)", "Err(%)");
    printf("%-15s %-10s %12s %12s %12s %8s\n",
           "---------------", "----------", "------------", "------------", "------------", "--------");

    for (int i = 0; i < nodes->count && i < 20; i++) {
        double cpu_surf = Xnode[i].newSurfArea;
        double gpu_surf = nodes->h_newSurfArea[i];
        double diff = gpu_surf - cpu_surf;
        double pct = (cpu_surf > 0.001) ? (diff / cpu_surf * 100.0) : 0.0;

        // Only print if significant difference or storage node
        if (fabs(pct) > 1.0 || Node[i].type == STORAGE) {
            const char* typeStr =
                (Node[i].type == JUNCTION) ? "JUNCTION" :
                (Node[i].type == OUTFALL) ? "OUTFALL" :
                (Node[i].type == STORAGE) ? "STORAGE" :
                (Node[i].type == DIVIDER) ? "DIVIDER" : "UNKNOWN";

            printf("%-15s %-10s %12.2f %12.2f %12.2f %8.2f\n",
                   Node[i].ID, typeStr, cpu_surf, gpu_surf, diff, pct);
        }
    }

    // Link surface areas
    printf("\nLINK SURFACE AREAS:\n");
    printf("%-15s %-10s %12s %12s %12s %12s %8s\n",
           "Link", "Type", "CPU_A1(ft²)", "GPU_A1(ft²)", "CPU_A2(ft²)", "GPU_A2(ft²)", "Err(%)");
    printf("%-15s %-10s %12s %12s %12s %12s %8s\n",
           "---------------", "----------", "------------", "------------", "------------", "------------", "--------");

    for (int j = 0; j < links->count && j < 20; j++) {
        double cpu_a1 = Link[j].surfArea1;
        double gpu_a1 = links->h_surfArea1[j];
        double cpu_a2 = Link[j].surfArea2;
        double gpu_a2 = links->h_surfArea2[j];

        double diff1 = gpu_a1 - cpu_a1;
        double diff2 = gpu_a2 - cpu_a2;
        double pct1 = (cpu_a1 > 0.001) ? (diff1 / cpu_a1 * 100.0) : 0.0;
        double pct2 = (cpu_a2 > 0.001) ? (diff2 / cpu_a2 * 100.0) : 0.0;
        double max_pct = fmax(fabs(pct1), fabs(pct2));

        // Only print if significant difference
        if (max_pct > 1.0) {
            const char* typeStr =
                (Link[j].type == CONDUIT) ? "CONDUIT" :
                (Link[j].type == PUMP) ? "PUMP" :
                (Link[j].type == ORIFICE) ? "ORIFICE" :
                (Link[j].type == WEIR) ? "WEIR" :
                (Link[j].type == OUTLET) ? "OUTLET" : "UNKNOWN";

            printf("%-15s %-10s %12.2f %12.2f %12.2f %12.2f %8.2f\n",
                   Link[j].ID, typeStr, cpu_a1, gpu_a1, cpu_a2, gpu_a2, max_pct);
        }
    }

    printf("===============================================\n\n");
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

    if (g_prevNodeCapacity != nodes->count) {
        if (g_prevNodeNewDepth) {
            free(g_prevNodeNewDepth);
            g_prevNodeNewDepth = NULL;
        }
        if (g_prevNodeNewVolume) {
            free(g_prevNodeNewVolume);
            g_prevNodeNewVolume = NULL;
        }
        if (nodes->count > 0) {
            g_prevNodeNewDepth = (double*)malloc(nodes->count * sizeof(double));
            g_prevNodeNewVolume = (double*)malloc(nodes->count * sizeof(double));
        }
        g_prevNodeCapacity = nodes->count;
    }
    if (g_prevNodeNewDepth) {
        for (int i = 0; i < nodes->count; i++) {
            g_prevNodeNewDepth[i] = nodes->h_newDepth[i];
            g_prevNodeNewVolume[i] = nodes->h_newVolume[i];
        }
    }

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

    // DISABLED: CPU storage recompute fallback - let GPU handle storage nodes
    if (0 && g_prevNodeNewDepth) {
        for (int i = 0; i < nodes->count; i++) {
            if (Node[i].type == STORAGE) {
                double prevDepth = g_prevNodeNewDepth ? g_prevNodeNewDepth[i] : Node[i].newDepth;
                double prevVolume = g_prevNodeNewVolume ? g_prevNodeNewVolume[i] : Node[i].newVolume;
                double prevSurf = node_getSurfArea(i, prevDepth);
                if (prevSurf < MinSurfArea) prevSurf = MinSurfArea;
                Xnode[i].newSurfArea = prevSurf;
                if (prevDepth > Node[i].fullDepth) {
                    Xnode[i].oldSurfArea = prevSurf;
                }
                Node[i].newDepth = prevDepth;
                Node[i].newVolume = prevVolume;

                // Estimate new volume/depth using mass balance before refinement
                double oldVolume = Node[i].oldVolume;
                double oldNet = Node[i].oldNetInflow;
                double newNet = Node[i].inflow - Node[i].outflow;
                double dV = 0.5 * (oldNet + newNet) * dt;
                double volumeCandidate = oldVolume + dV;
                double depthGuess = prevDepth;
                if (volumeCandidate <= 0.0) {
                    depthGuess = 0.0;
                    volumeCandidate = 0.0;
                } else {
                    double volumeForDepth = volumeCandidate;
                    if (volumeForDepth > Node[i].fullVolume) {
                        volumeForDepth = Node[i].fullVolume;
                        depthGuess = node_getDepth(i, volumeForDepth);
                        if (AllowPonding && Node[i].pondedArea > 0.0) {
                            double pondDepth = (volumeCandidate - Node[i].fullVolume) / Node[i].pondedArea;
                            if (pondDepth > 0.0)
                                depthGuess = Node[i].fullDepth + pondDepth;
                        }
                    } else {
                        depthGuess = node_getDepth(i, volumeForDepth);
                    }
                }
                Node[i].newDepth = depthGuess;
                Node[i].newVolume = volumeCandidate;
                setNodeDepth_hostWrapper(i, dt);
                if (i == 852) {
                    printf("storage host recompute node 852 prevDepth=%.6f prevVol=%.6f newDepth=%.6f newVol=%.6f overflow=%.6f\n",
                           prevDepth, prevVolume, Node[i].newDepth, Node[i].newVolume, Node[i].overflow);
                }
                nodes->h_newDepth[i] = Node[i].newDepth;
                nodes->h_newVolume[i] = Node[i].newVolume;
                nodes->h_overflow[i] = Node[i].overflow;
                nodes->h_newSurfArea[i] = Xnode[i].newSurfArea;
                nodes->h_oldSurfArea[i] = Xnode[i].oldSurfArea;
                nodes->h_sumdqdh[i] = Xnode[i].sumdqdh;
                nodes->h_dYdT[i] = Xnode[i].dYdT;
            }
        }
    }

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

    ensureDeviceCurvePointers();

    double ucfLength = UCF(LENGTH);
    double ucfVolume = UCF(VOLUME);
    GPU_CurveData* d_curves = g_gpuDeviceCurves;
    GPU_CurvePoints* d_curvePoints = g_gpuDeviceCurvePoints;

    // Reset convergence counter
    CUDA_CHECK(cudaMemsetAsync(d_convergedCount, 0, sizeof(int), stream));

    // Compute node depths
    kernel_findNodeDepths<<<gridSize, blockSize, 0, stream>>>(
        d_nodes, dt, allowPonding, surchargeMethod,
        minSurfArea, steps, omega, headTol,
        ucfLength, ucfVolume,
        d_curves, d_curvePoints);
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
// Forward declarations for external GPU functions and data
//=============================================================================

// External GPU data structures (defined in gpu_manager.cu and gpu_dwflow.cu)
extern GPU_CurveData* g_gpuDeviceCurves;
extern GPU_CurvePoints* g_gpuDeviceCurvePoints;
extern GPU_PumpData g_gpuPumps;
extern GPU_OrificeData g_gpuOrifices;
extern GPU_WeirData g_gpuWeirs;
extern GPU_OutletData g_gpuOutlets;

// External kernel context (from gpu_dwflow.cu)
typedef struct {
    GPU_LinkData links;
    GPU_ConduitData conduits;
    GPU_XsectData xsects;
    GPU_NodeData nodes;
    GPU_LinkData* d_links;
    GPU_ConduitData* d_conduits;
    GPU_XsectData* d_xsects;
    GPU_NodeData* d_nodes;
    int linkCapacity;
    int conduitCapacity;
    int xsectCapacity;
    int nodeCapacity;
    int nonConduitCount;
    int initialized;
    int xsectsInitialized;
    int linkStaticsInitialized;
    int conduitStaticsInitialized;
    int nodeStaticsInitialized;
    int resultsDirty;
    int linkStaticsUploaded;
    int conduitStaticsUploaded;
    int xsectsUploaded;
    int nodeStaticsUploaded;
} ConduitKernelContext;

extern ConduitKernelContext g_conduitKernelCtx;

// External helper functions from gpu_dwflow.cu
extern int ensureConduitKernelContext();
extern int ensureNonConduitStructuresInitialized(
    GPU_PumpData** out_d_gpuPumps,
    GPU_OrificeData** out_d_gpuOrifices,
    GPU_WeirData** out_d_gpuWeirs,
    GPU_OutletData** out_d_gpuOutlets,
    GPU_CurveData** out_d_gpuCurves,
    GPU_CurvePoints** out_d_gpuCurvePoints);

extern int launchLinkFlowKernels(
    GPU_LinkData* d_links,
    GPU_ConduitData* d_conduits,
    GPU_XsectData* d_xsects,
    GPU_NodeData* d_nodes,
    GPU_PumpData* d_gpuPumps,
    GPU_OrificeData* d_gpuOrifices,
    GPU_WeirData* d_gpuWeirs,
    GPU_OutletData* d_gpuOutlets,
    GPU_CurveData* d_gpuCurves,
    GPU_CurvePoints* d_gpuCurvePoints,
    double dt,
    int steps,
    double omega,
    int surchargeMethod,
    double crownCutoff,
    int inertDamping,
    cudaStream_t stream);

extern void copyNodesToGpu(GPU_NodeData* nodes);
extern void copyLinksToGpu(GPU_LinkData* links);
extern void copyConduitsToGpu(GPU_ConduitData* conduits);
extern void copyXsectsToGpu(GPU_XsectData* xsects);
extern void copyNodesFromGpu(GPU_NodeData* nodes);
extern void copyLinkIterStateFromGpu(GPU_LinkData* links);

extern int gpu_transferLinkIterationResultsFromDevice(GPU_LinkData* links, int count);
extern int gpu_transferNodeIterationStateFromDevice(GPU_NodeData* nodes, int count);

// External kernel wrappers
extern "C" int gpu_computeConduitFlows(
    GPU_LinkData* links,
    GPU_ConduitData* conduits,
    GPU_XsectData* xsects,
    GPU_NodeData* nodes,
    double dt,
    int steps,
    double omega,
    int surchargeMethod,
    double crownCutoff,
    int inertDamping);

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
    extern TLink* Link;
    extern TNode* Node;
    extern int Nobjects[];
    extern int Nlinks[];
    extern int RouteModel;
    extern int InertDamping;
    extern int SurchargeMethod;

    // Ensure both node and conduit kernel contexts are initialized
    if (ensureNodeKernelContext() != 0) {
        return -1;
    }
    if (ensureConduitKernelContext() != 0) {
        return -1;
    }

    // Get GPU contexts
    GPU_NodeData* nodes = &g_nodeKernelCtx.data;
    GPU_LinkData* links = &g_conduitKernelCtx.links;
    GPU_ConduitData* conduits = &g_conduitKernelCtx.conduits;
    GPU_XsectData* xsects = &g_conduitKernelCtx.xsects;

    // Get device pointers
    GPU_LinkData* d_links = g_conduitKernelCtx.d_links;
    GPU_ConduitData* d_conduits = g_conduitKernelCtx.d_conduits;
    GPU_XsectData* d_xsects = g_conduitKernelCtx.d_xsects;
    GPU_NodeData* d_nodes = g_conduitKernelCtx.d_nodes;

    // Ensure non-conduit structures (pumps, orifices, weirs, outlets) are initialized
    GPU_PumpData* d_gpuPumps = NULL;
    GPU_OrificeData* d_gpuOrifices = NULL;
    GPU_WeirData* d_gpuWeirs = NULL;
    GPU_OutletData* d_gpuOutlets = NULL;
    GPU_CurveData* d_gpuCurves = NULL;
    GPU_CurvePoints* d_gpuCurvePoints = NULL;

    if (ensureNonConduitStructuresInitialized(&d_gpuPumps, &d_gpuOrifices, &d_gpuWeirs,
                                               &d_gpuOutlets, &d_gpuCurves, &d_gpuCurvePoints) != 0) {
        return -1;
    }

    cudaStream_t stream = gpu_getStream();

    // Copy static geometry once (if not already uploaded)
    if (!g_conduitKernelCtx.xsectsInitialized) {
        copyXsectsToGpu(xsects);
        g_conduitKernelCtx.xsectsInitialized = 1;
    }

    // Initial transfer: Copy node, link, and conduit data to GPU ONCE
    copyNodesToGpu(nodes);
    copyLinksToGpu(links);
    copyConduitsToGpu(conduits);

    // Copy GPU data structures to device
    CUDA_CHECK(cudaMemcpy(d_links, links, sizeof(GPU_LinkData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conduits, conduits, sizeof(GPU_ConduitData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xsects, xsects, sizeof(GPU_XsectData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes, sizeof(GPU_NodeData), cudaMemcpyHostToDevice));

    // Allocate device memory for convergence counter
    int* d_convergedCount;
    int h_convergedCount;
    CUDA_CHECK(cudaMalloc(&d_convergedCount, sizeof(int)));

    // Picard iteration loop - ALL ON DEVICE
    int iter;
    int converged = 0;
    double crownCutoff = (surchargeMethod == EXTRAN) ? EXTRAN_CROWN_CUTOFF : SLOT_CROWN_CUTOFF;

    for (iter = 0; iter < maxIterations; iter++) {
        // === RESET ACCUMULATORS (before link flows) ===
        // This mirrors CPU's initNodeStates() which zeros inflow/outflow/surfArea/sumdqdh
        // before each Picard iteration
        int blockSize = DEFAULT_BLOCK_SIZE;
        int gridSize = GRID_SIZE(nodes->count, blockSize);
        kernel_resetNodeAccumulators<<<gridSize, blockSize, 0, stream>>>(
            d_nodes, allowPonding, iter + 1,
            UCF(LENGTH), d_gpuCurves, d_gpuCurvePoints);
        CUDA_CHECK_LAST_ERROR();

        // === LINK FLOWS (device-resident kernel launches) ===
        if (launchLinkFlowKernels(
                d_links, d_conduits, d_xsects, d_nodes,
                d_gpuPumps, d_gpuOrifices, d_gpuWeirs, d_gpuOutlets,
                d_gpuCurves, d_gpuCurvePoints,
                dt, iter + 1, omega,
                surchargeMethod, crownCutoff, InertDamping,
                stream) != 0)
        {
            cudaFree(d_convergedCount);
            return -1;
        }

        // === OUTFALL DEPTHS (set boundary conditions based on link flows) ===
        // This mirrors CPU's link_setOutfallDepth() at dynwave.c:817
        // Must be called AFTER link flows are computed but BEFORE node depths
        gridSize = GRID_SIZE(links->count, blockSize);
        kernel_setOutfallDepths<<<gridSize, blockSize, 0, stream>>>(
            d_nodes, d_links);
        CUDA_CHECK_LAST_ERROR();

        // === NODE DEPTHS (device-resident with convergence check) ===
        if (gpu_computeNodeDepthsWithConvergence(
                nodes, dt, allowPonding, surchargeMethod,
                minSurfArea, iter + 1, omega, headTol,
                d_convergedCount, stream) != 0)
        {
            cudaFree(d_convergedCount);
            return -1;
        }

        // === CONVERGENCE CHECK (only 4 bytes transferred) ===
        CUDA_CHECK(cudaMemcpy(&h_convergedCount, d_convergedCount,
                              sizeof(int), cudaMemcpyDeviceToHost));

        // Check if all nodes converged
        if (iter > 0 && h_convergedCount == nodes->count) {
            converged = 1;
            break;
        }
    }

    // Synchronize before transferring results
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Final transfer: Copy ALL results back to CPU ONCE
    gpu_transferLinkIterationResultsFromDevice(links, links->count);
    gpu_transferNodeIterationStateFromDevice(nodes, nodes->count);
    copyLinkIterStateFromGpu(links);
    copyNodesFromGpu(nodes);

    // Optional: Surface area diagnostic (enable with SWMM_DEBUG_SURF_AREA=1)
    static int diagnostic_enabled = -1;
    if (diagnostic_enabled == -1) {
        const char* debug_env = getenv("SWMM_DEBUG_SURF_AREA");
        diagnostic_enabled = (debug_env && atoi(debug_env) == 1) ? 1 : 0;
        if (diagnostic_enabled) {
            printf("\n*** Surface area diagnostics ENABLED (SWMM_DEBUG_SURF_AREA=1) ***\n");
        }
    }

    if (diagnostic_enabled) {
        // Note: This diagnostic currently only works in hybrid mode where CPU
        // also computes surface areas. In all-GPU mode, CPU values will be stale.
        // TODO: Add CPU recomputation for true comparison
        static int timestep_counter = 0;
        timestep_counter++;
        diagnosticSurfaceAreas(nodes, links, timestep_counter, iter + 1);
    }

    // Mark conduit results as dirty so gpu_flushConduitResults() will process them
    // This is CRITICAL for variable timestep calculation which needs current link flows
    g_conduitKernelCtx.resultsDirty = 1;

    // Cleanup
    cudaFree(d_convergedCount);

    // Set output parameters
    *outIterations = iter + 1;
    *outConverged = converged;

    return 0;
}
