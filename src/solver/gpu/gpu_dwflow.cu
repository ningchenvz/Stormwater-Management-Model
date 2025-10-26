//-----------------------------------------------------------------------------
//   gpu_dwflow.cu
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   GPU kernels for conduit flow calculations (dynamic wave routing).
//   Main kernel: gpu_findConduitFlows - computes new flows in all conduits
//
//   Note: This is Stage 1 implementation (simplified for regular conduits)
//
//-----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <stdio.h>
#include "gpu_config.h"
#include "gpu_structures.h"
#include "gpu_xsect_helpers.cuh"
#include "gpu_conduit_helpers.cuh"

extern "C" {
#include "headers.h"
}

//=============================================================================
// Static Context for Conduit Flow Kernel
//=============================================================================

typedef struct {
    GPU_LinkData links;
    GPU_ConduitData conduits;
    GPU_XsectData xsects;
    GPU_NodeData nodes;  // Need node data for flow calculations
    int linkCapacity;
    int conduitCapacity;
    int xsectCapacity;
    int nodeCapacity;
    int initialized;
} ConduitKernelContext;

static ConduitKernelContext g_conduitKernelCtx = {0};
static cudaEvent_t g_conduitKernelStartEvent = 0;
static cudaEvent_t g_conduitKernelStopEvent = 0;
static int g_conduitKernelEventsInitialized = 0;

//=============================================================================
// Helper Functions: Data Transfer
//=============================================================================

static void copyLinksToGpu(GPU_LinkData* gpuLinks)
{
    int count = gpuLinks->count;
    for (int j = 0; j < count; j++)
    {
        gpuLinks->type[j] = Link[j].type;
        gpuLinks->subIndex[j] = Link[j].subIndex;
        gpuLinks->node1[j] = Link[j].node1;
        gpuLinks->node2[j] = Link[j].node2;
        gpuLinks->offset1[j] = Link[j].offset1;
        gpuLinks->offset2[j] = Link[j].offset2;
        gpuLinks->qFull[j] = Link[j].qFull;
        gpuLinks->oldFlow[j] = Link[j].oldFlow;
        gpuLinks->newFlow[j] = Link[j].newFlow;
        gpuLinks->oldDepth[j] = Link[j].oldDepth;
        gpuLinks->newDepth[j] = Link[j].newDepth;
        gpuLinks->oldVolume[j] = Link[j].oldVolume;
        gpuLinks->newVolume[j] = Link[j].newVolume;
        gpuLinks->bypassed[j] = (char)Link[j].bypassed;
        gpuLinks->direction[j] = Link[j].direction;
        gpuLinks->flowClass[j] = Link[j].flowClass;
        gpuLinks->surfArea1[j] = Link[j].surfArea1;
        gpuLinks->surfArea2[j] = Link[j].surfArea2;
        gpuLinks->dqdh[j] = Link[j].dqdh;
        gpuLinks->froude[j] = Link[j].froude;
    }
}

static void copyConduitsToGpu(GPU_ConduitData* gpuConduits)
{
    int conduitCount = 0;
    // Count conduits
    for (int j = 0; j < Nobjects[LINK]; j++)
    {
        if (Link[j].type == CONDUIT) conduitCount++;
    }

    int k = 0;
    for (int j = 0; j < Nobjects[LINK]; j++)
    {
        if (Link[j].type == CONDUIT && k < gpuConduits->count)
        {
            int conduitIdx = Link[j].subIndex;
            gpuConduits->length[k] = Conduit[conduitIdx].length;
            gpuConduits->modLength[k] = Conduit[conduitIdx].modLength;
            gpuConduits->roughness[k] = Conduit[conduitIdx].roughness;
            gpuConduits->roughFactor[k] = Conduit[conduitIdx].roughFactor;
            gpuConduits->slope[k] = Conduit[conduitIdx].slope;
            gpuConduits->barrels[k] = (char)Conduit[conduitIdx].barrels;
            gpuConduits->q1[k] = Conduit[conduitIdx].q1;
            gpuConduits->q2[k] = Conduit[conduitIdx].q2;
            gpuConduits->a1[k] = Conduit[conduitIdx].a1;
            gpuConduits->a2[k] = Conduit[conduitIdx].a2;
            gpuConduits->q1Old[k] = Conduit[conduitIdx].q1Old;
            gpuConduits->q2Old[k] = Conduit[conduitIdx].q2Old;
            k++;
        }
    }
}

static void copyXsectsToGpu(GPU_XsectData* gpuXsects)
{
    int count = gpuXsects->count;
    for (int j = 0; j < count; j++)
    {
        int xtype = Link[j].xsect.type;
        gpuXsects->type[j] = xtype;
        gpuXsects->aFull[j] = Link[j].xsect.aFull;
        gpuXsects->rFull[j] = Link[j].xsect.rFull;
        gpuXsects->wMax[j] = Link[j].xsect.wMax;
        gpuXsects->yFull[j] = Link[j].xsect.yFull;

        // CRITICAL: Map geometry parameters based on cross-section shape type
        // These are used by gpu_xsect_helpers.cuh functions
        if (xtype == CIRCULAR || xtype == FILLED_CIRCULAR) {
            // For circular: geom1 = diameter
            gpuXsects->geom1[j] = Link[j].xsect.yFull;  // Diameter = full depth
            gpuXsects->geom2[j] = 0.0;
            gpuXsects->geom3[j] = 0.0;
        }
        else if (xtype == RECT_CLOSED || xtype == RECT_OPEN) {
            // For rectangular: geom1 = width, geom2 = height
            gpuXsects->geom1[j] = Link[j].xsect.wMax;   // Width
            gpuXsects->geom2[j] = Link[j].xsect.yFull;  // Height
            gpuXsects->geom3[j] = 0.0;
        }
        else if (xtype == TRAPEZOIDAL) {
            // For trapezoidal: geom1 = bottom width, geom2 = side slope
            // These are stored in multipurpose fields
            gpuXsects->geom1[j] = Link[j].xsect.yBot;   // Bottom width
            gpuXsects->geom2[j] = Link[j].xsect.sBot;   // Side slope
            gpuXsects->geom3[j] = 0.0;
        }
        else if (xtype == TRIANGULAR) {
            // For triangular: geom1 = side slope (bottom width = 0)
            gpuXsects->geom1[j] = Link[j].xsect.sBot;   // Side slope
            gpuXsects->geom2[j] = 0.0;
            gpuXsects->geom3[j] = 0.0;
        }
        else {
            // For unsupported shapes, zero out geometry (will use linear interpolation)
            gpuXsects->geom1[j] = 0.0;
            gpuXsects->geom2[j] = 0.0;
            gpuXsects->geom3[j] = 0.0;
        }
    }
}

static void copyLinksFromGpu(GPU_LinkData* gpuLinks)
{
    int count = gpuLinks->count;
    for (int j = 0; j < count; j++)
    {
        Link[j].newFlow = gpuLinks->newFlow[j];
        Link[j].newDepth = gpuLinks->newDepth[j];
        Link[j].newVolume = gpuLinks->newVolume[j];
        Link[j].dqdh = gpuLinks->dqdh[j];
        Link[j].froude = gpuLinks->froude[j];
        Link[j].surfArea1 = gpuLinks->surfArea1[j];
        Link[j].surfArea2 = gpuLinks->surfArea2[j];
    }
}

static void copyConduitsFromGpu(GPU_ConduitData* gpuConduits)
{
    int k = 0;
    for (int j = 0; j < Nobjects[LINK]; j++)
    {
        if (Link[j].type == CONDUIT && k < gpuConduits->count)
        {
            int conduitIdx = Link[j].subIndex;
            Conduit[conduitIdx].q1 = gpuConduits->q1[k];
            Conduit[conduitIdx].q2 = gpuConduits->q2[k];
            Conduit[conduitIdx].a1 = gpuConduits->a1[k];
            k++;
        }
    }
}

static void copyNodesToGpu(GPU_NodeData* gpuNodes)
{
    int count = gpuNodes->count;
    for (int i = 0; i < count; i++)
    {
        gpuNodes->type[i]        = Node[i].type;
        gpuNodes->invertElev[i]  = Node[i].invertElev;
        gpuNodes->newDepth[i]    = Node[i].newDepth;
        gpuNodes->inflow[i]      = Node[i].inflow;
        gpuNodes->outflow[i]     = Node[i].outflow;
    }
}

static void copyNodesFromGpu(GPU_NodeData* gpuNodes)
{
    int count = gpuNodes->count;
    for (int i = 0; i < count; i++)
    {
        // Update node inflows/outflows (updated by kernel)
        Node[i].inflow = gpuNodes->inflow[i];
        Node[i].outflow = gpuNodes->outflow[i];
    }
}

static int ensureConduitKernelContext()
{
    extern TNode* Node;
    extern int Nobjects[];

    int linkCount = Nobjects[LINK];
    int nodeCount = Nobjects[NODE];
    if (linkCount <= 0 || nodeCount <= 0) return -1;

    // Count conduits
    int conduitCount = 0;
    for (int j = 0; j < linkCount; j++)
    {
        if (Link[j].type == CONDUIT) conduitCount++;
    }

    if (!g_conduitKernelCtx.initialized ||
        linkCount != g_conduitKernelCtx.linkCapacity ||
        nodeCount != g_conduitKernelCtx.nodeCapacity)
    {
        if (g_conduitKernelCtx.initialized)
        {
            gpu_freeLinkData(&g_conduitKernelCtx.links);
            gpu_freeConduitData(&g_conduitKernelCtx.conduits);
            gpu_freeXsectData(&g_conduitKernelCtx.xsects);
            gpu_freeNodeData(&g_conduitKernelCtx.nodes);
            g_conduitKernelCtx.initialized = 0;
        }

        if (gpu_allocateLinkData(&g_conduitKernelCtx.links, linkCount) != 0) return -1;
        if (gpu_allocateConduitData(&g_conduitKernelCtx.conduits, conduitCount) != 0) return -1;
        if (gpu_allocateXsectData(&g_conduitKernelCtx.xsects, linkCount) != 0) return -1;
        if (gpu_allocateNodeData(&g_conduitKernelCtx.nodes, nodeCount) != 0) return -1;

        g_conduitKernelCtx.linkCapacity = linkCount;
        g_conduitKernelCtx.conduitCapacity = conduitCount;
        g_conduitKernelCtx.xsectCapacity = linkCount;
        g_conduitKernelCtx.nodeCapacity = nodeCount;
        g_conduitKernelCtx.initialized = 1;
    }

    if (!g_conduitKernelEventsInitialized)
    {
        cudaEventCreateWithFlags(&g_conduitKernelStartEvent, cudaEventDefault);
        cudaEventCreateWithFlags(&g_conduitKernelStopEvent, cudaEventDefault);
        g_conduitKernelEventsInitialized = 1;
    }

    return 0;
}

//=============================================================================
// Kernel: Find Conduit Flows
//=============================================================================

__global__ void kernel_findConduitFlows(
    GPU_LinkData* links,
    GPU_ConduitData* conduits,
    GPU_XsectData* xsects,
    GPU_NodeData* nodes,
    double dt,
    int steps,
    double omega,
    int surchargeMethod,
    double crownCutoff,
    int inertDamping)
//
//  Purpose: Computes new flow in all conduit links
//  Input:   links = GPU link data
//           conduits = GPU conduit data
//           xsects = GPU cross-section data
//           nodes = GPU node data
//           dt = time step (sec)
//           steps = current Picard iteration
//           omega = under-relaxation parameter
//           surchargeMethod = EXTRAN or SLOT
//           crownCutoff = crown cutoff fraction
//           inertDamping = damping option
//  Output:  Updates links->newFlow, links->newDepth, links->newVolume
//           Updates conduits->q1, conduits->q2, conduits->a1
//           Updates links->dqdh, links->froude
//
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= links->count) return;

    // Skip non-conduits (will handle pumps, orifices, etc. later)
    if (links->type[j] != 0) return;  // Assuming 0 = CONDUIT

    // Skip bypassed links
    if (links->bypassed[j]) return;

    int k = links->subIndex[j];  // Conduit index
    int n1 = links->node1[j];    // Upstream node
    int n2 = links->node2[j];    // Downstream node

    // Create xsect structure for this link from GPU arrays
    GPU_Xsect xsect;
    xsect.type = xsects->type[j];
    xsect.yFull = xsects->yFull[j];
    xsect.aFull = xsects->aFull[j];
    xsect.rFull = xsects->rFull[j];
    xsect.wMax = xsects->wMax[j];
    xsect.geom1 = xsects->geom1[j];  // Diameter, width, etc.
    xsect.geom2 = xsects->geom2[j];  // Height, side slope, etc.
    xsect.geom3 = xsects->geom3[j];  // Additional geometry

    // Call simplified conduit flow calculation
    double q, aMid, yMid, dqdh, froude;

    gpu_findConduitFlow_simplified(
        j, k, n1, n2,
        &xsect,
        // Node data
        nodes->newDepth[n1],
        nodes->invertElev[n1],
        nodes->newDepth[n2],
        nodes->invertElev[n2],
        // Link data
        links->offset1[j],
        links->offset2[j],
        links->oldFlow[j],
        links->flowClass[j],  // Flow classification from CPU
        // Conduit data
        (double)conduits->barrels[k],
        conduits->modLength[k],
        conduits->roughFactor[k],
        conduits->q1[k],
        conduits->a2[k],
        // Iteration parameters
        steps,
        omega,
        dt,
        surchargeMethod,
        crownCutoff,
        inertDamping,
        // Outputs
        &q,
        &aMid,
        &yMid,
        &dqdh,
        &froude);

    // Save results
    double barrels = (double)conduits->barrels[k];
    conduits->q1[k] = q;
    conduits->q2[k] = q;
    conduits->a1[k] = aMid;

    links->newFlow[j] = q * barrels;
    links->newDepth[j] = yMid;
    links->newVolume[j] = aMid * conduits->length[k] * barrels;
    links->dqdh[j] = dqdh;
    links->froude[j] = froude;

    // Debug first conduit on first few iterations
    if (k == 0 && steps < 5) {
        printf("GPU iter=%d: y1=%.6f y2=%.6f aMid=%.6f q=%.6f flowClass=%d\n",
               steps, nodes->newDepth[n1], nodes->newDepth[n2], aMid, q, links->flowClass[j]);
    }
}

//=============================================================================
// Kernel: Reset Node Flows
//=============================================================================

__global__ void kernel_resetNodeFlows(
    GPU_NodeData* nodes,
    int nodeCount)
//
//  Purpose: Resets node inflow and outflow to zero before accumulation
//
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nodeCount) return;

    nodes->inflow[i] = 0.0;
    nodes->outflow[i] = 0.0;
}

//=============================================================================
// Kernel: Update Node Inflows/Outflows
//=============================================================================

__global__ void kernel_updateNodeFlows(
    GPU_LinkData* links,
    GPU_NodeData* nodes,
    int linkCount)
//
//  Purpose: Updates node inflows and outflows based on link flows
//  Note: Uses atomic operations for thread safety
//
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= linkCount) return;

    int n1 = links->node1[j];
    int n2 = links->node2[j];
    double q = links->newFlow[j];

    // Update node flows using atomic operations
    if (q >= 0.0) {
        // Flow from n1 to n2
        atomicAdd(&nodes->outflow[n1], q);
        atomicAdd(&nodes->inflow[n2], q);
    } else {
        // Reverse flow
        atomicAdd(&nodes->inflow[n1], -q);
        atomicAdd(&nodes->outflow[n2], -q);
    }
}

//=============================================================================
// Host Function: Launch Conduit Flows Kernel
//=============================================================================

extern "C" {

int gpu_computeConduitFlows(
    GPU_LinkData* links_unused,
    GPU_ConduitData* conduits_unused,
    GPU_XsectData* xsects_unused,
    GPU_NodeData* nodes_unused,
    double dt,
    int steps,
    double omega,
    int surchargeMethod,
    double crownCutoff,
    int inertDamping)
//
//  Purpose: Copies SWMM link/conduit state to GPU, runs the conduit flow kernel,
//           and copies results back. Returns 0 on success or -1 on failure.
//
{
    extern TNode* Node;
    extern int Nobjects[];

    // Ensure GPU structures are allocated
    if (ensureConduitKernelContext() != 0) {
        return -1;
    }

    GPU_LinkData* links = &g_conduitKernelCtx.links;
    GPU_ConduitData* conduits = &g_conduitKernelCtx.conduits;
    GPU_XsectData* xsects = &g_conduitKernelCtx.xsects;
    GPU_NodeData* nodes = &g_conduitKernelCtx.nodes;

    // Copy data from CPU to GPU
    copyLinksToGpu(links);
    copyConduitsToGpu(conduits);
    copyXsectsToGpu(xsects);  // This includes geometry!
    copyNodesToGpu(nodes);

    // Determine launch configuration
    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = GRID_SIZE(links->count, blockSize);

    // Record start time
    cudaEventRecord(g_conduitKernelStartEvent, 0);

    // Launch conduit flow kernel
    kernel_findConduitFlows<<<gridSize, blockSize>>>(
        links, conduits, xsects, nodes,
        dt, steps, omega,
        surchargeMethod, crownCutoff, inertDamping);

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset node inflows/outflows before update using GPU kernel
    int nodeGridSize = GRID_SIZE(nodes->count, blockSize);
    kernel_resetNodeFlows<<<nodeGridSize, blockSize>>>(nodes, nodes->count);

    CUDA_CHECK_LAST_ERROR();

    // Launch node flow update kernel
    kernel_updateNodeFlows<<<gridSize, blockSize>>>(
        links, nodes, links->count);

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record end time
    cudaEventRecord(g_conduitKernelStopEvent, 0);
    cudaEventSynchronize(g_conduitKernelStopEvent);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, g_conduitKernelStartEvent, g_conduitKernelStopEvent);
    gpu_profiler_addKernelTime((double)elapsedMs);

    // Copy results back to CPU
    copyLinksFromGpu(links);
    copyConduitsFromGpu(conduits);
    copyNodesFromGpu(nodes);  // Node inflow/outflow updated by kernel

    return 0;
}

} // extern "C"
