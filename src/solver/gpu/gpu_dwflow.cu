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
#include "gpu_table_helpers.cuh"
#include "gpu_nonconduit_helpers.cuh"

#define GPU_PUMP_DEBUG_MAX 256

typedef struct {
    double dt;
    double qCurve;
    double qFinal;
    double oldVolume;
    double inflowBeforeUp;
    double outflowBeforeUp;
    double inflowAfterUp;
    double outflowAfterUp;
    double inflowBeforeDown;
    double inflowAfterDown;
    double oldDepth;
    double newSurf;
    int pumpIndex;
    int upNodeIndex;
    int downNodeIndex;
    int callIndex;
} GPUPumpDebugEntry;

__device__ GPUPumpDebugEntry g_gpuPumpDebugEntries[64 * GPU_PUMP_DEBUG_MAX];

extern "C" {
#include "headers.h"
#include "dynwave_data.h"
}

//=============================================================================
// Static Context for Conduit Flow Kernel
//=============================================================================

typedef struct {
    GPU_LinkData links;
    GPU_ConduitData conduits;
    GPU_XsectData xsects;
    GPU_NodeData nodes;  // Need node data for flow calculations
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

static ConduitKernelContext g_conduitKernelCtx = {0};
static cudaEvent_t g_conduitKernelStartEvent = 0;
static cudaEvent_t g_conduitKernelStopEvent = 0;
static int g_conduitKernelEventsInitialized = 0;
extern "C" void gpu_flushConduitResults(void);

//=============================================================================
// External GPU Data Structures for Non-Conduit Links
//=============================================================================
extern "C" {
    extern GPU_PumpData g_gpuPumps;
    extern GPU_OrificeData g_gpuOrifices;
    extern GPU_WeirData g_gpuWeirs;
    extern GPU_OutletData g_gpuOutlets;
    extern GPU_CurveData g_gpuCurves;
    extern GPU_CurvePoints g_gpuCurvePoints;
}

//=============================================================================
// Helper Functions: Data Transfer
//=============================================================================

static void copyLinksToGpu(GPU_LinkData* gpuLinks)
{
    if (!g_conduitKernelCtx.linkStaticsInitialized)
    {
        int count = gpuLinks->count;
        for (int j = 0; j < count; j++)
        {
            gpuLinks->h_type[j] = Link[j].type;
            gpuLinks->h_subIndex[j] = Link[j].subIndex;
            gpuLinks->h_node1[j] = Link[j].node1;
            gpuLinks->h_node2[j] = Link[j].node2;
            gpuLinks->h_offset1[j] = Link[j].offset1;
            gpuLinks->h_offset2[j] = Link[j].offset2;
            gpuLinks->h_qFull[j] = Link[j].qFull;
            gpuLinks->h_direction[j] = Link[j].direction;
            gpuLinks->h_hasFlapGate[j] = Link[j].hasFlapGate;
            if (Link[j].type == PUMP) {
                printf("Link %d (pump %d) node1=%d node2=%d\n",
                       j, Link[j].subIndex, Link[j].node1, Link[j].node2);
            }
        }
        g_conduitKernelCtx.linkStaticsInitialized = 1;
        g_conduitKernelCtx.linkStaticsUploaded = 0;
    }

    int count = gpuLinks->count;
    for (int j = 0; j < count; j++)
    {
        gpuLinks->h_oldFlow[j] = Link[j].oldFlow;
        gpuLinks->h_newFlow[j] = Link[j].newFlow;
        gpuLinks->h_oldDepth[j] = Link[j].oldDepth;
        gpuLinks->h_newDepth[j] = Link[j].newDepth;
        gpuLinks->h_oldVolume[j] = Link[j].oldVolume;
        gpuLinks->h_newVolume[j] = Link[j].newVolume;
        gpuLinks->h_bypassed[j] = (char)Link[j].bypassed;
        gpuLinks->h_flowClass[j] = Link[j].flowClass;
        gpuLinks->h_surfArea1[j] = Link[j].surfArea1;
        gpuLinks->h_surfArea2[j] = Link[j].surfArea2;
        gpuLinks->h_dqdh[j] = Link[j].dqdh;
        gpuLinks->h_froude[j] = Link[j].froude;
        gpuLinks->h_setting[j] = Link[j].setting;
        gpuLinks->h_targetSetting[j] = Link[j].targetSetting;
    }

    if (!g_conduitKernelCtx.linkStaticsUploaded)
    {
        gpu_transferLinkStaticToDevice(gpuLinks, gpuLinks->count);
        g_conduitKernelCtx.linkStaticsUploaded = 1;
    }
    gpu_transferLinkDynamicToDevice(gpuLinks, gpuLinks->count);
}

static void copyConduitsToGpu(GPU_ConduitData* gpuConduits)
{
    int k = 0;
    int needStaticUpload = !g_conduitKernelCtx.conduitStaticsInitialized;
    for (int j = 0; j < Nobjects[LINK]; j++)
    {
        if (Link[j].type == CONDUIT && k < gpuConduits->count)
        {
            int conduitIdx = Link[j].subIndex;
            if (!g_conduitKernelCtx.conduitStaticsInitialized)
            {
                gpuConduits->h_length[k] = Conduit[conduitIdx].length;
                gpuConduits->h_modLength[k] = Conduit[conduitIdx].modLength;
                gpuConduits->h_roughness[k] = Conduit[conduitIdx].roughness;
                gpuConduits->h_roughFactor[k] = Conduit[conduitIdx].roughFactor;
                gpuConduits->h_slope[k] = Conduit[conduitIdx].slope;
                gpuConduits->h_barrels[k] = (char)Conduit[conduitIdx].barrels;
            }
            gpuConduits->h_q1[k] = Conduit[conduitIdx].q1;
            gpuConduits->h_q2[k] = Conduit[conduitIdx].q2;
            gpuConduits->h_a1[k] = Conduit[conduitIdx].a1;
            gpuConduits->h_a2[k] = Conduit[conduitIdx].a2;
            gpuConduits->h_q1Old[k] = Conduit[conduitIdx].q1Old;
            gpuConduits->h_q2Old[k] = Conduit[conduitIdx].q2Old;
            k++;
        }
    }
    if (needStaticUpload)
    {
        g_conduitKernelCtx.conduitStaticsInitialized = 1;
        g_conduitKernelCtx.conduitStaticsUploaded = 0;
    }
    if (!g_conduitKernelCtx.conduitStaticsUploaded)
    {
        gpu_transferConduitStaticToDevice(gpuConduits, gpuConduits->count);
        g_conduitKernelCtx.conduitStaticsUploaded = 1;
    }
    gpu_transferConduitDynamicToDevice(gpuConduits, gpuConduits->count);
}

static void copyXsectsToGpu(GPU_XsectData* gpuXsects)
{
    int count = gpuXsects->count;
    for (int j = 0; j < count; j++)
    {
        int xtype = Link[j].xsect.type;
        gpuXsects->h_type[j] = xtype;
        gpuXsects->h_aFull[j] = Link[j].xsect.aFull;
        gpuXsects->h_rFull[j] = Link[j].xsect.rFull;
        gpuXsects->h_wMax[j] = Link[j].xsect.wMax;
        gpuXsects->h_yFull[j] = Link[j].xsect.yFull;

        // CRITICAL: Map geometry parameters based on cross-section shape type
        // These are used by gpu_xsect_helpers.cuh functions
        if (xtype == CIRCULAR || xtype == FILLED_CIRCULAR) {
            // For circular: geom1 = diameter
            gpuXsects->h_geom1[j] = Link[j].xsect.yFull;  // Diameter = full depth
            gpuXsects->h_geom2[j] = 0.0;
            gpuXsects->h_geom3[j] = 0.0;
        }
        else if (xtype == RECT_CLOSED || xtype == RECT_OPEN) {
            // For rectangular: geom1 = width, geom2 = height
            gpuXsects->h_geom1[j] = Link[j].xsect.wMax;   // Width
            gpuXsects->h_geom2[j] = Link[j].xsect.yFull;  // Height
            gpuXsects->h_geom3[j] = 0.0;
        }
        else if (xtype == TRAPEZOIDAL) {
            // For trapezoidal: geom1 = bottom width, geom2 = side slope
            // These are stored in multipurpose fields
            gpuXsects->h_geom1[j] = Link[j].xsect.yBot;   // Bottom width
            gpuXsects->h_geom2[j] = Link[j].xsect.sBot;   // Side slope
            gpuXsects->h_geom3[j] = 0.0;
        }
        else if (xtype == TRIANGULAR) {
            // For triangular: geom1 = side slope (bottom width = 0)
            gpuXsects->h_geom1[j] = Link[j].xsect.sBot;   // Side slope
            gpuXsects->h_geom2[j] = 0.0;
            gpuXsects->h_geom3[j] = 0.0;
        }
        else {
            // For unsupported shapes, zero out geometry (will use linear interpolation)
            gpuXsects->h_geom1[j] = 0.0;
            gpuXsects->h_geom2[j] = 0.0;
            gpuXsects->h_geom3[j] = 0.0;
        }
    }
    if (!g_conduitKernelCtx.xsectsUploaded)
    {
        gpu_transferXsectStaticToDevice(gpuXsects, gpuXsects->count);
        g_conduitKernelCtx.xsectsUploaded = 1;
    }
}

static void copyLinksFromGpu(GPU_LinkData* gpuLinks)
{
    int count = gpuLinks->count;
    for (int j = 0; j < count; j++)
    {
        Link[j].newFlow = gpuLinks->h_newFlow[j];
        Link[j].newDepth = gpuLinks->h_newDepth[j];
        Link[j].newVolume = gpuLinks->h_newVolume[j];
        Link[j].dqdh = gpuLinks->h_dqdh[j];
        Link[j].froude = gpuLinks->h_froude[j];
        Link[j].surfArea1 = gpuLinks->h_surfArea1[j];
        Link[j].surfArea2 = gpuLinks->h_surfArea2[j];
        Link[j].setting = gpuLinks->h_setting[j];
        Link[j].targetSetting = gpuLinks->h_targetSetting[j];
    }
}

static void copyLinkIterStateFromGpu(GPU_LinkData* gpuLinks)
{
    int count = gpuLinks->count;
    for (int j = 0; j < count; j++)
    {
        Link[j].newFlow = gpuLinks->h_newFlow[j];
        Link[j].newDepth = gpuLinks->h_newDepth[j];
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
            Conduit[conduitIdx].q1 = gpuConduits->h_q1[k];
            Conduit[conduitIdx].q2 = gpuConduits->h_q2[k];
            Conduit[conduitIdx].a1 = gpuConduits->h_a1[k];
            k++;
        }
    }
}

static void copyNodesToGpu(GPU_NodeData* gpuNodes)
{
    if (!g_conduitKernelCtx.nodeStaticsInitialized)
    {
        int count = gpuNodes->count;
        for (int i = 0; i < count; i++)
        {
            gpuNodes->h_type[i]        = Node[i].type;
            gpuNodes->h_invertElev[i]  = Node[i].invertElev;
            gpuNodes->h_fullDepth[i]   = Node[i].fullDepth;
            gpuNodes->h_surDepth[i]    = Node[i].surDepth;
            gpuNodes->h_pondedArea[i]  = Node[i].pondedArea;
            gpuNodes->h_crownElev[i]   = Node[i].crownElev;
            gpuNodes->h_fullVolume[i]  = Node[i].fullVolume;
            gpuNodes->h_degree[i]      = Node[i].degree;
        }
        g_conduitKernelCtx.nodeStaticsInitialized = 1;
        g_conduitKernelCtx.nodeStaticsUploaded = 0;
    }

    int count = gpuNodes->count;
    static int copyIteration = 0;
    for (int i = 0; i < count; i++)
    {
        gpuNodes->h_newDepth[i]    = Node[i].newDepth;
        gpuNodes->h_oldDepth[i]    = Node[i].oldDepth;
        gpuNodes->h_oldVolume[i]   = Node[i].oldVolume;
        if (i == 852 && copyIteration < 3000) {
            printf("copyNodesToGpu(iter=%d): node 852 oldVol=%.6f newVol=%.6f oldDepth=%.6f newDepth=%.6f\n",
                   copyIteration, Node[i].oldVolume, Node[i].newVolume,
                   Node[i].oldDepth, Node[i].newDepth);
        }
        gpuNodes->h_newVolume[i]   = Node[i].newVolume;
        gpuNodes->h_oldNetInflow[i]= Node[i].oldNetInflow;
        gpuNodes->h_inflow[i]      = Node[i].inflow;
        gpuNodes->h_outflow[i]     = Node[i].outflow;
        gpuNodes->h_overflow[i]    = Node[i].overflow;
        gpuNodes->h_newSurfArea[i] = Xnode[i].newSurfArea;
        gpuNodes->h_oldSurfArea[i] = Xnode[i].oldSurfArea;
        gpuNodes->h_sumdqdh[i]     = Xnode[i].sumdqdh;
        gpuNodes->h_converged[i]   = Xnode[i].converged;
        gpuNodes->h_dYdT[i]        = Xnode[i].dYdT;
    }

    if (!g_conduitKernelCtx.nodeStaticsUploaded)
    {
        gpu_transferNodeStaticToDevice(gpuNodes, gpuNodes->count);
        g_conduitKernelCtx.nodeStaticsUploaded = 1;
    }
    gpu_transferNodeDynamicToDevice(gpuNodes, gpuNodes->count);
    copyIteration++;
}

static void copyNodesFromGpu(GPU_NodeData* gpuNodes)
{
    int count = gpuNodes->count;
    for (int i = 0; i < count; i++)
    {
        // Update node inflows/outflows (updated by kernel)
        Node[i].inflow = gpuNodes->h_inflow[i];
        Node[i].outflow = gpuNodes->h_outflow[i];
        Xnode[i].newSurfArea = gpuNodes->h_newSurfArea[i];
        Xnode[i].sumdqdh     = gpuNodes->h_sumdqdh[i];
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
    int nonConduitCount = 0;
    for (int j = 0; j < linkCount; j++)
    {
        if (Link[j].type == CONDUIT) conduitCount++;
        if (!(Link[j].type == CONDUIT && Link[j].xsect.type != DUMMY)) {
            nonConduitCount++;
        }
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
            if (g_conduitKernelCtx.d_links)  { cudaFree(g_conduitKernelCtx.d_links);   g_conduitKernelCtx.d_links = nullptr; }
            if (g_conduitKernelCtx.d_conduits) { cudaFree(g_conduitKernelCtx.d_conduits); g_conduitKernelCtx.d_conduits = nullptr; }
            if (g_conduitKernelCtx.d_xsects) { cudaFree(g_conduitKernelCtx.d_xsects); g_conduitKernelCtx.d_xsects = nullptr; }
            if (g_conduitKernelCtx.d_nodes)  { cudaFree(g_conduitKernelCtx.d_nodes);   g_conduitKernelCtx.d_nodes = nullptr; }
            g_conduitKernelCtx.initialized = 0;
        }

        if (gpu_allocateLinkData(&g_conduitKernelCtx.links, linkCount) != 0) return -1;
        if (gpu_allocateConduitData(&g_conduitKernelCtx.conduits, conduitCount) != 0) return -1;
        if (gpu_allocateXsectData(&g_conduitKernelCtx.xsects, linkCount) != 0) return -1;
        if (gpu_allocateNodeData(&g_conduitKernelCtx.nodes, nodeCount) != 0) return -1;
        CUDA_CHECK(cudaMalloc((void**)&g_conduitKernelCtx.d_links, sizeof(GPU_LinkData)));
        CUDA_CHECK(cudaMalloc((void**)&g_conduitKernelCtx.d_conduits, sizeof(GPU_ConduitData)));
        CUDA_CHECK(cudaMalloc((void**)&g_conduitKernelCtx.d_xsects, sizeof(GPU_XsectData)));
        CUDA_CHECK(cudaMalloc((void**)&g_conduitKernelCtx.d_nodes, sizeof(GPU_NodeData)));

        g_conduitKernelCtx.linkCapacity = linkCount;
        g_conduitKernelCtx.conduitCapacity = conduitCount;
        g_conduitKernelCtx.xsectCapacity = linkCount;
        g_conduitKernelCtx.nodeCapacity = nodeCount;
        g_conduitKernelCtx.initialized = 1;
        g_conduitKernelCtx.xsectsInitialized = 0;
        g_conduitKernelCtx.linkStaticsInitialized = 0;
        g_conduitKernelCtx.conduitStaticsInitialized = 0;
        g_conduitKernelCtx.nodeStaticsInitialized = 0;
        g_conduitKernelCtx.resultsDirty = 0;
        g_conduitKernelCtx.linkStaticsUploaded = 0;
        g_conduitKernelCtx.conduitStaticsUploaded = 0;
        g_conduitKernelCtx.xsectsUploaded = 0;
        g_conduitKernelCtx.nodeStaticsUploaded = 0;
    }

    g_conduitKernelCtx.nonConduitCount = nonConduitCount;

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
//  Output:  Updates links->d_newFlow, links->d_newDepth, links->d_newVolume
//           Updates conduits->d_q1, conduits->d_q2, conduits->d_a1
//           Updates links->d_dqdh, links->d_froude
//
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= links->count) return;

    // Skip non-conduits
    if (links->d_type[j] != 0) return;  // 0 = CONDUIT

    // Skip bypassed links
    if (links->d_bypassed[j]) return;

    int k = links->d_subIndex[j];  // Conduit index
    int n1 = links->d_node1[j];    // Upstream node
    int n2 = links->d_node2[j];    // Downstream node

    // Create xsect structure for this link from GPU arrays
    GPU_Xsect xsect;
    xsect.type = xsects->d_type[j];
    xsect.yFull = xsects->d_yFull[j];
    xsect.aFull = xsects->d_aFull[j];
    xsect.rFull = xsects->d_rFull[j];
    xsect.wMax = xsects->d_wMax[j];
    xsect.geom1 = xsects->d_geom1[j];  // Diameter, width, etc.
    xsect.geom2 = xsects->d_geom2[j];  // Height, side slope, etc.
    xsect.geom3 = xsects->d_geom3[j];  // Additional geometry

    // Call simplified conduit flow calculation
    double q, aMid, yMid, dqdh, froude;
    double surfArea1, surfArea2;
    int flowClass;

    gpu_findConduitFlow_simplified(
        j, k, n1, n2,
        &xsect,
        // Node data
        nodes->d_newDepth[n1],
        nodes->d_invertElev[n1],
        nodes->d_newDepth[n2],
        nodes->d_invertElev[n2],
        // Link data
        links->d_offset1[j],
        links->d_offset2[j],
        links->d_oldFlow[j],
        // Conduit data
        (double)conduits->d_barrels[k],
        conduits->d_modLength[k],
        conduits->d_roughFactor[k],
        conduits->d_q1[k],
        conduits->d_a2[k],
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
        &froude,
        &surfArea1,
        &surfArea2,
        &flowClass);

    // Save results
    double barrels = (double)conduits->d_barrels[k];
    conduits->d_q1[k] = q;
    conduits->d_q2[k] = q;
    conduits->d_a1[k] = aMid;

    double qTotal = q * barrels;
    links->d_newFlow[j] = qTotal;
    links->d_newDepth[j] = yMid;
    links->d_newVolume[j] = aMid * conduits->d_length[k] * barrels;
    links->d_dqdh[j] = dqdh;
    links->d_froude[j] = froude;
    links->d_surfArea1[j] = surfArea1;
    links->d_surfArea2[j] = surfArea2;
    links->d_flowClass[j] = (signed char)flowClass;

    // Accumulate node-level terms
    atomicAdd(&nodes->d_newSurfArea[n1], surfArea1 * barrels);
    atomicAdd(&nodes->d_newSurfArea[n2], surfArea2 * barrels);
    atomicAdd(&nodes->d_sumdqdh[n1], dqdh);
    atomicAdd(&nodes->d_sumdqdh[n2], dqdh);

    if (qTotal >= 0.0) {
        atomicAdd(&nodes->d_outflow[n1], qTotal);
        atomicAdd(&nodes->d_inflow[n2], qTotal);
    } else {
        atomicAdd(&nodes->d_inflow[n1], -qTotal);
        atomicAdd(&nodes->d_outflow[n2], -qTotal);
    }

}

//=============================================================================
// Kernel: Find Pump Flows
//=============================================================================

__global__ void kernel_findPumpFlows(
    GPU_LinkData* links,
    GPU_PumpData* pumps,
    GPU_NodeData* nodes,
    GPU_CurveData* curves,
    GPU_CurvePoints* points,
    double ucfVolume,          // unit conversion factors
    double ucfLength,
    double ucfFlow,
    double dt,                 // time step
    int steps)
//
//  Purpose: Computes flow through all pump links
//  Note: Pumps do not use under-relaxation (omega = 1.0)
//
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Pump index

    if (k >= pumps->count) return;

    // Get link index for this pump
    int j = pumps->d_linkIndex[k];

    // Skip bypassed links or closed pumps
    if (links->d_bypassed[j] || links->d_setting[j] == 0.0) {
        links->d_newFlow[j] = 0.0;
        links->d_newDepth[j] = 0.0;
        links->d_dqdh[j] = 0.0;
        links->d_flowClass[j] = 0; // NO flow class
        return;
    }
    int n1 = links->d_node1[j];    // Upstream node
    int n2 = links->d_node2[j];    // Downstream node
    int curveIdx = pumps->d_pumpCurve[k];

    double qIn = 0.0;
    double dqdh = 0.0;
    char flowClass = 0; // NO

    int pumpType = pumps->d_type[k];
    double xMin = pumps->d_xMin[k];
    double xMax = pumps->d_xMax[k];
    double setting = links->d_setting[j];

    // Compute flow based on pump type
    switch (pumpType) {
        case 5: // IDEAL_PUMP
            qIn = gpu_pump_getIdealFlow(n1, nodes->d_inflow, nodes->d_overflow);
            break;

        case 0: // TYPE1_PUMP (volume curve)
            qIn = gpu_pump_getType1Flow(k, curveIdx, nodes->d_newVolume[n1],
                ucfVolume, ucfFlow, xMin, xMax, &flowClass, curves, points);
            break;

        case 1: // TYPE2_PUMP (depth curve, discrete)
            qIn = gpu_pump_getType2Flow(k, curveIdx, nodes->d_newDepth[n1],
                ucfLength, ucfFlow, xMin, xMax, &flowClass, curves, points);
            break;

        case 2: // TYPE3_PUMP (head curve, continuous)
        case 4: // TYPE5_PUMP (variable speed TYPE3)
        {
            double speed = (pumpType == 4) ? setting : 1.0;
            qIn = gpu_pump_getType3Flow(curveIdx,
                nodes->d_newDepth[n1], nodes->d_invertElev[n1],
                nodes->d_newDepth[n2], nodes->d_invertElev[n2],
                speed, ucfLength, ucfFlow, xMin, xMax,
                &flowClass, &dqdh, curves, points);
            break;
        }

        case 3: // TYPE4_PUMP (depth curve, continuous)
            qIn = gpu_pump_getType4Flow(curveIdx, nodes->d_newDepth[n1],
                ucfLength, ucfFlow, xMin, xMax, &flowClass, &dqdh, curves, points);
            break;
    }

    // No reverse flow through pumps
    if (qIn < 0.0) qIn = 0.0;

    // Apply setting
    qIn *= setting;
    // TODO: Implement parallel-safe pump flow modification
    // The CPU version uses Node[j].outflow which is built up sequentially,
    // but this doesn't work in parallel GPU execution
    // For now, we skip this check - may cause minor mass balance issues
    //
    // qIn = gpu_getModPumpFlow(
    //     k, n1, qIn, dt, pumpType,
    //     nodes->d_type,
    //     nodes->d_inflow,
    //     nodes->d_outflow,
    //     nodes->d_oldDepth,
    //     nodes->d_oldNetInflow,
    //     nodes->d_oldVolume,
    //     nodes->d_fullVolume,
    //     nodes->d_newSurfArea);

    // Update link state
    links->d_newFlow[j] = qIn;
    links->d_newDepth[j] = 0.0;  // Pumps have no depth
    links->d_dqdh[j] = dqdh;
    links->d_flowClass[j] = flowClass;

    // Update node flows
    if (qIn > 0.0) {
        atomicAdd(&nodes->d_outflow[n1], qIn);
        atomicAdd(&nodes->d_inflow[n2], qIn);
    }

    // Add dqdh to downstream node for TYPE3/TYPE5 pumps
    if (pumpType == 2 || pumpType == 4) {
        atomicAdd(&nodes->d_sumdqdh[n2], dqdh);
    }
}

//=============================================================================
// Kernel: Find Preliminary Pump Flows (Phase 1 of 2-pass processing)
//=============================================================================

__global__ void kernel_findPumpFlows_Preliminary(
    GPU_LinkData* links,
    GPU_PumpData* pumps,
    GPU_NodeData* nodes,
    GPU_CurveData* curves,
    GPU_CurvePoints* points,
    double ucfVolume,
    double ucfLength,
    double ucfFlow,
    double dt,
    int steps,
    double* d_prelimFlows,  // OUTPUT: preliminary pump flows before modification
    double* d_prelimDqdh,   // OUTPUT: preliminary dqdh values
    char* d_prelimFlowClass) // OUTPUT: preliminary flow classes
//
//  Purpose: Computes preliminary pump flows without updating node flows
//           This allows all links to finish before modifying pump flows
//
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Pump index

    if (k >= pumps->count) return;

    // Get link index for this pump
    int j = pumps->d_linkIndex[k];

    // Skip bypassed links or closed pumps
    if (links->d_bypassed[j] || links->d_setting[j] == 0.0) {
        d_prelimFlows[k] = 0.0;
        d_prelimDqdh[k] = 0.0;
        d_prelimFlowClass[k] = 0;
        return;
    }

    int n1 = links->d_node1[j];
    int n2 = links->d_node2[j];
    int curveIdx = pumps->d_pumpCurve[k];

    double qIn = 0.0;
    double dqdh = 0.0;
    char flowClass = 0;

    int pumpType = pumps->d_type[k];
    double xMin = pumps->d_xMin[k];
    double xMax = pumps->d_xMax[k];
    double setting = links->d_setting[j];

    // Compute flow based on pump type
    switch (pumpType) {
        case 5: // IDEAL_PUMP
            qIn = gpu_pump_getIdealFlow(n1, nodes->d_inflow, nodes->d_overflow);
            break;

        case 0: // TYPE1_PUMP (volume curve)
            qIn = gpu_pump_getType1Flow(k, curveIdx, nodes->d_newVolume[n1],
                ucfVolume, ucfFlow, xMin, xMax, &flowClass, curves, points);
            break;

        case 1: // TYPE2_PUMP (depth curve, discrete)
            qIn = gpu_pump_getType2Flow(k, curveIdx, nodes->d_newDepth[n1],
                ucfLength, ucfFlow, xMin, xMax, &flowClass, curves, points);
            break;

        case 2: // TYPE3_PUMP (head curve, continuous)
        case 4: // TYPE5_PUMP (variable speed TYPE3)
        {
            double speed = (pumpType == 4) ? setting : 1.0;
            qIn = gpu_pump_getType3Flow(curveIdx,
                nodes->d_newDepth[n1], nodes->d_invertElev[n1],
                nodes->d_newDepth[n2], nodes->d_invertElev[n2],
                speed, ucfLength, ucfFlow, xMin, xMax,
                &flowClass, &dqdh, curves, points);
            break;
        }

        case 3: // TYPE4_PUMP (depth curve, continuous)
            qIn = gpu_pump_getType4Flow(curveIdx, nodes->d_newDepth[n1],
                ucfLength, ucfFlow, xMin, xMax, &flowClass, &dqdh, curves, points);
            break;
    }

    // No reverse flow through pumps
    if (qIn < 0.0) qIn = 0.0;

    // Apply setting
    qIn *= setting;


    // Store preliminary results (don't update links/nodes yet)
    d_prelimFlows[k] = qIn;
    d_prelimDqdh[k] = dqdh;
    d_prelimFlowClass[k] = flowClass;
}

//=============================================================================
// Kernel: Accumulate Preliminary Pump Outflows (Phase 1b of 3-phase processing)
//=============================================================================

__global__ void kernel_accumulatePreliminaryPumpOutflows(
    GPU_LinkData* links,
    GPU_PumpData* pumps,
    GPU_NodeData* nodes,
    const double* d_prelimFlows)  // INPUT: preliminary flows from Phase 1
//
//  Purpose: Accumulate preliminary pump flows to node inflows/outflows
//           This happens BEFORE getModPumpFlow() modification
//           Allows Phase 2 to see complete outflow when applying modifications
//
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Pump index

    if (k >= pumps->count) return;

    int j = pumps->d_linkIndex[k];
    double qIn = d_prelimFlows[k];

    // Skip bypassed or zero flow
    if (links->d_bypassed[j] || qIn == 0.0) return;

    int n1 = links->d_node1[j];
    int n2 = links->d_node2[j];

    // Accumulate preliminary flows ONLY to downstream node inflow
    // DO NOT add to d_outflow yet - that will be done in Phase 2 after getModPumpFlow
    // This way d_nodeOutflow in Phase 2 only contains conduit flows (matching CPU behavior)
    // atomicAdd(&nodes->d_outflow[n1], qIn);  // REMOVED - do this in Phase 2 instead
    atomicAdd(&nodes->d_inflow[n2], qIn);
}

//=============================================================================
// Kernel: Process Pumps FULLY SEQUENTIALLY (Exact CPU Logic)
//=============================================================================

__device__ int g_gpuPumpDebugCounter[64];

__global__ void kernel_processPumpsSequentially(
    GPU_LinkData* links,
    GPU_PumpData* pumps,
    GPU_NodeData* nodes,
    GPU_CurveData* curves,
    GPU_CurvePoints* curvePoints,
    double dt,
    int routeModel,
    double ucfVolume,
    double ucfLength,
    double ucfFlow)
//
//  Purpose: Process ALL pumps sequentially in a single kernel, EXACTLY like CPU.
//           Each pump: compute flow → getModPumpFlow → update nodes → next pump
//           This ensures each pump sees the exact state the CPU sees.
//
{
    // Single thread processes ALL pumps sequentially
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    printf("GPU sequential pump kernel invoked dt=%.6f count=%d\n", dt, pumps->count);

    // Process each pump sequentially (EXACTLY like CPU for-loop)
    for (int k = 0; k < pumps->count; k++) {

    int j = pumps->d_linkIndex[k];

    // Skip if bypassed
    if (links->d_bypassed[j]) {
        links->d_newFlow[j] = 0.0;
        links->d_newDepth[j] = 0.0;
        links->d_dqdh[j] = 0.0;
        links->d_flowClass[j] = 0;
        continue;
    }

    int n1 = links->d_node1[j];
    int n2 = links->d_node2[j];

    bool logPump = (k == 1 || k == 15 || k == 20 || k == 22 || k == 24 || k == 25 || k == 26 ||
                    k == 32 || k == 35 || k == 38 || k == 41 || k == 43 || k == 45);
    int debugSlot = -1;
    if (logPump) {
        debugSlot = atomicAdd(&g_gpuPumpDebugCounter[k], 1);
        int debugPrintLimit = (k == 1 || k == 22 || k == 45) ? 4000 : 5;
        if (debugSlot < debugPrintLimit) {
            double oldVolDebug = nodes->d_oldVolume[n1];
            printf("GPU pump[%d] upstream node=%d downstream=%d oldVol=%.6f (call=%d)\n",
                   k, n1, n2, oldVolDebug, debugSlot);
        }
    }

    double inflowUpBefore = nodes->d_inflow[n1];
    double outflowUpBefore = nodes->d_outflow[n1];
    double inflowDownBefore = nodes->d_inflow[n2];

    // STEP 1: Compute pump flow from curve (exactly like CPU/Phase 1)
    double qIn = 0.0;
    double dqdh = 0.0;
    char flowClass = 0;

    int pumpType = pumps->d_type[k];
    int curveIdx = pumps->d_pumpCurve[k];
    double xMin = pumps->d_xMin[k];
    double xMax = pumps->d_xMax[k];
    double setting = links->d_setting[j];

    // Skip if pump is closed
    if (setting == 0.0) {
        links->d_newFlow[j] = 0.0;
        links->d_newDepth[j] = 0.0;
        links->d_dqdh[j] = 0.0;
        links->d_flowClass[j] = 0;
        continue;
    }

    // Compute flow based on pump type (inline from kernel_findPumpFlows_Preliminary)
    switch (pumpType) {
        case 5: // IDEAL_PUMP
            qIn = gpu_pump_getIdealFlow(n1, nodes->d_inflow, nodes->d_overflow);
            break;

        case 0: // TYPE1_PUMP (volume curve)
            qIn = gpu_pump_getType1Flow(k, curveIdx, nodes->d_newVolume[n1],
                ucfVolume, ucfFlow, xMin, xMax, &flowClass, curves, curvePoints);
            break;

        case 1: // TYPE2_PUMP (depth curve, discrete)
            qIn = gpu_pump_getType2Flow(k, curveIdx, nodes->d_newDepth[n1],
                ucfLength, ucfFlow, xMin, xMax, &flowClass, curves, curvePoints);
            break;

        case 2: // TYPE3_PUMP (head curve, continuous)
        case 4: // TYPE5_PUMP (variable speed TYPE3)
        {
            double speed = (pumpType == 4) ? setting : 1.0;
            qIn = gpu_pump_getType3Flow(curveIdx,
                nodes->d_newDepth[n1], nodes->d_invertElev[n1],
                nodes->d_newDepth[n2], nodes->d_invertElev[n2],
                speed, ucfLength, ucfFlow, xMin, xMax,
                &flowClass, &dqdh, curves, curvePoints);
            break;
        }

        case 3: // TYPE4_PUMP (depth curve, continuous)
            qIn = gpu_pump_getType4Flow(curveIdx, nodes->d_newDepth[n1],
                ucfLength, ucfFlow, xMin, xMax, &flowClass, &dqdh, curves, curvePoints);
            break;
    }

    // No reverse flow through pumps
    if (qIn < 0.0) qIn = 0.0;

    // Apply setting
    qIn *= setting;

    double qCurve = qIn;

    // Skip if zero flow
    if (qIn == 0.0) {
        links->d_newFlow[j] = 0.0;
        links->d_newDepth[j] = 0.0;
        links->d_dqdh[j] = 0.0;
        links->d_flowClass[j] = 0;
        continue;
    }

    // STEP 2: Apply getModPumpFlow() to prevent over-draining
    // At this point, d_nodeOutflow contains: conduits + previous pumps (EXACTLY like CPU!)
    double qOriginal = qIn;
    qIn = gpu_getModPumpFlow(
        k, n1, qIn, 0.0, dt, pumps->d_type[k],  // qPrelim=0 (not used in new approach)
        nodes->d_type,
        nodes->d_inflow,
        nodes->d_outflow,      // Contains conduits + pumps 0..(k-1)
        nodes->d_oldDepth,
        nodes->d_oldNetInflow,
        nodes->d_oldVolume,
        nodes->d_fullVolume,
        nodes->d_newSurfArea);

    if (logPump && debugSlot > -1 && debugSlot < 20) {
        printf("GPU pump[%d] post-mod: qFinal=%.6f dqdh=%.6f flowClass=%d oldDepth=%.6f newSurf=%.6f\n",
               k, qIn, dqdh, (int)flowClass,
               nodes->d_oldDepth[n1],
               nodes->d_newSurfArea[n1]);
    }

    // STEP 3: Update link state
    links->d_newFlow[j] = qIn;
    links->d_newDepth[j] = 0.0;
    links->d_dqdh[j] = dqdh;
    links->d_flowClass[j] = flowClass;

    // STEP 4: IMMEDIATELY update node flows (EXACTLY like CPU updateNodeFlows!)
    // No atomicAdd needed - we're sequential
    nodes->d_outflow[n1] += qIn;
    nodes->d_inflow[n2] += qIn;

    double inflowUpAfter = nodes->d_inflow[n1];
    double outflowUpAfter = nodes->d_outflow[n1];
    double inflowDownAfter = nodes->d_inflow[n2];

    if (logPump && debugSlot > -1 && debugSlot < 20) {
        printf("GPU pump[%d] node update: newOut=%.6f newIn=%.6f\n",
               k, nodes->d_outflow[n1], nodes->d_inflow[n2]);
    }

    if (logPump && debugSlot > -1 && debugSlot < GPU_PUMP_DEBUG_MAX) {
        GPUPumpDebugEntry* entry = &g_gpuPumpDebugEntries[k * GPU_PUMP_DEBUG_MAX + debugSlot];
        entry->dt = dt;
        entry->qCurve = qCurve;
        entry->qFinal = qIn;
        entry->oldVolume = nodes->d_oldVolume[n1];
        entry->inflowBeforeUp = inflowUpBefore;
        entry->outflowBeforeUp = outflowUpBefore;
        entry->inflowAfterUp = inflowUpAfter;
        entry->outflowAfterUp = outflowUpAfter;
        entry->inflowBeforeDown = inflowDownBefore;
        entry->inflowAfterDown = inflowDownAfter;
        entry->oldDepth = nodes->d_oldDepth[n1];
        entry->newSurf = nodes->d_newSurfArea[n1];
        entry->pumpIndex = k;
        entry->upNodeIndex = n1;
        entry->downNodeIndex = n2;
        entry->callIndex = debugSlot;
    }

    // Add dqdh contributions just like CPU updateNodeFlows()
    nodes->d_sumdqdh[n1] += dqdh;
    if (pumpType != TYPE4_PUMP) {
        nodes->d_sumdqdh[n2] += dqdh;
    }

    } // End of sequential for loop
}

//=============================================================================
// Kernel: Find Orifice Flows
//=============================================================================

__global__ void kernel_findOrificeFlows(
    GPU_LinkData* links,
    GPU_OrificeData* orifices,
    GPU_XsectData* xsects,
    GPU_NodeData* nodes,
    double omega,
    int routeModel)  // 0=DW, 1=KW
//
//  Purpose: Computes flow through all orifice links
//
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Orifice index

    if (k >= orifices->count) return;

    // Get link index for this orifice
    int j = orifices->d_linkIndex[k];

    if (links->d_bypassed[j]) return;
    int n1 = links->d_node1[j];    // Upstream node
    int n2 = links->d_node2[j];    // Downstream node

    // Get heads at nodes
    double h1, h2, dir;
    if (routeModel == 0) { // DW
        h1 = nodes->d_newDepth[n1] + nodes->d_invertElev[n1];
        h2 = nodes->d_newDepth[n2] + nodes->d_invertElev[n2];
    } else { // KW
        h1 = nodes->d_newDepth[n1] + nodes->d_invertElev[n1];
        h2 = nodes->d_invertElev[n1];
    }

    dir = (h1 >= h2) ? 1.0 : -1.0;

    // Exchange for reverse flow
    double y1 = nodes->d_newDepth[n1];
    if (dir < 0.0) {
        double temp = h1;
        h1 = h2;
        h2 = temp;
        y1 = nodes->d_newDepth[n2];
    }

    int orificeType = orifices->d_type[k];
    double hcrest = nodes->d_invertElev[n1] + links->d_offset1[j];
    double head;

    // Compute head on orifice
    if (orificeType == 1) { // BOTTOM_ORIFICE
        if (h1 < hcrest) head = 0.0;
        else if (h2 > hcrest) head = h1 - h2;
        else head = h1 - hcrest;
    } else { // SIDE_ORIFICE
        double hcrown = hcrest + xsects->d_yFull[j] * links->d_setting[j];
        double hmidpt = (hcrest + hcrown) / 2.0;

        if (h1 < hmidpt) head = h1 - hcrest;
        else if (h2 < hmidpt) head = h1 - hmidpt;
        else head = h1 - h2;
    }

    // Check for flap gate
    bool flapClosed = gpu_link_setFlapGate(j, n1, n2, dir, links->d_hasFlapGate[j],
        nodes->d_type, nodes->d_newDepth, nodes->d_outflow);

    // No flow if head negligible or flap closed
    if (head <= FUDGE || y1 <= FUDGE || flapClosed) {
        links->d_newFlow[j] = 0.0;
        links->d_newDepth[j] = 0.0;
        links->d_dqdh[j] = 0.0;
        links->d_flowClass[j] = 0; // DRY
        return;
    }

    // Compute flow
    double q = 0.0;
    double dqdh = 0.0;
    double hCrit = orifices->d_hCrit[k];
    double f = head / hCrit;
    f = fmin(f, 1.0);

    int xsectType = xsects->d_type[j];
    double yFull = xsects->d_yFull[j];
    double aFull = xsects->d_aFull[j];
    double rFull = xsects->d_rFull[j];
    double wMax = xsects->d_wMax[j];
    double geom1 = xsects->d_geom1[j];
    double geom2 = xsects->d_geom2[j];

    if (f < 1.0) {
        // Weir flow
        q = gpu_orifice_getWeirFlow(j, head, f, orifices->d_cWeir[k],
            xsectType, yFull, aFull, rFull, wMax, geom1, geom2);
        dqdh = (q > FUDGE) ? (1.5 * q / head) : 0.0;
    } else {
        // Orifice flow
        q = gpu_orifice_getOrificeFlow(j, head, orifices->d_cOrif[k],
            xsectType, yFull, aFull, rFull, wMax, geom1, geom2, &dqdh);
    }

    // Apply direction and under-relaxation
    q *= dir;
    double qOld = links->d_oldFlow[j];
    double qNew = qOld + omega * (q - qOld);

    // Update link state
    links->d_newFlow[j] = qNew;
    links->d_newDepth[j] = yFull * links->d_setting[j];
    links->d_dqdh[j] = dqdh;
    links->d_flowClass[j] = (hcrest > h2) ?
        ((dir == 1.0) ? 6 : 5) : 3; // DN_CRITICAL : UP_CRITICAL : SUBCRITICAL

    // Update node flows
    if (qNew >= 0.0) {
        atomicAdd(&nodes->d_outflow[n1], qNew);
        atomicAdd(&nodes->d_inflow[n2], qNew);
    } else {
        atomicAdd(&nodes->d_inflow[n1], -qNew);
        atomicAdd(&nodes->d_outflow[n2], -qNew);
    }

    // Add dqdh to nodes
    atomicAdd(&nodes->d_sumdqdh[n1], dqdh);
    atomicAdd(&nodes->d_sumdqdh[n2], dqdh);
}

//=============================================================================
// Kernel: Find Weir Flows
//=============================================================================

__global__ void kernel_findWeirFlows(
    GPU_LinkData* links,
    GPU_WeirData* weirs,
    GPU_XsectData* xsects,
    GPU_NodeData* nodes,
    double omega,
    int routeModel)
//
//  Purpose: Computes flow over all weir links
//
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Weir index

    if (k >= weirs->count) return;

    // Get link index for this weir
    int j = weirs->d_linkIndex[k];

    if (links->d_bypassed[j]) return;
    int n1 = links->d_node1[j];
    int n2 = links->d_node2[j];

    // Get heads
    double h1, h2, dir;
    if (routeModel == 0) { // DW
        h1 = nodes->d_newDepth[n1] + nodes->d_invertElev[n1];
        h2 = nodes->d_newDepth[n2] + nodes->d_invertElev[n2];
    } else {
        h1 = nodes->d_newDepth[n1] + nodes->d_invertElev[n1];
        h2 = nodes->d_invertElev[n1];
    }

    dir = (h1 > h2) ? 1.0 : -1.0;

    // Exchange for reverse flow
    if (dir < 0.0) {
        double temp = h1;
        h1 = h2;
        h2 = temp;
    }

    // Weir crest elevation
    double hcrest = nodes->d_invertElev[n1] + links->d_offset1[j];
    double hcrown = hcrest + xsects->d_yFull[j];

    // Adjust for partially open weir
    hcrest += (1.0 - links->d_setting[j]) * xsects->d_yFull[j];

    // Compute head relative to crest
    double head = h1 - hcrest;

    // Check for flap gate
    bool flapClosed = gpu_link_setFlapGate(j, n1, n2, dir, links->d_hasFlapGate[j],
        nodes->d_type, nodes->d_newDepth, nodes->d_outflow);

    // No flow if negligible head or flap closed
    if (head <= FUDGE || hcrest >= hcrown || flapClosed) {
        links->d_newFlow[j] = 0.0;
        links->d_newDepth[j] = 0.0;
        links->d_dqdh[j] = 0.0;
        links->d_flowClass[j] = 0; // DRY
        return;
    }

    double q = 0.0;
    double dqdh = 0.0;
    int weirType = weirs->d_type[k];

    // Check if head exceeds crown (surcharged)
    if (h1 >= hcrown && weirs->d_canSurcharge[k]) {
        // Use equivalent orifice
        double y = hcrown - hcrest;
        double headOrif = (h2 < (hcrest + hcrown) / 2.0) ?
            h1 - (hcrest + hcrown) / 2.0 : h1 - h2;

        q = gpu_weir_getOrificeFlow(headOrif, y, weirs->d_cSurcharge[k],
            xsects->d_type[j], xsects->d_yFull[j],
            xsects->d_aFull[j], xsects->d_rFull[j], xsects->d_wMax[j],
            xsects->d_geom1[j], xsects->d_geom2[j], &dqdh);

        links->d_newDepth[j] = y;
    } else {
        // Standard weir flow
        if (h1 >= hcrown) head = hcrown - hcrest;

        q = gpu_weir_getFlow(weirType, head, weirs->d_cDisch1[k],
            weirs->d_cDisch2[k], weirs->d_length[k], weirs->d_slope[k], &dqdh);

        // Apply Villemonte submergence correction
        if (h2 > hcrest) {
            double ratio = (h2 - hcrest) / (h1 - hcrest);
            double weirPower = (weirType == 0) ? 1.5 :
                              (weirType == 1) ? 5.0/3.0 :
                              (weirType == 2) ? 2.5 : 1.5;
            q *= pow(1.0 - pow(ratio, weirPower), 0.385);
        }

        links->d_newDepth[j] = fmin(head, xsects->d_yFull[j]);
    }

    // Apply direction and under-relaxation
    q *= dir;
    double qOld = links->d_oldFlow[j];
    double qNew = qOld + omega * (q - qOld);

    // Update link state
    links->d_newFlow[j] = qNew;
    links->d_dqdh[j] = dqdh;
    links->d_flowClass[j] = (hcrest > h2) ?
        ((dir == 1.0) ? 6 : 5) : 3; // DN_CRITICAL : UP_CRITICAL : SUBCRITICAL

    // Update node flows
    if (qNew >= 0.0) {
        atomicAdd(&nodes->d_outflow[n1], qNew);
        atomicAdd(&nodes->d_inflow[n2], qNew);
    } else {
        atomicAdd(&nodes->d_inflow[n1], -qNew);
        atomicAdd(&nodes->d_outflow[n2], -qNew);
    }

    // Add dqdh to nodes
    atomicAdd(&nodes->d_sumdqdh[n1], dqdh);
    atomicAdd(&nodes->d_sumdqdh[n2], dqdh);
}

//=============================================================================
// Kernel: Find Outlet Flows
//=============================================================================

__global__ void kernel_findOutletFlows(
    GPU_LinkData* links,
    GPU_OutletData* outlets,
    GPU_NodeData* nodes,
    GPU_CurveData* curves,
    GPU_CurvePoints* points,
    double ucfLength,
    double ucfFlow,
    double omega,
    int routeModel)
//
//  Purpose: Computes flow through all outlet links
//
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Outlet index

    if (k >= outlets->count) return;

    // Get link index for this outlet
    int j = outlets->d_linkIndex[k];

    if (links->d_bypassed[j]) return;
    int n1 = links->d_node1[j];
    int n2 = links->d_node2[j];

    // Get heads
    double h1, h2, dir;
    if (routeModel == 0) { // DW
        h1 = nodes->d_newDepth[n1] + nodes->d_invertElev[n1];
        h2 = nodes->d_newDepth[n2] + nodes->d_invertElev[n2];
    } else {
        h1 = nodes->d_newDepth[n1] + nodes->d_invertElev[n1];
        h2 = nodes->d_invertElev[n1];
    }

    dir = (h1 >= h2) ? 1.0 : -1.0;

    // Exchange for reverse flow
    double y1 = nodes->d_newDepth[n1];
    if (dir < 0.0) {
        double temp = h1;
        h1 = h2;
        h2 = temp;
        y1 = nodes->d_newDepth[n2];
    }

    // Compute head
    double hcrest = nodes->d_invertElev[n1] + links->d_offset1[j];
    double head;
    int curveType = outlets->d_curveType[k];

    if (curveType == 1 && routeModel == 0) { // NODE_HEAD and DW
        head = h1 - fmax(h2, hcrest);
    } else { // NODE_DEPTH
        head = h1 - hcrest;
    }

    // Check for flap gate
    bool flapClosed = gpu_link_setFlapGate(j, n1, n2, dir, links->d_hasFlapGate[j],
        nodes->d_type, nodes->d_newDepth, nodes->d_outflow);

    // No flow if negligible head or flap closed
    if (head <= FUDGE || y1 <= FUDGE || flapClosed) {
        links->d_newFlow[j] = 0.0;
        links->d_newDepth[j] = 0.0;
        links->d_flowClass[j] = 0; // DRY
        return;
    }

    // Compute flow
    int curveIdx = outlets->d_qCurve[k];
    double q = gpu_outlet_getFlow(k, curveIdx, head,
        outlets->d_qCoeff[k], outlets->d_qExpon[k],
        ucfLength, ucfFlow, curves, points);

    // Apply direction, setting, and under-relaxation
    q *= dir * links->d_setting[j];
    double qOld = links->d_oldFlow[j];
    double qNew = qOld + omega * (q - qOld);

    // Update link state
    links->d_newFlow[j] = qNew;
    links->d_newDepth[j] = head;
    links->d_flowClass[j] = 3; // SUBCRITICAL

    // Update node flows
    if (qNew >= 0.0) {
        atomicAdd(&nodes->d_outflow[n1], qNew);
        atomicAdd(&nodes->d_inflow[n2], qNew);
    } else {
        atomicAdd(&nodes->d_inflow[n1], -qNew);
        atomicAdd(&nodes->d_outflow[n2], -qNew);
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

    // Copy static geometry once per allocation
    if (!g_conduitKernelCtx.xsectsInitialized) {
        copyXsectsToGpu(xsects);
        g_conduitKernelCtx.xsectsInitialized = 1;
    }

    // Refresh link/conduit state only on first Picard iteration each time step
    if (steps == 0) {
        copyLinksToGpu(links);
        copyConduitsToGpu(conduits);
    }

    copyNodesToGpu(nodes);

    GPU_LinkData* d_links = g_conduitKernelCtx.d_links;
    GPU_ConduitData* d_conduits = g_conduitKernelCtx.d_conduits;
    GPU_XsectData* d_xsects = g_conduitKernelCtx.d_xsects;
    GPU_NodeData* d_nodes = g_conduitKernelCtx.d_nodes;

    CUDA_CHECK(cudaMemcpy(d_links, links, sizeof(GPU_LinkData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conduits, conduits, sizeof(GPU_ConduitData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xsects, xsects, sizeof(GPU_XsectData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes, sizeof(GPU_NodeData), cudaMemcpyHostToDevice));

    // Determine launch configuration
    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = GRID_SIZE(links->count, blockSize);

    cudaStream_t stream = gpu_getStream();

    // Record start time
    cudaEventRecord(g_conduitKernelStartEvent, stream);

    // Get unit conversion factors and routing model
    double ucfVolume = UCF(VOLUME);
    double ucfLength = UCF(LENGTH);
    double ucfFlow = UCF(FLOW);
    int routeModel = RouteModel;

    static int printedPumpCount = 0;
    if (!printedPumpCount) {
        printf("GPU pump count = %d\n", g_gpuPumps.count);
        if (g_gpuPumps.h_linkIndex != NULL) {
            for (int dbgIdx = 0; dbgIdx < g_gpuPumps.count; ++dbgIdx) {
                int linkIdx = g_gpuPumps.h_linkIndex[dbgIdx];
                printf("GPU pump map: k=%d link=%d\n", dbgIdx, linkIdx);
            }
        }
        printedPumpCount = 1;
    }

    // Allocate and copy device memory for non-conduit data structures (one-time only)
    static GPU_PumpData* d_gpuPumps = NULL;
    static GPU_OrificeData* d_gpuOrifices = NULL;
    static GPU_WeirData* d_gpuWeirs = NULL;
    static GPU_OutletData* d_gpuOutlets = NULL;
    static GPU_CurveData* d_gpuCurves = NULL;
    static GPU_CurvePoints* d_gpuCurvePoints = NULL;
    static int nonConduitStructuresInitialized = 0;

    // Pump preliminary flow buffers (for two-pass processing)
    static double* d_prelimPumpFlows = NULL;
    static double* d_prelimPumpDqdh = NULL;
    static char* d_prelimPumpFlowClass = NULL;
    static int prelimPumpBuffersAllocated = 0;

    // One-time allocation and transfer of data structures (device pointers)
    if (!nonConduitStructuresInitialized) {
        if (g_gpuPumps.count > 0) {
            CUDA_CHECK(cudaMalloc(&d_gpuPumps, sizeof(GPU_PumpData)));
            CUDA_CHECK(cudaMemcpy(d_gpuPumps, &g_gpuPumps, sizeof(GPU_PumpData), cudaMemcpyHostToDevice));

            // Allocate preliminary pump flow buffers
            if (!prelimPumpBuffersAllocated) {
                CUDA_CHECK(cudaMalloc(&d_prelimPumpFlows, g_gpuPumps.count * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_prelimPumpDqdh, g_gpuPumps.count * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_prelimPumpFlowClass, g_gpuPumps.count * sizeof(char)));
                prelimPumpBuffersAllocated = 1;
            }
        }
        if (g_gpuOrifices.count > 0) {
            CUDA_CHECK(cudaMalloc(&d_gpuOrifices, sizeof(GPU_OrificeData)));
            CUDA_CHECK(cudaMemcpy(d_gpuOrifices, &g_gpuOrifices, sizeof(GPU_OrificeData), cudaMemcpyHostToDevice));
        }
        if (g_gpuWeirs.count > 0) {
            CUDA_CHECK(cudaMalloc(&d_gpuWeirs, sizeof(GPU_WeirData)));
            CUDA_CHECK(cudaMemcpy(d_gpuWeirs, &g_gpuWeirs, sizeof(GPU_WeirData), cudaMemcpyHostToDevice));
        }
        if (g_gpuOutlets.count > 0) {
            CUDA_CHECK(cudaMalloc(&d_gpuOutlets, sizeof(GPU_OutletData)));
            CUDA_CHECK(cudaMemcpy(d_gpuOutlets, &g_gpuOutlets, sizeof(GPU_OutletData), cudaMemcpyHostToDevice));
        }
        if (g_gpuCurves.count > 0) {
            CUDA_CHECK(cudaMalloc(&d_gpuCurves, sizeof(GPU_CurveData)));
            CUDA_CHECK(cudaMalloc(&d_gpuCurvePoints, sizeof(GPU_CurvePoints)));
            CUDA_CHECK(cudaMemcpy(d_gpuCurves, &g_gpuCurves, sizeof(GPU_CurveData), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_gpuCurvePoints, &g_gpuCurvePoints, sizeof(GPU_CurvePoints), cudaMemcpyHostToDevice));
        }
        nonConduitStructuresInitialized = 1;
    }

    // Launch conduit flow kernel
    kernel_findConduitFlows<<<gridSize, blockSize, 0, stream>>>(
        d_links, d_conduits, d_xsects, d_nodes,
        dt, steps, omega,
        surchargeMethod, crownCutoff, inertDamping);

    CUDA_CHECK_LAST_ERROR();

    // RE-ENABLE weirs/orifices/outlets to test if they work without pumps
    // Launch orifice kernel if orifices exist and GPU data is initialized
    if (Nlinks[ORIFICE] > 0 && g_gpuOrifices.count > 0) {
        int orificeGridSize = GRID_SIZE(g_gpuOrifices.count, blockSize);
        kernel_findOrificeFlows<<<orificeGridSize, blockSize, 0, stream>>>(
            d_links, d_gpuOrifices, d_xsects, d_nodes, omega, routeModel);
        CUDA_CHECK_LAST_ERROR();
    }

    // Launch weir kernel if weirs exist and GPU data is initialized
    if (Nlinks[WEIR] > 0 && g_gpuWeirs.count > 0) {
        int weirGridSize = GRID_SIZE(g_gpuWeirs.count, blockSize);
        kernel_findWeirFlows<<<weirGridSize, blockSize, 0, stream>>>(
            d_links, d_gpuWeirs, d_xsects, d_nodes, omega, routeModel);
        CUDA_CHECK_LAST_ERROR();
    }

    // Launch outlet kernel if outlets exist and GPU data is initialized
    if (Nlinks[OUTLET] > 0 && g_gpuOutlets.count > 0) {
        int outletGridSize = GRID_SIZE(g_gpuOutlets.count, blockSize);
        kernel_findOutletFlows<<<outletGridSize, blockSize, 0, stream>>>(
            d_links, d_gpuOutlets, d_nodes, d_gpuCurves, d_gpuCurvePoints,
            ucfLength, ucfFlow, omega, routeModel);
        CUDA_CHECK_LAST_ERROR();
    }

    // TEMPORARY: Disable pumps to test Session18
    // FULLY SEQUENTIAL PUMP PROCESSING (Exact CPU Logic)
    // Process each pump: compute flow → getModPumpFlow → update nodes → next pump
    // This matches CPU behavior EXACTLY where each pump sees previous pumps' effects
    if (false && Nlinks[PUMP] > 0 && g_gpuPumps.count > 0) {
        kernel_processPumpsSequentially<<<1, 1, 0, stream>>>(
            d_links, d_gpuPumps, d_nodes, d_gpuCurves, d_gpuCurvePoints,
            dt, routeModel,
            ucfVolume, ucfLength, ucfFlow);
        CUDA_CHECK_LAST_ERROR();
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Record end time
    cudaEventRecord(g_conduitKernelStopEvent, stream);
    cudaEventSynchronize(g_conduitKernelStopEvent);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, g_conduitKernelStartEvent, g_conduitKernelStopEvent);
    gpu_profiler_addKernelTime((double)elapsedMs);

    gpu_transferLinkIterationResultsFromDevice(links, links->count);
    gpu_transferNodeIterationStateFromDevice(nodes, nodes->count);

    if (!g_gpuConfig.forceCuda && elapsedMs > g_gpuConfig.maxKernelTimeMs)
    {
        g_gpuConfig.useCuda = 0;
        char msg[196];
        snprintf(msg, sizeof(msg),
                 "\n ... CUDA acceleration disabled after this step (kernel %.1f ms exceeded %.1f ms threshold)",
                 elapsedMs, g_gpuConfig.maxKernelTimeMs);
        writecon(msg);
    }

    if (!g_gpuConfig.forceCuda && g_gpuPerfStats.kernelTimeMs > g_gpuConfig.maxTotalKernelTimeMs)
    {
        g_gpuConfig.useCuda = 0;
        char msg[196];
        snprintf(msg, sizeof(msg),
                 "\n ... CUDA acceleration disabled after this step (cumulative kernel time %.1f ms exceeded %.1f ms)",
                 g_gpuPerfStats.kernelTimeMs, g_gpuConfig.maxTotalKernelTimeMs);
        writecon(msg);
    }

    // Copy results back to CPU (only dynamic data the CPU needs immediately)
    copyLinkIterStateFromGpu(links);
    copyNodesFromGpu(nodes);  // Node inflow/outflow updated by kernel
    g_conduitKernelCtx.resultsDirty = 1;

    // --- DEBUG: Log link flow and node flow discrepancies for first few iterations
    static int debugIterationCount = 0;
    static FILE* debugFile = NULL;
    static FILE* nodeDebugFile = NULL;

    if (debugIterationCount < 200) {  // Log first 100 iterations
        if (debugFile == NULL) {
            debugFile = fopen("/tmp/gpu_link_flow_debug.txt", "w");
            if (debugFile) {
                fprintf(debugFile, "# GPU Link Flow Debug Log\n");
                fprintf(debugFile, "# Format: iteration, linkIndex, linkID, linkType, cpuFlow, gpuFlow, absDiff, relDiff\n");
            }
        }

        if (nodeDebugFile == NULL) {
            nodeDebugFile = fopen("/tmp/gpu_node_flow_debug.txt", "w");
            if (nodeDebugFile) {
                fprintf(nodeDebugFile, "# GPU Node Flow Debug Log\n");
                fprintf(nodeDebugFile, "# Format: iteration, nodeIndex, nodeID, cpuInflow, gpuInflow, cpuOutflow, gpuOutflow\n");
            }
        }

        if (debugFile && debugIterationCount % 10 == 0) {  // Every 10th iteration
            // Compare flows for all links (focus on non-conduits)
            for (int i = 0; i < Nobjects[LINK]; i++) {
                int linkType = Link[i].type;

                // Log non-conduits and any conduits with differences
                if (linkType != CONDUIT || debugIterationCount < 10) {
                    double cpuFlow = Link[i].newFlow;
                    double gpuFlow = links->h_newFlow[i];
                    double absDiff = fabs(cpuFlow - gpuFlow);
                    double relDiff = cpuFlow != 0.0 ? (absDiff / fabs(cpuFlow)) : 0.0;

                    // Log if non-conduit OR significant difference
                    if (linkType != CONDUIT || absDiff > 0.001) {
                        const char* typeStr =
                            linkType == CONDUIT ? "CONDUIT" :
                            linkType == PUMP ? "PUMP" :
                            linkType == ORIFICE ? "ORIFICE" :
                            linkType == WEIR ? "WEIR" :
                            linkType == OUTLET ? "OUTLET" : "UNKNOWN";

                        fprintf(debugFile, "%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f\n",
                                debugIterationCount, i, Link[i].ID, typeStr,
                                cpuFlow, gpuFlow, absDiff, relDiff);
                    }
                }
            }
            fflush(debugFile);

            // Log node flows for pump-connected nodes
            if (nodeDebugFile) {
                for (int i = 0; i < Nobjects[NODE]; i++) {
                    // Only log nodes that have non-zero CPU or GPU flows
                    double cpuInflow = Node[i].inflow;
                    double cpuOutflow = Node[i].outflow;
                    double gpuInflow = nodes->h_inflow[i];
                    double gpuOutflow = nodes->h_outflow[i];

                    if (cpuInflow > 0.001 || cpuOutflow > 0.001 ||
                        gpuInflow > 0.001 || gpuOutflow > 0.001) {
                        fprintf(nodeDebugFile, "%d,%d,%s,%.6f,%.6f,%.6f,%.6f\n",
                                debugIterationCount, i, Node[i].ID,
                                cpuInflow, gpuInflow, cpuOutflow, gpuOutflow);
                    }
                }
                fflush(nodeDebugFile);
            }

        }

        if (nodeDebugFile && debugIterationCount < 200) {
            int hostPumpCounts[64];
            GPUPumpDebugEntry hostPumpEntries[64 * GPU_PUMP_DEBUG_MAX];
            CUDA_CHECK(cudaMemcpyFromSymbol(hostPumpCounts, g_gpuPumpDebugCounter, sizeof(hostPumpCounts)));
            CUDA_CHECK(cudaMemcpyFromSymbol(hostPumpEntries, g_gpuPumpDebugEntries, sizeof(hostPumpEntries)));
            int targetPumps[] = {15, 20, 22, 24, 25, 26, 32, 35, 38, 41, 43, 45};
            int targetCount = sizeof(targetPumps) / sizeof(targetPumps[0]);
            for (int tp = 0; tp < targetCount; ++tp) {
                int pk = targetPumps[tp];
                fprintf(nodeDebugFile, "pump_debug_count iter=%d pump=%d count=%d\n", debugIterationCount, pk, hostPumpCounts[pk]);
                int limit = hostPumpCounts[pk];
                if (limit > GPU_PUMP_DEBUG_MAX) limit = GPU_PUMP_DEBUG_MAX;
                for (int e = 0; e < limit; ++e) {
                    GPUPumpDebugEntry* entry = &hostPumpEntries[pk * GPU_PUMP_DEBUG_MAX + e];
                    fprintf(nodeDebugFile,
                            "pump_debug iter=%d pump=%d node=%d downNode=%d call=%d dt=%.6f qCurve=%.6f qFinal=%.6f oldVol=%.6f inUp=%.6f->%.6f outUp=%.6f->%.6f inDown=%.6f->%.6f\n",
                            debugIterationCount,
                            entry->pumpIndex,
                            entry->upNodeIndex,
                            entry->downNodeIndex,
                            entry->callIndex,
                            entry->dt,
                            entry->qCurve,
                            entry->qFinal,
                            entry->oldVolume,
                            entry->inflowBeforeUp,
                            entry->inflowAfterUp,
                            entry->outflowBeforeUp,
                            entry->outflowAfterUp,
                            entry->inflowBeforeDown,
                            entry->inflowAfterDown);
                }
            }
            fflush(nodeDebugFile);
        }
        debugIterationCount++;

        if (debugIterationCount == 200) {
            if (debugFile) {
                fprintf(debugFile, "# Debug logging complete (100 iterations)\n");
                fclose(debugFile);
                debugFile = NULL;
                printf("\n  ... GPU flow debug log written to /tmp/gpu_link_flow_debug.txt\n");
            }
            if (nodeDebugFile) {
                fprintf(nodeDebugFile, "# Debug logging complete (100 iterations)\n");
                fclose(nodeDebugFile);
                nodeDebugFile = NULL;
                printf("  ... GPU node flow debug log written to /tmp/gpu_node_flow_debug.txt\n");
            }
        }
    }
    // --- END DEBUG

    return 0;
}

} // extern "C"

extern "C" void gpu_flushConduitResults(void)
{
    if (!g_gpuConfig.useCuda) return;
    if (!g_conduitKernelCtx.resultsDirty) return;

    gpu_transferLinkDynamicFromDevice(&g_conduitKernelCtx.links, g_conduitKernelCtx.links.count);
    gpu_transferConduitDynamicFromDevice(&g_conduitKernelCtx.conduits, g_conduitKernelCtx.conduits.count);
    copyLinksFromGpu(&g_conduitKernelCtx.links);
    copyConduitsFromGpu(&g_conduitKernelCtx.conduits);
    g_conduitKernelCtx.resultsDirty = 0;
}
