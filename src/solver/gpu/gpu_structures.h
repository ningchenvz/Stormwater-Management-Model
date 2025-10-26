//-----------------------------------------------------------------------------
//   gpu_structures.h
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   GPU-friendly Structure of Arrays (SoA) data structures for dynamic wave
//   routing. Converts SWMM's Array of Structures (AoS) to SoA for optimal
//   GPU memory access patterns (coalesced memory access).
//
//   Design Notes:
//   - CPU SWMM uses Array of Structures (AoS): Node[i].depth, Node[i].volume
//   - GPU benefits from Structure of Arrays (SoA): depth[i], volume[i]
//   - SoA enables coalesced memory access when adjacent GPU threads access
//     adjacent array elements
//   - Unified memory (DGX Spark) allows seamless CPU/GPU access
//   - Discrete GPUs require explicit cudaMemcpy between host/device
//
//-----------------------------------------------------------------------------

#ifndef GPU_STRUCTURES_H
#define GPU_STRUCTURES_H

#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// GPU Node Data (Structure of Arrays)
//-----------------------------------------------------------------------------
// Contains dynamic state variables for nodes needed in dynwave routing
//
typedef struct {
    int     count;              // Number of nodes

    // Static properties (read-only during routing)
    int*    type;               // Node type (JUNCTION, OUTFALL, STORAGE, DIVIDER)
    double* invertElev;         // Invert elevation (ft)
    double* fullDepth;          // Distance from invert to surface (ft)
    double* surDepth;           // Added depth under surcharge (ft)
    double* pondedArea;         // Area filled by ponded water (ft2)
    double* crownElev;          // Top of highest flowing conduit (ft)

    // Dynamic state (updated each iteration)
    double* oldDepth;           // Previous water depth (ft)
    double* newDepth;           // Current water depth (ft)
    double* oldVolume;          // Previous volume (ft3)
    double* newVolume;          // Current volume (ft3)
    double* fullVolume;         // Max storage available (ft3)
    double* oldNetInflow;       // Previous net inflow (cfs)
    double* inflow;             // Total inflow (cfs)
    double* outflow;            // Total outflow (cfs)
    double* overflow;           // Overflow rate (cfs)
    int*    degree;             // Number of outflow links

    // Extended node data (from TXnode in dynwave.c)
    char*   converged;          // TRUE if iterations done for this node
    double* newSurfArea;        // Current surface area (ft2)
    double* oldSurfArea;        // Previous surface area (ft2)
    double* sumdqdh;            // Sum of dqdh from adjoining links
    double* dYdT;               // Change in depth w.r.t. time (ft/sec)

} GPU_NodeData;

//-----------------------------------------------------------------------------
// GPU Link Data (Structure of Arrays)
//-----------------------------------------------------------------------------
// Contains dynamic state variables for links needed in dynwave routing
//
typedef struct {
    int     count;              // Number of links

    // Topology (connectivity)
    int*    type;               // Link type (CONDUIT, PUMP, ORIFICE, WEIR, OUTLET)
    int*    subIndex;           // Index to sub-category (Conduit, Pump, etc.)
    int*    node1;              // Upstream node index
    int*    node2;              // Downstream node index

    // Static properties
    double* offset1;            // Height above start node invert (ft)
    double* offset2;            // Height above end node invert (ft)
    double* qFull;              // Flow when full (cfs)

    // Dynamic state
    double* oldFlow;            // Previous flow rate (cfs)
    double* newFlow;            // Current flow rate (cfs)
    double* oldDepth;           // Previous flow depth (ft)
    double* newDepth;           // Current flow depth (ft)
    double* oldVolume;          // Previous flow volume (ft3)
    double* newVolume;          // Current flow volume (ft3)
    double* surfArea1;          // Upstream surface area (ft2)
    double* surfArea2;          // Downstream surface area (ft2)
    double* dqdh;               // Change in flow w.r.t. head (ft2/sec)
    double* froude;             // Froude number

    // Control flags
    char*   bypassed;           // Bypass dynwave calculation flag
    signed char* direction;     // Flow direction flag
    signed char* flowClass;     // Flow classification (DRY, SUBCRITICAL, etc.)

} GPU_LinkData;

//-----------------------------------------------------------------------------
// GPU Conduit Data (Structure of Arrays)
//-----------------------------------------------------------------------------
// Contains conduit-specific data for dynamic wave routing
// Note: Only true conduits (not pumps, orifices, etc.) have this data
//
typedef struct {
    int     count;              // Number of conduits

    // Static properties
    double* length;             // Conduit length (ft)
    double* modLength;          // Modified length (ft)
    double* roughness;          // Manning's n
    double* roughFactor;        // Roughness factor for DW routing
    double* slope;              // Conduit slope
    char*   barrels;            // Number of barrels

    // Dynamic wave parameters
    double* beta;               // Discharge factor
    double* qMax;               // Maximum flow (cfs)
    double* a1;                 // Upstream cross-sectional area (ft2)
    double* a2;                 // Downstream cross-sectional area (ft2)
    double* q1;                 // Upstream flow per barrel (cfs)
    double* q2;                 // Downstream flow per barrel (cfs)
    double* q1Old;              // Previous q1 (cfs)
    double* q2Old;              // Previous q2 (cfs)

    // Status flags
    char*   capacityLimited;    // Capacity limited flag
    char*   superCritical;      // Super-critical flow flag

} GPU_ConduitData;

//-----------------------------------------------------------------------------
// Cross-Section Data (needed for flow calculations)
//-----------------------------------------------------------------------------
// Simplified version of TXsect for GPU use
//
typedef struct {
    int     count;              // Number of cross-sections

    int*    type;               // Cross-section shape type
    double* aFull;              // Full cross-sectional area (ft2)
    double* rFull;              // Hydraulic radius when full (ft)
    double* wMax;               // Max width (ft)
    double* yFull;              // Full depth (ft)

    // Geometry parameters for different cross-section shapes
    double* geom1;              // Geometry parameter 1 (diameter, width, etc.)
    double* geom2;              // Geometry parameter 2 (height, side slope, etc.)
    double* geom3;              // Geometry parameter 3 (additional geometry)

} GPU_XsectData;

//-----------------------------------------------------------------------------
// Function Declarations
//-----------------------------------------------------------------------------

// Memory allocation functions
#ifdef BUILD_GPU
    int  gpu_allocateNodeData(GPU_NodeData* data, int nodeCount);
    int  gpu_allocateLinkData(GPU_LinkData* data, int linkCount);
    int  gpu_allocateConduitData(GPU_ConduitData* data, int conduitCount);
    int  gpu_allocateXsectData(GPU_XsectData* data, int xsectCount);

    void gpu_freeNodeData(GPU_NodeData* data);
    void gpu_freeLinkData(GPU_LinkData* data);
    void gpu_freeConduitData(GPU_ConduitData* data);
    void gpu_freeXsectData(GPU_XsectData* data);

    // Data transfer functions (CPU AoS -> GPU SoA)
    int  gpu_transferNodeData(GPU_NodeData* gpuData, void* cpuNodes, int count);
    int  gpu_transferLinkData(GPU_LinkData* gpuData, void* cpuLinks, int count);
    int  gpu_transferConduitData(GPU_ConduitData* gpuData, void* cpuConduits, int count);

    // Data transfer functions (GPU SoA -> CPU AoS)
    int  gpu_retrieveNodeData(void* cpuNodes, GPU_NodeData* gpuData, int count);
    int  gpu_retrieveLinkData(void* cpuLinks, GPU_LinkData* gpuData, int count);
    int  gpu_retrieveConduitData(void* cpuConduits, GPU_ConduitData* gpuData, int count);
#else
    // Stub implementations when GPU not available
    static inline int gpu_allocateNodeData(GPU_NodeData* data, int nodeCount) { return 0; }
    static inline int gpu_allocateLinkData(GPU_LinkData* data, int linkCount) { return 0; }
    static inline int gpu_allocateConduitData(GPU_ConduitData* data, int conduitCount) { return 0; }
    static inline int gpu_allocateXsectData(GPU_XsectData* data, int xsectCount) { return 0; }

    static inline void gpu_freeNodeData(GPU_NodeData* data) {}
    static inline void gpu_freeLinkData(GPU_LinkData* data) {}
    static inline void gpu_freeConduitData(GPU_ConduitData* data) {}
    static inline void gpu_freeXsectData(GPU_XsectData* data) {}

    static inline int gpu_transferNodeData(GPU_NodeData* gpuData, void* cpuNodes, int count) { return 0; }
    static inline int gpu_transferLinkData(GPU_LinkData* gpuData, void* cpuLinks, int count) { return 0; }
    static inline int gpu_transferConduitData(GPU_ConduitData* gpuData, void* cpuConduits, int count) { return 0; }

    static inline int gpu_retrieveNodeData(void* cpuNodes, GPU_NodeData* gpuData, int count) { return 0; }
    static inline int gpu_retrieveLinkData(void* cpuLinks, GPU_LinkData* gpuData, int count) { return 0; }
    static inline int gpu_retrieveConduitData(void* cpuConduits, GPU_ConduitData* gpuData, int count) { return 0; }
#endif

#ifdef __cplusplus
}
#endif

#endif // GPU_STRUCTURES_H
