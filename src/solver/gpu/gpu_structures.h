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
//   - EXPLICIT MEMORY: Separate host (h_) and device (d_) pointers for performance
//   - Pinned host memory (cudaMallocHost) for fast PCIe transfers
//   - Device memory (cudaMalloc) stays on GPU between timesteps
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
// EXPLICIT MEMORY LAYOUT:
//   h_* = Host (CPU) pointers - allocated with cudaMallocHost (pinned)
//   d_* = Device (GPU) pointers - allocated with cudaMalloc
//
typedef struct {
    int     count;              // Number of nodes

    // -------------------------------------------------------------------------
    // HOST POINTERS (CPU-side, pinned memory for fast transfers)
    // -------------------------------------------------------------------------

    // Static properties (read-only during routing)
    int*    h_type;             // Node type (JUNCTION, OUTFALL, STORAGE, DIVIDER)
    double* h_invertElev;       // Invert elevation (ft)
    double* h_fullDepth;        // Distance from invert to surface (ft)
    double* h_surDepth;         // Added depth under surcharge (ft)
    double* h_pondedArea;       // Area filled by ponded water (ft2)
    double* h_crownElev;        // Top of highest flowing conduit (ft)
    double* h_fullVolume;       // Max storage available (ft3)
    double* h_storageA0;        // storage parameter a0 (ft2)
    double* h_storageA1;        // storage parameter a1
    double* h_storageA2;        // storage parameter a2
    int*    h_storageShape;     // storage shape enum
    int*    h_storageCurve;     // storage area curve index

    // Dynamic state (updated each iteration)
    double* h_oldDepth;         // Previous water depth (ft)
    double* h_newDepth;         // Current water depth (ft)
    double* h_oldVolume;        // Previous volume (ft3)
    double* h_newVolume;        // Current volume (ft3)
    double* h_oldNetInflow;     // Previous net inflow (cfs)
    double* h_inflow;           // Total inflow (cfs)
    double* h_outflow;          // Total outflow (cfs)
    double* h_overflow;         // Overflow rate (cfs)
    int*    h_degree;           // Number of outflow links

    // Extended node data (from TXnode in dynwave.c)
    char*   h_converged;        // TRUE if iterations done for this node
    double* h_newSurfArea;      // Current surface area (ft2)
    double* h_oldSurfArea;      // Previous surface area (ft2)
    double* h_sumdqdh;          // Sum of dqdh from adjoining links
    double* h_dYdT;             // Change in depth w.r.t. time (ft/sec)

    // -------------------------------------------------------------------------
    // DEVICE POINTERS (GPU-side, device memory)
    // -------------------------------------------------------------------------

    // Static properties (read-only during routing)
    int*    d_type;             // Node type (JUNCTION, OUTFALL, STORAGE, DIVIDER)
    double* d_invertElev;       // Invert elevation (ft)
    double* d_fullDepth;        // Distance from invert to surface (ft)
    double* d_surDepth;         // Added depth under surcharge (ft)
    double* d_pondedArea;       // Area filled by ponded water (ft2)
    double* d_crownElev;        // Top of highest flowing conduit (ft)
    double* d_fullVolume;       // Max storage available (ft3)
    double* d_storageA0;
    double* d_storageA1;
    double* d_storageA2;
    int*    d_storageShape;
    int*    d_storageCurve;

    // Dynamic state (updated each iteration)
    double* d_oldDepth;         // Previous water depth (ft)
    double* d_newDepth;         // Current water depth (ft)
    double* d_oldVolume;        // Previous volume (ft3)
    double* d_newVolume;        // Current volume (ft3)
    double* d_oldNetInflow;     // Previous net inflow (cfs)
    double* d_inflow;           // Total inflow (cfs)
    double* d_outflow;          // Total outflow (cfs)
    double* d_overflow;         // Overflow rate (cfs)
    int*    d_degree;           // Number of outflow links

    // Extended node data (from TXnode in dynwave.c)
    char*   d_converged;        // TRUE if iterations done for this node
    double* d_newSurfArea;      // Current surface area (ft2)
    double* d_oldSurfArea;      // Previous surface area (ft2)
    double* d_sumdqdh;          // Sum of dqdh from adjoining links
    double* d_dYdT;             // Change in depth w.r.t. time (ft/sec)

} GPU_NodeData;

//-----------------------------------------------------------------------------
// GPU Link Data (Structure of Arrays)
//-----------------------------------------------------------------------------
// Contains dynamic state variables for links needed in dynwave routing
//
// EXPLICIT MEMORY LAYOUT:
//   h_* = Host (CPU) pointers - allocated with cudaMallocHost (pinned)
//   d_* = Device (GPU) pointers - allocated with cudaMalloc
//
typedef struct {
    int     count;              // Number of links

    // -------------------------------------------------------------------------
    // HOST POINTERS (CPU-side, pinned memory for fast transfers)
    // -------------------------------------------------------------------------

    // Topology (connectivity)
    int*    h_type;             // Link type (CONDUIT, PUMP, ORIFICE, WEIR, OUTLET)
    int*    h_subIndex;         // Index to sub-category (Conduit, Pump, etc.)
    int*    h_node1;            // Upstream node index
    int*    h_node2;            // Downstream node index

    // Static properties
    double* h_offset1;          // Height above start node invert (ft)
    double* h_offset2;          // Height above end node invert (ft)
    double* h_qFull;            // Flow when full (cfs)

    // Dynamic state
    double* h_oldFlow;          // Previous flow rate (cfs)
    double* h_newFlow;          // Current flow rate (cfs)
    double* h_oldDepth;         // Previous flow depth (ft)
    double* h_newDepth;         // Current flow depth (ft)
    double* h_oldVolume;        // Previous flow volume (ft3)
    double* h_newVolume;        // Current flow volume (ft3)
    double* h_surfArea1;        // Upstream surface area (ft2)
    double* h_surfArea2;        // Downstream surface area (ft2)
    double* h_dqdh;             // Change in flow w.r.t. head (ft2/sec)
    double* h_froude;           // Froude number
    double* h_setting;          // Current control setting (0-1)
    double* h_targetSetting;    // Target control setting (0-1)

    // Control flags
    char*   h_bypassed;         // Bypass dynwave calculation flag
    signed char* h_direction;   // Flow direction flag
    signed char* h_flowClass;   // Flow classification (DRY, SUBCRITICAL, etc.)
    char*   h_hasFlapGate;      // TRUE if link has flap gate

    // -------------------------------------------------------------------------
    // DEVICE POINTERS (GPU-side, device memory)
    // -------------------------------------------------------------------------

    // Topology (connectivity)
    int*    d_type;             // Link type (CONDUIT, PUMP, ORIFICE, WEIR, OUTLET)
    int*    d_subIndex;         // Index to sub-category (Conduit, Pump, etc.)
    int*    d_node1;            // Upstream node index
    int*    d_node2;            // Downstream node index

    // Static properties
    double* d_offset1;          // Height above start node invert (ft)
    double* d_offset2;          // Height above end node invert (ft)
    double* d_qFull;            // Flow when full (cfs)

    // Dynamic state
    double* d_oldFlow;          // Previous flow rate (cfs)
    double* d_newFlow;          // Current flow rate (cfs)
    double* d_oldDepth;         // Previous flow depth (ft)
    double* d_newDepth;         // Current flow depth (ft)
    double* d_oldVolume;        // Previous flow volume (ft3)
    double* d_newVolume;        // Current flow volume (ft3)
    double* d_surfArea1;        // Upstream surface area (ft2)
    double* d_surfArea2;        // Downstream surface area (ft2)
    double* d_dqdh;             // Change in flow w.r.t. head (ft2/sec)
    double* d_froude;           // Froude number
    double* d_setting;          // Current control setting (0-1)
    double* d_targetSetting;    // Target control setting (0-1)

    // Control flags
    char*   d_bypassed;         // Bypass dynwave calculation flag
    signed char* d_direction;   // Flow direction flag
    signed char* d_flowClass;   // Flow classification (DRY, SUBCRITICAL, etc.)
    char*   d_hasFlapGate;      // TRUE if link has flap gate

} GPU_LinkData;

//-----------------------------------------------------------------------------
// GPU Conduit Data (Structure of Arrays)
//-----------------------------------------------------------------------------
// Contains conduit-specific data for dynamic wave routing
// Note: Only true conduits (not pumps, orifices, etc.) have this data
//
// EXPLICIT MEMORY LAYOUT:
//   h_* = Host (CPU) pointers - allocated with cudaMallocHost (pinned)
//   d_* = Device (GPU) pointers - allocated with cudaMalloc
//
typedef struct {
    int     count;              // Number of conduits

    // -------------------------------------------------------------------------
    // HOST POINTERS (CPU-side, pinned memory for fast transfers)
    // -------------------------------------------------------------------------

    // Static properties
    double* h_length;           // Conduit length (ft)
    double* h_modLength;        // Modified length (ft)
    double* h_roughness;        // Manning's n
    double* h_roughFactor;      // Roughness factor for DW routing
    double* h_slope;            // Conduit slope
    char*   h_barrels;          // Number of barrels

    // Dynamic wave parameters
    double* h_beta;             // Discharge factor
    double* h_qMax;             // Maximum flow (cfs)
    double* h_a1;               // Upstream cross-sectional area (ft2)
    double* h_a2;               // Downstream cross-sectional area (ft2)
    double* h_q1;               // Upstream flow per barrel (cfs)
    double* h_q2;               // Downstream flow per barrel (cfs)
    double* h_q1Old;            // Previous q1 (cfs)
    double* h_q2Old;            // Previous q2 (cfs)

    // Status flags
    char*   h_capacityLimited;  // Capacity limited flag
    char*   h_superCritical;    // Super-critical flow flag

    // -------------------------------------------------------------------------
    // DEVICE POINTERS (GPU-side, device memory)
    // -------------------------------------------------------------------------

    // Static properties
    double* d_length;           // Conduit length (ft)
    double* d_modLength;        // Modified length (ft)
    double* d_roughness;        // Manning's n
    double* d_roughFactor;      // Roughness factor for DW routing
    double* d_slope;            // Conduit slope
    char*   d_barrels;          // Number of barrels

    // Dynamic wave parameters
    double* d_beta;             // Discharge factor
    double* d_qMax;             // Maximum flow (cfs)
    double* d_a1;               // Upstream cross-sectional area (ft2)
    double* d_a2;               // Downstream cross-sectional area (ft2)
    double* d_q1;               // Upstream flow per barrel (cfs)
    double* d_q2;               // Downstream flow per barrel (cfs)
    double* d_q1Old;            // Previous q1 (cfs)
    double* d_q2Old;            // Previous q2 (cfs)

    // Status flags
    char*   d_capacityLimited;  // Capacity limited flag
    char*   d_superCritical;    // Super-critical flow flag

} GPU_ConduitData;

//-----------------------------------------------------------------------------
// GPU Pump Data (Structure of Arrays)
//-----------------------------------------------------------------------------
typedef struct {
    int     count;

    // Host (pinned memory)
    int*    h_linkIndex;     // Maps pump index k → link index j
    int*    h_type;
    int*    h_pumpCurve;
    double* h_initSetting;
    double* h_yOn;
    double* h_yOff;
    double* h_xMin;
    double* h_xMax;

    // Device pointers
    int*    d_linkIndex;     // Maps pump index k → link index j
    int*    d_type;
    int*    d_pumpCurve;
    double* d_initSetting;
    double* d_yOn;
    double* d_yOff;
    double* d_xMin;
    double* d_xMax;
} GPU_PumpData;

//-----------------------------------------------------------------------------
// GPU Orifice Data (Structure of Arrays)
//-----------------------------------------------------------------------------
typedef struct {
    int     count;

    // Host (pinned memory)
    int*    h_linkIndex;     // Maps orifice index k → link index j
    int*    h_type;
    int*    h_shape;
    double* h_cDisch;
    double* h_orate;
    double* h_cOrif;
    double* h_hCrit;
    double* h_cWeir;
    double* h_length;
    double* h_surfArea;

    // Device pointers
    int*    d_linkIndex;     // Maps orifice index k → link index j
    int*    d_type;
    int*    d_shape;
    double* d_cDisch;
    double* d_orate;
    double* d_cOrif;
    double* d_hCrit;
    double* d_cWeir;
    double* d_length;
    double* d_surfArea;
} GPU_OrificeData;

//-----------------------------------------------------------------------------
// GPU Weir Data (Structure of Arrays)
//-----------------------------------------------------------------------------
typedef struct {
    int     count;

    // Host (pinned memory)
    int*    h_linkIndex;     // Maps weir index k → link index j
    int*    h_type;
    double* h_cDisch1;
    double* h_cDisch2;
    double* h_endCon;
    int*    h_canSurcharge;
    double* h_roadWidth;
    int*    h_roadSurface;
    int*    h_cdCurve;
    double* h_cSurcharge;
    double* h_length;
    double* h_slope;
    double* h_surfArea;

    // Device pointers
    int*    d_linkIndex;     // Maps weir index k → link index j
    int*    d_type;
    double* d_cDisch1;
    double* d_cDisch2;
    double* d_endCon;
    int*    d_canSurcharge;
    double* d_roadWidth;
    int*    d_roadSurface;
    int*    d_cdCurve;
    double* d_cSurcharge;
    double* d_length;
    double* d_slope;
    double* d_surfArea;
} GPU_WeirData;

//-----------------------------------------------------------------------------
// GPU Outlet Data (Structure of Arrays)
//-----------------------------------------------------------------------------
typedef struct {
    int     count;

    // Host (pinned memory)
    int*    h_linkIndex;     // Maps outlet index k → link index j
    double* h_qCoeff;
    double* h_qExpon;
    int*    h_qCurve;
    int*    h_curveType;

    // Device pointers
    int*    d_linkIndex;     // Maps outlet index k → link index j
    double* d_qCoeff;
    double* d_qExpon;
    int*    d_qCurve;
    int*    d_curveType;
} GPU_OutletData;

//-----------------------------------------------------------------------------
// GPU Curve Data (Structure of Arrays for TTable/TCurve)
//-----------------------------------------------------------------------------
// CPU TTable uses linked list (not GPU-friendly)
// GPU_CurveData uses arrays for coalesced memory access
//
// EXPLICIT MEMORY LAYOUT:
//   h_* = Host (CPU) pointers - allocated with cudaMallocHost (pinned)
//   d_* = Device (GPU) pointers - allocated with cudaMalloc
//
typedef struct {
    int     count;              // Number of curves

    // Host arrays (pinned memory)
    int*    h_curveType;        // Curve type (PUMP1_CURVE, PUMP2_CURVE, etc.)
    int*    h_dataStart;        // Start index in global x/y arrays for this curve
    int*    h_dataCount;        // Number of (x,y) points in this curve
    double* h_dxMin;            // Minimum x spacing for this curve

    // Device arrays
    int*    d_curveType;
    int*    d_dataStart;
    int*    d_dataCount;
    double* d_dxMin;

} GPU_CurveData;

//-----------------------------------------------------------------------------
// GPU Curve Points (global arrays of all x/y pairs)
//-----------------------------------------------------------------------------
// All curve data points stored in contiguous arrays
//
typedef struct {
    int     totalPoints;        // Total number of (x,y) points across all curves

    // Host arrays
    double* h_xValues;          // All x-values in sequence
    double* h_yValues;          // All y-values in sequence

    // Device arrays
    double* d_xValues;
    double* d_yValues;

} GPU_CurvePoints;

//-----------------------------------------------------------------------------
// Global GPU Data Structures (defined in gpu_manager.cu)
//-----------------------------------------------------------------------------
#ifdef BUILD_GPU
extern GPU_NodeData     g_gpuNodes;
extern GPU_LinkData     g_gpuLinks;
extern GPU_ConduitData  g_gpuConduits;
extern GPU_PumpData     g_gpuPumps;
extern GPU_OrificeData  g_gpuOrifices;
extern GPU_WeirData     g_gpuWeirs;
extern GPU_OutletData   g_gpuOutlets;
extern GPU_CurveData    g_gpuCurves;
extern GPU_CurvePoints  g_gpuCurvePoints;
extern GPU_CurveData*   g_gpuDeviceCurves;
extern GPU_CurvePoints* g_gpuDeviceCurvePoints;
#endif

//-----------------------------------------------------------------------------
// Cross-Section Data (needed for flow calculations)
//-----------------------------------------------------------------------------
// Simplified version of TXsect for GPU use
//
// EXPLICIT MEMORY LAYOUT:
//   h_* = Host (CPU) pointers - allocated with cudaMallocHost (pinned)
//   d_* = Device (GPU) pointers - allocated with cudaMalloc
//
typedef struct {
    int     count;              // Number of cross-sections

    // -------------------------------------------------------------------------
    // HOST POINTERS (CPU-side, pinned memory for fast transfers)
    // -------------------------------------------------------------------------

    int*    h_type;             // Cross-section shape type
    double* h_aFull;            // Full cross-sectional area (ft2)
    double* h_rFull;            // Hydraulic radius when full (ft)
    double* h_wMax;             // Max width (ft)
    double* h_yFull;            // Full depth (ft)

    // Geometry parameters for different cross-section shapes
    double* h_geom1;            // Geometry parameter 1 (diameter, width, etc.)
    double* h_geom2;            // Geometry parameter 2 (height, side slope, etc.)
    double* h_geom3;            // Geometry parameter 3 (additional geometry)

    // -------------------------------------------------------------------------
    // DEVICE POINTERS (GPU-side, device memory)
    // -------------------------------------------------------------------------

    int*    d_type;             // Cross-section shape type
    double* d_aFull;            // Full cross-sectional area (ft2)
    double* d_rFull;            // Hydraulic radius when full (ft)
    double* d_wMax;             // Max width (ft)
    double* d_yFull;            // Full depth (ft)

    // Geometry parameters for different cross-section shapes
    double* d_geom1;            // Geometry parameter 1 (diameter, width, etc.)
    double* d_geom2;            // Geometry parameter 2 (height, side slope, etc.)
    double* d_geom3;            // Geometry parameter 3 (additional geometry)

} GPU_XsectData;

//-----------------------------------------------------------------------------
// Function Declarations
//-----------------------------------------------------------------------------

// Memory allocation functions
#ifdef BUILD_GPU
    int  gpu_allocateNodeData(GPU_NodeData* data, int nodeCount);
    int  gpu_allocateLinkData(GPU_LinkData* data, int linkCount);
    int  gpu_allocateConduitData(GPU_ConduitData* data, int conduitCount);
    int  gpu_allocatePumpData(GPU_PumpData* data, int pumpCount);
    int  gpu_allocateOrificeData(GPU_OrificeData* data, int orificeCount);
    int  gpu_allocateWeirData(GPU_WeirData* data, int weirCount);
    int  gpu_allocateOutletData(GPU_OutletData* data, int outletCount);
    int  gpu_allocateXsectData(GPU_XsectData* data, int xsectCount);
    int  gpu_allocateCurveData(GPU_CurveData* data, int curveCount);
    int  gpu_allocateCurvePoints(GPU_CurvePoints* data, int totalPoints);

    void gpu_freeNodeData(GPU_NodeData* data);
    void gpu_freeLinkData(GPU_LinkData* data);
    void gpu_freeConduitData(GPU_ConduitData* data);
    void gpu_freePumpData(GPU_PumpData* data);
    void gpu_freeOrificeData(GPU_OrificeData* data);
    void gpu_freeWeirData(GPU_WeirData* data);
    void gpu_freeOutletData(GPU_OutletData* data);
    void gpu_freeXsectData(GPU_XsectData* data);
    void gpu_freeCurveData(GPU_CurveData* data);
    void gpu_freeCurvePoints(GPU_CurvePoints* data);

    // Data transfer functions (CPU AoS -> GPU SoA)
    int  gpu_transferNodeData(GPU_NodeData* gpuData, void* cpuNodes, int count);
    int  gpu_transferLinkData(GPU_LinkData* gpuData, void* cpuLinks, int count);
    int  gpu_transferConduitData(GPU_ConduitData* gpuData, void* cpuConduits, int count);

    // Data transfer functions (GPU SoA -> CPU AoS)
    int  gpu_retrieveNodeData(void* cpuNodes, GPU_NodeData* gpuData, int count);
    int  gpu_retrieveLinkData(void* cpuLinks, GPU_LinkData* gpuData, int count);
    int  gpu_retrieveConduitData(void* cpuConduits, GPU_ConduitData* gpuData, int count);

    // -------------------------------------------------------------------------
    // EXPLICIT MEMORY TRANSFER HELPERS (Host ↔ Device)
    // -------------------------------------------------------------------------
    // These functions provide fine-grained control over data transfers between
    // pinned host memory (h_*) and device memory (d_*).
    //
    // Benefits:
    //   - Transfer only changed data (not entire arrays)
    //   - Asynchronous transfers with CUDA streams
    //   - Explicit control over transfer timing
    //   - Eliminate page fault overhead from unified memory
    //
    // Usage pattern:
    //   1. gpu_transferStaticToDevice()  - Once at simulation start
    //   2. gpu_transferDynamicToDevice() - Before each timestep
    //   3. [GPU kernels execute]
    //   4. gpu_transferDynamicFromDevice() - After convergence
    //   5. gpu_transferFinalFromDevice() - Once at simulation end

    // Transfer static (read-only) data to GPU (once at start)
    int  gpu_transferNodeStaticToDevice(GPU_NodeData* data, int count);
    int  gpu_transferLinkStaticToDevice(GPU_LinkData* data, int count);
    int  gpu_transferConduitStaticToDevice(GPU_ConduitData* data, int count);
    int  gpu_transferPumpStaticToDevice(GPU_PumpData* data, int count);
    int  gpu_transferOrificeStaticToDevice(GPU_OrificeData* data, int count);
    int  gpu_transferWeirStaticToDevice(GPU_WeirData* data, int count);
    int  gpu_transferOutletStaticToDevice(GPU_OutletData* data, int count);
    int  gpu_transferXsectStaticToDevice(GPU_XsectData* data, int count);
    int  gpu_transferCurveDataToDevice(GPU_CurveData* data);
    int  gpu_transferCurvePointsToDevice(GPU_CurvePoints* data);

    // Transfer dynamic data to GPU (before each timestep)
    int  gpu_transferNodeDynamicToDevice(GPU_NodeData* data, int count);
    int  gpu_transferLinkDynamicToDevice(GPU_LinkData* data, int count);
    int  gpu_transferConduitDynamicToDevice(GPU_ConduitData* data, int count);
    int  gpu_transferPumpDynamicToDevice(GPU_PumpData* data, int count);
    int  gpu_transferOrificeDynamicToDevice(GPU_OrificeData* data, int count);
    int  gpu_transferWeirDynamicToDevice(GPU_WeirData* data, int count);
    int  gpu_transferOutletDynamicToDevice(GPU_OutletData* data, int count);

    // Transfer results from GPU (after convergence)
    int  gpu_transferNodeDynamicFromDevice(GPU_NodeData* data, int count);
    int  gpu_transferLinkDynamicFromDevice(GPU_LinkData* data, int count);
    int  gpu_transferConduitDynamicFromDevice(GPU_ConduitData* data, int count);
    int  gpu_transferPumpDynamicFromDevice(GPU_PumpData* data, int count);
    int  gpu_transferOrificeDynamicFromDevice(GPU_OrificeData* data, int count);
    int  gpu_transferWeirDynamicFromDevice(GPU_WeirData* data, int count);
    int  gpu_transferOutletDynamicFromDevice(GPU_OutletData* data, int count);
    int  gpu_transferLinkIterationResultsFromDevice(GPU_LinkData* data, int count);
    int  gpu_transferNodeIterationStateFromDevice(GPU_NodeData* data, int count);

    // Async transfer variants (non-blocking, use with CUDA streams)
    int  gpu_transferNodeDynamicToDeviceAsync(GPU_NodeData* data, int count, void* stream);
    int  gpu_transferLinkDynamicToDeviceAsync(GPU_LinkData* data, int count, void* stream);
    int  gpu_transferNodeDynamicFromDeviceAsync(GPU_NodeData* data, int count, void* stream);
    int  gpu_transferLinkDynamicFromDeviceAsync(GPU_LinkData* data, int count, void* stream);

    // Partial range transfers (only transfer touched elements)
    int  gpu_transferNodeRangeToDevice(GPU_NodeData* data, int startIdx, int endIdx);
    int  gpu_transferLinkRangeToDevice(GPU_LinkData* data, int startIdx, int endIdx);
    int  gpu_transferNodeRangeFromDevice(GPU_NodeData* data, int startIdx, int endIdx);
    int  gpu_transferLinkRangeFromDevice(GPU_LinkData* data, int startIdx, int endIdx);

    // -------------------------------------------------------------------------
    // PERSISTENT PICARD ITERATION (GPU-side convergence checking)
    // -------------------------------------------------------------------------
    // Runs complete Picard iteration loop on GPU with minimal CPU-GPU transfers.
    // Only transfers convergence counter (4 bytes) per iteration instead of
    // full node arrays (N * sizeof(char)).
    //
    // Performance benefits:
    //   - Eliminates per-iteration kernel launch overhead (~50-100μs)
    //   - Reduces CPU-GPU transfers from O(N*iters) to O(1)
    //   - Expected speedup: 2-5x for typical Picard iterations
    int  gpu_runPersistentPicardIteration(
        double dt,
        int allowPonding,
        int surchargeMethod,
        double minSurfArea,
        double omega,
        double headTol,
        int maxIterations,
        int* outIterations,
        int* outConverged);

#else
    // Stub implementations when GPU not available
    static inline int gpu_allocateNodeData(GPU_NodeData* data, int nodeCount) { return 0; }
    static inline int gpu_allocateLinkData(GPU_LinkData* data, int linkCount) { return 0; }
    static inline int gpu_allocateConduitData(GPU_ConduitData* data, int conduitCount) { return 0; }
    static inline int gpu_allocatePumpData(GPU_PumpData* data, int pumpCount) { return 0; }
    static inline int gpu_allocateOrificeData(GPU_OrificeData* data, int orificeCount) { return 0; }
    static inline int gpu_allocateWeirData(GPU_WeirData* data, int weirCount) { return 0; }
    static inline int gpu_allocateOutletData(GPU_OutletData* data, int outletCount) { return 0; }
    static inline int gpu_allocateXsectData(GPU_XsectData* data, int xsectCount) { return 0; }
    static inline int gpu_allocateCurveData(GPU_CurveData* data, int curveCount) { return 0; }
    static inline int gpu_allocateCurvePoints(GPU_CurvePoints* data, int totalPoints) { return 0; }

    static inline void gpu_freeNodeData(GPU_NodeData* data) {}
    static inline void gpu_freeLinkData(GPU_LinkData* data) {}
    static inline void gpu_freeConduitData(GPU_ConduitData* data) {}
    static inline void gpu_freePumpData(GPU_PumpData* data) {}
    static inline void gpu_freeOrificeData(GPU_OrificeData* data) {}
    static inline void gpu_freeWeirData(GPU_WeirData* data) {}
    static inline void gpu_freeOutletData(GPU_OutletData* data) {}
    static inline void gpu_freeXsectData(GPU_XsectData* data) {}
    static inline void gpu_freeCurveData(GPU_CurveData* data) {}
    static inline void gpu_freeCurvePoints(GPU_CurvePoints* data) {}

    static inline int gpu_transferNodeData(GPU_NodeData* gpuData, void* cpuNodes, int count) { return 0; }
    static inline int gpu_transferLinkData(GPU_LinkData* gpuData, void* cpuLinks, int count) { return 0; }
    static inline int gpu_transferConduitData(GPU_ConduitData* gpuData, void* cpuConduits, int count) { return 0; }

    static inline int gpu_retrieveNodeData(void* cpuNodes, GPU_NodeData* gpuData, int count) { return 0; }
    static inline int gpu_retrieveLinkData(void* cpuLinks, GPU_LinkData* gpuData, int count) { return 0; }
    static inline int gpu_retrieveConduitData(void* cpuConduits, GPU_ConduitData* gpuData, int count) { return 0; }

    // Stub implementations for explicit memory transfer helpers
    static inline int gpu_transferNodeStaticToDevice(GPU_NodeData* data, int count) { return 0; }
    static inline int gpu_transferLinkStaticToDevice(GPU_LinkData* data, int count) { return 0; }
    static inline int gpu_transferConduitStaticToDevice(GPU_ConduitData* data, int count) { return 0; }
    static inline int gpu_transferPumpStaticToDevice(GPU_PumpData* data, int count) { return 0; }
    static inline int gpu_transferOrificeStaticToDevice(GPU_OrificeData* data, int count) { return 0; }
    static inline int gpu_transferWeirStaticToDevice(GPU_WeirData* data, int count) { return 0; }
    static inline int gpu_transferOutletStaticToDevice(GPU_OutletData* data, int count) { return 0; }
    static inline int gpu_transferXsectStaticToDevice(GPU_XsectData* data, int count) { return 0; }
    static inline int gpu_transferCurveDataToDevice(GPU_CurveData* data) { return 0; }
    static inline int gpu_transferCurvePointsToDevice(GPU_CurvePoints* data) { return 0; }

    static inline int gpu_transferNodeDynamicToDevice(GPU_NodeData* data, int count) { return 0; }
    static inline int gpu_transferLinkDynamicToDevice(GPU_LinkData* data, int count) { return 0; }
    static inline int gpu_transferConduitDynamicToDevice(GPU_ConduitData* data, int count) { return 0; }
    static inline int gpu_transferPumpDynamicToDevice(GPU_PumpData* data, int count) { return 0; }
    static inline int gpu_transferOrificeDynamicToDevice(GPU_OrificeData* data, int count) { return 0; }
    static inline int gpu_transferWeirDynamicToDevice(GPU_WeirData* data, int count) { return 0; }
    static inline int gpu_transferOutletDynamicToDevice(GPU_OutletData* data, int count) { return 0; }

    static inline int gpu_transferNodeDynamicFromDevice(GPU_NodeData* data, int count) { return 0; }
    static inline int gpu_transferLinkDynamicFromDevice(GPU_LinkData* data, int count) { return 0; }
    static inline int gpu_transferConduitDynamicFromDevice(GPU_ConduitData* data, int count) { return 0; }
    static inline int gpu_transferPumpDynamicFromDevice(GPU_PumpData* data, int count) { return 0; }
    static inline int gpu_transferOrificeDynamicFromDevice(GPU_OrificeData* data, int count) { return 0; }
    static inline int gpu_transferWeirDynamicFromDevice(GPU_WeirData* data, int count) { return 0; }
    static inline int gpu_transferOutletDynamicFromDevice(GPU_OutletData* data, int count) { return 0; }
    static inline int gpu_transferLinkIterationResultsFromDevice(GPU_LinkData* data, int count) { return 0; }
    static inline int gpu_transferNodeIterationStateFromDevice(GPU_NodeData* data, int count) { return 0; }

    static inline int gpu_transferNodeDynamicToDeviceAsync(GPU_NodeData* data, int count, void* stream) { return 0; }
    static inline int gpu_transferLinkDynamicToDeviceAsync(GPU_LinkData* data, int count, void* stream) { return 0; }
    static inline int gpu_transferNodeDynamicFromDeviceAsync(GPU_NodeData* data, int count, void* stream) { return 0; }
    static inline int gpu_transferLinkDynamicFromDeviceAsync(GPU_LinkData* data, int count, void* stream) { return 0; }

    static inline int gpu_transferNodeRangeToDevice(GPU_NodeData* data, int startIdx, int endIdx) { return 0; }
    static inline int gpu_transferLinkRangeToDevice(GPU_LinkData* data, int startIdx, int endIdx) { return 0; }
    static inline int gpu_transferNodeRangeFromDevice(GPU_NodeData* data, int startIdx, int endIdx) { return 0; }
    static inline int gpu_transferLinkRangeFromDevice(GPU_LinkData* data, int startIdx, int endIdx) { return 0; }
#endif

#ifdef __cplusplus
}
#endif

#endif // GPU_STRUCTURES_H
