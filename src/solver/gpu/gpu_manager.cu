//-----------------------------------------------------------------------------
//   gpu_manager.cu
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   GPU memory management and device initialization.
//
//-----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_config.h"
#include "gpu_structures.h"

// Global GPU configuration
GPUConfig g_gpuConfig = {0};
GPUPerfStats g_gpuPerfStats = {0};

// Global GPU data structures for dynamic wave routing
GPU_NodeData g_gpuNodes = {0};
GPU_LinkData g_gpuLinks = {0};
GPU_ConduitData g_gpuConduits = {0};
GPU_XsectData g_gpuXsects = {0};

// Global GPU data structures for non-conduit links
GPU_PumpData g_gpuPumps = {0};
GPU_OrificeData g_gpuOrifices = {0};
GPU_WeirData g_gpuWeirs = {0};
GPU_OutletData g_gpuOutlets = {0};

// Global GPU curve data (for pumps and outlets)
GPU_CurveData g_gpuCurves = {0};
GPU_CurvePoints g_gpuCurvePoints = {0};
GPU_CurveData* g_gpuDeviceCurves = NULL;
GPU_CurvePoints* g_gpuDeviceCurvePoints = NULL;

static cudaStream_t g_gpuStream = 0;

//=============================================================================

int gpu_initialize(void)
//
//  Purpose: Initializes GPU device and queries capabilities
//  Returns: 0 if successful, CUDA error code otherwise
//
{
    int deviceCount = 0;
    cudaError_t err;

    // Check for CUDA devices
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to get device count: %s\n",
                cudaGetErrorString(err));
        g_gpuConfig.available = 0;
        return err;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        g_gpuConfig.available = 0;
        return cudaErrorNoDevice;
    }

    g_gpuConfig.deviceCount = deviceCount;
    g_gpuConfig.activeDevice = 0;  // Use first device by default

    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, g_gpuConfig.activeDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to get device properties: %s\n",
                cudaGetErrorString(err));
        g_gpuConfig.available = 0;
        return err;
    }

    // Store device capabilities
    g_gpuConfig.computeCapability = prop.major * 10 + prop.minor;
    g_gpuConfig.totalMemory = prop.totalGlobalMem;
    g_gpuConfig.multiProcessorCount = prop.multiProcessorCount;
    g_gpuConfig.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    g_gpuConfig.warpSize = prop.warpSize;

    // Check unified memory support
    g_gpuConfig.unifiedMemory = prop.managedMemory;
    g_gpuConfig.concurrentManagedAccess = prop.concurrentManagedAccess;

    // Get available memory
    size_t freeMem, totalMem;
    err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Warning: Failed to get memory info: %s\n",
                cudaGetErrorString(err));
        g_gpuConfig.availableMemory = 0;
    } else {
        g_gpuConfig.availableMemory = freeMem;
    }

    // Set device
    err = cudaSetDevice(g_gpuConfig.activeDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to set device: %s\n",
                cudaGetErrorString(err));
        g_gpuConfig.available = 0;
        return err;
    }

    // Mark as available and enabled by default
    g_gpuConfig.available = 1;
    g_gpuConfig.enabled = 1;
    g_gpuConfig.useCuda = 0;
    g_gpuConfig.forceCuda = 0;
    g_gpuConfig.maxKernelTimeMs = GPU_MAX_KERNEL_MS_DEFAULT;
    g_gpuConfig.maxTotalKernelTimeMs = GPU_MAX_TOTAL_KERNEL_MS_DEFAULT;

    if (!g_gpuStream)
    {
        cudaStreamCreateWithFlags(&g_gpuStream, cudaStreamNonBlocking);
    }

    // Set default thresholds
    g_gpuConfig.minLinksForGPU = GPU_MIN_LINKS_DEFAULT;
    g_gpuConfig.minNodesForGPU = GPU_MIN_NODES_DEFAULT;
    g_gpuConfig.minConduitsForGPU = GPU_MIN_CONDUITS_DEFAULT;
    gpu_profiler_reset();

    printf("\n... GPU Initialized: %s\n", prop.name);
    printf("... Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("... Total Memory: %.2f GB\n",
           g_gpuConfig.totalMemory / (1024.0 * 1024.0 * 1024.0));
    printf("... Available Memory: %.2f GB\n",
           g_gpuConfig.availableMemory / (1024.0 * 1024.0 * 1024.0));
    printf("... Unified Memory: %s\n",
           g_gpuConfig.unifiedMemory ? "Supported" : "Not Supported");
    if (g_gpuConfig.unifiedMemory && g_gpuConfig.concurrentManagedAccess) {
        printf("... Concurrent Managed Access: Enabled (optimal for CPU/GPU interleaving)\n");
    }
    printf("... GPU failover threshold: %.1f ms per iteration (%.1f ms total)\n",
           g_gpuConfig.maxKernelTimeMs, g_gpuConfig.maxTotalKernelTimeMs);

    return cudaSuccess;
}

cudaStream_t gpu_getStream(void)
{
    return g_gpuStream;
}

//=============================================================================

void gpu_cleanup(void)
//
//  Purpose: Cleanup GPU resources
//
{
    if (g_gpuConfig.available) {
        if (g_gpuStream) {
            cudaStreamDestroy(g_gpuStream);
            g_gpuStream = 0;
        }
        cudaDeviceReset();
        g_gpuConfig.available = 0;
        g_gpuConfig.enabled = 0;
        g_gpuConfig.useCuda = 0;
    }
    g_gpuDeviceCurves = NULL;
    g_gpuDeviceCurvePoints = NULL;
    gpu_profiler_reset();
}

//=============================================================================

int gpu_isAvailable(void)
//
//  Purpose: Check if GPU is available
//  Returns: 1 if available, 0 otherwise
//
{
    return g_gpuConfig.available;
}

//=============================================================================

int gpu_isEnabled(void)
//
//  Purpose: Check if GPU acceleration is enabled
//  Returns: 1 if enabled, 0 otherwise
//
{
    return g_gpuConfig.available && g_gpuConfig.enabled;
}

//=============================================================================

void gpu_setEnabled(int enabled)
//
//  Purpose: Enable or disable GPU acceleration
//
{
    g_gpuConfig.enabled = enabled;
}

//=============================================================================

void gpu_printInfo(void)
//
//  Purpose: Print detailed GPU information
//
{
    if (!g_gpuConfig.available) {
        printf("\n  GPU Acceleration: Not Available\n");
        return;
    }

    printf("\n  GPU Acceleration: %s\n",
           g_gpuConfig.enabled ? "Enabled" : "Disabled");
    printf("  GPU Device ID: %d\n", g_gpuConfig.activeDevice);
    printf("  Compute Capability: %d.%d\n",
           g_gpuConfig.computeCapability / 10,
           g_gpuConfig.computeCapability % 10);
    printf("  Multiprocessors: %d\n", g_gpuConfig.multiProcessorCount);
    printf("  Max Threads/Block: %d\n", g_gpuConfig.maxThreadsPerBlock);
    printf("  Warp Size: %d\n", g_gpuConfig.warpSize);
    printf("  Unified Memory: %s\n", g_gpuConfig.unifiedMemory ? "Yes" : "No");
    printf("  Concurrent Access: %s\n", g_gpuConfig.concurrentManagedAccess ? "Yes" : "No");
    printf("  GPU Min Links: %d\n", g_gpuConfig.minLinksForGPU);
    printf("  GPU Min Nodes: %d\n", g_gpuConfig.minNodesForGPU);
    printf("  GPU Min Conduits: %d\n", g_gpuConfig.minConduitsForGPU);
}

//=============================================================================

void gpu_profiler_reset(void)
{
    memset(&g_gpuPerfStats, 0, sizeof(GPUPerfStats));
}

//=============================================================================

void gpu_profiler_addKernelTime(double ms)
{
    g_gpuPerfStats.kernelTimeMs += ms;
    g_gpuPerfStats.kernelLaunches++;
}

//=============================================================================

void gpu_profiler_addMemcpyTime(double ms)
{
    g_gpuPerfStats.memcpyTimeMs += ms;
    g_gpuPerfStats.memcpyCalls++;
}

//=============================================================================

void gpu_profiler_printSummary(void)
//
//  Purpose: Print simple CUDA timing summary for diagnostics
//
{
    if (!g_gpuConfig.useCuda) return;

    printf("\n  CUDA Performance Summary:\n");
    printf("    Kernel launches : %6d  (%.3f ms total)\n",
           g_gpuPerfStats.kernelLaunches, g_gpuPerfStats.kernelTimeMs);
    printf("    Memcpy/prefetch : %6d  (%.3f ms total)\n",
           g_gpuPerfStats.memcpyCalls, g_gpuPerfStats.memcpyTimeMs);
}

//=============================================================================
// Non-Conduit Link GPU Initialization
//=============================================================================

extern "C" {
#include "headers.h"
}

extern "C" int gpu_initializeNonConduitData(void)
//
//  Purpose: Initialize GPU data structures for non-conduit links
//           (pumps, orifices, weirs, outlets) and curves
//  Returns: 0 if successful, error code otherwise
//
//  This function:
//   1. Converts CPU Curve[] (linked lists) to GPU SoA format
//   2. Allocates and initializes pump/orifice/weir/outlet data
//   3. Transfers all data to GPU
//
{
    printf("\n  Initializing non-conduit GPU data...\n");

    // Step 1: Convert Curves from linked list to SoA format
    int numCurves = Nobjects[CURVE];
    if (numCurves > 0) {
        // Count total curve points
        int totalPoints = 0;
        for (int i = 0; i < numCurves; i++) {
            TTableEntry* entry = Curve[i].firstEntry;
            while (entry) {
                totalPoints++;
                entry = entry->next;
            }
        }

        printf("    Converting %d curves (%d total points) to GPU format...\n",
               numCurves, totalPoints);

        // Allocate GPU curve structures
        if (gpu_allocateCurveData(&g_gpuCurves, numCurves) != 0) {
            fprintf(stderr, "    ERROR: Failed to allocate GPU curve data\n");
            return -1;
        }

        if (gpu_allocateCurvePoints(&g_gpuCurvePoints, totalPoints) != 0) {
            fprintf(stderr, "    ERROR: Failed to allocate GPU curve points\n");
            return -1;
        }

        // Fill curve metadata and flatten points
        int pointOffset = 0;
        for (int i = 0; i < numCurves; i++) {
            g_gpuCurves.h_curveType[i] = Curve[i].curveType;
            g_gpuCurves.h_dxMin[i] = Curve[i].dxMin;
            g_gpuCurves.h_dataStart[i] = pointOffset;

            // Count and copy points for this curve
            int curvePointCount = 0;
            TTableEntry* entry = Curve[i].firstEntry;
            while (entry) {
                g_gpuCurvePoints.h_xValues[pointOffset + curvePointCount] = entry->x;
                g_gpuCurvePoints.h_yValues[pointOffset + curvePointCount] = entry->y;
                curvePointCount++;
                entry = entry->next;
            }

            g_gpuCurves.h_dataCount[i] = curvePointCount;
            pointOffset += curvePointCount;
        }

        // Transfer curves to GPU
        if (gpu_transferCurveDataToDevice(&g_gpuCurves) != 0 ||
            gpu_transferCurvePointsToDevice(&g_gpuCurvePoints) != 0) {
            fprintf(stderr, "    ERROR: Failed to transfer curve data to GPU\n");
            return -1;
        }

        // Allocate (or refresh) device-side curve metadata structs
        if (g_gpuCurves.count > 0) {
            if (g_gpuDeviceCurves == NULL) {
                CUDA_CHECK(cudaMalloc((void**)&g_gpuDeviceCurves, sizeof(GPU_CurveData)));
            }
            CUDA_CHECK(cudaMemcpy(g_gpuDeviceCurves, &g_gpuCurves,
                                  sizeof(GPU_CurveData), cudaMemcpyHostToDevice));
        }

        if (g_gpuCurvePoints.totalPoints > 0) {
            if (g_gpuDeviceCurvePoints == NULL) {
                CUDA_CHECK(cudaMalloc((void**)&g_gpuDeviceCurvePoints, sizeof(GPU_CurvePoints)));
            }
            CUDA_CHECK(cudaMemcpy(g_gpuDeviceCurvePoints, &g_gpuCurvePoints,
                                  sizeof(GPU_CurvePoints), cudaMemcpyHostToDevice));
        }

        printf("    Curves transferred to GPU successfully\n");

        // ===== DEBUG: Validate curve data integrity =====
        printf("\n    === CURVE DATA VALIDATION ===\n");
        printf("    Total curves: %d, Total points: %d\n", numCurves, totalPoints);

        // Verify pointOffset matches totalPoints
        if (pointOffset != totalPoints) {
            printf("    ERROR: pointOffset (%d) != totalPoints (%d)\n", pointOffset, totalPoints);
        } else {
            printf("    ✓ Point offset verification passed\n");
        }

        // Dump first few curves and storage curves specifically
        for (int i = 0; i < numCurves; i++) {
            // Only print storage curves (curveType == STORAGE_CURVE) or first 3 curves
            // Storage curve type should be 2 based on enums.h
            int start = g_gpuCurves.h_dataStart[i];
            int count = g_gpuCurves.h_dataCount[i];

            // Print all storage curves (typically curve indices used by storage nodes)
            // We'll print curves 0-15 to catch most storage curves
            if (i <= 15 || count == 0) {
                printf("    Curve %d: type=%d start=%d count=%d dxMin=%.6f\n",
                       i, g_gpuCurves.h_curveType[i], start, count, g_gpuCurves.h_dxMin[i]);

                if (count > 0 && count <= 20) {
                    printf("      CPU linked list: ");
                    TTableEntry* entry = Curve[i].firstEntry;
                    int idx = 0;
                    while (entry && idx < 10) {
                        printf("(%.3f,%.1f) ", entry->x, entry->y);
                        entry = entry->next;
                        idx++;
                    }
                    if (count > 10) printf("... (%d more)", count - 10);
                    printf("\n");

                    printf("      GPU SoA arrays:  ");
                    for (int j = 0; j < count && j < 10; j++) {
                        printf("(%.3f,%.1f) ",
                               g_gpuCurvePoints.h_xValues[start + j],
                               g_gpuCurvePoints.h_yValues[start + j]);
                    }
                    if (count > 10) printf("... (%d more)", count - 10);
                    printf("\n");

                    // Verify CPU vs GPU match
                    entry = Curve[i].firstEntry;
                    bool mismatch = false;
                    for (int j = 0; j < count; j++) {
                        if (!entry) {
                            printf("      ERROR: CPU curve has fewer points than expected\n");
                            mismatch = true;
                            break;
                        }
                        if (fabs(entry->x - g_gpuCurvePoints.h_xValues[start + j]) > 1e-9 ||
                            fabs(entry->y - g_gpuCurvePoints.h_yValues[start + j]) > 1e-9) {
                            printf("      ERROR: Point %d mismatch: CPU(%.6f,%.6f) GPU(%.6f,%.6f)\n",
                                   j, entry->x, entry->y,
                                   g_gpuCurvePoints.h_xValues[start + j],
                                   g_gpuCurvePoints.h_yValues[start + j]);
                            mismatch = true;
                            break;
                        }
                        entry = entry->next;
                    }
                    if (!mismatch) {
                        printf("      ✓ CPU/GPU data match verified\n");
                    }
                }
                printf("\n");
            }
        }
        printf("    === END CURVE VALIDATION ===\n\n");
    }

    // Step 2: Initialize Pump Data
    int numPumps = Nlinks[PUMP];
    if (numPumps > 0) {
        printf("    Initializing %d pumps...\n", numPumps);

        if (gpu_allocatePumpData(&g_gpuPumps, numPumps) != 0) {
            fprintf(stderr, "    ERROR: Failed to allocate GPU pump data\n");
            return -1;
        }

        // Find link indices for each pump (reverse map: pump k → link j)
        for (int j = 0; j < Nobjects[LINK]; j++) {
            if (Link[j].type == PUMP) {
                int k = Link[j].subIndex;
                g_gpuPumps.h_linkIndex[k] = j;
            }
        }

        for (int i = 0; i < numPumps; i++) {
            g_gpuPumps.h_type[i] = Pump[i].type;
            g_gpuPumps.h_pumpCurve[i] = Pump[i].pumpCurve;
            g_gpuPumps.h_initSetting[i] = Pump[i].initSetting;
            g_gpuPumps.h_yOn[i] = Pump[i].yOn;
            g_gpuPumps.h_yOff[i] = Pump[i].yOff;
            g_gpuPumps.h_xMin[i] = Pump[i].xMin;
            g_gpuPumps.h_xMax[i] = Pump[i].xMax;
        }

        if (gpu_transferPumpStaticToDevice(&g_gpuPumps, numPumps) != 0) {
            fprintf(stderr, "    ERROR: Failed to transfer pump data to GPU\n");
            return -1;
        }
    }

    // Step 3: Initialize Orifice Data
    int numOrifices = Nlinks[ORIFICE];
    if (numOrifices > 0) {
        printf("    Initializing %d orifices...\n", numOrifices);

        if (gpu_allocateOrificeData(&g_gpuOrifices, numOrifices) != 0) {
            fprintf(stderr, "    ERROR: Failed to allocate GPU orifice data\n");
            return -1;
        }

        // Find link indices for each orifice (reverse map: orifice k → link j)
        for (int j = 0; j < Nobjects[LINK]; j++) {
            if (Link[j].type == ORIFICE) {
                int k = Link[j].subIndex;
                g_gpuOrifices.h_linkIndex[k] = j;
            }
        }

        for (int i = 0; i < numOrifices; i++) {
            g_gpuOrifices.h_type[i] = Orifice[i].type;
            g_gpuOrifices.h_shape[i] = Orifice[i].shape;
            g_gpuOrifices.h_cDisch[i] = Orifice[i].cDisch;
            g_gpuOrifices.h_orate[i] = Orifice[i].orate;
            g_gpuOrifices.h_cOrif[i] = Orifice[i].cOrif;
            g_gpuOrifices.h_hCrit[i] = Orifice[i].hCrit;
            g_gpuOrifices.h_cWeir[i] = Orifice[i].cWeir;
            g_gpuOrifices.h_length[i] = Orifice[i].length;
            g_gpuOrifices.h_surfArea[i] = Orifice[i].surfArea;
        }

        if (gpu_transferOrificeStaticToDevice(&g_gpuOrifices, numOrifices) != 0) {
            fprintf(stderr, "    ERROR: Failed to transfer orifice data to GPU\n");
            return -1;
        }
    }

    // Step 4: Initialize Weir Data
    int numWeirs = Nlinks[WEIR];
    if (numWeirs > 0) {
        printf("    Initializing %d weirs...\n", numWeirs);

        if (gpu_allocateWeirData(&g_gpuWeirs, numWeirs) != 0) {
            fprintf(stderr, "    ERROR: Failed to allocate GPU weir data\n");
            return -1;
        }

        // Find link indices for each weir (reverse map: weir k → link j)
        for (int j = 0; j < Nobjects[LINK]; j++) {
            if (Link[j].type == WEIR) {
                int k = Link[j].subIndex;
                g_gpuWeirs.h_linkIndex[k] = j;
            }
        }

        for (int i = 0; i < numWeirs; i++) {
            g_gpuWeirs.h_type[i] = Weir[i].type;
            g_gpuWeirs.h_cDisch1[i] = Weir[i].cDisch1;
            g_gpuWeirs.h_cDisch2[i] = Weir[i].cDisch2;
            g_gpuWeirs.h_endCon[i] = Weir[i].endCon;
            g_gpuWeirs.h_canSurcharge[i] = Weir[i].canSurcharge;
            g_gpuWeirs.h_roadWidth[i] = Weir[i].roadWidth;
            g_gpuWeirs.h_roadSurface[i] = Weir[i].roadSurface;
            g_gpuWeirs.h_cdCurve[i] = Weir[i].cdCurve;
            g_gpuWeirs.h_cSurcharge[i] = Weir[i].cSurcharge;
            g_gpuWeirs.h_length[i] = Weir[i].length;
            g_gpuWeirs.h_slope[i] = Weir[i].slope;
            g_gpuWeirs.h_surfArea[i] = Weir[i].surfArea;
        }

        if (gpu_transferWeirStaticToDevice(&g_gpuWeirs, numWeirs) != 0) {
            fprintf(stderr, "    ERROR: Failed to transfer weir data to GPU\n");
            return -1;
        }
    }

    // Step 5: Initialize Outlet Data
    int numOutlets = Nlinks[OUTLET];
    if (numOutlets > 0) {
        printf("    Initializing %d outlets...\n", numOutlets);

        if (gpu_allocateOutletData(&g_gpuOutlets, numOutlets) != 0) {
            fprintf(stderr, "    ERROR: Failed to allocate GPU outlet data\n");
            return -1;
        }

        // Find link indices for each outlet (reverse map: outlet k → link j)
        for (int j = 0; j < Nobjects[LINK]; j++) {
            if (Link[j].type == OUTLET) {
                int k = Link[j].subIndex;
                g_gpuOutlets.h_linkIndex[k] = j;
            }
        }

        for (int i = 0; i < numOutlets; i++) {
            g_gpuOutlets.h_qCoeff[i] = Outlet[i].qCoeff;
            g_gpuOutlets.h_qExpon[i] = Outlet[i].qExpon;
            g_gpuOutlets.h_qCurve[i] = Outlet[i].qCurve;
            g_gpuOutlets.h_curveType[i] = Outlet[i].curveType;
        }

        if (gpu_transferOutletStaticToDevice(&g_gpuOutlets, numOutlets) != 0) {
            fprintf(stderr, "    ERROR: Failed to transfer outlet data to GPU\n");
            return -1;
        }
    }

    printf("  Non-conduit GPU initialization complete!\n");
    printf("    Pumps: %d, Orifices: %d, Weirs: %d, Outlets: %d\n",
           numPumps, numOrifices, numWeirs, numOutlets);

    return 0;
}

extern "C" void gpu_freeNonConduitData(void)
//
//  Purpose: Free GPU memory for non-conduit link data
//
{
    gpu_freeCurveData(&g_gpuCurves);
    gpu_freeCurvePoints(&g_gpuCurvePoints);
    gpu_freePumpData(&g_gpuPumps);
    gpu_freeOrificeData(&g_gpuOrifices);
    gpu_freeWeirData(&g_gpuWeirs);
    gpu_freeOutletData(&g_gpuOutlets);
}
