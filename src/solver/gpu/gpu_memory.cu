//-----------------------------------------------------------------------------
//   gpu_memory.cu
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   GPU memory management functions for SWMM data structures.
//   Supports both unified memory (DGX Spark) and discrete GPU architectures.
//
//-----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "gpu_config.h"
#include "gpu_structures.h"

//=============================================================================
// Helper Macros
//=============================================================================

#define CUDA_ALLOC_MANAGED(ptr, size) \
    do { \
        if (g_gpuConfig.unifiedMemory) { \
            CUDA_CHECK(cudaMallocManaged(ptr, size)); \
        } else { \
            CUDA_CHECK(cudaMalloc(ptr, size)); \
        } \
    } while(0)

#define CUDA_FREE_SAFE(ptr) \
    do { \
        if (ptr != NULL) { \
            cudaFree(ptr); \
            ptr = NULL; \
        } \
    } while(0)

//=============================================================================
// Node Data Allocation/Deallocation
//=============================================================================

int gpu_allocateNodeData(GPU_NodeData* data, int nodeCount)
//
//  Purpose: Allocates EXPLICIT memory for node data arrays
//  Input:   data = pointer to GPU_NodeData structure
//           nodeCount = number of nodes
//  Returns: 0 if successful, CUDA error code otherwise
//
//  Memory Strategy:
//    - h_* pointers: Pinned host memory (cudaMallocHost) for fast PCIe transfers
//    - d_* pointers: Device memory (cudaMalloc) that stays on GPU
//    - NO unified memory - we control all transfers explicitly
//
{
    if (data == NULL || nodeCount <= 0) return -1;

    // Initialize structure
    memset(data, 0, sizeof(GPU_NodeData));
    data->count = nodeCount;

    size_t intSize = nodeCount * sizeof(int);
    size_t doubleSize = nodeCount * sizeof(double);
    size_t charSize = nodeCount * sizeof(char);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    // -------------------------------------------------------------------------
    // ALLOCATE HOST (CPU) MEMORY - Pinned for fast transfers
    // -------------------------------------------------------------------------

    // Static properties
    CUDA_CHECK(cudaMallocHost((void**)&data->h_type, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_invertElev, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_fullDepth, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_surDepth, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_pondedArea, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_crownElev, doubleSize));
    hostBytes += doubleSize;

    // Dynamic state
    CUDA_CHECK(cudaMallocHost((void**)&data->h_oldDepth, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_newDepth, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_oldVolume, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_newVolume, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_fullVolume, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_oldNetInflow, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_inflow, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_outflow, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_overflow, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_degree, intSize));
    hostBytes += intSize;

    // Extended data
    CUDA_CHECK(cudaMallocHost((void**)&data->h_converged, charSize));
    hostBytes += charSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_newSurfArea, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_oldSurfArea, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_sumdqdh, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_dYdT, doubleSize));
    hostBytes += doubleSize;

    // -------------------------------------------------------------------------
    // ALLOCATE DEVICE (GPU) MEMORY
    // -------------------------------------------------------------------------

    // Static properties
    CUDA_CHECK(cudaMalloc((void**)&data->d_type, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_invertElev, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_fullDepth, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_surDepth, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_pondedArea, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_crownElev, doubleSize));
    deviceBytes += doubleSize;

    // Dynamic state
    CUDA_CHECK(cudaMalloc((void**)&data->d_oldDepth, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_newDepth, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_oldVolume, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_newVolume, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_fullVolume, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_oldNetInflow, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_inflow, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_outflow, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_overflow, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_degree, intSize));
    deviceBytes += intSize;

    // Extended data
    CUDA_CHECK(cudaMalloc((void**)&data->d_converged, charSize));
    deviceBytes += charSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_newSurfArea, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_oldSurfArea, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_sumdqdh, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_dYdT, doubleSize));
    deviceBytes += doubleSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d nodes (%.2f MB: %.2f MB host + %.2f MB device)\n",
           nodeCount, hostMB + deviceMB, hostMB, deviceMB);

    return 0;
}

//=============================================================================

void gpu_freeNodeData(GPU_NodeData* data)
//
//  Purpose: Frees EXPLICIT GPU memory for node data (both host and device)
//
{
    if (data == NULL) return;

    // Free host (pinned) memory
    if (data->h_type) cudaFreeHost(data->h_type);
    if (data->h_invertElev) cudaFreeHost(data->h_invertElev);
    if (data->h_fullDepth) cudaFreeHost(data->h_fullDepth);
    if (data->h_surDepth) cudaFreeHost(data->h_surDepth);
    if (data->h_pondedArea) cudaFreeHost(data->h_pondedArea);
    if (data->h_crownElev) cudaFreeHost(data->h_crownElev);
    if (data->h_oldDepth) cudaFreeHost(data->h_oldDepth);
    if (data->h_newDepth) cudaFreeHost(data->h_newDepth);
    if (data->h_oldVolume) cudaFreeHost(data->h_oldVolume);
    if (data->h_newVolume) cudaFreeHost(data->h_newVolume);
    if (data->h_fullVolume) cudaFreeHost(data->h_fullVolume);
    if (data->h_oldNetInflow) cudaFreeHost(data->h_oldNetInflow);
    if (data->h_inflow) cudaFreeHost(data->h_inflow);
    if (data->h_outflow) cudaFreeHost(data->h_outflow);
    if (data->h_overflow) cudaFreeHost(data->h_overflow);
    if (data->h_degree) cudaFreeHost(data->h_degree);
    if (data->h_converged) cudaFreeHost(data->h_converged);
    if (data->h_newSurfArea) cudaFreeHost(data->h_newSurfArea);
    if (data->h_oldSurfArea) cudaFreeHost(data->h_oldSurfArea);
    if (data->h_sumdqdh) cudaFreeHost(data->h_sumdqdh);
    if (data->h_dYdT) cudaFreeHost(data->h_dYdT);

    // Free device memory
    CUDA_FREE_SAFE(data->d_type);
    CUDA_FREE_SAFE(data->d_invertElev);
    CUDA_FREE_SAFE(data->d_fullDepth);
    CUDA_FREE_SAFE(data->d_surDepth);
    CUDA_FREE_SAFE(data->d_pondedArea);
    CUDA_FREE_SAFE(data->d_crownElev);
    CUDA_FREE_SAFE(data->d_oldDepth);
    CUDA_FREE_SAFE(data->d_newDepth);
    CUDA_FREE_SAFE(data->d_oldVolume);
    CUDA_FREE_SAFE(data->d_newVolume);
    CUDA_FREE_SAFE(data->d_fullVolume);
    CUDA_FREE_SAFE(data->d_oldNetInflow);
    CUDA_FREE_SAFE(data->d_inflow);
    CUDA_FREE_SAFE(data->d_outflow);
    CUDA_FREE_SAFE(data->d_overflow);
    CUDA_FREE_SAFE(data->d_degree);
    CUDA_FREE_SAFE(data->d_converged);
    CUDA_FREE_SAFE(data->d_newSurfArea);
    CUDA_FREE_SAFE(data->d_oldSurfArea);
    CUDA_FREE_SAFE(data->d_sumdqdh);
    CUDA_FREE_SAFE(data->d_dYdT);

    data->count = 0;
}

//=============================================================================
// Link Data Allocation/Deallocation
//=============================================================================

int gpu_allocateLinkData(GPU_LinkData* data, int linkCount)
//
//  Purpose: Allocates EXPLICIT memory for link data arrays
//  Memory Strategy: h_* (pinned host) + d_* (device)
//
{
    if (data == NULL || linkCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_LinkData));
    data->count = linkCount;

    size_t intSize = linkCount * sizeof(int);
    size_t doubleSize = linkCount * sizeof(double);
    size_t charSize = linkCount * sizeof(char);
    size_t scharSize = linkCount * sizeof(signed char);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    // -------------------------------------------------------------------------
    // ALLOCATE HOST (CPU) MEMORY - Pinned
    // -------------------------------------------------------------------------

    // Topology
    CUDA_CHECK(cudaMallocHost((void**)&data->h_type, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_subIndex, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_node1, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_node2, intSize));
    hostBytes += intSize;

    // Static properties
    CUDA_CHECK(cudaMallocHost((void**)&data->h_offset1, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_offset2, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_qFull, doubleSize));
    hostBytes += doubleSize;

    // Dynamic state
    CUDA_CHECK(cudaMallocHost((void**)&data->h_oldFlow, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_newFlow, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_oldDepth, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_newDepth, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_oldVolume, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_newVolume, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_surfArea1, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_surfArea2, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_dqdh, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_froude, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_setting, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_targetSetting, doubleSize));
    hostBytes += doubleSize;

    // Flags
    CUDA_CHECK(cudaMallocHost((void**)&data->h_bypassed, charSize));
    hostBytes += charSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_direction, scharSize));
    hostBytes += scharSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_flowClass, scharSize));
    hostBytes += scharSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_hasFlapGate, charSize));
    hostBytes += charSize;

    // -------------------------------------------------------------------------
    // ALLOCATE DEVICE (GPU) MEMORY
    // -------------------------------------------------------------------------

    // Topology
    CUDA_CHECK(cudaMalloc((void**)&data->d_type, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_subIndex, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_node1, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_node2, intSize));
    deviceBytes += intSize;

    // Static properties
    CUDA_CHECK(cudaMalloc((void**)&data->d_offset1, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_offset2, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_qFull, doubleSize));
    deviceBytes += doubleSize;

    // Dynamic state
    CUDA_CHECK(cudaMalloc((void**)&data->d_oldFlow, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_newFlow, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_oldDepth, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_newDepth, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_oldVolume, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_newVolume, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_surfArea1, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_surfArea2, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_dqdh, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_froude, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_setting, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_targetSetting, doubleSize));
    deviceBytes += doubleSize;

    // Flags
    CUDA_CHECK(cudaMalloc((void**)&data->d_bypassed, charSize));
    deviceBytes += charSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_direction, scharSize));
    deviceBytes += scharSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_flowClass, scharSize));
    deviceBytes += scharSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_hasFlapGate, charSize));
    deviceBytes += charSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d links (%.2f MB: %.2f MB host + %.2f MB device)\n",
           linkCount, hostMB + deviceMB, hostMB, deviceMB);

    return 0;
}

//=============================================================================

void gpu_freeLinkData(GPU_LinkData* data)
//
//  Purpose: Frees EXPLICIT GPU memory for link data (both host and device)
//
{
    if (data == NULL) return;

    // Free host (pinned) memory
    if (data->h_type) cudaFreeHost(data->h_type);
    if (data->h_subIndex) cudaFreeHost(data->h_subIndex);
    if (data->h_node1) cudaFreeHost(data->h_node1);
    if (data->h_node2) cudaFreeHost(data->h_node2);
    if (data->h_offset1) cudaFreeHost(data->h_offset1);
    if (data->h_offset2) cudaFreeHost(data->h_offset2);
    if (data->h_qFull) cudaFreeHost(data->h_qFull);
    if (data->h_oldFlow) cudaFreeHost(data->h_oldFlow);
    if (data->h_newFlow) cudaFreeHost(data->h_newFlow);
    if (data->h_oldDepth) cudaFreeHost(data->h_oldDepth);
    if (data->h_newDepth) cudaFreeHost(data->h_newDepth);
    if (data->h_oldVolume) cudaFreeHost(data->h_oldVolume);
    if (data->h_newVolume) cudaFreeHost(data->h_newVolume);
    if (data->h_surfArea1) cudaFreeHost(data->h_surfArea1);
    if (data->h_surfArea2) cudaFreeHost(data->h_surfArea2);
    if (data->h_dqdh) cudaFreeHost(data->h_dqdh);
    if (data->h_froude) cudaFreeHost(data->h_froude);
    if (data->h_setting) cudaFreeHost(data->h_setting);
    if (data->h_targetSetting) cudaFreeHost(data->h_targetSetting);
    if (data->h_bypassed) cudaFreeHost(data->h_bypassed);
    if (data->h_direction) cudaFreeHost(data->h_direction);
    if (data->h_flowClass) cudaFreeHost(data->h_flowClass);
    if (data->h_hasFlapGate) cudaFreeHost(data->h_hasFlapGate);

    // Free device memory
    CUDA_FREE_SAFE(data->d_type);
    CUDA_FREE_SAFE(data->d_subIndex);
    CUDA_FREE_SAFE(data->d_node1);
    CUDA_FREE_SAFE(data->d_node2);
    CUDA_FREE_SAFE(data->d_offset1);
    CUDA_FREE_SAFE(data->d_offset2);
    CUDA_FREE_SAFE(data->d_qFull);
    CUDA_FREE_SAFE(data->d_oldFlow);
    CUDA_FREE_SAFE(data->d_newFlow);
    CUDA_FREE_SAFE(data->d_oldDepth);
    CUDA_FREE_SAFE(data->d_newDepth);
    CUDA_FREE_SAFE(data->d_oldVolume);
    CUDA_FREE_SAFE(data->d_newVolume);
    CUDA_FREE_SAFE(data->d_surfArea1);
    CUDA_FREE_SAFE(data->d_surfArea2);
    CUDA_FREE_SAFE(data->d_dqdh);
    CUDA_FREE_SAFE(data->d_froude);
    CUDA_FREE_SAFE(data->d_setting);
    CUDA_FREE_SAFE(data->d_targetSetting);
    CUDA_FREE_SAFE(data->d_bypassed);
    CUDA_FREE_SAFE(data->d_direction);
    CUDA_FREE_SAFE(data->d_flowClass);
    CUDA_FREE_SAFE(data->d_hasFlapGate);

    data->count = 0;
}

//=============================================================================
// Conduit Data Allocation/Deallocation
//=============================================================================

int gpu_allocateConduitData(GPU_ConduitData* data, int conduitCount)
//
//  Purpose: Allocates EXPLICIT memory for conduit data arrays
//  Memory Strategy: h_* (pinned host) + d_* (device)
//
{
    if (data == NULL || conduitCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_ConduitData));
    data->count = conduitCount;

    size_t doubleSize = conduitCount * sizeof(double);
    size_t charSize = conduitCount * sizeof(char);

    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    // -------------------------------------------------------------------------
    // ALLOCATE HOST (CPU) MEMORY - Pinned
    // -------------------------------------------------------------------------

    // Static properties
    CUDA_CHECK(cudaMallocHost((void**)&data->h_length, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_modLength, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_roughness, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_roughFactor, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_slope, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_barrels, charSize));
    hostBytes += charSize;

    // Dynamic wave parameters
    CUDA_CHECK(cudaMallocHost((void**)&data->h_beta, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_qMax, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_a1, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_a2, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_q1, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_q2, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_q1Old, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_q2Old, doubleSize));
    hostBytes += doubleSize;

    // Flags
    CUDA_CHECK(cudaMallocHost((void**)&data->h_capacityLimited, charSize));
    hostBytes += charSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_superCritical, charSize));
    hostBytes += charSize;

    // -------------------------------------------------------------------------
    // ALLOCATE DEVICE (GPU) MEMORY
    // -------------------------------------------------------------------------

    // Static properties
    CUDA_CHECK(cudaMalloc((void**)&data->d_length, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_modLength, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_roughness, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_roughFactor, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_slope, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_barrels, charSize));
    deviceBytes += charSize;

    // Dynamic wave parameters
    CUDA_CHECK(cudaMalloc((void**)&data->d_beta, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_qMax, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_a1, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_a2, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_q1, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_q2, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_q1Old, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_q2Old, doubleSize));
    deviceBytes += doubleSize;

    // Flags
    CUDA_CHECK(cudaMalloc((void**)&data->d_capacityLimited, charSize));
    deviceBytes += charSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_superCritical, charSize));
    deviceBytes += charSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d conduits (%.2f MB: %.2f MB host + %.2f MB device)\n",
           conduitCount, hostMB + deviceMB, hostMB, deviceMB);

    return 0;
}

//=============================================================================

void gpu_freeConduitData(GPU_ConduitData* data)
//
//  Purpose: Frees EXPLICIT GPU memory for conduit data (both host and device)
//
{
    if (data == NULL) return;

    // Free host (pinned) memory
    if (data->h_length) cudaFreeHost(data->h_length);
    if (data->h_modLength) cudaFreeHost(data->h_modLength);
    if (data->h_roughness) cudaFreeHost(data->h_roughness);
    if (data->h_roughFactor) cudaFreeHost(data->h_roughFactor);
    if (data->h_slope) cudaFreeHost(data->h_slope);
    if (data->h_barrels) cudaFreeHost(data->h_barrels);
    if (data->h_beta) cudaFreeHost(data->h_beta);
    if (data->h_qMax) cudaFreeHost(data->h_qMax);
    if (data->h_a1) cudaFreeHost(data->h_a1);
    if (data->h_a2) cudaFreeHost(data->h_a2);
    if (data->h_q1) cudaFreeHost(data->h_q1);
    if (data->h_q2) cudaFreeHost(data->h_q2);
    if (data->h_q1Old) cudaFreeHost(data->h_q1Old);
    if (data->h_q2Old) cudaFreeHost(data->h_q2Old);
    if (data->h_capacityLimited) cudaFreeHost(data->h_capacityLimited);
    if (data->h_superCritical) cudaFreeHost(data->h_superCritical);

    // Free device memory
    CUDA_FREE_SAFE(data->d_length);
    CUDA_FREE_SAFE(data->d_modLength);
    CUDA_FREE_SAFE(data->d_roughness);
    CUDA_FREE_SAFE(data->d_roughFactor);
    CUDA_FREE_SAFE(data->d_slope);
    CUDA_FREE_SAFE(data->d_barrels);
    CUDA_FREE_SAFE(data->d_beta);
    CUDA_FREE_SAFE(data->d_qMax);
    CUDA_FREE_SAFE(data->d_a1);
    CUDA_FREE_SAFE(data->d_a2);
    CUDA_FREE_SAFE(data->d_q1);
    CUDA_FREE_SAFE(data->d_q2);
    CUDA_FREE_SAFE(data->d_q1Old);
    CUDA_FREE_SAFE(data->d_q2Old);
    CUDA_FREE_SAFE(data->d_capacityLimited);
    CUDA_FREE_SAFE(data->d_superCritical);

    data->count = 0;
}

//=============================================================================
// Pump Data Allocation/Deallocation
//=============================================================================

int gpu_allocatePumpData(GPU_PumpData* data, int pumpCount)
{
    if (data == NULL || pumpCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_PumpData));
    data->count = pumpCount;

    size_t intSize = pumpCount * sizeof(int);
    size_t doubleSize = pumpCount * sizeof(double);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    CUDA_CHECK(cudaMallocHost((void**)&data->h_linkIndex, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_type, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_pumpCurve, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_initSetting, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_yOn, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_yOff, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_xMin, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_xMax, doubleSize));
    hostBytes += doubleSize;

    CUDA_CHECK(cudaMalloc((void**)&data->d_linkIndex, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_type, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_pumpCurve, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_initSetting, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_yOn, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_yOff, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_xMin, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_xMax, doubleSize));
    deviceBytes += doubleSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d pumps (%.2f MB: %.2f MB host + %.2f MB device)\n",
           pumpCount, hostMB + deviceMB, hostMB, deviceMB);
    return 0;
}

void gpu_freePumpData(GPU_PumpData* data)
{
    if (data == NULL) return;

    if (data->h_type) cudaFreeHost(data->h_type);
    if (data->h_pumpCurve) cudaFreeHost(data->h_pumpCurve);
    if (data->h_initSetting) cudaFreeHost(data->h_initSetting);
    if (data->h_yOn) cudaFreeHost(data->h_yOn);
    if (data->h_yOff) cudaFreeHost(data->h_yOff);
    if (data->h_xMin) cudaFreeHost(data->h_xMin);
    if (data->h_xMax) cudaFreeHost(data->h_xMax);

    CUDA_FREE_SAFE(data->d_type);
    CUDA_FREE_SAFE(data->d_pumpCurve);
    CUDA_FREE_SAFE(data->d_initSetting);
    CUDA_FREE_SAFE(data->d_yOn);
    CUDA_FREE_SAFE(data->d_yOff);
    CUDA_FREE_SAFE(data->d_xMin);
    CUDA_FREE_SAFE(data->d_xMax);

    data->count = 0;
}

//=============================================================================
// Orifice Data Allocation/Deallocation
//=============================================================================

int gpu_allocateOrificeData(GPU_OrificeData* data, int orificeCount)
{
    if (data == NULL || orificeCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_OrificeData));
    data->count = orificeCount;

    size_t intSize = orificeCount * sizeof(int);
    size_t doubleSize = orificeCount * sizeof(double);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    CUDA_CHECK(cudaMallocHost((void**)&data->h_linkIndex, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_type, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_shape, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_cDisch, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_orate, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_cOrif, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_hCrit, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_cWeir, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_length, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_surfArea, doubleSize));
    hostBytes += doubleSize;

    CUDA_CHECK(cudaMalloc((void**)&data->d_linkIndex, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_type, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_shape, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_cDisch, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_orate, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_cOrif, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_hCrit, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_cWeir, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_length, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_surfArea, doubleSize));
    deviceBytes += doubleSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d orifices (%.2f MB: %.2f MB host + %.2f MB device)\n",
           orificeCount, hostMB + deviceMB, hostMB, deviceMB);
    return 0;
}

void gpu_freeOrificeData(GPU_OrificeData* data)
{
    if (data == NULL) return;

    if (data->h_linkIndex) cudaFreeHost(data->h_linkIndex);
    if (data->h_type) cudaFreeHost(data->h_type);
    if (data->h_shape) cudaFreeHost(data->h_shape);
    if (data->h_cDisch) cudaFreeHost(data->h_cDisch);
    if (data->h_orate) cudaFreeHost(data->h_orate);
    if (data->h_cOrif) cudaFreeHost(data->h_cOrif);
    if (data->h_hCrit) cudaFreeHost(data->h_hCrit);
    if (data->h_cWeir) cudaFreeHost(data->h_cWeir);
    if (data->h_length) cudaFreeHost(data->h_length);
    if (data->h_surfArea) cudaFreeHost(data->h_surfArea);

    CUDA_FREE_SAFE(data->d_linkIndex);
    CUDA_FREE_SAFE(data->d_type);
    CUDA_FREE_SAFE(data->d_shape);
    CUDA_FREE_SAFE(data->d_cDisch);
    CUDA_FREE_SAFE(data->d_orate);
    CUDA_FREE_SAFE(data->d_cOrif);
    CUDA_FREE_SAFE(data->d_hCrit);
    CUDA_FREE_SAFE(data->d_cWeir);
    CUDA_FREE_SAFE(data->d_length);
    CUDA_FREE_SAFE(data->d_surfArea);

    data->count = 0;
}

//=============================================================================
// Weir Data Allocation/Deallocation
//=============================================================================

int gpu_allocateWeirData(GPU_WeirData* data, int weirCount)
{
    if (data == NULL || weirCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_WeirData));
    data->count = weirCount;

    size_t intSize = weirCount * sizeof(int);
    size_t doubleSize = weirCount * sizeof(double);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    CUDA_CHECK(cudaMallocHost((void**)&data->h_linkIndex, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_type, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_cDisch1, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_cDisch2, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_endCon, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_canSurcharge, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_roadWidth, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_roadSurface, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_cdCurve, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_cSurcharge, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_length, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_slope, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_surfArea, doubleSize));
    hostBytes += doubleSize;

    CUDA_CHECK(cudaMalloc((void**)&data->d_linkIndex, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_type, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_cDisch1, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_cDisch2, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_endCon, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_canSurcharge, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_roadWidth, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_roadSurface, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_cdCurve, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_cSurcharge, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_length, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_slope, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_surfArea, doubleSize));
    deviceBytes += doubleSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d weirs (%.2f MB: %.2f MB host + %.2f MB device)\n",
           weirCount, hostMB + deviceMB, hostMB, deviceMB);
    return 0;
}

void gpu_freeWeirData(GPU_WeirData* data)
{
    if (data == NULL) return;

    if (data->h_linkIndex) cudaFreeHost(data->h_linkIndex);
    if (data->h_type) cudaFreeHost(data->h_type);
    if (data->h_cDisch1) cudaFreeHost(data->h_cDisch1);
    if (data->h_cDisch2) cudaFreeHost(data->h_cDisch2);
    if (data->h_endCon) cudaFreeHost(data->h_endCon);
    if (data->h_canSurcharge) cudaFreeHost(data->h_canSurcharge);
    if (data->h_roadWidth) cudaFreeHost(data->h_roadWidth);
    if (data->h_roadSurface) cudaFreeHost(data->h_roadSurface);
    if (data->h_cdCurve) cudaFreeHost(data->h_cdCurve);
    if (data->h_cSurcharge) cudaFreeHost(data->h_cSurcharge);
    if (data->h_length) cudaFreeHost(data->h_length);
    if (data->h_slope) cudaFreeHost(data->h_slope);
    if (data->h_surfArea) cudaFreeHost(data->h_surfArea);

    CUDA_FREE_SAFE(data->d_linkIndex);
    CUDA_FREE_SAFE(data->d_type);
    CUDA_FREE_SAFE(data->d_cDisch1);
    CUDA_FREE_SAFE(data->d_cDisch2);
    CUDA_FREE_SAFE(data->d_endCon);
    CUDA_FREE_SAFE(data->d_canSurcharge);
    CUDA_FREE_SAFE(data->d_roadWidth);
    CUDA_FREE_SAFE(data->d_roadSurface);
    CUDA_FREE_SAFE(data->d_cdCurve);
    CUDA_FREE_SAFE(data->d_cSurcharge);
    CUDA_FREE_SAFE(data->d_length);
    CUDA_FREE_SAFE(data->d_slope);
    CUDA_FREE_SAFE(data->d_surfArea);

    data->count = 0;
}

//=============================================================================
// Outlet Data Allocation/Deallocation
//=============================================================================

int gpu_allocateOutletData(GPU_OutletData* data, int outletCount)
{
    if (data == NULL || outletCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_OutletData));
    data->count = outletCount;

    size_t intSize = outletCount * sizeof(int);
    size_t doubleSize = outletCount * sizeof(double);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    CUDA_CHECK(cudaMallocHost((void**)&data->h_linkIndex, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_qCoeff, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_qExpon, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_qCurve, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_curveType, intSize));
    hostBytes += intSize;

    CUDA_CHECK(cudaMalloc((void**)&data->d_linkIndex, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_qCoeff, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_qExpon, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_qCurve, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_curveType, intSize));
    deviceBytes += intSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d outlets (%.2f MB: %.2f MB host + %.2f MB device)\n",
           outletCount, hostMB + deviceMB, hostMB, deviceMB);
    return 0;
}

void gpu_freeOutletData(GPU_OutletData* data)
{
    if (data == NULL) return;

    if (data->h_linkIndex) cudaFreeHost(data->h_linkIndex);
    if (data->h_qCoeff) cudaFreeHost(data->h_qCoeff);
    if (data->h_qExpon) cudaFreeHost(data->h_qExpon);
    if (data->h_qCurve) cudaFreeHost(data->h_qCurve);
    if (data->h_curveType) cudaFreeHost(data->h_curveType);

    CUDA_FREE_SAFE(data->d_linkIndex);
    CUDA_FREE_SAFE(data->d_qCoeff);
    CUDA_FREE_SAFE(data->d_qExpon);
    CUDA_FREE_SAFE(data->d_qCurve);
    CUDA_FREE_SAFE(data->d_curveType);

    data->count = 0;
}

//=============================================================================
// Cross-Section Data Allocation/Deallocation
//=============================================================================

int gpu_allocateXsectData(GPU_XsectData* data, int xsectCount)
//
//  Purpose: Allocates EXPLICIT memory for cross-section data arrays
//  Memory Strategy: h_* (pinned host) + d_* (device)
//
{
    if (data == NULL || xsectCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_XsectData));
    data->count = xsectCount;

    size_t intSize = xsectCount * sizeof(int);
    size_t doubleSize = xsectCount * sizeof(double);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    // -------------------------------------------------------------------------
    // ALLOCATE HOST (CPU) MEMORY - Pinned
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaMallocHost((void**)&data->h_type, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_aFull, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_rFull, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_wMax, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_yFull, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_geom1, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_geom2, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_geom3, doubleSize));
    hostBytes += doubleSize;

    // -------------------------------------------------------------------------
    // ALLOCATE DEVICE (GPU) MEMORY
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaMalloc((void**)&data->d_type, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_aFull, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_rFull, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_wMax, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_yFull, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_geom1, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_geom2, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_geom3, doubleSize));
    deviceBytes += doubleSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d cross-sections (%.2f MB: %.2f MB host + %.2f MB device)\n",
           xsectCount, hostMB + deviceMB, hostMB, deviceMB);

    return 0;
}

//=============================================================================

void gpu_freeXsectData(GPU_XsectData* data)
//
//  Purpose: Frees EXPLICIT GPU memory for cross-section data (both host and device)
//
{
    if (data == NULL) return;

    // Free host (pinned) memory
    if (data->h_type) cudaFreeHost(data->h_type);
    if (data->h_aFull) cudaFreeHost(data->h_aFull);
    if (data->h_rFull) cudaFreeHost(data->h_rFull);
    if (data->h_wMax) cudaFreeHost(data->h_wMax);
    if (data->h_yFull) cudaFreeHost(data->h_yFull);
    if (data->h_geom1) cudaFreeHost(data->h_geom1);
    if (data->h_geom2) cudaFreeHost(data->h_geom2);
    if (data->h_geom3) cudaFreeHost(data->h_geom3);

    // Free device memory
    CUDA_FREE_SAFE(data->d_type);
    CUDA_FREE_SAFE(data->d_aFull);
    CUDA_FREE_SAFE(data->d_rFull);
    CUDA_FREE_SAFE(data->d_wMax);
    CUDA_FREE_SAFE(data->d_yFull);
    CUDA_FREE_SAFE(data->d_geom1);
    CUDA_FREE_SAFE(data->d_geom2);
    CUDA_FREE_SAFE(data->d_geom3);

    data->count = 0;
}

//=============================================================================
// Data Transfer Functions (CPU AoS -> GPU SoA)
//=============================================================================
// Note: These functions require knowledge of SWMM's internal structures
//       Since gpu_memory.cu cannot directly include headers.h (circular deps),
//       we use void* and cast internally or provide wrapper functions

int gpu_transferNodeDataFromArrays(
    GPU_NodeData* gpuData,
    // Arrays from CPU
    int* type,
    double* invertElev,
    double* fullDepth,
    double* surDepth,
    double* pondedArea,
    double* crownElev,
    double* oldDepth,
    double* newDepth,
    double* oldVolume,
    double* newVolume,
    double* oldNetInflow,
    double* inflow,
    double* outflow,
    int count)
//
//  Purpose: LEGACY - Transfers node data from CPU arrays to GPU (EXPLICIT memory version)
//  Input:   gpuData = pre-allocated GPU_NodeData structure
//           Arrays = pointers to CPU data arrays
//           count = number of nodes
//  Returns: 0 if successful, error code otherwise
//
//  Note: Updated for EXPLICIT memory model
//        Step 1: Copy CPU arrays → pinned host arrays (h_*)
//        Step 2: Transfer pinned host → device (d_*)
//
{
    if (gpuData == NULL || count <= 0) return -1;
    if (gpuData->count != count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Step 1: Copy CPU arrays to pinned host memory (h_*)
    memcpy(gpuData->h_type, type, intSize);
    memcpy(gpuData->h_invertElev, invertElev, doubleSize);
    memcpy(gpuData->h_fullDepth, fullDepth, doubleSize);
    memcpy(gpuData->h_surDepth, surDepth, doubleSize);
    memcpy(gpuData->h_pondedArea, pondedArea, doubleSize);
    memcpy(gpuData->h_crownElev, crownElev, doubleSize);
    memcpy(gpuData->h_oldDepth, oldDepth, doubleSize);
    memcpy(gpuData->h_newDepth, newDepth, doubleSize);
    memcpy(gpuData->h_oldVolume, oldVolume, doubleSize);
    memcpy(gpuData->h_newVolume, newVolume, doubleSize);
    memcpy(gpuData->h_oldNetInflow, oldNetInflow, doubleSize);
    memcpy(gpuData->h_inflow, inflow, doubleSize);
    memcpy(gpuData->h_outflow, outflow, doubleSize);

    // Step 2: Transfer host → device using explicit copy
    CUDA_CHECK(cudaMemcpy(gpuData->d_type, gpuData->h_type, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_invertElev, gpuData->h_invertElev, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_fullDepth, gpuData->h_fullDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_surDepth, gpuData->h_surDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_pondedArea, gpuData->h_pondedArea, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_crownElev, gpuData->h_crownElev, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_oldDepth, gpuData->h_oldDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_newDepth, gpuData->h_newDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_oldVolume, gpuData->h_oldVolume, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_newVolume, gpuData->h_newVolume, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_oldNetInflow, gpuData->h_oldNetInflow, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_inflow, gpuData->h_inflow, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData->d_outflow, gpuData->h_outflow, doubleSize, cudaMemcpyHostToDevice));

    return 0;
}

//=============================================================================

int gpu_retrieveNodeDataToArrays(
    // GPU data
    GPU_NodeData* gpuData,
    // CPU arrays to fill
    double* newDepth,
    double* newVolume,
    double* overflow,
    int count)
//
//  Purpose: LEGACY - Retrieves computed node data from GPU (EXPLICIT memory version)
//  Input:   gpuData = GPU_NodeData structure with computed results
//           Arrays = pointers to CPU arrays to fill
//           count = number of nodes
//  Returns: 0 if successful, error code otherwise
//
//  Note: Updated for EXPLICIT memory model
//        Step 1: Transfer device (d_*) → pinned host (h_*)
//        Step 2: Copy pinned host → CPU arrays
//
{
    if (gpuData == NULL || count <= 0) return -1;
    if (gpuData->count != count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Step 1: Transfer device → host
    CUDA_CHECK(cudaMemcpy(gpuData->h_newDepth, gpuData->d_newDepth, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpuData->h_newVolume, gpuData->d_newVolume, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpuData->h_overflow, gpuData->d_overflow, doubleSize, cudaMemcpyDeviceToHost));

    // Step 2: Copy pinned host → CPU arrays
    memcpy(newDepth, gpuData->h_newDepth, doubleSize);
    memcpy(newVolume, gpuData->h_newVolume, doubleSize);
    memcpy(overflow, gpuData->h_overflow, doubleSize);

    return 0;
}

//=============================================================================
// EXPLICIT MEMORY TRANSFER HELPERS
//=============================================================================
// These functions provide fine-grained control over Host ↔ Device transfers
// for the explicit memory model (separate h_* and d_* pointers).
//
// Design:
//   - Static data: Transfer once at simulation start (read-only on GPU)
//   - Dynamic data: Transfer before each timestep / after convergence
//   - Async variants: Use CUDA streams for overlapping compute and transfer
//   - Range variants: Transfer only touched elements (future optimization)
//=============================================================================

//-----------------------------------------------------------------------------
// Node Data Transfer - Static Properties (Once at start)
//-----------------------------------------------------------------------------

int gpu_transferNodeStaticToDevice(GPU_NodeData* data, int count)
//
//  Purpose: Transfer static (read-only) node properties to GPU
//  Input:   data = GPU_NodeData with populated h_* arrays
//           count = number of nodes
//  Returns: 0 on success, error code otherwise
//
//  When to call: Once at simulation start, after initializing h_* arrays
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Transfer static properties (read-only during routing)
    CUDA_CHECK(cudaMemcpy(data->d_type, data->h_type, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_invertElev, data->h_invertElev, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_fullDepth, data->h_fullDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_surDepth, data->h_surDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_pondedArea, data->h_pondedArea, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_crownElev, data->h_crownElev, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_fullVolume, data->h_fullVolume, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_degree, data->h_degree, intSize, cudaMemcpyHostToDevice));

    return 0;
}

//-----------------------------------------------------------------------------
// Node Data Transfer - Dynamic State (Before each timestep)
//-----------------------------------------------------------------------------

int gpu_transferNodeDynamicToDevice(GPU_NodeData* data, int count)
//
//  Purpose: Transfer dynamic node state to GPU before Picard iteration
//  Input:   data = GPU_NodeData with updated h_* arrays
//           count = number of nodes
//  Returns: 0 on success, error code otherwise
//
//  When to call: Before each timestep's Picard iteration loop
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Transfer dynamic state (updated each iteration)
    CUDA_CHECK(cudaMemcpy(data->d_oldDepth, data->h_oldDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_newDepth, data->h_newDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_oldVolume, data->h_oldVolume, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_newVolume, data->h_newVolume, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_oldNetInflow, data->h_oldNetInflow, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_inflow, data->h_inflow, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_outflow, data->h_outflow, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_overflow, data->h_overflow, doubleSize, cudaMemcpyHostToDevice));

    // Transfer extended data
    CUDA_CHECK(cudaMemcpy(data->d_converged, data->h_converged, charSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_newSurfArea, data->h_newSurfArea, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_oldSurfArea, data->h_oldSurfArea, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_sumdqdh, data->h_sumdqdh, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_dYdT, data->h_dYdT, doubleSize, cudaMemcpyHostToDevice));

    return 0;
}

//-----------------------------------------------------------------------------
// Node Data Transfer - Results from GPU (After convergence)
//-----------------------------------------------------------------------------

int gpu_transferNodeDynamicFromDevice(GPU_NodeData* data, int count)
//
//  Purpose: Retrieve computed node results from GPU after convergence
//  Input:   data = GPU_NodeData to fill h_* arrays
//           count = number of nodes
//  Returns: 0 on success, error code otherwise
//
//  When to call: After Picard iteration converges for the timestep
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Retrieve computed results
    CUDA_CHECK(cudaMemcpy(data->h_newDepth, data->d_newDepth, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_newVolume, data->d_newVolume, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_overflow, data->d_overflow, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_inflow, data->d_inflow, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_outflow, data->d_outflow, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_converged, data->d_converged, charSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_newSurfArea, data->d_newSurfArea, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_sumdqdh, data->d_sumdqdh, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_dYdT, data->d_dYdT, doubleSize, cudaMemcpyDeviceToHost));

    return 0;
}

//-----------------------------------------------------------------------------
// Node Data Transfer - Iteration State (After conduit kernel)
//-----------------------------------------------------------------------------

int gpu_transferNodeIterationStateFromDevice(GPU_NodeData* data, int count)
//
//  Purpose: Retrieve only the node fields required between Picard iterations.
//           Full node state is copied later after convergence.
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    CUDA_CHECK(cudaMemcpy(data->h_inflow, data->d_inflow, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_outflow, data->d_outflow, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_newSurfArea, data->d_newSurfArea, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_sumdqdh, data->d_sumdqdh, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_converged, data->d_converged, charSize, cudaMemcpyDeviceToHost));

    // CRITICAL: Also transfer newDepth and newVolume for CPU non-conduits
    // Pumps need these values to compute flows correctly
    CUDA_CHECK(cudaMemcpy(data->h_newDepth, data->d_newDepth, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_newVolume, data->d_newVolume, doubleSize, cudaMemcpyDeviceToHost));

    return 0;
}

//-----------------------------------------------------------------------------
// Link Data Transfer - Static Properties
//-----------------------------------------------------------------------------

int gpu_transferLinkStaticToDevice(GPU_LinkData* data, int count)
//
//  Purpose: Transfer static (read-only) link properties to GPU
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);

    size_t charSize = count * sizeof(char);
    // Topology
    CUDA_CHECK(cudaMemcpy(data->d_type, data->h_type, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_subIndex, data->h_subIndex, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_node1, data->h_node1, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_node2, data->h_node2, intSize, cudaMemcpyHostToDevice));

    // Static properties
    CUDA_CHECK(cudaMemcpy(data->d_offset1, data->h_offset1, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_offset2, data->h_offset2, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_qFull, data->h_qFull, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_hasFlapGate, data->h_hasFlapGate, charSize, cudaMemcpyHostToDevice));

    return 0;
}

int gpu_transferPumpStaticToDevice(GPU_PumpData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(data->d_linkIndex, data->h_linkIndex, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_type, data->h_type, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_pumpCurve, data->h_pumpCurve, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_initSetting, data->h_initSetting, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_yOn, data->h_yOn, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_yOff, data->h_yOff, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_xMin, data->h_xMin, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_xMax, data->h_xMax, doubleSize, cudaMemcpyHostToDevice));
    return 0;
}

int gpu_transferOrificeStaticToDevice(GPU_OrificeData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(data->d_linkIndex, data->h_linkIndex, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_type, data->h_type, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_shape, data->h_shape, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_cDisch, data->h_cDisch, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_orate, data->h_orate, doubleSize, cudaMemcpyHostToDevice));
    return 0;
}

int gpu_transferWeirStaticToDevice(GPU_WeirData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(data->d_linkIndex, data->h_linkIndex, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_type, data->h_type, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_cDisch1, data->h_cDisch1, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_cDisch2, data->h_cDisch2, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_endCon, data->h_endCon, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_canSurcharge, data->h_canSurcharge, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_roadWidth, data->h_roadWidth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_roadSurface, data->h_roadSurface, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_cdCurve, data->h_cdCurve, intSize, cudaMemcpyHostToDevice));
    return 0;
}

int gpu_transferOutletStaticToDevice(GPU_OutletData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(data->d_linkIndex, data->h_linkIndex, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_qCoeff, data->h_qCoeff, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_qExpon, data->h_qExpon, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_qCurve, data->h_qCurve, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_curveType, data->h_curveType, intSize, cudaMemcpyHostToDevice));
    return 0;
}

//-----------------------------------------------------------------------------
// Link Data Transfer - Dynamic State
//-----------------------------------------------------------------------------

int gpu_transferLinkDynamicToDevice(GPU_LinkData* data, int count)
//
//  Purpose: Transfer dynamic link state to GPU before iteration
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);
    size_t scharSize = count * sizeof(signed char);

    // Dynamic state
    CUDA_CHECK(cudaMemcpy(data->d_oldFlow, data->h_oldFlow, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_newFlow, data->h_newFlow, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_oldDepth, data->h_oldDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_newDepth, data->h_newDepth, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_oldVolume, data->h_oldVolume, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_newVolume, data->h_newVolume, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_surfArea1, data->h_surfArea1, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_surfArea2, data->h_surfArea2, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_dqdh, data->h_dqdh, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_froude, data->h_froude, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_setting, data->h_setting, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_targetSetting, data->h_targetSetting, doubleSize, cudaMemcpyHostToDevice));

    // Control flags
    CUDA_CHECK(cudaMemcpy(data->d_bypassed, data->h_bypassed, charSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_direction, data->h_direction, scharSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_flowClass, data->h_flowClass, scharSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_hasFlapGate, data->h_hasFlapGate, charSize, cudaMemcpyHostToDevice));

    return 0;
}

//-----------------------------------------------------------------------------
// Link Data Transfer - Results from GPU
//-----------------------------------------------------------------------------

int gpu_transferLinkDynamicFromDevice(GPU_LinkData* data, int count)
//
//  Purpose: Retrieve computed link results from GPU
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);
    size_t scharSize = count * sizeof(signed char);

    // Retrieve results
    CUDA_CHECK(cudaMemcpy(data->h_newFlow, data->d_newFlow, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_newDepth, data->d_newDepth, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_newVolume, data->d_newVolume, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_surfArea1, data->d_surfArea1, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_surfArea2, data->d_surfArea2, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_dqdh, data->d_dqdh, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_froude, data->d_froude, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_setting, data->d_setting, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_targetSetting, data->d_targetSetting, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_bypassed, data->d_bypassed, charSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_direction, data->d_direction, scharSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_flowClass, data->d_flowClass, scharSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_hasFlapGate, data->d_hasFlapGate, charSize, cudaMemcpyDeviceToHost));

    return 0;
}

//-----------------------------------------------------------------------------
// Link Data Transfer - Iteration Results (minimal subset)
//-----------------------------------------------------------------------------

int gpu_transferLinkIterationResultsFromDevice(GPU_LinkData* data, int count)
//
//  Purpose: Retrieve only the per-iteration fields the CPU needs immediately
//           (new flow and depth). Full state is copied later during flush.
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(data->h_newFlow, data->d_newFlow, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_newDepth, data->d_newDepth, doubleSize, cudaMemcpyDeviceToHost));

    return 0;
}

int gpu_transferPumpDynamicToDevice(GPU_PumpData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;
    // No per-iteration pump state currently needs transfer.
    return 0;
}

int gpu_transferPumpDynamicFromDevice(GPU_PumpData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;
    // No per-iteration pump state currently produced on GPU.
    return 0;
}

int gpu_transferOrificeDynamicToDevice(GPU_OrificeData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    CUDA_CHECK(cudaMemcpy(data->d_cOrif, data->h_cOrif, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_hCrit, data->h_hCrit, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_cWeir, data->h_cWeir, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_length, data->h_length, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_surfArea, data->h_surfArea, doubleSize, cudaMemcpyHostToDevice));
    return 0;
}

int gpu_transferOrificeDynamicFromDevice(GPU_OrificeData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    CUDA_CHECK(cudaMemcpy(data->h_cOrif, data->d_cOrif, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_hCrit, data->d_hCrit, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_cWeir, data->d_cWeir, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_length, data->d_length, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_surfArea, data->d_surfArea, doubleSize, cudaMemcpyDeviceToHost));
    return 0;
}

int gpu_transferWeirDynamicToDevice(GPU_WeirData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    CUDA_CHECK(cudaMemcpy(data->d_cSurcharge, data->h_cSurcharge, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_length, data->h_length, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_slope, data->h_slope, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_surfArea, data->h_surfArea, doubleSize, cudaMemcpyHostToDevice));
    return 0;
}

int gpu_transferWeirDynamicFromDevice(GPU_WeirData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    CUDA_CHECK(cudaMemcpy(data->h_cSurcharge, data->d_cSurcharge, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_length, data->d_length, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_slope, data->d_slope, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_surfArea, data->d_surfArea, doubleSize, cudaMemcpyDeviceToHost));
    return 0;
}

int gpu_transferOutletDynamicToDevice(GPU_OutletData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;
    // Outlet data currently static.
    return 0;
}

int gpu_transferOutletDynamicFromDevice(GPU_OutletData* data, int count)
{
    if (data == NULL || count <= 0 || count != data->count) return -1;
    // Outlet data currently static.
    return 0;
}

//-----------------------------------------------------------------------------
// Conduit Data Transfer - Static Properties
//-----------------------------------------------------------------------------

int gpu_transferConduitStaticToDevice(GPU_ConduitData* data, int count)
//
//  Purpose: Transfer static conduit properties to GPU
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Static properties
    CUDA_CHECK(cudaMemcpy(data->d_length, data->h_length, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_modLength, data->h_modLength, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_roughness, data->h_roughness, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_roughFactor, data->h_roughFactor, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_slope, data->h_slope, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_barrels, data->h_barrels, charSize, cudaMemcpyHostToDevice));

    return 0;
}

//-----------------------------------------------------------------------------
// Conduit Data Transfer - Dynamic State
//-----------------------------------------------------------------------------

int gpu_transferConduitDynamicToDevice(GPU_ConduitData* data, int count)
//
//  Purpose: Transfer dynamic conduit state to GPU
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Dynamic wave parameters
    CUDA_CHECK(cudaMemcpy(data->d_beta, data->h_beta, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_qMax, data->h_qMax, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_a1, data->h_a1, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_a2, data->h_a2, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_q1, data->h_q1, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_q2, data->h_q2, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_q1Old, data->h_q1Old, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_q2Old, data->h_q2Old, doubleSize, cudaMemcpyHostToDevice));

    // Status flags
    CUDA_CHECK(cudaMemcpy(data->d_capacityLimited, data->h_capacityLimited, charSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_superCritical, data->h_superCritical, charSize, cudaMemcpyHostToDevice));

    return 0;
}

//-----------------------------------------------------------------------------
// Conduit Data Transfer - Results from GPU
//-----------------------------------------------------------------------------

int gpu_transferConduitDynamicFromDevice(GPU_ConduitData* data, int count)
//
//  Purpose: Retrieve computed conduit results from GPU
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Retrieve results
    CUDA_CHECK(cudaMemcpy(data->h_beta, data->d_beta, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_qMax, data->d_qMax, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_a1, data->d_a1, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_a2, data->d_a2, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_q1, data->d_q1, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_q2, data->d_q2, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_q1Old, data->d_q1Old, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_q2Old, data->d_q2Old, doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_capacityLimited, data->d_capacityLimited, charSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data->h_superCritical, data->d_superCritical, charSize, cudaMemcpyDeviceToHost));

    return 0;
}

//-----------------------------------------------------------------------------
// Cross-Section Data Transfer
//-----------------------------------------------------------------------------

int gpu_transferXsectStaticToDevice(GPU_XsectData* data, int count)
//
//  Purpose: Transfer cross-section data to GPU (static, once at start)
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(data->d_type, data->h_type, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_aFull, data->h_aFull, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_rFull, data->h_rFull, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_wMax, data->h_wMax, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_yFull, data->h_yFull, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_geom1, data->h_geom1, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_geom2, data->h_geom2, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_geom3, data->h_geom3, doubleSize, cudaMemcpyHostToDevice));

    return 0;
}

//=============================================================================
// ASYNC TRANSFER VARIANTS (with CUDA Streams)
//=============================================================================
// These enable overlap of computation and transfer for advanced optimization
//=============================================================================

int gpu_transferNodeDynamicToDeviceAsync(GPU_NodeData* data, int count, void* cudaStream)
//
//  Purpose: Asynchronous transfer of node data to GPU (non-blocking)
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    cudaStream_t stream = (cudaStream_t)cudaStream;
    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Async transfers
    CUDA_CHECK(cudaMemcpyAsync(data->d_oldDepth, data->h_oldDepth, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_newDepth, data->h_newDepth, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_oldVolume, data->h_oldVolume, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_newVolume, data->h_newVolume, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_inflow, data->h_inflow, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_outflow, data->h_outflow, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_converged, data->h_converged, charSize, cudaMemcpyHostToDevice, stream));

    return 0;
}

int gpu_transferLinkDynamicToDeviceAsync(GPU_LinkData* data, int count, void* cudaStream)
//
//  Purpose: Asynchronous transfer of link data to GPU (non-blocking)
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    cudaStream_t stream = (cudaStream_t)cudaStream;
    size_t doubleSize = count * sizeof(double);

    // Async transfers
    CUDA_CHECK(cudaMemcpyAsync(data->d_oldFlow, data->h_oldFlow, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_newFlow, data->h_newFlow, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_newDepth, data->h_newDepth, doubleSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->d_dqdh, data->h_dqdh, doubleSize, cudaMemcpyHostToDevice, stream));

    return 0;
}

int gpu_transferNodeDynamicFromDeviceAsync(GPU_NodeData* data, int count, void* cudaStream)
//
//  Purpose: Asynchronous retrieve of node results from GPU (non-blocking)
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    cudaStream_t stream = (cudaStream_t)cudaStream;
    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Async transfers
    CUDA_CHECK(cudaMemcpyAsync(data->h_newDepth, data->d_newDepth, doubleSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->h_newVolume, data->d_newVolume, doubleSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->h_overflow, data->d_overflow, doubleSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->h_converged, data->d_converged, charSize, cudaMemcpyDeviceToHost, stream));

    return 0;
}

int gpu_transferLinkDynamicFromDeviceAsync(GPU_LinkData* data, int count, void* cudaStream)
//
//  Purpose: Asynchronous retrieve of link results from GPU (non-blocking)
//
{
    if (data == NULL || count <= 0 || count != data->count) return -1;

    cudaStream_t stream = (cudaStream_t)cudaStream;
    size_t doubleSize = count * sizeof(double);

    // Async transfers
    CUDA_CHECK(cudaMemcpyAsync(data->h_newFlow, data->d_newFlow, doubleSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->h_newDepth, data->d_newDepth, doubleSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(data->h_dqdh, data->d_dqdh, doubleSize, cudaMemcpyDeviceToHost, stream));

    return 0;
}

//=============================================================================
// RANGE TRANSFER VARIANTS (Partial Array Transfers)
//=============================================================================
// Transfer only a range [startIdx, endIdx) for optimization
// Future: Track "dirty" flags to transfer only touched elements
//=============================================================================

int gpu_transferNodeRangeToDevice(GPU_NodeData* data, int startIdx, int endIdx)
//
//  Purpose: Transfer a range of node elements to GPU
//  Note: Future optimization for large models with sparse updates
//
{
    if (data == NULL || startIdx < 0 || endIdx > data->count || startIdx >= endIdx) return -1;

    int count = endIdx - startIdx;
    size_t doubleSize = count * sizeof(double);
    size_t charSize = count * sizeof(char);

    // Transfer range
    CUDA_CHECK(cudaMemcpy(&data->d_newDepth[startIdx], &data->h_newDepth[startIdx], doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data->d_inflow[startIdx], &data->h_inflow[startIdx], doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data->d_converged[startIdx], &data->h_converged[startIdx], charSize, cudaMemcpyHostToDevice));

    return 0;
}

int gpu_transferLinkRangeToDevice(GPU_LinkData* data, int startIdx, int endIdx)
{
    if (data == NULL || startIdx < 0 || endIdx > data->count || startIdx >= endIdx) return -1;

    int count = endIdx - startIdx;
    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(&data->d_newFlow[startIdx], &data->h_newFlow[startIdx], doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&data->d_newDepth[startIdx], &data->h_newDepth[startIdx], doubleSize, cudaMemcpyHostToDevice));

    return 0;
}

int gpu_transferNodeRangeFromDevice(GPU_NodeData* data, int startIdx, int endIdx)
{
    if (data == NULL || startIdx < 0 || endIdx > data->count || startIdx >= endIdx) return -1;

    int count = endIdx - startIdx;
    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(&data->h_newDepth[startIdx], &data->d_newDepth[startIdx], doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&data->h_newVolume[startIdx], &data->d_newVolume[startIdx], doubleSize, cudaMemcpyDeviceToHost));

    return 0;
}

int gpu_transferLinkRangeFromDevice(GPU_LinkData* data, int startIdx, int endIdx)
{
    if (data == NULL || startIdx < 0 || endIdx > data->count || startIdx >= endIdx) return -1;

    int count = endIdx - startIdx;
    size_t doubleSize = count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(&data->h_newFlow[startIdx], &data->d_newFlow[startIdx], doubleSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&data->h_newDepth[startIdx], &data->d_newDepth[startIdx], doubleSize, cudaMemcpyDeviceToHost));

    return 0;
}

//=============================================================================
// Curve Data Allocation/Deallocation
//=============================================================================

int gpu_allocateCurveData(GPU_CurveData* data, int curveCount)
//
//  Purpose: Allocates EXPLICIT memory for curve metadata arrays
//  Memory Strategy: h_* (pinned host) + d_* (device)
//
{
    if (data == NULL || curveCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_CurveData));
    data->count = curveCount;

    size_t intSize = curveCount * sizeof(int);
    size_t doubleSize = curveCount * sizeof(double);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    // -------------------------------------------------------------------------
    // ALLOCATE HOST (CPU) MEMORY - Pinned
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaMallocHost((void**)&data->h_curveType, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_dataStart, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_dataCount, intSize));
    hostBytes += intSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_dxMin, doubleSize));
    hostBytes += doubleSize;

    // -------------------------------------------------------------------------
    // ALLOCATE DEVICE (GPU) MEMORY
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaMalloc((void**)&data->d_curveType, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_dataStart, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_dataCount, intSize));
    deviceBytes += intSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_dxMin, doubleSize));
    deviceBytes += doubleSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d curves (%.2f MB: %.2f MB host + %.2f MB device)\n",
           curveCount, hostMB + deviceMB, hostMB, deviceMB);

    return 0;
}

void gpu_freeCurveData(GPU_CurveData* data)
//
//  Purpose: Frees EXPLICIT GPU memory for curve metadata
//
{
    if (data == NULL) return;

    // Free host (pinned) memory
    if (data->h_curveType) cudaFreeHost(data->h_curveType);
    if (data->h_dataStart) cudaFreeHost(data->h_dataStart);
    if (data->h_dataCount) cudaFreeHost(data->h_dataCount);
    if (data->h_dxMin) cudaFreeHost(data->h_dxMin);

    // Free device memory
    CUDA_FREE_SAFE(data->d_curveType);
    CUDA_FREE_SAFE(data->d_dataStart);
    CUDA_FREE_SAFE(data->d_dataCount);
    CUDA_FREE_SAFE(data->d_dxMin);

    memset(data, 0, sizeof(GPU_CurveData));
}

int gpu_allocateCurvePoints(GPU_CurvePoints* data, int totalPoints)
//
//  Purpose: Allocates EXPLICIT memory for curve point data (x,y values)
//  Memory Strategy: h_* (pinned host) + d_* (device)
//
{
    if (data == NULL || totalPoints <= 0) return -1;

    memset(data, 0, sizeof(GPU_CurvePoints));
    data->totalPoints = totalPoints;

    size_t doubleSize = totalPoints * sizeof(double);
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    // -------------------------------------------------------------------------
    // ALLOCATE HOST (CPU) MEMORY - Pinned
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaMallocHost((void**)&data->h_xValues, doubleSize));
    hostBytes += doubleSize;
    CUDA_CHECK(cudaMallocHost((void**)&data->h_yValues, doubleSize));
    hostBytes += doubleSize;

    // -------------------------------------------------------------------------
    // ALLOCATE DEVICE (GPU) MEMORY
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaMalloc((void**)&data->d_xValues, doubleSize));
    deviceBytes += doubleSize;
    CUDA_CHECK(cudaMalloc((void**)&data->d_yValues, doubleSize));
    deviceBytes += doubleSize;

    double hostMB = (double)hostBytes / (1024.0 * 1024.0);
    double deviceMB = (double)deviceBytes / (1024.0 * 1024.0);
    printf("... Allocated EXPLICIT memory for %d curve points (%.2f MB: %.2f MB host + %.2f MB device)\n",
           totalPoints, hostMB + deviceMB, hostMB, deviceMB);

    return 0;
}

void gpu_freeCurvePoints(GPU_CurvePoints* data)
//
//  Purpose: Frees EXPLICIT GPU memory for curve point data
//
{
    if (data == NULL) return;

    // Free host (pinned) memory
    if (data->h_xValues) cudaFreeHost(data->h_xValues);
    if (data->h_yValues) cudaFreeHost(data->h_yValues);

    // Free device memory
    CUDA_FREE_SAFE(data->d_xValues);
    CUDA_FREE_SAFE(data->d_yValues);

    memset(data, 0, sizeof(GPU_CurvePoints));
}

int gpu_transferCurveDataToDevice(GPU_CurveData* data)
//
//  Purpose: Transfer curve metadata to GPU (one-time, at initialization)
//
{
    if (data == NULL || data->count == 0) return -1;

    size_t intSize = data->count * sizeof(int);
    size_t doubleSize = data->count * sizeof(double);

    CUDA_CHECK(cudaMemcpy(data->d_curveType, data->h_curveType, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_dataStart, data->h_dataStart, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_dataCount, data->h_dataCount, intSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_dxMin, data->h_dxMin, doubleSize, cudaMemcpyHostToDevice));

    return 0;
}

int gpu_transferCurvePointsToDevice(GPU_CurvePoints* data)
//
//  Purpose: Transfer curve point data to GPU (one-time, at initialization)
//
{
    if (data == NULL || data->totalPoints == 0) return -1;

    size_t doubleSize = data->totalPoints * sizeof(double);

    CUDA_CHECK(cudaMemcpy(data->d_xValues, data->h_xValues, doubleSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_yValues, data->h_yValues, doubleSize, cudaMemcpyHostToDevice));

    return 0;
}
