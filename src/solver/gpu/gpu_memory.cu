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
//  Purpose: Allocates GPU memory for node data arrays
//  Input:   data = pointer to GPU_NodeData structure
//           nodeCount = number of nodes
//  Returns: 0 if successful, CUDA error code otherwise
//
{
    if (data == NULL || nodeCount <= 0) return -1;

    // Initialize structure
    memset(data, 0, sizeof(GPU_NodeData));
    data->count = nodeCount;

    size_t intSize = nodeCount * sizeof(int);
    size_t doubleSize = nodeCount * sizeof(double);
    size_t charSize = nodeCount * sizeof(char);

    // Allocate static properties
    CUDA_ALLOC_MANAGED((void**)&data->type, intSize);
    CUDA_ALLOC_MANAGED((void**)&data->invertElev, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->fullDepth, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->surDepth, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->pondedArea, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->crownElev, doubleSize);

    // Allocate dynamic state
    CUDA_ALLOC_MANAGED((void**)&data->oldDepth, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->newDepth, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->oldVolume, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->newVolume, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->fullVolume, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->oldNetInflow, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->inflow, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->outflow, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->overflow, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->degree, intSize);

    // Allocate extended data
    CUDA_ALLOC_MANAGED((void**)&data->converged, charSize);
    CUDA_ALLOC_MANAGED((void**)&data->newSurfArea, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->oldSurfArea, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->sumdqdh, doubleSize);

    printf("... Allocated GPU memory for %d nodes (%.2f MB)\n",
           nodeCount, (6 * doubleSize + 11 * doubleSize + intSize + charSize) / (1024.0 * 1024.0));

    return 0;
}

//=============================================================================

void gpu_freeNodeData(GPU_NodeData* data)
//
//  Purpose: Frees GPU memory for node data
//
{
    if (data == NULL) return;

    CUDA_FREE_SAFE(data->type);
    CUDA_FREE_SAFE(data->invertElev);
    CUDA_FREE_SAFE(data->fullDepth);
    CUDA_FREE_SAFE(data->surDepth);
    CUDA_FREE_SAFE(data->pondedArea);
    CUDA_FREE_SAFE(data->crownElev);
    CUDA_FREE_SAFE(data->oldDepth);
    CUDA_FREE_SAFE(data->newDepth);
    CUDA_FREE_SAFE(data->oldVolume);
    CUDA_FREE_SAFE(data->newVolume);
    CUDA_FREE_SAFE(data->fullVolume);
    CUDA_FREE_SAFE(data->oldNetInflow);
    CUDA_FREE_SAFE(data->inflow);
    CUDA_FREE_SAFE(data->outflow);
    CUDA_FREE_SAFE(data->overflow);
    CUDA_FREE_SAFE(data->degree);
    CUDA_FREE_SAFE(data->converged);
    CUDA_FREE_SAFE(data->newSurfArea);
    CUDA_FREE_SAFE(data->oldSurfArea);
    CUDA_FREE_SAFE(data->sumdqdh);

    data->count = 0;
}

//=============================================================================
// Link Data Allocation/Deallocation
//=============================================================================

int gpu_allocateLinkData(GPU_LinkData* data, int linkCount)
//
//  Purpose: Allocates GPU memory for link data arrays
//
{
    if (data == NULL || linkCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_LinkData));
    data->count = linkCount;

    size_t intSize = linkCount * sizeof(int);
    size_t doubleSize = linkCount * sizeof(double);
    size_t charSize = linkCount * sizeof(char);
    size_t scharSize = linkCount * sizeof(signed char);

    // Allocate topology
    CUDA_ALLOC_MANAGED((void**)&data->type, intSize);
    CUDA_ALLOC_MANAGED((void**)&data->subIndex, intSize);
    CUDA_ALLOC_MANAGED((void**)&data->node1, intSize);
    CUDA_ALLOC_MANAGED((void**)&data->node2, intSize);

    // Allocate static properties
    CUDA_ALLOC_MANAGED((void**)&data->offset1, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->offset2, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->qFull, doubleSize);

    // Allocate dynamic state
    CUDA_ALLOC_MANAGED((void**)&data->oldFlow, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->newFlow, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->oldDepth, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->newDepth, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->oldVolume, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->newVolume, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->surfArea1, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->surfArea2, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->dqdh, doubleSize);

    // Allocate flags
    CUDA_ALLOC_MANAGED((void**)&data->bypassed, charSize);
    CUDA_ALLOC_MANAGED((void**)&data->direction, scharSize);

    printf("... Allocated GPU memory for %d links (%.2f MB)\n",
           linkCount, (4 * intSize + 12 * doubleSize + charSize + scharSize) / (1024.0 * 1024.0));

    return 0;
}

//=============================================================================

void gpu_freeLinkData(GPU_LinkData* data)
//
//  Purpose: Frees GPU memory for link data
//
{
    if (data == NULL) return;

    CUDA_FREE_SAFE(data->type);
    CUDA_FREE_SAFE(data->subIndex);
    CUDA_FREE_SAFE(data->node1);
    CUDA_FREE_SAFE(data->node2);
    CUDA_FREE_SAFE(data->offset1);
    CUDA_FREE_SAFE(data->offset2);
    CUDA_FREE_SAFE(data->qFull);
    CUDA_FREE_SAFE(data->oldFlow);
    CUDA_FREE_SAFE(data->newFlow);
    CUDA_FREE_SAFE(data->oldDepth);
    CUDA_FREE_SAFE(data->newDepth);
    CUDA_FREE_SAFE(data->oldVolume);
    CUDA_FREE_SAFE(data->newVolume);
    CUDA_FREE_SAFE(data->surfArea1);
    CUDA_FREE_SAFE(data->surfArea2);
    CUDA_FREE_SAFE(data->dqdh);
    CUDA_FREE_SAFE(data->bypassed);
    CUDA_FREE_SAFE(data->direction);

    data->count = 0;
}

//=============================================================================
// Conduit Data Allocation/Deallocation
//=============================================================================

int gpu_allocateConduitData(GPU_ConduitData* data, int conduitCount)
//
//  Purpose: Allocates GPU memory for conduit data arrays
//
{
    if (data == NULL || conduitCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_ConduitData));
    data->count = conduitCount;

    size_t doubleSize = conduitCount * sizeof(double);
    size_t charSize = conduitCount * sizeof(char);

    // Allocate static properties
    CUDA_ALLOC_MANAGED((void**)&data->length, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->modLength, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->roughness, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->roughFactor, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->slope, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->barrels, charSize);

    // Allocate dynamic wave parameters
    CUDA_ALLOC_MANAGED((void**)&data->beta, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->qMax, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->a1, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->a2, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->q1, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->q2, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->q1Old, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->q2Old, doubleSize);

    // Allocate flags
    CUDA_ALLOC_MANAGED((void**)&data->capacityLimited, charSize);
    CUDA_ALLOC_MANAGED((void**)&data->superCritical, charSize);

    printf("... Allocated GPU memory for %d conduits (%.2f MB)\n",
           conduitCount, (14 * doubleSize + 3 * charSize) / (1024.0 * 1024.0));

    return 0;
}

//=============================================================================

void gpu_freeConduitData(GPU_ConduitData* data)
//
//  Purpose: Frees GPU memory for conduit data
//
{
    if (data == NULL) return;

    CUDA_FREE_SAFE(data->length);
    CUDA_FREE_SAFE(data->modLength);
    CUDA_FREE_SAFE(data->roughness);
    CUDA_FREE_SAFE(data->roughFactor);
    CUDA_FREE_SAFE(data->slope);
    CUDA_FREE_SAFE(data->barrels);
    CUDA_FREE_SAFE(data->beta);
    CUDA_FREE_SAFE(data->qMax);
    CUDA_FREE_SAFE(data->a1);
    CUDA_FREE_SAFE(data->a2);
    CUDA_FREE_SAFE(data->q1);
    CUDA_FREE_SAFE(data->q2);
    CUDA_FREE_SAFE(data->q1Old);
    CUDA_FREE_SAFE(data->q2Old);
    CUDA_FREE_SAFE(data->capacityLimited);
    CUDA_FREE_SAFE(data->superCritical);

    data->count = 0;
}

//=============================================================================
// Cross-Section Data Allocation/Deallocation
//=============================================================================

int gpu_allocateXsectData(GPU_XsectData* data, int xsectCount)
//
//  Purpose: Allocates GPU memory for cross-section data arrays
//
{
    if (data == NULL || xsectCount <= 0) return -1;

    memset(data, 0, sizeof(GPU_XsectData));
    data->count = xsectCount;

    size_t intSize = xsectCount * sizeof(int);
    size_t doubleSize = xsectCount * sizeof(double);

    CUDA_ALLOC_MANAGED((void**)&data->type, intSize);
    CUDA_ALLOC_MANAGED((void**)&data->aFull, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->rFull, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->wMax, doubleSize);
    CUDA_ALLOC_MANAGED((void**)&data->yFull, doubleSize);

    printf("... Allocated GPU memory for %d cross-sections (%.2f MB)\n",
           xsectCount, (intSize + 4 * doubleSize) / (1024.0 * 1024.0));

    return 0;
}

//=============================================================================

void gpu_freeXsectData(GPU_XsectData* data)
//
//  Purpose: Frees GPU memory for cross-section data
//
{
    if (data == NULL) return;

    CUDA_FREE_SAFE(data->type);
    CUDA_FREE_SAFE(data->aFull);
    CUDA_FREE_SAFE(data->rFull);
    CUDA_FREE_SAFE(data->wMax);
    CUDA_FREE_SAFE(data->yFull);

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
//  Purpose: Transfers node data from CPU arrays to GPU SoA structure
//  Input:   gpuData = pre-allocated GPU_NodeData structure
//           Arrays = pointers to CPU data arrays
//           count = number of nodes
//  Returns: 0 if successful, error code otherwise
//
//  Note: This function assumes gpuData is already allocated via
//        gpu_allocateNodeData(). It copies data from CPU arrays
//        into the GPU SoA structure. With unified memory, this is
//        just memcpy. With discrete GPUs, would need cudaMemcpy.
//
{
    if (gpuData == NULL || count <= 0) return -1;
    if (gpuData->count != count) return -1;

    size_t intSize = count * sizeof(int);
    size_t doubleSize = count * sizeof(double);

    // Copy static properties
    if (g_gpuConfig.unifiedMemory) {
        // Unified memory: simple memcpy
        memcpy(gpuData->type, type, intSize);
        memcpy(gpuData->invertElev, invertElev, doubleSize);
        memcpy(gpuData->fullDepth, fullDepth, doubleSize);
        memcpy(gpuData->surDepth, surDepth, doubleSize);
        memcpy(gpuData->pondedArea, pondedArea, doubleSize);
        memcpy(gpuData->crownElev, crownElev, doubleSize);
        memcpy(gpuData->oldDepth, oldDepth, doubleSize);
        memcpy(gpuData->newDepth, newDepth, doubleSize);
        memcpy(gpuData->oldVolume, oldVolume, doubleSize);
        memcpy(gpuData->newVolume, newVolume, doubleSize);
        memcpy(gpuData->oldNetInflow, oldNetInflow, doubleSize);
        memcpy(gpuData->inflow, inflow, doubleSize);
        memcpy(gpuData->outflow, outflow, doubleSize);
    } else {
        // Discrete GPU: explicit copy to device
        CUDA_CHECK(cudaMemcpy(gpuData->type, type, intSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->invertElev, invertElev, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->fullDepth, fullDepth, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->surDepth, surDepth, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->pondedArea, pondedArea, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->crownElev, crownElev, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->oldDepth, oldDepth, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->newDepth, newDepth, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->oldVolume, oldVolume, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->newVolume, newVolume, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->oldNetInflow, oldNetInflow, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->inflow, inflow, doubleSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpuData->outflow, outflow, doubleSize, cudaMemcpyHostToDevice));
    }

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
//  Purpose: Retrieves computed node data from GPU back to CPU arrays
//  Input:   gpuData = GPU_NodeData structure with computed results
//           Arrays = pointers to CPU arrays to fill
//           count = number of nodes
//  Returns: 0 if successful, error code otherwise
//
{
    if (gpuData == NULL || count <= 0) return -1;
    if (gpuData->count != count) return -1;

    size_t doubleSize = count * sizeof(double);

    // Copy results back
    if (g_gpuConfig.unifiedMemory) {
        // Unified memory: simple memcpy
        memcpy(newDepth, gpuData->newDepth, doubleSize);
        memcpy(newVolume, gpuData->newVolume, doubleSize);
        memcpy(overflow, gpuData->overflow, doubleSize);
    } else {
        // Discrete GPU: explicit copy from device
        CUDA_CHECK(cudaMemcpy(newDepth, gpuData->newDepth, doubleSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(newVolume, gpuData->newVolume, doubleSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(overflow, gpuData->overflow, doubleSize, cudaMemcpyDeviceToHost));
    }

    return 0;
}
