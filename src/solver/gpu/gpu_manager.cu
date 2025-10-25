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
#include "gpu_config.h"

// Global GPU configuration
GPUConfig g_gpuConfig = {0};

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

    // Set default thresholds
    g_gpuConfig.minLinksForGPU = GPU_MIN_LINKS_DEFAULT;
    g_gpuConfig.minNodesForGPU = GPU_MIN_NODES_DEFAULT;

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

    return cudaSuccess;
}

//=============================================================================

void gpu_cleanup(void)
//
//  Purpose: Cleanup GPU resources
//
{
    if (g_gpuConfig.available) {
        cudaDeviceReset();
        g_gpuConfig.available = 0;
        g_gpuConfig.enabled = 0;
    }
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
}
