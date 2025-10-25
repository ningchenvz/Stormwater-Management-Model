//-----------------------------------------------------------------------------
//   gpu_config.h
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   GPU configuration, feature detection, and common macros.
//
//-----------------------------------------------------------------------------

#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

// Check if CUDA is available
#ifdef BUILD_GPU
    #define GPU_AVAILABLE 1
#else
    #define GPU_AVAILABLE 0
#endif

// CUDA error checking macro
#ifdef BUILD_GPU
    #include <cuda_runtime.h>
    #include <stdio.h>

    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                        __FILE__, __LINE__, cudaGetErrorString(err)); \
                return err; \
            } \
        } while (0)

    #define CUDA_CHECK_LAST_ERROR() \
        do { \
            cudaError_t err = cudaGetLastError(); \
            if (err != cudaSuccess) { \
                fprintf(stderr, "CUDA Kernel Error at %s:%d: %s\n", \
                        __FILE__, __LINE__, cudaGetErrorString(err)); \
                return err; \
            } \
        } while (0)
#else
    #define CUDA_CHECK(call) 0
    #define CUDA_CHECK_LAST_ERROR() 0
#endif

// GPU kernel launch parameters
#define DEFAULT_BLOCK_SIZE 256
#define DEFAULT_THREADS_PER_BLOCK 256
#define MAX_THREADS_PER_BLOCK 1024

// Minimum model size to use GPU (overhead not worth it for small models)
#define GPU_MIN_LINKS_DEFAULT 500
#define GPU_MIN_NODES_DEFAULT 500

// GPU memory alignment
#define GPU_MEMORY_ALIGNMENT 256

// Compute grid dimensions
#define GRID_SIZE(n, block_size) (((n) + (block_size) - 1) / (block_size))

// GPU feature flags
typedef struct {
    int available;              // GPU is available
    int enabled;                // User enabled GPU acceleration
    int deviceCount;            // Number of CUDA devices
    int activeDevice;           // Active device ID
    int computeCapability;      // Compute capability (major * 10 + minor)
    size_t totalMemory;         // Total GPU memory (bytes)
    size_t availableMemory;     // Available GPU memory (bytes)
    int multiProcessorCount;    // Number of SMs
    int maxThreadsPerBlock;     // Max threads per block
    int warpSize;               // Warp size (typically 32)
    int minLinksForGPU;         // Minimum links to use GPU
    int minNodesForGPU;         // Minimum nodes to use GPU
    int unifiedMemory;          // Supports unified memory (managed memory)
    int concurrentManagedAccess;// Can access managed memory concurrently from CPU/GPU
} GPUConfig;

// Global GPU configuration (defined in gpu_manager.cu)
extern GPUConfig g_gpuConfig;

// GPU initialization and cleanup
#ifdef BUILD_GPU
    int gpu_initialize(void);
    void gpu_cleanup(void);
    int gpu_isAvailable(void);
    int gpu_isEnabled(void);
    void gpu_setEnabled(int enabled);
    void gpu_printInfo(void);

    // GPU test functions
    int gpu_runAllTests(void);
    int gpu_test_vectorAdd(int n);
    int gpu_test_nodeStructure(int nodeCount);
    int gpu_test_massBalance(int nodeCount);
#else
    static inline int gpu_initialize(void) { return 0; }
    static inline void gpu_cleanup(void) {}
    static inline int gpu_isAvailable(void) { return 0; }
    static inline int gpu_isEnabled(void) { return 0; }
    static inline void gpu_setEnabled(int enabled) { (void)enabled; }
    static inline void gpu_printInfo(void) {}
    static inline int gpu_runAllTests(void) { return 0; }
#endif

#ifdef __cplusplus
}
#endif

#endif // GPU_CONFIG_H
