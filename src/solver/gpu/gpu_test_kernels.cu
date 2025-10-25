//-----------------------------------------------------------------------------
//   gpu_test_kernels.cu
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   Simple test kernels to verify GPU pipeline functionality.
//   These kernels perform basic operations to validate memory allocation,
//   data transfer, and kernel execution before implementing complex
//   dynamic wave routing kernels.
//
//-----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "gpu_config.h"
#include "gpu_structures.h"

//=============================================================================
// Test Kernel 1: Vector Addition
//=============================================================================

__global__ void test_vectorAdd(double* a, double* b, double* c, int n)
//
//  Purpose: Simple vector addition kernel for testing GPU pipeline
//  Input:   a, b = input vectors
//           n = vector length
//  Output:  c = output vector (c[i] = a[i] + b[i])
//
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

//=============================================================================
// Test Kernel 2: Node Depth Initialization
//=============================================================================

__global__ void test_initNodeDepths(GPU_NodeData* nodes)
//
//  Purpose: Initialize node depths to test SoA structure access
//  Input:   nodes = pointer to GPU_NodeData structure
//  Output:  Sets newDepth = oldDepth for all nodes
//
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nodes->count) {
        nodes->newDepth[i] = nodes->oldDepth[i];
        nodes->newVolume[i] = nodes->oldVolume[i];
        nodes->converged[i] = 0;  // FALSE
    }
}

//=============================================================================
// Test Kernel 3: Link Flow Initialization
//=============================================================================

__global__ void test_initLinkFlows(GPU_LinkData* links)
//
//  Purpose: Initialize link flows to test SoA structure access
//  Input:   links = pointer to GPU_LinkData structure
//  Output:  Sets newFlow = oldFlow for all links
//
{
    int i = blockIdx.x * blockDim.x * threadIdx.x;
    if (i < links->count) {
        links->newFlow[i] = links->oldFlow[i];
        links->newDepth[i] = links->oldDepth[i];
        links->newVolume[i] = links->oldVolume[i];
    }
}

//=============================================================================
// Test Kernel 4: Compute Simple Node Balance
//=============================================================================

__global__ void test_computeNodeBalance(GPU_NodeData* nodes, double dt)
//
//  Purpose: Simple mass balance calculation for nodes
//  Input:   nodes = pointer to GPU_NodeData structure
//           dt = time step (sec)
//  Output:  Updates newVolume based on inflow/outflow
//
//  Formula: newVolume = oldVolume + (inflow - outflow) * dt
//
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nodes->count) {
        double dV = (nodes->inflow[i] - nodes->outflow[i]) * dt;
        nodes->newVolume[i] = nodes->oldVolume[i] + dV;

        // Simple depth calculation (assuming constant surface area for testing)
        double surfArea = nodes->newSurfArea[i];
        if (surfArea > 0.0) {
            double dDepth = dV / surfArea;
            nodes->newDepth[i] = nodes->oldDepth[i] + dDepth;
        }
    }
}

//=============================================================================
// Test Kernel 5: Convergence Check (Reduction)
//=============================================================================

__global__ void test_checkConvergence(GPU_NodeData* nodes, double tol, int* convergedCount)
//
//  Purpose: Check if node depths have converged
//  Input:   nodes = pointer to GPU_NodeData structure
//           tol = convergence tolerance (ft)
//  Output:  convergedCount = number of converged nodes (atomic increment)
//
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nodes->count) {
        double depthChange = fabs(nodes->newDepth[i] - nodes->oldDepth[i]);

        if (depthChange < tol) {
            nodes->converged[i] = 1;  // TRUE
            atomicAdd(convergedCount, 1);
        } else {
            nodes->converged[i] = 0;  // FALSE
        }
    }
}

//=============================================================================
// Host Functions to Launch Test Kernels
//=============================================================================

extern "C" {

int gpu_test_vectorAdd(int n)
//
//  Purpose: Tests basic GPU memory allocation and kernel execution
//  Returns: 0 if successful, error code otherwise
//
{
    double *a, *b, *c;
    double *h_a, *h_b, *h_c;
    int errors = 0;

    printf("\n=== Test 1: Vector Addition ===\n");

    // Allocate host memory
    h_a = (double*)malloc(n * sizeof(double));
    h_b = (double*)malloc(n * sizeof(double));
    h_c = (double*)malloc(n * sizeof(double));

    // Initialize input
    for (int i = 0; i < n; i++) {
        h_a[i] = (double)i;
        h_b[i] = (double)(i * 2);
    }

    // Allocate device memory
    if (g_gpuConfig.unifiedMemory) {
        CUDA_CHECK(cudaMallocManaged(&a, n * sizeof(double)));
        CUDA_CHECK(cudaMallocManaged(&b, n * sizeof(double)));
        CUDA_CHECK(cudaMallocManaged(&c, n * sizeof(double)));

        // Copy data
        memcpy(a, h_a, n * sizeof(double));
        memcpy(b, h_b, n * sizeof(double));
    } else {
        CUDA_CHECK(cudaMalloc(&a, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&b, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c, n * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(a, h_a, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Launch kernel
    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = GRID_SIZE(n, blockSize);

    printf("Launching kernel: gridSize=%d, blockSize=%d\n", gridSize, blockSize);
    test_vectorAdd<<<gridSize, blockSize>>>(a, b, c, n);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    if (!g_gpuConfig.unifiedMemory) {
        CUDA_CHECK(cudaMemcpy(h_c, c, n * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        memcpy(h_c, c, n * sizeof(double));
    }

    // Verify results
    for (int i = 0; i < n; i++) {
        double expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-10) {
            printf("ERROR at index %d: expected %.2f, got %.2f\n", i, expected, h_c[i]);
            errors++;
            if (errors > 10) break;
        }
    }

    if (errors == 0) {
        printf("✓ Vector addition test PASSED\n");
    } else {
        printf("✗ Vector addition test FAILED (%d errors)\n", errors);
    }

    // Cleanup
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(h_a);
    free(h_b);
    free(h_c);

    return (errors == 0) ? 0 : -1;
}

//=============================================================================

int gpu_test_nodeStructure(int nodeCount)
//
//  Purpose: Tests GPU_NodeData structure allocation and kernel access
//  Returns: 0 if successful, error code otherwise
//
{
    GPU_NodeData nodes;
    int errors = 0;

    printf("\n=== Test 2: Node Structure Access ===\n");

    // Allocate GPU memory
    if (gpu_allocateNodeData(&nodes, nodeCount) != 0) {
        printf("✗ Failed to allocate node data\n");
        return -1;
    }

    // Initialize test data on CPU (works with unified memory)
    for (int i = 0; i < nodeCount; i++) {
        nodes.oldDepth[i] = (double)i * 0.5;
        nodes.oldVolume[i] = (double)i * 10.0;
        nodes.inflow[i] = 5.0;
        nodes.outflow[i] = 3.0;
        nodes.newSurfArea[i] = 100.0;
    }

    // If discrete GPU, copy to device
    if (!g_gpuConfig.unifiedMemory) {
        // Would need explicit memcpy here for discrete GPU
    }

    // Launch test kernel
    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = GRID_SIZE(nodeCount, blockSize);

    printf("Launching initNodeDepths kernel\n");
    test_initNodeDepths<<<gridSize, blockSize>>>(&nodes);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    for (int i = 0; i < nodeCount; i++) {
        if (fabs(nodes.newDepth[i] - nodes.oldDepth[i]) > 1e-10) {
            printf("ERROR at node %d: newDepth=%.2f, oldDepth=%.2f\n",
                   i, nodes.newDepth[i], nodes.oldDepth[i]);
            errors++;
            if (errors > 10) break;
        }
    }

    if (errors == 0) {
        printf("✓ Node structure test PASSED\n");
    } else {
        printf("✗ Node structure test FAILED (%d errors)\n", errors);
    }

    // Cleanup
    gpu_freeNodeData(&nodes);

    return (errors == 0) ? 0 : -1;
}

//=============================================================================

int gpu_test_massBalance(int nodeCount)
//
//  Purpose: Tests simple mass balance calculation on GPU
//  Returns: 0 if successful, error code otherwise
//
{
    GPU_NodeData nodes;
    double dt = 1.0;  // 1 second time step
    int errors = 0;

    printf("\n=== Test 3: Mass Balance Calculation ===\n");

    // Allocate and initialize
    if (gpu_allocateNodeData(&nodes, nodeCount) != 0) {
        printf("✗ Failed to allocate node data\n");
        return -1;
    }

    for (int i = 0; i < nodeCount; i++) {
        nodes.oldDepth[i] = 1.0;
        nodes.oldVolume[i] = 100.0;
        nodes.inflow[i] = 10.0;   // 10 cfs inflow
        nodes.outflow[i] = 5.0;   // 5 cfs outflow
        nodes.newSurfArea[i] = 100.0;  // 100 ft2 surface area
    }

    // Launch kernel
    int blockSize = DEFAULT_BLOCK_SIZE;
    int gridSize = GRID_SIZE(nodeCount, blockSize);

    test_computeNodeBalance<<<gridSize, blockSize>>>(&nodes, dt);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify: newVolume should be oldVolume + (10-5)*1 = oldVolume + 5
    for (int i = 0; i < nodeCount; i++) {
        double expected = nodes.oldVolume[i] + 5.0;
        if (fabs(nodes.newVolume[i] - expected) > 1e-6) {
            printf("ERROR at node %d: expected volume=%.2f, got=%.2f\n",
                   i, expected, nodes.newVolume[i]);
            errors++;
            if (errors > 10) break;
        }
    }

    if (errors == 0) {
        printf("✓ Mass balance test PASSED\n");
    } else {
        printf("✗ Mass balance test FAILED (%d errors)\n", errors);
    }

    gpu_freeNodeData(&nodes);
    return (errors == 0) ? 0 : -1;
}

//=============================================================================

int gpu_runAllTests()
//
//  Purpose: Runs all GPU test kernels
//  Returns: 0 if all tests pass, -1 otherwise
//
{
    int result = 0;

    printf("\n====================================================\n");
    printf("  SWMM-GPU Test Suite\n");
    printf("====================================================\n");

    result |= gpu_test_vectorAdd(10000);
    result |= gpu_test_nodeStructure(1000);
    result |= gpu_test_massBalance(1000);

    printf("\n====================================================\n");
    if (result == 0) {
        printf("  ✓ All tests PASSED\n");
    } else {
        printf("  ✗ Some tests FAILED\n");
    }
    printf("====================================================\n");

    return result;
}

} // extern "C"
