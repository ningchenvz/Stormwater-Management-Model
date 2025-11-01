//-----------------------------------------------------------------------------
//   gpu_table_helpers.cuh
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/27/2025
//
//   CUDA device functions for table/curve lookups used by pump and outlet links
//
//   Design Notes:
//   - TTable (linked list) converted to GPU-friendly SoA arrays
//   - Linear interpolation for table lookups
//   - Slope calculations for dqdh computations
//
//-----------------------------------------------------------------------------

#ifndef GPU_TABLE_HELPERS_CUH
#define GPU_TABLE_HELPERS_CUH

#include <cuda_runtime.h>
#include "gpu_structures.h"

//=============================================================================
// Device Helper: Linear Interpolation
//=============================================================================
__device__ __forceinline__
double gpu_table_interpolate(double x, double x1, double y1, double x2, double y2)
{
    // Linear interpolation between (x1,y1) and (x2,y2)
    if (x2 == x1) return y1;
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

//=============================================================================
// Device Helper: Table Lookup (with interpolation)
//=============================================================================
// Equivalent to CPU table_lookup()
// Returns interpolated y-value for given x
//
__device__ __forceinline__
double gpu_table_lookup(
    int curveIdx,
    double x,
    const int* d_dataStart,
    const int* d_dataCount,
    const double* d_xValues,
    const double* d_yValues)
{
    int start = d_dataStart[curveIdx];
    int count = d_dataCount[curveIdx];

    if (count == 0) return 0.0;
    if (count == 1) return d_yValues[start];

    // Find bracketing points via binary search
    int left = 0;
    int right = count - 1;

    // Handle out-of-range cases
    if (x <= d_xValues[start + left]) {
        return d_yValues[start + left];
    }
    if (x >= d_xValues[start + right]) {
        return d_yValues[start + right];
    }

    // Binary search for bracketing interval
    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (x < d_xValues[start + mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    // Interpolate between points
    double x1 = d_xValues[start + left];
    double y1 = d_yValues[start + left];
    double x2 = d_xValues[start + right];
    double y2 = d_yValues[start + right];

    return gpu_table_interpolate(x, x1, y1, x2, y2);
}

//=============================================================================
// Device Helper: Interval Lookup (step function, no interpolation)
//=============================================================================
// Equivalent to CPU table_intervalLookup()
// Returns y-value for interval containing x (step function)
//
__device__ __forceinline__
double gpu_table_intervalLookup(
    int curveIdx,
    double x,
    const int* d_dataStart,
    const int* d_dataCount,
    const double* d_xValues,
    const double* d_yValues)
{
    int start = d_dataStart[curveIdx];
    int count = d_dataCount[curveIdx];

    if (count == 0) return 0.0;
    if (count == 1) return d_yValues[start];

    // Return first y-value if x is below range
    if (x <= d_xValues[start]) {
        return d_yValues[start];
    }

    // Find the interval containing x
    for (int i = 1; i < count; i++) {
        if (x < d_xValues[start + i]) {
            return d_yValues[start + i - 1];
        }
    }

    // x is beyond last interval, return last y-value
    return d_yValues[start + count - 1];
}

//=============================================================================
// Device Helper: Get Slope at Point
//=============================================================================
// Equivalent to CPU table_getSlope()
// Returns dy/dx at given x (for dqdh calculation)
//
__device__ __forceinline__
double gpu_table_getSlope(
    int curveIdx,
    double x,
    const int* d_dataStart,
    const int* d_dataCount,
    const double* d_xValues,
    const double* d_yValues)
{
    int start = d_dataStart[curveIdx];
    int count = d_dataCount[curveIdx];

    if (count < 2) return 0.0;

    // Handle out-of-range cases - use slope of first/last segment
    if (x <= d_xValues[start]) {
        double dx = d_xValues[start + 1] - d_xValues[start];
        if (dx == 0.0) return 0.0;
        return (d_yValues[start + 1] - d_yValues[start]) / dx;
    }

    if (x >= d_xValues[start + count - 1]) {
        double dx = d_xValues[start + count - 1] - d_xValues[start + count - 2];
        if (dx == 0.0) return 0.0;
        return (d_yValues[start + count - 1] - d_yValues[start + count - 2]) / dx;
    }

    // Find bracketing interval
    for (int i = 0; i < count - 1; i++) {
        if (x >= d_xValues[start + i] && x <= d_xValues[start + i + 1]) {
            double dx = d_xValues[start + i + 1] - d_xValues[start + i];
            if (dx == 0.0) return 0.0;
            return (d_yValues[start + i + 1] - d_yValues[start + i]) / dx;
        }
    }

    return 0.0;
}

//=============================================================================
// Device Helper: Check if Point is in Curve Range
//=============================================================================
__device__ __forceinline__
bool gpu_table_inRange(
    int curveIdx,
    double x,
    double* xMin,
    double* xMax,
    const int* d_dataStart,
    const int* d_dataCount,
    const double* d_xValues)
{
    int start = d_dataStart[curveIdx];
    int count = d_dataCount[curveIdx];

    if (count == 0) {
        *xMin = 0.0;
        *xMax = 0.0;
        return false;
    }

    *xMin = d_xValues[start];
    *xMax = d_xValues[start + count - 1];

    return (x >= *xMin && x <= *xMax);
}

#endif // GPU_TABLE_HELPERS_CUH
