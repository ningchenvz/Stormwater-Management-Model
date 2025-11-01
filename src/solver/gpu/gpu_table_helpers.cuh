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
#include <math.h>
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
// Device Helper: Table Lookup with Linear Extrapolation (table_lookupEx)
//=============================================================================
__device__ __forceinline__
double gpu_table_lookupEx(
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

    double x1 = d_xValues[start];
    double y1 = d_yValues[start];

    if (x <= x1) {
        if (x1 > 0.0) return (x / x1) * y1;
        return y1;
    }

    double slope = 0.0;
    for (int idx = 1; idx < count; ++idx) {
        double x2 = d_xValues[start + idx];
        double y2 = d_yValues[start + idx];
        if (x2 != x1) slope = (y2 - y1) / (x2 - x1);
        if (x <= x2) {
            return gpu_table_interpolate(x, x1, y1, x2, y2);
        }
        x1 = x2;
        y1 = y2;
    }

    if (slope < 0.0) slope = 0.0;
    return y1 + slope * (x - x1);
}

//=============================================================================
// Device Helper: Storage Volume Integration (table_getStorageVolume)
//=============================================================================
__device__ __forceinline__
double gpu_table_getStorageVolume(
    int curveIdx,
    double depth,
    const int* d_dataStart,
    const int* d_dataCount,
    const double* d_xValues,
    const double* d_yValues)
{
    int start = d_dataStart[curveIdx];
    int count = d_dataCount[curveIdx];

    if (count == 0) return 0.0;

    double v = 0.0;
    double x1 = d_xValues[start];
    double a1 = d_yValues[start];

    if (depth <= x1) {
        if (x1 < 1.0e-6) return 0.0;
        return (a1 / x1) * depth * depth * 0.5;
    }

    double dx = 0.0;
    double dy = 0.0;

    for (int idx = 1; idx < count; ++idx) {
        double x2 = d_xValues[start + idx];
        double a2 = d_yValues[start + idx];
        if (x2 >= depth) {
            double aInterp = gpu_table_interpolate(depth, x1, a1, x2, a2);
            return v + (a1 + aInterp) * (depth - x1) * 0.5;
        }
        dx = x2 - x1;
        dy = a2 - a1;
        v += (a1 + a2) * dx * 0.5;
        x1 = x2;
        a1 = a2;
    }

    if (dx > 1.0e-6) {
        double s = dy / dx;
        double a = a1 + s * (depth - x1);
        if (a < 0.0 && fabs(s) > 1.0e-12) {
            v = v - a1 * a1 / (2.0 * s);
        }
        else {
            v = v + (a1 + a) * (depth - x1) * 0.5;
        }
    }

    return v;
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
