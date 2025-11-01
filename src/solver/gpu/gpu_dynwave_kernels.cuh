//-----------------------------------------------------------------------------
//   gpu_dynwave_kernels.cuh
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   CUDA device functions and helper functions for dynamic wave routing.
//   These are the GPU equivalents of helper functions in dynwave.c, node.c, etc.
//
//   Note: .cuh extension indicates CUDA header file with device code
//
//-----------------------------------------------------------------------------

#ifndef GPU_DYNWAVE_KERNELS_CUH
#define GPU_DYNWAVE_KERNELS_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "gpu_table_helpers.cuh"

//-----------------------------------------------------------------------------
// Constants (from dynwave.c)
//-----------------------------------------------------------------------------
#define GPU_OMEGA               0.5     // under-relaxation parameter
#define GPU_DEFAULT_SURFAREA    12.566  // Min. nodal surface area (~4 ft diam.)
#define GPU_FUDGE               0.0001  // Small depth adjustment

// Node types (from enums.h)
#define GPU_JUNCTION    0
#define GPU_OUTFALL     1
#define GPU_STORAGE     2
#define GPU_DIVIDER     3

// Surcharge method (from enums.h)
#define GPU_EXTRAN      0
#define GPU_SLOT        1

// Storage shapes (from enums.h)
#define GPU_STORAGE_TABULAR     0
#define GPU_STORAGE_FUNCTIONAL  1
#define GPU_STORAGE_CYLINDRICAL 2
#define GPU_STORAGE_CONICAL     3
#define GPU_STORAGE_PARABOLOID  4
#define GPU_STORAGE_PYRAMIDAL   5

//-----------------------------------------------------------------------------
// Helper Device Functions
//-----------------------------------------------------------------------------

__device__ inline double gpu_MAX(double a, double b)
{
    return (a > b) ? a : b;
}

__device__ inline double gpu_MIN(double a, double b)
{
    return (a < b) ? a : b;
}

//=============================================================================

__device__ double gpu_storage_getVolume(
    double depth,
    double fullDepth,
    double fullVolume,
    double storageA0,
    double storageA1,
    double storageA2,
    int storageShape,
    int storageCurve,
    double ucfLength,
    double ucfVolume,
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
//
//  Purpose: Computes storage node volume from depth using shape parameters
//
{
    if (depth <= 0.0) return 0.0;
    if (fullVolume > 0.0 && depth >= fullDepth) return fullVolume;

    switch (storageShape) {
        case GPU_STORAGE_TABULAR:
            if (storageCurve >= 0 && curves != NULL && points != NULL) {
                double volUser = gpu_table_getStorageVolume(
                    storageCurve,
                    depth * ucfLength,
                    curves->d_dataStart,
                    curves->d_dataCount,
                    points->d_xValues,
                    points->d_yValues);
                return volUser / ucfVolume;
            }
            return 0.0;

        case GPU_STORAGE_FUNCTIONAL:
        {
            double d = depth * ucfLength;
            double n = storageA2 + 1.0;
            double v = storageA0 * d;
            if (storageA1 != 0.0)
                v += storageA1 / n * pow(d, n);
            return v / ucfVolume;
        }

        case GPU_STORAGE_CYLINDRICAL:
        case GPU_STORAGE_CONICAL:
        case GPU_STORAGE_PARABOLOID:
        case GPU_STORAGE_PYRAMIDAL:
        {
            double d = depth * ucfLength;
            double v = d * (storageA0 + d * (storageA1 / 2.0 + d * storageA2 / 3.0));
            return v / ucfVolume;
        }

        default:
            return 0.0;
    }
}

//=============================================================================

__device__ double gpu_node_getVolume(
    int nodeType,
    double depth,
    double fullDepth,
    double fullVolume,
    double storageA0,
    double storageA1,
    double storageA2,
    int storageShape,
    int storageCurve,
    double ucfLength,
    double ucfVolume,
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
//
//  Purpose: Computes volume stored at a node from its water depth
//
{
    if (nodeType == GPU_STORAGE) {
        return gpu_storage_getVolume(depth, fullDepth, fullVolume,
                                     storageA0, storageA1, storageA2,
                                     storageShape, storageCurve,
                                     ucfLength, ucfVolume,
                                     curves, points);
    }

    if (fullDepth > 0.0)
        return fullVolume * (depth / fullDepth);

    return 0.0;
}

//=============================================================================

__device__ double gpu_getFloodedDepth(
    int canPond,
    double dV,
    double yNew,
    double yMax,
    double dt,
    double fullVolume,
    double oldVolume,
    double* overflow,
    double* newVolume)
//
//  Purpose: Computes depth, volume and overflow for a flooded node
//           (mirrors CPU getFloodedDepth)
{
    if (canPond == 0) {
        double oflow = dV / dt;
        if (oflow < GPU_FUDGE) oflow = 0.0;
        *overflow = oflow;
        *newVolume = fullVolume;
        return yMax;
    }

    double volume = oldVolume + dV;
    if (volume < fullVolume) volume = fullVolume;
    double reference = gpu_MAX(oldVolume, fullVolume);
    double excess = volume - reference;
    double oflow = excess / dt;
    if (oflow < GPU_FUDGE) oflow = 0.0;

    *overflow = oflow;
    *newVolume = volume;
    return yNew;
}

//=============================================================================

__device__ void gpu_setNodeDepth(
    int i,
    double dt,
    int allowPonding,
    int surchargeMethod,
    double minSurfArea,
    int steps,
    double omega,
    // Node data
    int nodeType,
    double invertElev,
    double fullDepth,
    double surDepth,
    double pondedArea,
    double crownElev,
    double oldDepth,
    double oldVolume,
    double oldNetInflow,
    double inflow,
    double outflow,
    double fullVolume,
    int degree,
    double storageA0,
    double storageA1,
    double storageA2,
    int storageShape,
    int storageCurve,
    // Xnode data
    double newSurfArea,
    double oldSurfArea_in,
    double sumdqdh,
    // Unit conversions & lookup tables
    double ucfLength,
    double ucfVolume,
    const GPU_CurveData* curves,
    const GPU_CurvePoints* curvePoints,
    // Previous iteration value
    double newDepth_last,
    // Outputs
    double* newDepth_out,
    double* newVolume_out,
    double* overflow_out,
    double* oldSurfArea_out,
    double* dYdT_out)
//
//  Purpose: Sets depth at non-outfall node after current time step
//  Note: This is a direct port of setNodeDepth() from dynwave.c
//        Simplified for regular nodes (storage nodes not fully supported yet)
//
{
    int canPond, isPonded, isSurcharged;
    double dQ, dV, dy, yMax, yOld, yLast, yNew, yCrown, surfArea;
    double denom, corr, f;

    // --- see if node can pond water above it
    canPond = (allowPonding && pondedArea > 0.0);
    isPonded = (canPond && newDepth_last > fullDepth);

    // --- initialize values
    yCrown = crownElev - invertElev;
    yOld = oldDepth;
    yLast = newDepth_last;
    *overflow_out = 0.0;
    surfArea = newSurfArea;
    surfArea = gpu_MAX(surfArea, minSurfArea);

    // --- determine average net flow volume into node over the time step
    dQ = inflow - outflow;
    dV = 0.5 * (oldNetInflow + dQ) * dt;

    // --- determine if node is EXTRAN surcharged
    isSurcharged = 0;
    if (surchargeMethod == GPU_EXTRAN) {
        if (isPonded) {
            isSurcharged = 0;
        }
        else if (nodeType == GPU_STORAGE) {
            isSurcharged = (surDepth > 0.0 && yLast > fullDepth);
        }
        else {
            isSurcharged = (yCrown > 0.0 && yLast > yCrown);
        }
    }

    // --- if node not surcharged, base depth change on surface area
    if (!isSurcharged) {
        dy = dV / surfArea;
        yNew = yOld + dy;

        // --- save non-ponded surface area
        if (!isPonded) *oldSurfArea_out = surfArea;
        else *oldSurfArea_out = oldSurfArea_in;

        // --- apply under-relaxation
        if (steps > 0) {
            yNew = (1.0 - omega) * yLast + omega * yNew;
        }

        // --- don't allow ponded node to drop much below full depth
        if (isPonded && yNew < fullDepth) {
            yNew = fullDepth - GPU_FUDGE;
        }
    }
    // --- if node surcharged, base depth change on dqdh
    else {
        // --- correction factor for upstream terminal nodes
        corr = 1.0;
        if (degree < 0) corr = 0.6;

        // --- allow surface area to influence dqdh if depth close to crown
        denom = sumdqdh;
        if (yLast < 1.25 * yCrown) {
            f = (yLast - yCrown) / yCrown;
            denom += (oldSurfArea_in / dt - sumdqdh) * exp(-15.0 * f);
        }

        // --- compute new depth estimate
        if (denom == 0.0) dy = 0.0;
        else dy = corr * dQ / denom;

        yNew = yLast + dy;
        if (yNew < yCrown) yNew = yCrown - GPU_FUDGE;

        // --- don't allow newly ponded node to rise much above full depth
        if (canPond && yNew > fullDepth) {
            yNew = fullDepth + GPU_FUDGE;
        }

        *oldSurfArea_out = oldSurfArea_in;
    }

    // --- depth cannot be negative
    if (yNew < 0.0) yNew = 0.0;

    // --- determine max non-flooded depth
    yMax = fullDepth;
    if (!canPond) yMax += surDepth;

    // --- handle flooded conditions
    if (yNew > yMax) {
        yNew = gpu_getFloodedDepth(
            canPond,
            dV,
            yNew,
            yMax,
            dt,
            fullVolume,
            oldVolume,
            overflow_out,
            newVolume_out);
    }
    else {
        *newVolume_out = gpu_node_getVolume(
            nodeType,
            yNew,
            fullDepth,
            fullVolume,
            storageA0,
            storageA1,
            storageA2,
            storageShape,
            storageCurve,
            ucfLength,
            ucfVolume,
            curves,
            curvePoints);
    }

    // --- compute rate of depth change
    *dYdT_out = fabs(yNew - yOld) / dt;

    // --- save new depth
    *newDepth_out = yNew;
}

#endif // GPU_DYNWAVE_KERNELS_CUH
