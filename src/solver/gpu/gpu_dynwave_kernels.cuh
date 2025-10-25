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

__device__ double gpu_node_getVolume(
    int nodeType,
    double depth,
    double fullDepth,
    double fullVolume)
//
//  Purpose: Computes volume stored at a node from its water depth
//  Input:   nodeType = type of node
//           depth = water depth (ft)
//           fullDepth = depth when node is full (ft)
//           fullVolume = volume when node is full (ft3)
//  Returns: Volume of water at node (ft3)
//
//  Note: Simplified version for non-storage nodes
//        Storage nodes require curve lookup (not implemented yet)
//
{
    // For now, only handle non-storage nodes (linear interpolation)
    if (nodeType == GPU_STORAGE) {
        // TODO: Implement storage_getVolume equivalent
        // For now, use linear approximation
        if (fullDepth > 0.0)
            return fullVolume * (depth / fullDepth);
        else
            return 0.0;
    }
    else {
        // Regular nodes: linear relationship
        if (fullDepth > 0.0)
            return fullVolume * (depth / fullDepth);
        else
            return 0.0;
    }
}

//=============================================================================

__device__ double gpu_getFloodedDepth(
    int i,
    int canPond,
    double dV,
    double yNew,
    double yMax,
    double dt,
    double pondedArea,
    double fullVolume,
    double* overflow,      // output
    double* newVolume)     // output
//
//  Purpose: Computes depth, volume and overflow for a flooded node
//  Input:   i = node index
//           canPond = TRUE if water can pond over node
//           dV = change in volume over time step (ft3)
//           yNew = current depth at node (ft)
//           yMax = max. depth at node before ponding (ft)
//           dt = time step (sec)
//           pondedArea = surface area for ponding (ft2)
//           fullVolume = volume when full (ft3)
//  Output:  overflow = overflow rate (cfs)
//           newVolume = new volume (ft3)
//  Returns: Adjusted depth at node when flooded (ft)
//
{
    if (canPond == 0) {
        // No ponding allowed - overflow
        *overflow = dV / dt;
        *newVolume = fullVolume;
        return yMax;
    }
    else {
        // Ponding allowed
        double dY = (yNew - yMax);
        if (pondedArea > 0.0) {
            dY = dV / pondedArea;
        }
        yNew = yMax + dY;
        *overflow = 0.0;
        *newVolume = fullVolume + dV;
        return yNew;
    }
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
    double oldNetInflow,
    double inflow,
    double outflow,
    double fullVolume,
    int degree,
    // Xnode data
    double newSurfArea,
    double oldSurfArea_in,
    double sumdqdh,
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
        yNew = gpu_getFloodedDepth(i, canPond, dV, yNew, yMax, dt,
                                    pondedArea, fullVolume,
                                    overflow_out, newVolume_out);
    }
    else {
        *newVolume_out = gpu_node_getVolume(nodeType, yNew, fullDepth, fullVolume);
    }

    // --- compute rate of depth change
    *dYdT_out = fabs(yNew - yOld) / dt;

    // --- save new depth
    *newDepth_out = yNew;
}

#endif // GPU_DYNWAVE_KERNELS_CUH
