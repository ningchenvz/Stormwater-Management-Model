//-----------------------------------------------------------------------------
//   gpu_conduit_helpers.cuh
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   GPU device functions for conduit flow calculations.
//   Simplified implementation of dwflow_findConduitFlow for Phase 4 Stage 1.
//
//   Limitations in this version:
//   - Regular conduits only (no force mains, culverts)
//   - No flap gates
//   - No normal flow limitation
//   - No evaporation/seepage losses
//   - Simplified local loss calculations
//
//-----------------------------------------------------------------------------

#ifndef GPU_CONDUIT_HELPERS_CUH
#define GPU_CONDUIT_HELPERS_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "gpu_xsect_helpers.cuh"

//-----------------------------------------------------------------------------
// Constants
//-----------------------------------------------------------------------------
#define GPU_MAXVELOCITY     50.0     // Maximum velocity (ft/sec)
#define GPU_MIN_SURFAREA    12.566   // Minimum surface area (ft2)

__device__ inline double gpu_MAX(double a, double b)
{
    return (a > b) ? a : b;
}

__device__ inline double gpu_MIN(double a, double b)
{
    return (a < b) ? a : b;
}

// Flow classification
#define GPU_DRY             0
#define GPU_UP_DRY          1
#define GPU_DN_DRY          2
#define GPU_SUBCRITICAL     3
#define GPU_SUPCRITICAL     4

// Inertial damping
#define GPU_NO_DAMPING      0
#define GPU_PARTIAL_DAMPING 1
#define GPU_FULL_DAMPING    2

// Sign function
#define GPU_SGN(x) ((x) < 0 ? -1 : 1)

__device__ inline int gpu_classifyFlow(double y1, double y2)
{
    if (y1 <= GPU_FUDGE && y2 <= GPU_FUDGE) return GPU_DRY;
    if (y1 <= GPU_FUDGE) return GPU_UP_DRY;
    if (y2 <= GPU_FUDGE) return GPU_DN_DRY;
    return GPU_SUBCRITICAL;
}

__device__ void gpu_computeSurfaceAreas(
    GPU_Xsect* xsect,
    double length,
    double offset1,
    double offset2,
    double y1,
    double y2,
    int flowClass,
    int surchargeMethod,
    double crownCutoff,
    double* surfArea1_out,
    double* surfArea2_out)
//
//  Purpose: Approximates conduit surface area contribution at each node
//
{
    double surfArea1 = 0.0;
    double surfArea2 = 0.0;

    double flowDepth1 = gpu_MAX(y1, GPU_FUDGE);
    double flowDepth2 = gpu_MAX(y2, GPU_FUDGE);
    double flowDepthMid = 0.5 * (flowDepth1 + flowDepth2);
    flowDepthMid = gpu_MAX(flowDepthMid, GPU_FUDGE);

    double width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
    double width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
    double widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);

    switch (flowClass)
    {
        case GPU_DRY:
            surfArea1 = GPU_FUDGE * length * 0.5;
            surfArea2 = surfArea1;
            break;

        case GPU_UP_DRY:
            flowDepth1 = GPU_FUDGE;
            width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
            flowDepthMid = gpu_MAX(0.5 * (flowDepth1 + flowDepth2), GPU_FUDGE);
            widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
            surfArea2 = (widthMid + width2) * length * 0.25;
            if (offset1 <= 0.0)
            {
                surfArea1 = (width1 + widthMid) * length * 0.25;
            }
            else
            {
                surfArea1 = GPU_FUDGE * length * 0.25;
            }
            break;

        case GPU_DN_DRY:
            flowDepth2 = GPU_FUDGE;
            width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
            flowDepthMid = gpu_MAX(0.5 * (flowDepth1 + flowDepth2), GPU_FUDGE);
            widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
            surfArea1 = (width1 + widthMid) * length * 0.25;
            if (offset2 <= 0.0)
            {
                surfArea2 = (width2 + widthMid) * length * 0.25;
            }
            else
            {
                surfArea2 = GPU_FUDGE * length * 0.25;
            }
            break;

        default:
            // Treat subcritical/supercritical the same for surface area
            surfArea1 = (width1 + widthMid) * length * 0.25;
            surfArea2 = (widthMid + width2) * length * 0.25;
            break;
    }

    *surfArea1_out = surfArea1;
    *surfArea2_out = surfArea2;
}

//=============================================================================

__device__ double gpu_link_getFroude(
    double v,
    double y,
    double aFull)
//
//  Purpose: Computes Froude number
//  Input:   v = velocity (ft/sec)
//           y = flow depth (ft)
//           aFull = full cross-sectional area (ft2)
//  Returns: Froude number
//
{
    double hyd_depth;

    if (y <= 0.0 || aFull <= 0.0) return 0.0;

    // Hydraulic depth = area / top width
    // For simplification, approximate as y for now
    hyd_depth = y;

    if (hyd_depth > 0.0) {
        return fabs(v) / sqrt(GPU_GRAVITY * hyd_depth);
    }
    return 0.0;
}

//=============================================================================

__device__ void gpu_findConduitFlow_simplified(
    // Link indices and properties
    int j,                      // link index
    int k,                      // conduit index
    int n1,                     // upstream node
    int n2,                     // downstream node
    // Cross-section
    GPU_Xsect* xsect,
    // Node data
    double node1_newDepth,
    double node1_invertElev,
    double node2_newDepth,
    double node2_invertElev,
    // Link data
    double offset1,
    double offset2,
    double oldFlow,
    // Conduit data
    double barrels,
    double modLength,
    double roughFactor,
    double q1_last,             // flow from previous iteration
    double a2_old,              // area from previous time step
    // Iteration parameters
    int steps,
    double omega,
    double dt,
    int surchargeMethod,
    double crownCutoff,
    int inertDamping,
    // Outputs
    double* q_out,              // new flow (cfs per barrel)
    double* aMid_out,           // mid-conduit area
    double* yMid_out,           // mid-conduit depth
    double* dqdh_out,           // derivative of flow w.r.t. head
    double* froude_out,         // Froude number
    double* surfArea1_out,      // upstream surface area contribution
    double* surfArea2_out,      // downstream surface area contribution
    int* flowClass_out)         // updated flow class
//
//  Purpose: Simplified GPU version of dwflow_findConduitFlow
//  Note: This is Stage 1 implementation - regular conduits only
//
{
    double z1, z2;              // invert elevations
    double h1, h2;              // flow heads
    double y1, y2;              // flow depths
    double a1, a2;              // flow areas
    double r1;                  // upstream hydraulic radius
    double yMid, rMid, aMid;    // mid-stream values
    double aWtd, rWtd;          // weighted values
    double qLast;               // flow from previous iteration
    double qOld;                // flow from previous time step
    double aOld;                // area from previous time step
    double v;                   // velocity
    double rho;                 // upstream weighting factor
    double sigma;               // inertial damping factor
    double length;              // effective conduit length
    double wSlot;               // Preissmann slot width
    double dq1, dq2, dq3, dq4;  // terms in momentum equation
    double denom;               // denominator
    double q;                   // new flow
    double froude;              // Froude number
    int isFull;                 // TRUE if flowing full
    int flowClassLocal;         // Simplified flow classification

    // Get flow from last time step & previous iteration
    qOld = oldFlow / barrels;
    qLast = q1_last;

    // Get current heads at upstream and downstream ends
    z1 = node1_invertElev + offset1;
    z2 = node2_invertElev + offset2;
    h1 = node1_newDepth + node1_invertElev;
    h2 = node2_newDepth + node2_invertElev;
    h1 = gpu_MAX(h1, z1);
    h2 = gpu_MAX(h2, z2);

    // Get flow depths in conduit
    y1 = h1 - z1;
    y2 = h2 - z2;
    y1 = gpu_MAX(y1, GPU_FUDGE);
    y2 = gpu_MAX(y2, GPU_FUDGE);

    // Flow depths can't exceed full depth if slot not used
    if (surchargeMethod != GPU_SLOT_METHOD) {
        y1 = gpu_MIN(y1, xsect->yFull);
        y2 = gpu_MIN(y2, xsect->yFull);
    }

    flowClassLocal = gpu_classifyFlow(y1, y2);

    // Get area from previous time step
    aOld = a2_old;
    aOld = gpu_MAX(aOld, GPU_FUDGE);

    // Use Courant-modified length
    length = modLength;

    // Compute area at each end & hydraulic radius at upstream end
    wSlot = gpu_getSlotWidth(xsect, y1, surchargeMethod, crownCutoff);
    a1 = gpu_getArea(xsect, y1, wSlot);
    r1 = gpu_getHydRad(xsect, y1);

    wSlot = gpu_getSlotWidth(xsect, y2, surchargeMethod, crownCutoff);
    a2 = gpu_getArea(xsect, y2, wSlot);

    // Compute mid-stream values
    yMid = 0.5 * (y1 + y2);
    wSlot = gpu_getSlotWidth(xsect, yMid, surchargeMethod, crownCutoff);
    aMid = gpu_getArea(xsect, yMid, wSlot);
    rMid = gpu_getHydRad(xsect, yMid);

    // Check if flowing full
    isFull = (y1 >= xsect->yFull && y2 >= xsect->yFull) ? 1 : 0;

    // TEMPORARY: Completely disable dry condition check for debugging
    // This will allow flow computation even with very small/zero depths
    /*
    if (aMid <= GPU_FUDGE) {
        *q_out = 0.0;
        *aMid_out = 0.5 * (a1 + a2);
        *yMid_out = gpu_MIN(yMid, xsect->yFull);
        *dqdh_out = GPU_GRAVITY * dt * aMid / length * barrels;
        *froude_out = 0.0;
        return;
    }
    */

    // Compute velocity from last flow estimate
    v = qLast / aMid;
    if (fabs(v) > GPU_MAXVELOCITY) {
        v = GPU_MAXVELOCITY * GPU_SGN(qLast);
    }

    // Compute Froude number
    froude = gpu_link_getFroude(v, yMid, xsect->aFull);
    if (froude > 1.0 && flowClassLocal == GPU_SUBCRITICAL)
    {
        flowClassLocal = GPU_SUPCRITICAL;
    }

    // Find inertial damping factor (sigma)
    if (froude <= 0.5) sigma = 1.0;
    else if (froude >= 1.0) sigma = 0.0;
    else sigma = 2.0 * (1.0 - froude);

    // Get upstream-weighted area & hydraulic radius
    rho = 1.0;
    if (!isFull && qLast > 0.0 && h1 >= h2) rho = sigma;
    aWtd = a1 + (aMid - a1) * rho;
    rWtd = r1 + (rMid - r1) * rho;

    // Determine inertial damping
    if (inertDamping == GPU_NO_DAMPING) sigma = 1.0;
    else if (inertDamping == GPU_FULL_DAMPING) sigma = 0.0;

    // Use full damping for surcharged closed conduits
    if (isFull && !gpu_xsect_isOpen(xsect->type)) sigma = 0.0;

    // Compute terms of momentum equation:

    // 1. Friction slope term
    dq1 = dt * roughFactor / pow(rWtd, 1.33333) * fabs(v);

    // 2. Energy slope term
    dq2 = dt * GPU_GRAVITY * aWtd * (h2 - h1) / length;

    // 3 & 4. Inertial terms
    dq3 = 0.0;
    dq4 = 0.0;
    if (sigma > 0.0) {
        dq3 = 2.0 * v * (aMid - aOld) * sigma;
        dq4 = dt * v * v * (a2 - a1) / length * sigma;
    }

    // Combine terms to find new conduit flow
    denom = 1.0 + dq1;  // Simplified: no local losses in Stage 1
    q = (qOld - dq2 + dq3 + dq4) / denom;

    // Compute derivative of flow w.r.t. head
    *dqdh_out = 1.0 / denom * GPU_GRAVITY * dt * aWtd / length * barrels;

    // Apply under-relaxation
    if (steps > 0) {
        q = (1.0 - omega) * qLast + omega * q;
        // Don't allow flow direction change without first being zero
        if (q * qLast < 0.0) q = 0.001 * GPU_SGN(q);
    }

    // Don't allow flow out of a dry node
    if (q > GPU_FUDGE && node1_newDepth <= GPU_FUDGE) q = GPU_FUDGE;
    if (q < -GPU_FUDGE && node2_newDepth <= GPU_FUDGE) q = -GPU_FUDGE;

    // Set outputs
    *q_out = q;
    *aMid_out = aMid;
    *yMid_out = gpu_MIN(yMid, xsect->yFull);
    double surfArea1 = 0.0;
    double surfArea2 = 0.0;
    gpu_computeSurfaceAreas(
        xsect,
        length,
        offset1,
        offset2,
        y1,
        y2,
        flowClassLocal,
        surchargeMethod,
        crownCutoff,
        &surfArea1,
        &surfArea2);

    *froude_out = froude;
    *surfArea1_out = surfArea1;
    *surfArea2_out = surfArea2;
    *flowClass_out = flowClassLocal;
}

#endif // GPU_CONDUIT_HELPERS_CUH
