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
#include "gpu_link_helpers.cuh"

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

// Node types (from enums.h)
#define GPU_JUNCTION        0
#define GPU_OUTFALL         1
#define GPU_STORAGE         2
#define GPU_DIVIDER         3

// Flow classification (matches enums.h)
#define GPU_DRY             0
#define GPU_UP_DRY          1
#define GPU_DN_DRY          2
#define GPU_SUBCRITICAL     3
#define GPU_SUPCRITICAL     4
#define GPU_UP_CRITICAL     5  // NEW: Upstream end at critical depth
#define GPU_DN_CRITICAL     6  // NEW: Downstream end at critical depth

// Inertial damping
#define GPU_NO_DAMPING      0
#define GPU_PARTIAL_DAMPING 1
#define GPU_FULL_DAMPING    2

// Sign function
#define GPU_SGN(x) ((x) < 0 ? -1 : 1)

// DEPRECATED: Replaced by gpu_getFlowClass() which includes critical flow classifications
// __device__ inline int gpu_classifyFlow(double y1, double y2)
// {
//     if (y1 <= GPU_FUDGE && y2 <= GPU_FUDGE) return GPU_DRY;
//     if (y1 <= GPU_FUDGE) return GPU_UP_DRY;
//     if (y2 <= GPU_FUDGE) return GPU_DN_DRY;
//     return GPU_SUBCRITICAL;
// }

//=============================================================================
// GPU Port: getFlowClass()
// From: src/solver/dwflow.c:297-413
//=============================================================================
__device__ int gpu_getFlowClass(
    double q,
    double h1,
    double h2,
    double y1,
    double y2,
    double offset1,
    double offset2,
    int node1Type,
    int node2Type,
    double node1NewDepth,
    double node2NewDepth,
    double node1InvertElev,
    double node2InvertElev,
    GPU_Xsect* xsect,
    double conduitBeta,
    double conduitQmax,
    double* yC_out,      // critical depth
    double* yN_out,      // normal depth
    double* fasnh_out)   // fraction between norm & crit
//
//  Input:   q  = current conduit flow (cfs)
//           h1 = head at upstream end of conduit (ft)
//           h2 = head at downstream end of conduit (ft)
//           y1 = upstream flow depth (ft)
//           y2 = downstream flow depth (ft)
//           offset1/2 = offsets of conduit inverts (ft)
//           node1/2Type = node types (GPU_OUTFALL, etc.)
//           node1/2NewDepth = current node depths (ft)
//           node1/2InvertElev = node invert elevations (ft)
//           xsect = cross-section data
//           conduitBeta, conduitQmax = conduit parameters
//  Output:  returns flow classification code
//           *yC_out = critical flow depth (ft)
//           *yN_out = normal flow depth (ft)
//           *fasnh_out = fraction between norm. & crit. depth
//  Purpose: determines flow class for a conduit based on depths at each end
//
{
    int flowClass;
    double ycMin, ycMax;
    double z1, z2;

    // Get upstream & downstream conduit invert offsets
    z1 = offset1;
    z2 = offset2;

    // Base offset of an outfall conduit on outfall's depth
    if (node1Type == GPU_OUTFALL) z1 = gpu_MAX(0.0, (z1 - node1NewDepth));
    if (node2Type == GPU_OUTFALL) z2 = gpu_MAX(0.0, (z2 - node2NewDepth));

    // Default class is SUBCRITICAL
    flowClass = GPU_SUBCRITICAL;
    *fasnh_out = 1.0;
    *yC_out = 0.0;
    *yN_out = 0.0;

    // Case where both ends of conduit are wet
    if (y1 > GPU_FUDGE && y2 > GPU_FUDGE)
    {
        if (q < 0.0)  // Flow reversal
        {
            // Upstream end at critical depth if flow depth is
            // below conduit's critical depth and an upstream offset exists
            if (z1 > 0.0)
            {
                *yN_out = gpu_link_getYnorm(xsect, fabs(q), conduitBeta, conduitQmax);
                *yC_out = gpu_link_getYcrit(xsect, fabs(q));
                ycMin = gpu_MIN(*yN_out, *yC_out);
                if (y1 < ycMin) flowClass = GPU_UP_CRITICAL;
            }
        }
        else  // Normal direction flow
        {
            // Downstream end at smaller of critical and normal depth
            // if downstream flow depth below this and a downstream offset exists
            if (z2 > 0.0)
            {
                *yN_out = gpu_link_getYnorm(xsect, fabs(q), conduitBeta, conduitQmax);
                *yC_out = gpu_link_getYcrit(xsect, fabs(q));
                ycMin = gpu_MIN(*yN_out, *yC_out);
                ycMax = gpu_MAX(*yN_out, *yC_out);

                if (y2 < ycMin)
                {
                    flowClass = GPU_DN_CRITICAL;
                }
                else if (y2 < ycMax)
                {
                    // Interpolate fasnh between normal and critical
                    if (ycMax - ycMin < GPU_FUDGE)
                        *fasnh_out = 0.0;
                    else
                        *fasnh_out = (ycMax - y2) / (ycMax - ycMin);
                }
            }
        }
    }

    // Case where no flow at either end of conduit
    else if (y1 <= GPU_FUDGE && y2 <= GPU_FUDGE)
    {
        flowClass = GPU_DRY;
    }

    // Case where downstream end of pipe is wet, upstream dry
    else if (y2 > GPU_FUDGE)
    {
        // Flow classification is UP_DRY if downstream head <
        // invert of upstream end of conduit
        if (h2 < node1InvertElev + offset1)
        {
            flowClass = GPU_UP_DRY;
        }
        // Otherwise, the downstream head will be >= upstream
        // conduit invert creating a flow reversal and upstream end
        // should be at critical depth, providing that an upstream
        // offset exists (otherwise subcritical condition is maintained)
        else if (z1 > 0.0)
        {
            *yN_out = gpu_link_getYnorm(xsect, fabs(q), conduitBeta, conduitQmax);
            *yC_out = gpu_link_getYcrit(xsect, fabs(q));
            flowClass = GPU_UP_CRITICAL;
        }
    }

    // Case where upstream end of pipe is wet, downstream dry
    else
    {
        // Flow classification is DN_DRY if upstream head <
        // invert of downstream end of conduit
        if (h1 < node2InvertElev + offset2)
        {
            flowClass = GPU_DN_DRY;
        }
        // Otherwise flow at downstream end should be at critical depth
        // providing that a downstream offset exists (otherwise
        // subcritical condition is maintained)
        else if (z2 > 0.0)
        {
            *yN_out = gpu_link_getYnorm(xsect, fabs(q), conduitBeta, conduitQmax);
            *yC_out = gpu_link_getYcrit(xsect, fabs(q));
            flowClass = GPU_DN_CRITICAL;
        }
    }

    return flowClass;
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
    double fasnh,           // Fraction between normal and critical depth
    double criticalDepth,   // Critical depth from flow classification
    double normalDepth,     // Normal depth from flow classification
    double* surfArea1_out,
    double* surfArea2_out)
//
//  Purpose: Approximates conduit surface area contribution at each node
//  Port of: src/solver/dwflow.c:findSurfArea() (lines 452-550)
//
{
    double surfArea1 = 0.0;
    double surfArea2 = 0.0;

    double flowDepth1 = gpu_MAX(y1, GPU_FUDGE);
    double flowDepth2 = gpu_MAX(y2, GPU_FUDGE);
    double flowDepthMid;
    double width1, width2, widthMid;

    switch (flowClass)
    {
        case GPU_DRY:
            // Both ends dry - minimal surface area
            surfArea1 = GPU_FUDGE * length * 0.5;
            surfArea2 = surfArea1;
            break;

        case GPU_UP_DRY:
            // Upstream dry, downstream wet
            flowDepth1 = GPU_FUDGE;
            width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
            flowDepthMid = gpu_MAX(0.5 * (flowDepth1 + flowDepth2), GPU_FUDGE);
            widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
            width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
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
            // Downstream dry, upstream wet
            flowDepth2 = GPU_FUDGE;
            width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
            flowDepthMid = gpu_MAX(0.5 * (flowDepth1 + flowDepth2), GPU_FUDGE);
            widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
            width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
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

        case GPU_SUBCRITICAL:
            // Normal subcritical flow - both ends contribute
            // Apply fasnh scaling to downstream contribution
            flowDepthMid = 0.5 * (flowDepth1 + flowDepth2);
            if (flowDepthMid < GPU_FUDGE) flowDepthMid = GPU_FUDGE;
            width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
            width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
            widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
            surfArea1 = (width1 + widthMid) * length * 0.25;
            surfArea2 = (widthMid + width2) * length * 0.25 * fasnh;  // Apply fasnh scaling!
            break;

        case GPU_UP_CRITICAL:
            // Upstream at critical depth - only downstream contributes
            // Use critical/normal depth for upstream end
            flowDepth1 = criticalDepth;
            if (normalDepth < criticalDepth) flowDepth1 = normalDepth;
            flowDepth1 = gpu_MAX(flowDepth1, GPU_FUDGE);

            flowDepthMid = 0.5 * (flowDepth1 + flowDepth2);
            if (flowDepthMid < GPU_FUDGE) flowDepthMid = GPU_FUDGE;

            width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
            widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);

            surfArea1 = 0.0;                                    // No upstream contribution!
            surfArea2 = (widthMid + width2) * length * 0.5;    // Half-length at downstream!
            break;

        case GPU_DN_CRITICAL:
            // Downstream at critical depth - only upstream contributes
            // Use critical/normal depth for downstream end
            flowDepth2 = criticalDepth;
            if (normalDepth < criticalDepth) flowDepth2 = normalDepth;
            flowDepth2 = gpu_MAX(flowDepth2, GPU_FUDGE);

            flowDepthMid = 0.5 * (flowDepth1 + flowDepth2);
            if (flowDepthMid < GPU_FUDGE) flowDepthMid = GPU_FUDGE;

            width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
            widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);

            surfArea1 = (width1 + widthMid) * length * 0.5;    // Half-length at upstream!
            surfArea2 = 0.0;                                    // No downstream contribution!
            break;

        default:
            // Fallback for any unexpected flow class
            flowDepthMid = 0.5 * (flowDepth1 + flowDepth2);
            if (flowDepthMid < GPU_FUDGE) flowDepthMid = GPU_FUDGE;
            width1 = gpu_getWidth(xsect, flowDepth1, surchargeMethod, crownCutoff);
            width2 = gpu_getWidth(xsect, flowDepth2, surchargeMethod, crownCutoff);
            widthMid = gpu_getWidth(xsect, flowDepthMid, surchargeMethod, crownCutoff);
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
    int node1_type,             // Node type (for flow classification)
    int node2_type,             // Node type (for flow classification)
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
    double conduitBeta,         // Discharge factor (for normal depth)
    double conduitQmax,         // Maximum flow (for normal depth)
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

    // Classify flow using full logic from CPU dwflow.c
    double yC = 0.0, yN = 0.0, fasnh = 1.0;
    flowClassLocal = gpu_getFlowClass(
        qLast, h1, h2, y1, y2,
        offset1, offset2,
        node1_type, node2_type,
        node1_newDepth, node2_newDepth,
        node1_invertElev, node2_invertElev,
        xsect, conduitBeta, conduitQmax,
        &yC, &yN, &fasnh);

    // Override with supercritical if Froude > 1.0 (for inertial damping)
    int flowClassForDamping = flowClassLocal;
    if (froude > 1.0 && flowClassLocal == GPU_SUBCRITICAL)
    {
        flowClassForDamping = GPU_SUPCRITICAL;
    }

    // Find inertial damping factor (sigma) - use Froude-based class
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

    // Compute surface areas with full flow classification logic
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
        fasnh,      // Pass fasnh scaling factor
        yC,         // Pass critical depth
        yN,         // Pass normal depth
        &surfArea1,
        &surfArea2);

    *froude_out = froude;
    *surfArea1_out = surfArea1;
    *surfArea2_out = surfArea2;
    *flowClass_out = flowClassLocal;

#ifdef GPU_DEBUG_SURF
    // DEBUG: Log surface areas for first few conduits to diagnose Session18 issues
    // Focus on early timesteps only
    if (steps <= 3) {
        printf("GPU_SURF[step=%d link=%d node1=%d node2=%d]: flowClass=%d fasnh=%.3f yC=%.3f yN=%.3f\n",
               steps, j, n1, n2, flowClassLocal, fasnh, yC, yN);
        printf("  y1=%.3f y2=%.3f → surfArea1=%.3f surfArea2=%.3f (length=%.1f)\n",
               y1, y2, surfArea1, surfArea2, length);
    }
#endif
}

#endif // GPU_CONDUIT_HELPERS_CUH
