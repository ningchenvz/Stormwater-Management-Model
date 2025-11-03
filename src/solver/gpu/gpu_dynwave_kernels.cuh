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
// External device variables
//-----------------------------------------------------------------------------
extern __device__ int g_depthRoutingStepCounter;

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

__device__ double gpu_storage_getSurfArea(
    double depth,
    double storageA0,
    double storageA1,
    double storageA2,
    int storageShape,
    int storageCurve,
    double ucfLength,
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
//
//  Purpose: Computes storage node surface area from depth using shape parameters
//           Mirrors CPU storage_getSurfArea() in node.c
//
{
    if (depth <= 0.0) return 0.0;

    double area = 0.0;
    double d = depth * ucfLength;

    switch (storageShape) {
        case GPU_STORAGE_TABULAR:
            // Debug: check for NULL pointers
            if (storageCurve == 0 && depth > 0.1) {
                static int null_check_done = 0;
                if (!null_check_done) {
                    printf("GPU_SURF_DEBUG curve=%d curves=%p points=%p\n",
                           storageCurve, curves, points);
                    null_check_done = 1;
                }
            }

            if (storageCurve >= 0 && curves != NULL && points != NULL) {
                // Debug: print curve info for curveIdx=0 and 10
                bool debug_this = ((storageCurve == 0 || storageCurve == 10) && depth > 0.1);

                if (debug_this) {
                    int start = curves->d_dataStart[storageCurve];
                    int count = curves->d_dataCount[storageCurve];
                    printf("\nGPU_SURF_DEBUG curve=%d depth_internal=%.6f ucfLength=%.6f\n",
                           storageCurve, depth, ucfLength);
                    printf("  d_user = depth * ucfLength = %.6f * %.6f = %.6f\n",
                           depth, ucfLength, d);
                    printf("  Curve points (x=depth_user, y=area_user): ");
                    for (int j = 0; j < count; j++) {
                        printf("(%.3f,%.1f) ", points->d_xValues[start+j], points->d_yValues[start+j]);
                    }
                    printf("\n");
                }

                // Use table lookup for surface area
                area = gpu_table_lookupEx(
                    storageCurve,
                    d,
                    curves->d_dataStart,
                    curves->d_dataCount,
                    points->d_xValues,
                    points->d_yValues);

                if (debug_this) {
                    double area_internal = area / (ucfLength * ucfLength);
                    printf("  area_user (from lookup) = %.2f\n", area);
                    printf("  area_internal = area_user / (ucfLength²) = %.2f / %.6f = %.2f ft²\n",
                           area, ucfLength * ucfLength, area_internal);
                }
            }
            break;

        case GPU_STORAGE_FUNCTIONAL:
            // area = a0 + a1 * d^a2
            area = storageA0 + storageA1 * pow(d, storageA2);
            break;

        case GPU_STORAGE_CYLINDRICAL:
        case GPU_STORAGE_CONICAL:
        case GPU_STORAGE_PARABOLOID:
        case GPU_STORAGE_PYRAMIDAL:
            // area = a0 + a1*d + a2*d^2
            area = storageA0 + d * (storageA1 + d * storageA2);
            break;

        default:
            return 0.0;
    }

    // Convert from user units to internal units (ft^2)
    return area / (ucfLength * ucfLength);
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

__device__ double gpu_node_getSurfArea(
    int nodeType,
    double depth,
    double storageA0,
    double storageA1,
    double storageA2,
    int storageShape,
    int storageCurve,
    double ucfLength,
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
//
//  Purpose: Computes surface area at a node from its water depth
//           Mirrors CPU node_getSurfArea() in node.c
//
{
    if (nodeType == GPU_STORAGE) {
        return gpu_storage_getSurfArea(depth, storageA0, storageA1, storageA2,
                                        storageShape, storageCurve,
                                        ucfLength, curves, points);
    }

    // All other node types (junctions, outfalls, dividers) return 0.0
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

    // DEBUG: Trace Node 1 (J2) for routing steps 1-3, all iterations
    // This is the problematic node that's not converging
    if (i == 1 && g_depthRoutingStepCounter >= 1 && g_depthRoutingStepCounter <= 3) {
        const char* typeStr = (nodeType == GPU_JUNCTION) ? "JUNC" : (nodeType == GPU_STORAGE) ? "STOR" : "OTHR";
        printf("NODE%d_%s[routingStep=%d iter=%d]: yOld=%.6f yLast=%.6f\n",
               i, typeStr, g_depthRoutingStepCounter, steps, yOld, yLast);
        printf("  inflow=%.6f outflow=%.6f oldNetInflow=%.6f dt=%.3f surfArea=%.3f newSurfArea=%.6f minSurfArea=%.6f\n",
               inflow, outflow, oldNetInflow, dt, surfArea, newSurfArea, minSurfArea);
    }

    // DEBUG: Trace STOR-10 (node 926) surface area accumulation (first 3 routing steps)
    #ifdef GPU_DEBUG_SURF
    if (i == 926 && steps <= 3 && nodeType == GPU_STORAGE) {
        printf("STOR10_DEPTH[step=%d]: newSurfArea(conduits)=%.3f minSurfArea=%.3f surfArea(used)=%.3f\n",
               steps, newSurfArea, minSurfArea, surfArea);
        printf("  inflow=%.6f outflow=%.6f dQ=%.6f oldNet=%.6f\n",
               inflow, outflow, inflow - outflow, oldNetInflow);
        printf("  oldVol=%.3f yOld=%.6f yLast=%.6f fullDepth=%.3f dt=%.6f\n",
               oldVolume, yOld, yLast, fullDepth, dt);
        printf("  dV_calc=(0.5*(%.6f + %.6f)*%.6f)=%.6f → dy=%.6f\n",
               oldNetInflow, inflow - outflow, dt,
               0.5 * (oldNetInflow + (inflow - outflow)) * dt,
               0.5 * (oldNetInflow + (inflow - outflow)) * dt / surfArea);
    }
    #endif

    // --- determine average net flow volume into node over the time step
    dQ = inflow - outflow;
    dV = 0.5 * (oldNetInflow + dQ) * dt;

    // DEBUG: Comprehensive Node 1 (J2) input logging for routing steps 1-3, iterations 0-2
    if (i == 1 && g_depthRoutingStepCounter >= 1 && g_depthRoutingStepCounter <= 3 && steps <= 2) {
        printf("GPU_NODE1_INPUTS[routingStep=%d iter=%d]:\n", g_depthRoutingStepCounter, steps);
        printf("  oldDepth=%.6f oldVolume=%.6f oldNetInflow=%.6f\n", yOld, oldVolume, oldNetInflow);
        printf("  inflow=%.6f outflow=%.6f dQ=%.6f\n", inflow, outflow, dQ);
        printf("  dV=%.6f dt=%.6f surfArea=%.6f\n", dV, dt, surfArea);
        printf("  newDepth_last=%.6f fullDepth=%.6f\n", yLast, fullDepth);
    }

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
            #ifdef GPU_DEBUG_SURF
            if (i == 926 && steps <= 3) {
                printf("  RELAX[i=%d step=%d]: yLast=%.6f yNew_raw=%.6f omega=%.3f\n",
                       i, steps, yLast, yNew, omega);
            }
            #endif
            yNew = (1.0 - omega) * yLast + omega * yNew;
            #ifdef GPU_DEBUG_SURF
            if (i == 926 && steps <= 3) {
                printf("  RELAX[i=%d step=%d]: yNew_relaxed=%.6f (moved %.6f)\n",
                       i, steps, yNew, yNew - yLast);
            }
            #endif
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
