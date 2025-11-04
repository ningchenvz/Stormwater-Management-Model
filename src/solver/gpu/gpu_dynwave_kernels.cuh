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
                double depthUser = depth * ucfLength;
                double volUser = gpu_table_getStorageVolume(
                    storageCurve,
                    depthUser,
                    curves->d_dataStart,
                    curves->d_dataCount,
                    points->d_xValues,
                    points->d_yValues);
                double volInternal = volUser / ucfVolume;

                // Log for curve 10 (STOR-10)
                if (storageCurve == 10 && depth > 0.0) {
                    printf("    gpu_storage_getVolume[curve=%d]: depth_internal=%.6f → depth_user=%.6f → vol_user=%.6f → vol_internal=%.6f\n",
                           storageCurve, depth, depthUser, volUser, volInternal);
                    printf("      CHECK: volInternal isnan=%d isinf=%d\n", isnan(volInternal), isinf(volInternal));
                }

                return volInternal;
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

    // DEBUG: Trace problem nodes (247, 928) and STOR-10 (926) - NaN INVESTIGATION
    bool debugThis = false; // Disabled for clean testing
    // bool debugThis = ((i == 247 || i == 928 || i == 926) && g_depthRoutingStepCounter <= 5);

    if (debugThis) {
        const char* nodeTypeStr = (nodeType == 0) ? "JUNC" : (nodeType == 2) ? "STOR" : "OTHER";
        printf("\n=== NODE_%d_%s[routeStep=%d iter=%d] ===\n", i, nodeTypeStr, g_depthRoutingStepCounter, steps);
        printf("  INPUTS: oldVol=%.6f oldDepth=%.6f yLast=%.6f\n", oldVolume, yOld, yLast);
        printf("  INPUTS: inflow=%.6f outflow=%.6f oldNetInflow=%.6f\n", inflow, outflow, oldNetInflow);
        printf("  INPUTS: newSurfArea=%.6f minSurfArea=%.6f → surfArea(used)=%.6f\n",
               newSurfArea, minSurfArea, surfArea);
        printf("  INPUTS: fullDepth=%.3f fullVolume=%.3f dt=%.6f\n", fullDepth, fullVolume, dt);
        printf("  CHECK isnan: oldDepth=%d yLast=%d inflow=%d outflow=%d surfArea=%d\n",
               isnan(yOld), isnan(yLast), isnan(inflow), isnan(outflow), isnan(surfArea));
        if (nodeType == GPU_STORAGE) {
            printf("  STORAGE: shape=%d curve=%d a0=%.3f a1=%.3f a2=%.3f\n",
                   storageShape, storageCurve, storageA0, storageA1, storageA2);
        }
    }

    // --- determine average net flow volume into node over the time step
    dQ = inflow - outflow;
    dV = 0.5 * (oldNetInflow + dQ) * dt;

    // DEBUG: Log volume integration for ALL storage nodes during first 3 routing steps, iteration 0 only
    if (nodeType == GPU_STORAGE && g_depthRoutingStepCounter <= 3 && steps == 0) {
        printf("VOL_INT[step=%d iter=%d node=%d]: oldNetIn=%.6f in=%.6f out=%.6f dQ=%.6f dV=%.6f surfArea=%.6f oldVol=%.6f newVol=%.6f\n",
               g_depthRoutingStepCounter, steps, i, oldNetInflow, inflow, outflow, dQ, dV, surfArea, oldVolume, oldVolume + dV);
    }

    // DEBUG: CRITICAL - WW-1002 (node 830) and WW-1003 (node 831) detailed balance
    // These are the storage nodes draining at half the correct rate
    if ((i == 830 || i == 831) && g_depthRoutingStepCounter >= 1 && g_depthRoutingStepCounter <= 5) {
        const char* name = (i == 830) ? "WW-1002" : "WW-1003";
        printf("GPU_STORAGE_%s[step=%d iter=%d]: oldNetIn=%.9f oldVol=%.9f oldDepth=%.9f\n",
               name, g_depthRoutingStepCounter, steps, oldNetInflow, oldVolume, yOld);
        printf("  FLOWS: in=%.9f out=%.9f dQ=%.9f\n", inflow, outflow, dQ);
        printf("  INTEGRATION: dV=%.9f = 0.5*(%.9f + %.9f)*%.6f, dt=%.6f\n",
               dV, oldNetInflow, dQ, dt, dt);
        printf("  RESULT: newVol=%.9f newDepth_prelim=%.9f surfArea=%.9f\n",
               oldVolume + dV, yOld + (dV/surfArea), surfArea);
    }

    // DEBUG: COMPREHENSIVE logging for node 852 (WW-258 storage) - Iteration 6 storage volume tracking
    if (i == 852 && g_depthRoutingStepCounter >= 1 && g_depthRoutingStepCounter <= 3) {
        printf("GPU_STOR852[step=%d iter=%d]: INPUTS: oldVol=%.9f oldDepth=%.9f oldNetIn=%.9f\n",
               g_depthRoutingStepCounter, steps, oldVolume, yOld, oldNetInflow);
        printf("  FLOWS: inflow=%.9f outflow=%.9f dQ=%.9f\n", inflow, outflow, dQ);
        printf("  INTEGRATION: dV=%.9f (=0.5*(%.9f+%.9f)*%.6f) dt=%.6f\n",
               dV, oldNetInflow, dQ, dt, dt);
        printf("  AREA: surfArea=%.9f newSurfArea=%.9f\n", surfArea, newSurfArea);
        printf("  GEOMETRY: fullVol=%.6f fullDepth=%.6f storageA0=%.6f storageA1=%.6f storageA2=%.6f\n",
               fullVolume, fullDepth, storageA0, storageA1, storageA2);
    }

    // DEBUG: Comprehensive Node 1 (J2) input logging for routing steps 150-151, iterations 0-1
    if (i == 1 && g_depthRoutingStepCounter >= 150 && g_depthRoutingStepCounter <= 151 && steps <= 1) {
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
            if (debugThis) {
                printf("  CALC: dQ=%.6f dV=%.6f dy=%.6f yNew_raw=%.6f\n",
                       dQ, dV, dy, yNew);
                printf("  CHECK isnan: dQ=%d dV=%d dy=%d yNew_raw=%d\n",
                       isnan(dQ), isnan(dV), isnan(dy), isnan(yNew));
                printf("  RELAX: omega=%.3f yLast=%.6f\n", omega, yLast);
            }
            yNew = (1.0 - omega) * yLast + omega * yNew;
            if (debugThis) {
                printf("  RELAX: yNew_relaxed=%.6f (moved %.6f)\n",
                       yNew, yNew - yLast);
                printf("  CHECK isnan: yNew_relaxed=%d\n", isnan(yNew));
            }
        } else {
            if (debugThis) {
                printf("  CALC: dQ=%.6f dV=%.6f dy=%.6f yNew_raw=%.6f (NO RELAX step=0)\n",
                       dQ, dV, dy, yNew);
                printf("  CHECK isnan: dQ=%d dV=%d dy=%d yNew_raw=%d\n",
                       isnan(dQ), isnan(dV), isnan(dy), isnan(yNew));
            }
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
        if (debugThis) {
            printf("  BEFORE VOLUME CALC: yNew=%.6f (about to call gpu_node_getVolume)\n", yNew);
        }

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

        if (debugThis) {
            printf("  AFTER VOLUME CALC: newVolume=%.6f\n", *newVolume_out);
            printf("  CHECK NaN: newVolume isnan=%d isinf=%d\n",
                   isnan(*newVolume_out), isinf(*newVolume_out));
        }
    }

    // --- compute rate of depth change
    *dYdT_out = fabs(yNew - yOld) / dt;

    // DEBUG: Log dYdT calculation for first 10 nodes with significant dYdT
    if (g_depthRoutingStepCounter == 0 && *dYdT_out > 0.01 && i < 100) {
        printf("  GPU dYdT[node=%d step=%d iter=%d]: yNew=%.6f yOld=%.6f dt=%.6f → dYdT=%.6f\n",
               i, g_depthRoutingStepCounter, steps, yNew, yOld, dt, *dYdT_out);
    }

    // DEBUG: Detailed trace for node 12 (J-229-OUT) at early steps to diagnose depth halving
    if (i == 12 && g_depthRoutingStepCounter <= 3 && steps <= 2) {
        printf("GPU_NODE12[step=%d iter=%d]: oldDepth=%.9f yLast=%.9f inflow=%.6f outflow=%.6f\n",
               g_depthRoutingStepCounter, steps, yOld, yLast, inflow, outflow);
        printf("  oldNetInflow=%.9f dQ=%.6f dV=%.9f surfArea=%.9f newSurfArea=%.9f\n",
               oldNetInflow, dQ, dV, surfArea, newSurfArea);
        printf("  dy_raw=%.9f yNew=%.9f omega=%.3f dYdT=%.6f dt=%.6f\n",
               dy, yNew, omega, *dYdT_out, dt);
    }

    // DEBUG: Capture final converged depth for first 20 nodes at end of step 5 (iteration 0 of step 6)
    // This shows the exact state being passed from step 5 → step 6
    if (g_depthRoutingStepCounter == 6 && steps == 0 && i < 20) {
        printf("GPU_STEP5_FINAL[node=%d]: depth=%.9f inflow=%.6f outflow=%.6f\n",
               i, yOld, inflow, outflow);
    }

    // --- save new depth
    *newDepth_out = yNew;

    // DEBUG: Log OUTPUTS for node 852 (continuation of STOR852 logging above)
    if (i == 852 && g_depthRoutingStepCounter >= 1 && g_depthRoutingStepCounter <= 3) {
        printf("  OUTPUTS: dy=%.9f yNew=%.9f newVol=%.9f overflow=%.9f dYdT=%.9f\n",
               dy, yNew, *newVolume_out, *overflow_out, *dYdT_out);
        printf("  DEPTH_CHANGE: yOld=%.9f → yNew=%.9f (delta=%.9f)\n",
               yOld, yNew, yNew - yOld);
        printf("  VOLUME_CHANGE: oldVol=%.9f → newVol=%.9f (delta=%.9f)\n\n",
               oldVolume, *newVolume_out, *newVolume_out - oldVolume);
    }

    if (debugThis) {
        printf("  FINAL: newDepth=%.6f newVolume=%.6f dYdT=%.6f\n",
               *newDepth_out, *newVolume_out, *dYdT_out);
        printf("  CHECK isnan: newDepth=%d newVolume=%d dYdT=%d\n",
               isnan(*newDepth_out), isnan(*newVolume_out), isnan(*dYdT_out));
        printf("=== END NODE_%d ===\n\n", i);
    }
}

#endif // GPU_DYNWAVE_KERNELS_CUH
