//-----------------------------------------------------------------------------
//   gpu_nonconduit_helpers.cuh
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/27/2025
//
//   GPU device functions for non-conduit link flow calculations
//   (pumps, orifices, weirs, outlets)
//
//   Based on CPU implementations in link.c
//
//-----------------------------------------------------------------------------

#ifndef GPU_NONCONDUIT_HELPERS_CUH
#define GPU_NONCONDUIT_HELPERS_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "gpu_table_helpers.cuh"
#include "gpu_xsect_helpers.cuh"

// Include C enums but avoid extern "C" wrapper
#include "../enums.h"
#include "../consts.h"

//=============================================================================
// Inline Helper: Create GPU_Xsect from parameters
//=============================================================================
__device__ __forceinline__
GPU_Xsect makeGPU_Xsect(int type, double yFull, double aFull, double rFull,
                        double wMax, double geom1, double geom2, double geom3)
{
    GPU_Xsect xs;
    xs.type = type;
    xs.yFull = yFull;
    xs.aFull = aFull;
    xs.rFull = rFull;
    xs.wMax = wMax;
    xs.geom1 = geom1;
    xs.geom2 = geom2;
    xs.geom3 = geom3;
    return xs;
}

//=============================================================================
// Device Helper: Flap Gate Logic
//=============================================================================
// Equivalent to link_setFlapGate() in link.c
//
__device__ __forceinline__
bool gpu_link_setFlapGate(
    int j,                       // link index
    int n1,                      // upstream node
    int n2,                      // downstream node
    double dir,                  // flow direction
    char hasFlapGate,            // does link have flap gate
    const int* d_nodeType,       // node types
    const double* d_nodeNewDepth, // node depths
    const double* d_nodeOutflow) // node outflows
{
    // No flap gate - allow flow
    if (!hasFlapGate) return false;

    // Negative flow direction - gate closes
    if (dir < 0.0) return true;

    // Check if downstream node is an outfall with flap gate restriction
    if (d_nodeType[n2] == OUTFALL && d_nodeNewDepth[n2] > FUDGE) {
        return true;
    }

    return false;
}

//=============================================================================
// PUMP FLOW CALCULATIONS
//=============================================================================

//-----------------------------------------------------------------------------
// Device Helper: Pump Flow (IDEAL_PUMP type)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_pump_getIdealFlow(
    int n1,                      // upstream node index
    const double* d_nodeInflow,  // node inflows
    const double* d_nodeOverflow) // node overflows
{
    return d_nodeInflow[n1] + d_nodeOverflow[n1];
}

//-----------------------------------------------------------------------------
// Device Helper: Pump Flow (TYPE1 - volume curve)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_pump_getType1Flow(
    int pumpIdx,
    int curveIdx,
    double volume,               // wet well volume (internal units)
    double ucfVolume,            // unit conversion for volume
    double ucfFlow,              // unit conversion for flow
    double xMin, double xMax,    // curve range
    char* flowClass,             // output: flow class
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
{
    double vol = volume * ucfVolume;
    double qIn = gpu_table_intervalLookup(
        curveIdx, vol,
        curves->d_dataStart,
        curves->d_dataCount,
        points->d_xValues,
        points->d_yValues) / ucfFlow;

    // Check if off pump curve
    if (vol < xMin || vol > xMax) {
        *flowClass = 1; // YES - off curve
    }

    return qIn;
}

//-----------------------------------------------------------------------------
// Device Helper: Pump Flow (TYPE2 - depth curve, discrete)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_pump_getType2Flow(
    int pumpIdx,
    int curveIdx,
    double depth,                // wet well depth (internal units)
    double ucfLength,            // unit conversion for length
    double ucfFlow,              // unit conversion for flow
    double xMin, double xMax,    // curve range
    char* flowClass,             // output: flow class
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
{
    double d = depth * ucfLength;
    double qIn = gpu_table_intervalLookup(
        curveIdx, d,
        curves->d_dataStart,
        curves->d_dataCount,
        points->d_xValues,
        points->d_yValues) / ucfFlow;

    // Check if off pump curve
    if (d < xMin || d > xMax) {
        *flowClass = 1; // YES - off curve
    }

    return qIn;
}

//-----------------------------------------------------------------------------
// Device Helper: Pump Flow (TYPE3 - head curve, continuous)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_pump_getType3Flow(
    int curveIdx,
    double upstreamDepth,
    double upstreamInvert,
    double downstreamDepth,
    double downstreamInvert,
    double speed,                // speed setting (for TYPE5)
    double ucfLength,
    double ucfFlow,
    double xMin, double xMax,
    char* flowClass,
    double* dqdh,                // output: dQ/dH
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
{
    // Compute head difference
    double head = ((downstreamDepth + downstreamInvert) -
                   (upstreamDepth + upstreamInvert)) / (speed * speed);
    head = fmax(head, 0.0) * ucfLength;

    double qIn = gpu_table_lookup(
        curveIdx, head,
        curves->d_dataStart,
        curves->d_dataCount,
        points->d_xValues,
        points->d_yValues) / ucfFlow;

    // Compute dQ/dH (slope of pump curve)
    // Reverse sign since flow decreases with increasing head
    *dqdh = -gpu_table_getSlope(
        curveIdx, head,
        curves->d_dataStart,
        curves->d_dataCount,
        points->d_xValues,
        points->d_yValues) * ucfLength / ucfFlow / speed;

    // Check if off pump curve
    if (head < xMin || head > xMax) {
        *flowClass = 1; // YES - off curve
    }

    return qIn;
}

//-----------------------------------------------------------------------------
// Device Helper: Pump Flow (TYPE4 - depth curve, continuous)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_pump_getType4Flow(
    int curveIdx,
    double depth,
    double ucfLength,
    double ucfFlow,
    double xMin, double xMax,
    char* flowClass,
    double* dqdh,                // output: dQ/dH
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
{
    const double dh = 0.001;     // small increment for numerical derivative

    double qIn = gpu_table_lookup(
        curveIdx, depth * ucfLength,
        curves->d_dataStart,
        curves->d_dataCount,
        points->d_xValues,
        points->d_yValues) / ucfFlow;

    // Compute dQ/dH numerically
    double qIn1 = gpu_table_lookup(
        curveIdx, (depth + dh) * ucfLength,
        curves->d_dataStart,
        curves->d_dataCount,
        points->d_xValues,
        points->d_yValues) / ucfFlow;
    *dqdh = (qIn1 - qIn) / dh;

    // Check if off pump curve
    double d = depth * ucfLength;
    if (d < xMin) {
        *flowClass = DN_DRY;
    } else if (d > xMax) {
        *flowClass = UP_DRY;
    }

    return qIn;
}

//=============================================================================
// ORIFICE FLOW CALCULATIONS
//=============================================================================

//-----------------------------------------------------------------------------
// Device Helper: Orifice Flow (weir-like behavior when partially filled)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_orifice_getWeirFlow(
    int linkIdx,
    double head,
    double f,                    // fraction of critical depth filled
    double cWeir,                // weir discharge coefficient
    int xsectType,               // cross-section type
    double yFull,                // full depth
    double aFull,                // full area
    double rFull,                // full hydraulic radius
    double wMax,                 // max width
    double geom1, double geom2)  // cross-section geometry parameters
{
    // Get width at depth y = f * yFull
    double y = f * yFull;
    GPU_Xsect xs = makeGPU_Xsect(xsectType, yFull, aFull, rFull, wMax, geom1, geom2, 0.0);
    double width = gpu_xsect_getWofY(&xs, y);

    // Weir flow equation: Q = C * L * H^1.5
    double q = cWeir * width * pow(head, 1.5);
    return q;
}

//-----------------------------------------------------------------------------
// Device Helper: Orifice Flow (orifice behavior when fully filled)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_orifice_getOrificeFlow(
    int linkIdx,
    double head,
    double cOrif,                // orifice discharge coefficient
    int xsectType,               // cross-section type
    double yFull,                // full depth
    double aFull,                // full area
    double rFull,                // full hydraulic radius
    double wMax,                 // max width
    double geom1, double geom2,  // cross-section geometry parameters
    double* dqdh)                // output: dQ/dH
{
    // Get cross-sectional area
    GPU_Xsect xs = makeGPU_Xsect(xsectType, yFull, aFull, rFull, wMax, geom1, geom2, 0.0);
    double area = gpu_xsect_getAofY(&xs, yFull);

    // Orifice flow equation: Q = C * A * sqrt(2gH)
    double q = cOrif * area * sqrt(2.0 * GRAVITY * head);

    // Compute dQ/dH = C * A * sqrt(g/(2H))
    if (head > FUDGE) {
        *dqdh = cOrif * area * sqrt(GRAVITY / (2.0 * head));
    } else {
        *dqdh = 0.0;
    }

    return q;
}

//=============================================================================
// WEIR FLOW CALCULATIONS
//=============================================================================

//-----------------------------------------------------------------------------
// Device Helper: Weir Flow (standard weir equation)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_weir_getFlow(
    int weirType,
    double head,
    double cDisch1,              // discharge coefficient 1
    double cDisch2,              // discharge coefficient 2 (for trapezoidal)
    double length,               // weir length
    double slope,                // side slope (for triangular)
    double* dqdh)                // output: dQ/dH
{
    double q = 0.0;
    double expon;

    switch (weirType) {
        case TRANSVERSE_WEIR:
            // Q = C * L * H^1.5
            expon = 1.5;
            q = cDisch1 * length * pow(head, expon);
            if (head > FUDGE) {
                *dqdh = expon * q / head;
            }
            break;

        case SIDEFLOW_WEIR:
            // Q = C * L * H^(5/3)
            expon = 5.0/3.0;
            q = cDisch1 * length * pow(head, expon);
            if (head > FUDGE) {
                *dqdh = expon * q / head;
            }
            break;

        case VNOTCH_WEIR:
            // Q = C * slope * H^2.5
            expon = 2.5;
            q = cDisch1 * slope * pow(head, expon);
            if (head > FUDGE) {
                *dqdh = expon * q / head;
            }
            break;

        case TRAPEZOIDAL_WEIR:
            // Q = C1 * L * H^1.5 + C2 * slope * H^2.5
            q = cDisch1 * length * pow(head, 1.5) +
                cDisch2 * slope * pow(head, 2.5);
            if (head > FUDGE) {
                *dqdh = (1.5 * cDisch1 * length * pow(head, 0.5) +
                         2.5 * cDisch2 * slope * pow(head, 1.5));
            }
            break;

        default:
            *dqdh = 0.0;
            break;
    }

    return q;
}

//-----------------------------------------------------------------------------
// Device Helper: Weir Flow as Orifice (when surcharged)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_weir_getOrificeFlow(
    double head,
    double y,                    // weir opening height
    double cSurcharge,           // surcharge coefficient
    int xsectType,
    double yFull,
    double aFull,                // full area
    double rFull,                // full hydraulic radius
    double wMax,                 // max width
    double geom1, double geom2,
    double* dqdh)
{
    // Treat weir as equivalent orifice
    GPU_Xsect xs = makeGPU_Xsect(xsectType, yFull, aFull, rFull, wMax, geom1, geom2, 0.0);
    double area = gpu_xsect_getAofY(&xs, y);
    double q = cSurcharge * area * sqrt(2.0 * GRAVITY * head);

    if (head > FUDGE) {
        *dqdh = cSurcharge * area * sqrt(GRAVITY / (2.0 * head));
    } else {
        *dqdh = 0.0;
    }

    return q;
}

//=============================================================================
// OUTLET FLOW CALCULATIONS
//=============================================================================

//-----------------------------------------------------------------------------
// Device Helper: Outlet Flow (rating curve or power function)
//-----------------------------------------------------------------------------
__device__ __forceinline__
double gpu_outlet_getFlow(
    int outletIdx,
    int curveIdx,                // -1 if using power function
    double head,                 // head across outlet
    double qCoeff,               // coefficient for power function
    double qExpon,               // exponent for power function
    double ucfLength,            // unit conversion for length
    double ucfFlow,              // unit conversion for flow
    const GPU_CurveData* curves,
    const GPU_CurvePoints* points)
{
    double h = head * ucfLength;

    // Use rating curve if provided
    if (curveIdx >= 0) {
        return gpu_table_lookup(
            curveIdx, h,
            curves->d_dataStart,
            curves->d_dataCount,
            points->d_xValues,
            points->d_yValues) / ucfFlow;
    }

    // Otherwise use power function: Q = qCoeff * H^qExpon
    return qCoeff * pow(h, qExpon) / ucfFlow;
}

//=============================================================================
// SURFACE AREA CALCULATIONS
//=============================================================================

//-----------------------------------------------------------------------------
// Device Helper: Non-Conduit Surface Area
//-----------------------------------------------------------------------------
// Computes surface areas at ends of non-conduit links for node depth updates
//
__device__ __forceinline__
void gpu_findNonConduitSurfArea(
    int linkType,
    int n1, int n2,              // node indices
    double linkSurfArea,         // computed link surface area
    double* surfArea1,           // output: upstream surface area
    double* surfArea2,           // output: downstream surface area
    const int* d_nodeType,
    const double* d_nodeNewDepth,
    const double* d_nodeCrownElev,
    const double* d_nodeInvertElev)
{
    *surfArea1 = 0.0;
    *surfArea2 = 0.0;

    // Only assign surface area to storage nodes that are not full
    if (d_nodeType[n1] == STORAGE &&
        d_nodeNewDepth[n1] < (d_nodeCrownElev[n1] - d_nodeInvertElev[n1])) {
        *surfArea1 = linkSurfArea;
    }

    if (d_nodeType[n2] == STORAGE &&
        d_nodeNewDepth[n2] < (d_nodeCrownElev[n2] - d_nodeInvertElev[n2])) {
        *surfArea2 = linkSurfArea;
    }
}

#endif // GPU_NONCONDUIT_HELPERS_CUH
