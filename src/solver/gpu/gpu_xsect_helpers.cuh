//-----------------------------------------------------------------------------
//   gpu_xsect_helpers.cuh
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    10/25/2025
//
//   GPU device functions for cross-section geometry calculations.
//   These are the GPU equivalents of functions in xsect.c and dwflow.c.
//
//-----------------------------------------------------------------------------

#ifndef GPU_XSECT_HELPERS_CUH
#define GPU_XSECT_HELPERS_CUH

#include <cuda_runtime.h>
#include <math.h>

//-----------------------------------------------------------------------------
// Cross-Section Type Constants (from enums.h)
//-----------------------------------------------------------------------------
#define GPU_DUMMY           0
#define GPU_CIRCULAR        1
#define GPU_FILLED_CIRCULAR 2
#define GPU_RECT_CLOSED     3
#define GPU_RECT_OPEN       4
#define GPU_TRAPEZOIDAL     5
#define GPU_TRIANGULAR      6
#define GPU_PARABOLIC       7
#define GPU_POWER_FUNCTION  8
#define GPU_RECT_TRIANG     9
#define GPU_RECT_ROUND      10
#define GPU_MOD_BASKET      11
#define GPU_HORIZ_ELLIPSE   12
#define GPU_VERT_ELLIPSE    13
#define GPU_ARCH            14
#define GPU_EGGSHAPED       15
#define GPU_HORSESHOE       16
#define GPU_GOTHIC          17
#define GPU_CATENARY        18
#define GPU_SEMIELLIPTICAL  19
#define GPU_BASKETHANDLE    20
#define GPU_SEMICIRCULAR    21
#define GPU_IRREGULAR       22
#define GPU_CUSTOM          23
#define GPU_FORCE_MAIN      24
#define GPU_STREET_XSECT    25

// Surcharge method
#define GPU_SLOT_METHOD     1

// Constants
#define GPU_FUDGE           0.0001
#define GPU_GRAVITY         32.2      // ft/sec2
#define GPU_PI              3.141592653589793

//-----------------------------------------------------------------------------
// Simplified Cross-Section Data Structure for GPU
//-----------------------------------------------------------------------------
// Note: We use a simplified structure with pre-computed values
// Full xsection lookup tables are too complex for initial GPU implementation

typedef struct {
    int     type;           // Cross-section type
    double  yFull;          // Full depth (ft)
    double  aFull;          // Full area (ft2)
    double  rFull;          // Full hydraulic radius (ft)
    double  pFull;          // Full wetted perimeter (ft)
    double  wMax;           // Maximum width (ft)
    double  sBot;           // Bottom slope / side slope (for triangular, trapezoidal)
    double  rBot;           // Bottom radius (for parabolic)
    // Additional parameters for specific shapes
    double  geom1;          // Shape parameter 1 (width, diameter, etc.)
    double  geom2;          // Shape parameter 2 (height, side slope, etc.)
    double  geom3;          // Shape parameter 3 (additional geometry)
} GPU_Xsect;

//-----------------------------------------------------------------------------
// Helper Device Functions
//-----------------------------------------------------------------------------

__device__ inline int gpu_xsect_isOpen(int xsectType)
//
//  Purpose: Determines if cross-section is open or closed
//  Input:   xsectType = cross-section type code
//  Returns: 1 if open channel, 0 if closed conduit
//
{
    // Open channel types
    switch (xsectType) {
        case GPU_RECT_OPEN:
        case GPU_TRAPEZOIDAL:
        case GPU_TRIANGULAR:
        case GPU_PARABOLIC:
        case GPU_POWER_FUNCTION:
        case GPU_IRREGULAR:
        case GPU_STREET_XSECT:
            return 1;
        default:
            return 0;
    }
}

//=============================================================================

__device__ double gpu_getSlotWidth(
    GPU_Xsect* xsect,
    double y,
    int surchargeMethod,
    double crownCutoff)
//
//  Purpose: Computes Preissmann slot width for surcharged closed conduits
//  Input:   xsect = pointer to cross-section data
//           y = flow depth (ft)
//           surchargeMethod = EXTRAN or SLOT
//           crownCutoff = fraction of full depth for crown
//  Returns: Slot width (ft) or 0 if not applicable
//
//  Note: Preissmann slot is a fictitious narrow slot at the crown of
//        closed conduits to allow pressurized flow calculations
//
{
    double yNorm = y / xsect->yFull;

    // Return 0 if slot method not used or if open channel
    if (surchargeMethod != GPU_SLOT_METHOD ||
        gpu_xsect_isOpen(xsect->type) ||
        yNorm < crownCutoff) {
        return 0.0;
    }

    // For depth > 1.78 * full depth, slot width = 1% of max width
    if (yNorm > 1.78) {
        return 0.01 * xsect->wMax;
    }

    // Otherwise use Sjoberg formula
    return xsect->wMax * 0.5423 * exp(-pow(yNorm, 2.4));
}

//=============================================================================

__device__ double gpu_circular_getWofY(double y, double d)
//
//  Purpose: Computes top width for circular section
//
{
    if (y <= 0.0) return 0.0;
    if (y >= d) return d;

    double theta = 2.0 * acos(1.0 - 2.0 * y / d);
    return d * sin(theta / 2.0);
}

//=============================================================================

__device__ double gpu_circular_getAofY(double y, double d)
//
//  Purpose: Computes area for circular cross-section
//  Input:   y = flow depth (ft)
//           d = diameter (ft)
//  Returns: Flow area (ft2)
//
{
    double theta;

    if (y <= 0.0) return 0.0;
    if (y >= d) return GPU_PI * d * d / 4.0;

    // Use circular segment formula
    theta = 2.0 * acos(1.0 - 2.0 * y / d);
    return d * d * (theta - sin(theta)) / 8.0;
}

//=============================================================================

__device__ double gpu_circular_getRofY(double y, double d)
//
//  Purpose: Computes hydraulic radius for circular cross-section
//  Input:   y = flow depth (ft)
//           d = diameter (ft)
//  Returns: Hydraulic radius (ft)
//
{
    double a, w, theta;

    if (y <= 0.0) return 0.0;
    if (y >= d) return d / 4.0;

    theta = 2.0 * acos(1.0 - 2.0 * y / d);
    a = d * d * (theta - sin(theta)) / 8.0;
    w = d * sin(theta / 2.0);

    return (w > 0.0) ? a / w : 0.0;
}

//=============================================================================

__device__ double gpu_rect_getWofY(double /*y*/, double w)
//
//  Purpose: Returns top width for rectangular cross-section
//
{
    return (w > 0.0) ? w : 0.0;
}

//=============================================================================

__device__ double gpu_rect_getAofY(double y, double w, double h)
//
//  Purpose: Computes area for rectangular cross-section
//  Input:   y = flow depth (ft)
//           w = width (ft)
//           h = height (ft)
//  Returns: Flow area (ft2)
//
{
    if (y <= 0.0) return 0.0;
    if (y >= h) return w * h;
    return w * y;
}

//=============================================================================

__device__ double gpu_rect_getRofY(double y, double w, double h)
//
//  Purpose: Computes hydraulic radius for rectangular cross-section
//  Input:   y = flow depth (ft)
//           w = width (ft)
//           h = height (ft)
//  Returns: Hydraulic radius (ft)
//
{
    double p;

    if (y <= 0.0) return 0.0;
    if (y >= h) {
        p = w + 2.0 * h;
        return (w * h) / p;
    }

    p = w + 2.0 * y;
    return (w * y) / p;
}

//=============================================================================

__device__ double gpu_trapezoidal_getWofY(double y, double b, double s)
//
//  Purpose: Returns top width for trapezoidal section
//
{
    if (y <= 0.0) return b;
    return b + 2.0 * s * y;
}

//=============================================================================

__device__ double gpu_trapezoidal_getAofY(double y, double b, double s)
//
//  Purpose: Computes area for trapezoidal cross-section
//  Input:   y = flow depth (ft)
//           b = bottom width (ft)
//           s = side slope (run/rise)
//  Returns: Flow area (ft2)
//
{
    if (y <= 0.0) return 0.0;
    return (b + s * y) * y;
}

//=============================================================================

__device__ double gpu_trapezoidal_getRofY(double y, double b, double s)
//
//  Purpose: Computes hydraulic radius for trapezoidal cross-section
//  Input:   y = flow depth (ft)
//           b = bottom width (ft)
//           s = side slope (run/rise)
//  Returns: Hydraulic radius (ft)
//
{
    double a, p;

    if (y <= 0.0) return 0.0;

    a = (b + s * y) * y;
    p = b + 2.0 * y * sqrt(1.0 + s * s);

    return (p > 0.0) ? a / p : 0.0;
}

//=============================================================================

__device__ double gpu_xsect_getWofY(GPU_Xsect* xsect, double y)
//
//  Purpose: Computes top width for supported shapes
//
{
    if (y <= 0.0) return 0.0;

    switch (xsect->type) {
        case GPU_CIRCULAR:
        case GPU_FILLED_CIRCULAR:
            return gpu_circular_getWofY(y, xsect->geom1);

        case GPU_RECT_CLOSED:
        case GPU_RECT_OPEN:
            return gpu_rect_getWofY(y, xsect->geom1);

        case GPU_TRAPEZOIDAL:
            return gpu_trapezoidal_getWofY(y, xsect->geom1, xsect->geom2);

        case GPU_TRIANGULAR:
            return gpu_trapezoidal_getWofY(y, 0.0, xsect->geom1);

        default:
            // Linear approximation for unsupported shapes
            return xsect->wMax * (y / xsect->yFull);
    }
}

//=============================================================================

__device__ double gpu_getWidth(
    GPU_Xsect* xsect,
    double y,
    int surchargeMethod,
    double crownCutoff)
//
//  Purpose: Computes effective top width including slot when surcharged
//
{
    double wSlot = gpu_getSlotWidth(xsect, y, surchargeMethod, crownCutoff);
    if (wSlot > 0.0) return wSlot;

    if ((y / xsect->yFull) >= crownCutoff && !gpu_xsect_isOpen(xsect->type)) {
        y = crownCutoff * xsect->yFull;
    }
    return gpu_xsect_getWofY(xsect, y);
}

//=============================================================================

__device__ double gpu_xsect_getAofY(GPU_Xsect* xsect, double y)
//
//  Purpose: Computes flow area for any cross-section type
//  Input:   xsect = pointer to cross-section data
//           y = flow depth (ft)
//  Returns: Flow area (ft2)
//
//  Note: Simplified implementation for common shapes only
//        Full implementation would use lookup tables for complex shapes
//
{
    if (y <= 0.0) return 0.0;
    if (y >= xsect->yFull) return xsect->aFull;

    switch (xsect->type) {
        case GPU_CIRCULAR:
        case GPU_FILLED_CIRCULAR:
            return gpu_circular_getAofY(y, xsect->geom1);

        case GPU_RECT_CLOSED:
        case GPU_RECT_OPEN:
            return gpu_rect_getAofY(y, xsect->geom1, xsect->yFull);

        case GPU_TRAPEZOIDAL:
            return gpu_trapezoidal_getAofY(y, xsect->geom1, xsect->geom2);

        case GPU_TRIANGULAR:
            return gpu_trapezoidal_getAofY(y, 0.0, xsect->geom1);

        default:
            // For unsupported shapes, use linear interpolation as approximation
            return xsect->aFull * (y / xsect->yFull);
    }
}

//=============================================================================

__device__ double gpu_xsect_getRofY(GPU_Xsect* xsect, double y)
//
//  Purpose: Computes hydraulic radius for any cross-section type
//  Input:   xsect = pointer to cross-section data
//           y = flow depth (ft)
//  Returns: Hydraulic radius (ft)
//
{
    if (y <= 0.0) return 0.0;
    if (y >= xsect->yFull) return xsect->rFull;

    switch (xsect->type) {
        case GPU_CIRCULAR:
        case GPU_FILLED_CIRCULAR:
            return gpu_circular_getRofY(y, xsect->geom1);

        case GPU_RECT_CLOSED:
        case GPU_RECT_OPEN:
            return gpu_rect_getRofY(y, xsect->geom1, xsect->yFull);

        case GPU_TRAPEZOIDAL:
            return gpu_trapezoidal_getRofY(y, xsect->geom1, xsect->geom2);

        case GPU_TRIANGULAR:
            return gpu_trapezoidal_getRofY(y, 0.0, xsect->geom1);

        default:
            // For unsupported shapes, use linear interpolation as approximation
            return xsect->rFull * (y / xsect->yFull);
    }
}

//=============================================================================

__device__ double gpu_getArea(
    GPU_Xsect* xsect,
    double y,
    double wSlot)
//
//  Purpose: Computes flow area including Preissmann slot if applicable
//  Input:   xsect = pointer to cross-section data
//           y = flow depth (ft)
//           wSlot = Preissmann slot width (ft)
//  Returns: Flow area (ft2)
//
{
    // If flow exceeds full depth, add slot area
    if (y >= xsect->yFull) {
        return xsect->aFull + (y - xsect->yFull) * wSlot;
    }

    // Otherwise use standard area calculation
    return gpu_xsect_getAofY(xsect, y);
}

//=============================================================================

__device__ double gpu_getHydRad(GPU_Xsect* xsect, double y)
//
//  Purpose: Computes hydraulic radius
//  Input:   xsect = pointer to cross-section data
//           y = flow depth (ft)
//  Returns: Hydraulic radius (ft)
//
{
    // For surcharged flow, use full hydraulic radius
    if (y >= xsect->yFull) return xsect->rFull;

    // Otherwise compute from depth
    return gpu_xsect_getRofY(xsect, y);
}

#endif // GPU_XSECT_HELPERS_CUH
