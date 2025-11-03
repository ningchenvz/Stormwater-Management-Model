//-----------------------------------------------------------------------------
//   gpu_link_helpers.cuh
//
//   Project: EPA SWMM-GPU
//   Version: 5.2
//   Date:    11/02/2025
//
//   CUDA device functions for link calculations (normal depth, critical depth)
//   Ports from src/solver/link.c and src/solver/xsect.c
//
//-----------------------------------------------------------------------------

#ifndef GPU_LINK_HELPERS_CUH
#define GPU_LINK_HELPERS_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "gpu_structures.h"
#include "gpu_xsect_helpers.cuh"

// Constants
#define GPU_GRAVITY 32.2  // ft/s^2
#define GPU_PI 3.141592653589793

//=============================================================================
// Forward declarations of xsect helper functions we need to port
//=============================================================================

__device__ double gpu_xsect_getYofA(GPU_Xsect* xsect, double a);
__device__ double gpu_xsect_getAofS(GPU_Xsect* xsect, double s);
__device__ double gpu_xsect_getYcrit(GPU_Xsect* xsect, double q);

//=============================================================================
// GPU Port: link_getYnorm()
// From: src/solver/link.c:783
//=============================================================================
__device__ inline double gpu_link_getYnorm(
    GPU_Xsect* xsect,
    double q,
    double conduitBeta,
    double conduitQmax)
//
//  Input:   xsect = cross-section data
//           q = link flow rate (cfs)
//           conduitBeta = conduit beta parameter (from Conduit[k].beta)
//           conduitQmax = conduit max flow (from Conduit[k].qMax)
//  Output:  returns normal depth (ft)
//  Purpose: computes normal depth for given flow rate
//
{
    double s, a, y;

    q = fabs(q);
    if (q > conduitQmax) q = conduitQmax;
    if (q <= 0.0) return 0.0;

    s = q / conduitBeta;
    a = gpu_xsect_getAofS(xsect, s);
    y = gpu_xsect_getYofA(xsect, a);
    return y;
}

//=============================================================================
// GPU Port: link_getYcrit()
// From: src/solver/link.c:770
//=============================================================================
__device__ inline double gpu_link_getYcrit(
    GPU_Xsect* xsect,
    double q)
//
//  Input:   xsect = cross-section data
//           q = link flow rate (cfs)
//  Output:  returns critical depth (ft)
//  Purpose: computes critical depth for given flow rate
//
{
    return gpu_xsect_getYcrit(xsect, q);
}

//=============================================================================
// GPU Port: xsect_getYcrit()
// From: src/solver/xsect.c:1257
// Initial implementation: Common cross-section types only
//=============================================================================
__device__ double gpu_xsect_getYcrit(GPU_Xsect* xsect, double q)
//
//  Input:   xsect = cross-section data
//           q = flow rate (cfs)
//  Output:  returns critical depth (ft)
//  Purpose: computes critical depth at a specific flow rate
//
{
    double q2g = (q * q) / GPU_GRAVITY;
    double y, r;

    if (q2g == 0.0) return 0.0;

    switch (xsect->type)
    {
        case GPU_DUMMY:
            return 0.0;

        case GPU_RECT_OPEN:
        case GPU_RECT_CLOSED:
            // Analytical: y = (q2g / w^2)^(1/3)
            y = pow(q2g / (xsect->wMax * xsect->wMax), 1.0/3.0);
            break;

        case GPU_TRIANGULAR:
            // Analytical: y = (2 * q2g / s^2)^(1/5)
            y = pow(2.0 * q2g / (xsect->sBot * xsect->sBot), 1.0/5.0);
            break;

        case GPU_PARABOLIC:
            // Analytical: y = (27/32 * q2g * c)^(1/4)
            y = pow(27.0/32.0 * q2g / (xsect->rBot * xsect->rBot), 1.0/4.0);
            break;

        case GPU_POWER_FUNCTION:
            y = 1.0 / (2.0 * xsect->sBot + 3.0);
            y = pow(q2g * (xsect->sBot + 1.0) / (xsect->rBot * xsect->rBot), y);
            break;

        case GPU_CIRCULAR:
        case GPU_FORCE_MAIN:
        default:
            // Use iterative method for circular and other complex shapes
            // First estimate using equivalent circular conduit: 1.01 * (q2g / yFull)^(1/4)
            y = 1.01 * pow(q2g / xsect->yFull, 1.0/4.0);
            if (y >= xsect->yFull) y = 0.97 * xsect->yFull;

            // Find ratio of conduit area to equiv. circular area
            r = xsect->aFull / (GPU_PI / 4.0 * xsect->yFull * xsect->yFull);

            // Adjust y by r^(1/4)
            y = y * pow(r, 1.0/4.0);

            // Use Newton-Raphson iteration to refine (max 20 iterations)
            // Iterative solution of: q = sqrt(g * A^3 / W)
            for (int iter = 0; iter < 20; iter++)
            {
                double a = gpu_xsect_getAofY(xsect, y);
                double w = gpu_xsect_getWofY(xsect, y);

                if (w <= 0.0) break;

                double f = q - sqrt(GPU_GRAVITY * a * a * a / w);
                if (fabs(f) < 0.001) break;  // Converged

                // Newton step: y_new = y - f / f'
                // f' ≈ -sqrt(g) * d/dy[A^(3/2) / sqrt(W)]
                double dady = w;  // dA/dy = W (top width)
                double dwdy = 0.0;  // Approximate dW/dy ≈ 0 for simplicity

                double df = -sqrt(GPU_GRAVITY) * (1.5 * sqrt(a) * dady / sqrt(w));

                if (fabs(df) < 0.0001) break;

                double dy = -f / df;
                y += dy;

                // Keep y in valid range
                if (y < 0.0) y = 0.001;
                if (y > xsect->yFull) y = xsect->yFull;
            }
            break;
    }

    return y;
}

//=============================================================================
// GPU Port: xsect_getAofS()
// From: src/solver/xsect.c:1149
// Initial implementation: Common cross-section types only
//=============================================================================
__device__ double gpu_xsect_getAofS(GPU_Xsect* xsect, double s)
//
//  Input:   xsect = cross-section data
//           s = section factor (AR^(2/3)) (ft^(8/3))
//  Output:  returns area (ft2)
//  Purpose: computes flow area given section factor
//
{
    double a;

    switch (xsect->type)
    {
        case GPU_DUMMY:
            return 0.0;

        case GPU_CIRCULAR:
        case GPU_FORCE_MAIN:
            // Use lookup table like CPU does (matches xsect.c::circ_getAofS)
            {
                extern __device__ const double S_CIRC[];
                extern __device__ const int N_S_CIRC;

                // Compute sFull = aFull * rFull^(2/3) for circular
                double rFull = xsect->yFull / 4.0;
                double sFull = xsect->aFull * pow(rFull, 2.0/3.0);

                double psi = s / sFull;
                if (psi == 0.0) return 0.0;
                if (psi >= 1.0) return xsect->aFull;

                // Use lookup table for section factor inversion
                a = xsect->aFull * gpu_invLookup(psi, S_CIRC, N_S_CIRC);
            }
            break;

        case GPU_EGGSHAPED:
            {
                extern __device__ const double S_EGG[];
                extern __device__ const int N_S_EGG;
                double sFull = xsect->aFull * pow(xsect->rFull, 2.0/3.0);
                double psi = s / sFull;
                if (psi == 0.0) return 0.0;
                if (psi >= 1.0) return xsect->aFull;
                a = xsect->aFull * gpu_invLookup(psi, S_EGG, N_S_EGG);
            }
            break;

        case GPU_HORSESHOE:
            {
                extern __device__ const double S_HORSESHOE[];
                extern __device__ const int N_S_HORSESHOE;
                double sFull = xsect->aFull * pow(xsect->rFull, 2.0/3.0);
                double psi = s / sFull;
                if (psi == 0.0) return 0.0;
                if (psi >= 1.0) return xsect->aFull;
                a = xsect->aFull * gpu_invLookup(psi, S_HORSESHOE, N_S_HORSESHOE);
            }
            break;

        case GPU_GOTHIC:
            {
                extern __device__ const double S_GOTHIC[];
                extern __device__ const int N_S_GOTHIC;
                double sFull = xsect->aFull * pow(xsect->rFull, 2.0/3.0);
                double psi = s / sFull;
                if (psi == 0.0) return 0.0;
                if (psi >= 1.0) return xsect->aFull;
                a = xsect->aFull * gpu_invLookup(psi, S_GOTHIC, N_S_GOTHIC);
            }
            break;

        case GPU_CATENARY:
            {
                extern __device__ const double S_CATENARY[];
                extern __device__ const int N_S_CATENARY;
                double sFull = xsect->aFull * pow(xsect->rFull, 2.0/3.0);
                double psi = s / sFull;
                if (psi == 0.0) return 0.0;
                if (psi >= 1.0) return xsect->aFull;
                a = xsect->aFull * gpu_invLookup(psi, S_CATENARY, N_S_CATENARY);
            }
            break;

        case GPU_SEMIELLIPTICAL:
            {
                extern __device__ const double S_SEMIELLIP[];
                extern __device__ const int N_S_SEMIELLIP;
                double sFull = xsect->aFull * pow(xsect->rFull, 2.0/3.0);
                double psi = s / sFull;
                if (psi == 0.0) return 0.0;
                if (psi >= 1.0) return xsect->aFull;
                a = xsect->aFull * gpu_invLookup(psi, S_SEMIELLIP, N_S_SEMIELLIP);
            }
            break;

        case GPU_BASKETHANDLE:
            {
                extern __device__ const double S_BASKETHANDLE[];
                extern __device__ const int N_S_BASKETHANDLE;
                double sFull = xsect->aFull * pow(xsect->rFull, 2.0/3.0);
                double psi = s / sFull;
                if (psi == 0.0) return 0.0;
                if (psi >= 1.0) return xsect->aFull;
                a = xsect->aFull * gpu_invLookup(psi, S_BASKETHANDLE, N_S_BASKETHANDLE);
            }
            break;

        case GPU_SEMICIRCULAR:
            {
                extern __device__ const double S_SEMICIRC[];
                extern __device__ const int N_S_SEMICIRC;
                double sFull = xsect->aFull * pow(xsect->rFull, 2.0/3.0);
                double psi = s / sFull;
                if (psi == 0.0) return 0.0;
                if (psi >= 1.0) return xsect->aFull;
                a = xsect->aFull * gpu_invLookup(psi, S_SEMICIRC, N_S_SEMICIRC);
            }
            break;

        case GPU_RECT_OPEN:
        case GPU_RECT_CLOSED:
            // For rectangle: S = A * R^(2/3) = w*y * (w*y/(w+2*y))^(2/3)
            // Solve iteratively
            {
                double w = xsect->wMax;
                double y = pow(s / w, 3.0/5.0);  // Initial guess

                for (int iter = 0; iter < 20; iter++)
                {
                    a = w * y;
                    double p = w + 2.0 * y;
                    double r = a / p;
                    double f = a * pow(r, 2.0/3.0) - s;

                    if (fabs(f) < 0.001) break;

                    // dS/dy ≈ w * (w/(w+2*y))^(2/3) * (1 - 2*y/(3*(w+2*y)))
                    double df = w * pow(r, 2.0/3.0) * (1.0 - 2.0*y/(3.0*p));

                    if (fabs(df) < 0.0001) break;

                    y -= f / df;

                    if (y < 0.0) y = 0.001;
                    if (y > xsect->yFull) y = xsect->yFull;
                }
                a = w * y;
            }
            break;

        case GPU_TRAPEZOIDAL:
            // For trapezoid: Iterative solution
            {
                double b = xsect->wMax - 2.0 * xsect->sBot * xsect->yFull;  // Bottom width
                double slope = xsect->sBot;
                double y = pow(s / b, 3.0/5.0);  // Initial guess

                for (int iter = 0; iter < 20; iter++)
                {
                    a = (b + slope * y) * y;
                    double p = b + 2.0 * y * sqrt(1.0 + slope * slope);
                    double r = a / p;
                    double f = a * pow(r, 2.0/3.0) - s;

                    if (fabs(f) < 0.001) break;

                    // Numerical derivative
                    double dy_step = 0.001;
                    double y2 = y + dy_step;
                    double a2 = (b + slope * y2) * y2;
                    double p2 = b + 2.0 * y2 * sqrt(1.0 + slope * slope);
                    double r2 = a2 / p2;
                    double f2 = a2 * pow(r2, 2.0/3.0) - s;
                    double df = (f2 - f) / dy_step;

                    if (fabs(df) < 0.0001) break;

                    y -= f / df;

                    if (y < 0.0) y = 0.001;
                    if (y > xsect->yFull) y = xsect->yFull;
                }
                a = (b + slope * y) * y;
            }
            break;

        default:
            // For other shapes, use simple approximation
            // A ≈ aFull * (S / SFull)^(3/5) where SFull = aFull * rFull^(2/3)
            {
                double rFull = xsect->aFull / xsect->pFull;
                double sFull = xsect->aFull * pow(rFull, 2.0/3.0);
                a = xsect->aFull * pow(s / sFull, 3.0/5.0);
                if (a > xsect->aFull) a = xsect->aFull;
            }
            break;
    }

    return a;
}

//=============================================================================
// GPU Port: xsect_getYofA()
// From: src/solver/xsect.c:773
// Initial implementation: Common cross-section types only
//=============================================================================
__device__ double gpu_xsect_getYofA(GPU_Xsect* xsect, double a)
//
//  Input:   xsect = cross-section data
//           a = area (ft2)
//  Output:  returns depth (ft)
//  Purpose: computes depth at a given area
//
{
    double y;

    if (a <= 0.0) return 0.0;
    if (a >= xsect->aFull) return xsect->yFull;

    switch (xsect->type)
    {
        case GPU_DUMMY:
            return 0.0;

        case GPU_CIRCULAR:
        case GPU_FORCE_MAIN:
            // Use lookup table like CPU does (matches xsect.c::circ_getYofA)
            {
                extern __device__ const double Y_CIRC[];
                extern __device__ const int N_Y_CIRC;

                double alpha = a / xsect->aFull;

                // Use special function for small a/aFull (alpha < 0.04)
                // For now, just use lookup table for all values
                y = xsect->yFull * gpu_lookup(alpha, Y_CIRC, N_Y_CIRC);
            }
            break;

        case GPU_EGGSHAPED:
            {
                extern __device__ const double Y_EGG[];
                extern __device__ const int N_Y_EGG;
                double alpha = a / xsect->aFull;
                y = xsect->yFull * gpu_lookup(alpha, Y_EGG, N_Y_EGG);
            }
            break;

        case GPU_HORSESHOE:
            {
                extern __device__ const double Y_HORSESHOE[];
                extern __device__ const int N_Y_HORSESHOE;
                double alpha = a / xsect->aFull;
                y = xsect->yFull * gpu_lookup(alpha, Y_HORSESHOE, N_Y_HORSESHOE);
            }
            break;

        case GPU_GOTHIC:
            {
                extern __device__ const double Y_GOTHIC[];
                extern __device__ const int N_Y_GOTHIC;
                double alpha = a / xsect->aFull;
                y = xsect->yFull * gpu_lookup(alpha, Y_GOTHIC, N_Y_GOTHIC);
            }
            break;

        case GPU_CATENARY:
            {
                extern __device__ const double Y_CATENARY[];
                extern __device__ const int N_Y_CATENARY;
                double alpha = a / xsect->aFull;
                y = xsect->yFull * gpu_lookup(alpha, Y_CATENARY, N_Y_CATENARY);
            }
            break;

        case GPU_SEMIELLIPTICAL:
            {
                extern __device__ const double Y_SEMIELLIP[];
                extern __device__ const int N_Y_SEMIELLIP;
                double alpha = a / xsect->aFull;
                y = xsect->yFull * gpu_lookup(alpha, Y_SEMIELLIP, N_Y_SEMIELLIP);
            }
            break;

        case GPU_BASKETHANDLE:
            {
                extern __device__ const double Y_BASKETHANDLE[];
                extern __device__ const int N_Y_BASKETHANDLE;
                double alpha = a / xsect->aFull;
                y = xsect->yFull * gpu_lookup(alpha, Y_BASKETHANDLE, N_Y_BASKETHANDLE);
            }
            break;

        case GPU_SEMICIRCULAR:
            {
                extern __device__ const double Y_SEMICIRC[];
                extern __device__ const int N_Y_SEMICIRC;
                double alpha = a / xsect->aFull;
                y = xsect->yFull * gpu_lookup(alpha, Y_SEMICIRC, N_Y_SEMICIRC);
            }
            break;

        case GPU_RECT_OPEN:
        case GPU_RECT_CLOSED:
            // Analytical: A = w * y, so y = A / w
            y = a / xsect->wMax;
            break;

        case GPU_TRIANGULAR:
            // Analytical: A = s * y^2, so y = sqrt(A / s)
            y = sqrt(a / xsect->sBot);
            break;

        case GPU_TRAPEZOIDAL:
            // Analytical: A = (b + s*y) * y where b = bottom width
            // Solve quadratic: s*y^2 + b*y - A = 0
            {
                double b = xsect->wMax - 2.0 * xsect->sBot * xsect->yFull;
                double s = xsect->sBot;

                // y = (-b + sqrt(b^2 + 4*s*A)) / (2*s)
                if (s > 0.0001)
                {
                    y = (-b + sqrt(b*b + 4.0*s*a)) / (2.0*s);
                }
                else
                {
                    y = a / b;  // Rectangular if slope is zero
                }
            }
            break;

        case GPU_PARABOLIC:
            // Analytical: A = (2/3) * w * y where w = top width
            // For parabola: w = 2*y/rBot, so A = (4/3) * y^2 / rBot
            // Thus: y = sqrt(3*A*rBot / 4)
            y = sqrt(3.0 * a * xsect->rBot / 4.0);
            break;

        default:
            // For other shapes, use linear interpolation approximation
            y = xsect->yFull * (a / xsect->aFull);
            break;
    }

    return y;
}

#endif // GPU_LINK_HELPERS_CUH
