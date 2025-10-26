//-----------------------------------------------------------------------------
//   dynwave_data.h
//
//   Shared structures between dynwave CPU code and GPU helpers.
//-----------------------------------------------------------------------------

#ifndef DYNWAVE_DATA_H
#define DYNWAVE_DATA_H

typedef struct
{
    char    converged;   // TRUE if node iterations have converged
    double  newSurfArea; // current surface area (ft2)
    double  oldSurfArea; // previous surface area (ft2)
    double  sumdqdh;     // sum of dqdh from adjoining links
    double  dYdT;        // change in depth w.r.t. time (ft/sec)
} TXnode;

extern TXnode* Xnode;

#endif // DYNWAVE_DATA_H
