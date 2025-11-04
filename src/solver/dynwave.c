//-----------------------------------------------------------------------------
//   dynwave.c
//
//   Project:  EPA SWMM5
//   Version:  5.2
//   Date:     07/13/23  (Build 5.2.4)
//   Author:   L. Rossman
//             M. Tryby (EPA)
//             R. Dickinson (CDM)
//
//   Dynamic wave flow routing functions.
//
//   This module solves the dynamic wave flow routing equations using
//   Picard Iterations (i.e., a method of successive approximations)
//   to solve the explicit form of the continuity and momentum equations
//   for conduits.
//
//   Update History
//   ==============
//   Build 5.1.002:
//   - Only non-ponded nodal surface area is saved for use in
//     surcharge algorithm.
//   Build 5.1.007:
//   - Node losses added to node outflow variable instead of treated
//     as a separate item when computing change in node flow volume.
//   Build 5.1.008:
//   - Module-specific constants moved here from project.c.
//   - Support added for user-specified minimum variable time step.
//   - Node crown elevations found here instead of in flowrout.c module.
//   - OpenMP use to parallelize findLinkFlows() & findNodeDepths().
//   - Bug in finding complete list of capacity limited links fixed.
//   Build 5.1.011:
//   - Added test for failed memory allocation.
//   - Fixed illegal array index bug for Ideal Pumps.
//   Build 5.1.013:
//   - Include omp.h protected against lack of compiler support for OpenMP.
//   - SurchargeMethod option used to decide how node surcharging is handled.
//   - Storage nodes allowed to pressurize if their surcharge depth > 0.
//   - Minimum flow needed to compute a Courant time step modified.
//   Build 5.1.014:
//   - updateNodeFlows() modified to subtract conduit evap. and seepage losses
//     from downstream node inflow instead of upstream node outflow.
//   Build 5.1.015:
//   - Roll back the 5.1.014 change for conduit losses in updateNodeFlows().
//   Build 5.2.0:
//   - Support added for reporting most frequent non-converging links.
//   Build 5.2.4:
//   - Conduit evap+seepage outflow split evenly between outflow from
//     conduit's upstream and non-outfall downstream nodes.
//-----------------------------------------------------------------------------
#define _CRT_SECURE_NO_DEPRECATE

#include <stdlib.h>
#include <math.h>
#include "headers.h"
#include "dynwave_data.h"
#ifdef BUILD_GPU
#include "gpu_config.h"
#include "gpu_structures.h"

int gpu_runNodeDepthKernel(
    double dt,
    int allowPonding,
    int surchargeMethod,
    double minSurfArea,
    int steps,
    double omega,
    double headTol);

int gpu_computeConduitFlows(
    GPU_LinkData* links,
    GPU_ConduitData* conduits,
    GPU_XsectData* xsects,
    GPU_NodeData* nodes,
    double dt,
    int steps,
    double omega,
    int surchargeMethod,
    double crownCutoff,
    int inertDamping);

void gpu_flushConduitResults(void);
int gpu_initializeNonConduitData(void);
void gpu_freeNonConduitData(void);

// External GPU data structures (assumed to be allocated/managed elsewhere)
extern GPU_LinkData g_gpuLinks;
extern GPU_ConduitData g_gpuConduits;
extern GPU_XsectData g_gpuXsects;
extern GPU_NodeData g_gpuNodes;

// Non-conduit link data structures
extern GPU_PumpData g_gpuPumps;
extern GPU_OrificeData g_gpuOrifices;
extern GPU_WeirData g_gpuWeirs;
extern GPU_OutletData g_gpuOutlets;

// Curve data for pumps and outlets
extern GPU_CurveData g_gpuCurves;
extern GPU_CurvePoints g_gpuCurvePoints;
#endif

//-----------------------------------------------------------------------------
//     Constants 
//-----------------------------------------------------------------------------
static const double MINTIMESTEP         = 0.001;  // min. time step (sec)
static const double OMEGA               = 0.5;    // under-relaxation parameter
static const double DEFAULT_SURFAREA    = 12.566; // Min. nodal surface area (~4 ft diam.)
static const double DEFAULT_HEADTOL     = 0.005;  // Default head tolerance (ft)
static const double EXTRAN_CROWN_CUTOFF = 0.96;   // crown cutoff for EXTRAN
static const double SLOT_CROWN_CUTOFF   = 0.985257; // crown cutoff for SLOT
static const int    DEFAULT_MAXTRIALS   = 8;      // Max. trials per time step


//-----------------------------------------------------------------------------
//  Data Structures
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//  Shared Variables
//-----------------------------------------------------------------------------
static double  VariableStep;           // size of variable time step (sec)
TXnode* Xnode = NULL;                  // extended nodal information

static double  Omega;                  // actual under-relaxation parameter
static int     Steps;                  // number of Picard iterations

//-----------------------------------------------------------------------------
//  Function declarations
//-----------------------------------------------------------------------------
static void   initRoutingStep(void);
static void   initNodeStates(void);
static void   findBypassedLinks();
static void   findLimitedLinks();

static void   findLinkFlows(double dt);
static int    isTrueConduit(int link);
static void   findNonConduitFlow(int link, double dt);
static void   findNonConduitSurfArea(int link);
static double getModPumpFlow(int link, double q, double dt);
static void   updateNodeFlows(int link);
static void   updateConvergenceStats();

static int    findNodeDepths(double dt);
static void   setNodeDepth(int node, double dt);
static double getFloodedDepth(int node, int canPond, double dV, double yNew,
              double yMax, double dt);

static double getVariableStep(double maxStep);
static double getLinkStep(double tMin, int *minLink);
static double getNodeStep(double tMin, int *minNode);

//=============================================================================

void dynwave_init()
//
//  Input:   none
//  Output:  none
//  Purpose: initializes dynamic wave routing method.
//
{
    int i, j;
    double z;

    VariableStep = 0.0;
    Xnode = (TXnode *) calloc(Nobjects[NODE], sizeof(TXnode));
    if ( Xnode == NULL )
    {
        report_writeErrorMsg(ERR_MEMORY,
            " Not enough memory for dynamic wave routing.");
        return;
    }
    
    // --- initialize node surface areas & crown elev.
    for (i = 0; i < Nobjects[NODE]; i++ )
    {
        Xnode[i].newSurfArea = 0.0;
        Xnode[i].oldSurfArea = 0.0;
        Node[i].crownElev = Node[i].invertElev;
    }

    // --- initialize links & update node crown elevations
    for (i = 0; i < Nobjects[LINK]; i++)
    {
        j = Link[i].node1;
        z = Node[j].invertElev + Link[i].offset1 + Link[i].xsect.yFull;
        Node[j].crownElev = MAX(Node[j].crownElev, z);
        
        j = Link[i].node2;
        z = Node[j].invertElev + Link[i].offset2 + Link[i].xsect.yFull;
        Node[j].crownElev = MAX(Node[j].crownElev, z);
        Link[i].flowClass = DRY;
        Link[i].dqdh = 0.0;
    }

    // --- set crown cutoff for finding top width of closed conduits
    if ( SurchargeMethod == SLOT ) CrownCutoff = SLOT_CROWN_CUTOFF;
    else                           CrownCutoff = EXTRAN_CROWN_CUTOFF;

#ifdef BUILD_GPU
    // --- initialize GPU data for non-conduit links and curves if GPU is enabled
    // NOTE: Curves are needed for TABULAR storage nodes, so initialize even if no non-conduit links
    if (g_gpuConfig.useCuda && (Nlinks[PUMP] > 0 || Nlinks[ORIFICE] > 0 ||
                                 Nlinks[WEIR] > 0 || Nlinks[OUTLET] > 0 ||
                                 Nobjects[CURVE] > 0))
    {
        if (gpu_initializeNonConduitData() != 0) {
            printf("\n  WARNING: Failed to initialize GPU non-conduit data, disabling GPU\n");
            g_gpuConfig.useCuda = 0;
        }
    }
#endif
}

//=============================================================================

void  dynwave_close()
//
//  Input:   none
//  Output:  none
//  Purpose: frees memory allocated for dynamic wave routing method.
//
{
#ifdef BUILD_GPU
    gpu_freeNonConduitData();
#endif
    FREE(Xnode);
}

//=============================================================================

void dynwave_validate()
//
//  Input:   none
//  Output:  none
//  Purpose: adjusts dynamic wave routing options.
//
{
    if ( MinRouteStep > RouteStep ) MinRouteStep = RouteStep;
    if ( MinRouteStep < MINTIMESTEP ) MinRouteStep = MINTIMESTEP;
    if ( MinSurfArea == 0.0 ) MinSurfArea = DEFAULT_SURFAREA;
    else MinSurfArea /= UCF(LENGTH) * UCF(LENGTH);
    if ( HeadTol == 0.0 ) HeadTol = DEFAULT_HEADTOL;
    else HeadTol /= UCF(LENGTH);
    if ( MaxTrials == 0 ) MaxTrials = DEFAULT_MAXTRIALS;
}

//=============================================================================

double dynwave_getRoutingStep(double fixedStep)
//
//  Input:   fixedStep = user-supplied fixed time step (sec)
//  Output:  returns routing time step (sec)
//  Purpose: computes variable routing time step if applicable.
//
{
    // --- use user-supplied fixed step if variable step option turned off
    //     or if its smaller than the min. allowable variable time step
    if ( CourantFactor == 0.0 ) return fixedStep;
    if ( fixedStep < MINTIMESTEP ) return fixedStep;

    // --- at start of simulation (when current variable step is zero)
    //     use the minimum allowable time step
    if ( VariableStep == 0.0 )
    {
        VariableStep = MinRouteStep;
    }

    // --- otherwise compute variable step based on current flow solution
    else VariableStep = getVariableStep(fixedStep);

    // FAST PATCH DISABLED: Testing if proper fix (deferred timestep calculation) is sufficient
    // The proper fix moves getRoutingStep() to AFTER routing_execute() completes
    // TODO: Re-enable if proper fix alone doesn't prevent 30s+pump explosion
#if 0
    static int routingStepCounter = 0;
    if (g_gpuConfig.useCuda && routingStepCounter < 5) {
        double maxEarlyStep = 1.0;  // Cap at 1 second for first 5 steps
        if (VariableStep > maxEarlyStep) {
            if (routingStepCounter < 3) {
                printf("  PATCH: Capping step %d from %.3f to %.3f sec (GPU cold-start protection)\n",
                       routingStepCounter, VariableStep, maxEarlyStep);
            }
            VariableStep = maxEarlyStep;
        }
    }
    routingStepCounter++;
#endif

    // --- adjust step to be a multiple of a millisecond
    VariableStep = floor(1000.0 * VariableStep) / 1000.0;

    // DEBUG: Log computed variable step
    static int getStepCallCount = 0;
    if (getStepCallCount < 10) {
        printf("DEBUG_GETSTEP[call=%d]: fixedStep=%.6f → VariableStep=%.6f (GPU=%d)\n",
               getStepCallCount, fixedStep, VariableStep, g_gpuConfig.useCuda);
    }
    getStepCallCount++;

    return VariableStep;
}

//=============================================================================

int dynwave_execute(double tStep)
//
//  Input:   links = array of topo sorted links indexes
//           tStep = time step (sec)
//  Output:  returns number of iterations used
//  Purpose: routes flows through drainage network over current time step.
//
{
    int converged;

    // --- initialize
    if ( ErrorCode ) return 0;
    Steps = 0;
    converged = FALSE;
    Omega = OMEGA;
    initRoutingStep();

    // --- DEBUG: Log node states for mass balance tracking
    static FILE* massBalanceLog = NULL;
    static int routingStepCount = 0;
    if (routingStepCount < 20 && g_gpuConfig.useCuda) {  // First 20 routing steps
        if (massBalanceLog == NULL) {
            massBalanceLog = fopen("/tmp/gpu_mass_balance.txt", "w");
            if (massBalanceLog) {
                fprintf(massBalanceLog, "# Mass Balance Debug Log\n");
                fprintf(massBalanceLog, "# Format: step,nodeID,oldDepth,newDepth,oldVolume,newVolume,inflow,outflow\n");
            }
        }
    }
    routingStepCount++;

    // --- Picard iteration loop (both CPU and GPU use same structure)
    // --- Link flows must be recomputed each iteration because they depend
    //     on node heads, which change as node depths are updated

    // DEBUG: Log first few routing steps
    static int routingStepDebugCount = 0;
    int logThisStep = (routingStepDebugCount < 5);
    if (logThisStep) {
        printf("=== ROUTING STEP %d: tStep = %.6f sec (GPU=%d) ===\n", routingStepDebugCount, tStep, g_gpuConfig.useCuda);
    }
    routingStepDebugCount++;

#ifdef BUILD_GPU
    // Try GPU Picard iteration if CUDA is enabled
    if (g_gpuConfig.useCuda) {
        int iterations, gpuConverged;
        int result = gpu_runPersistentPicardIteration(
            tStep, AllowPonding, SurchargeMethod, MinSurfArea,
            Omega, HeadTol, MaxTrials, &iterations, &gpuConverged);

        if (result == 0) {
            // GPU Picard succeeded
            Steps = iterations;
            converged = gpuConverged;
            if (!converged) updateConvergenceStats();
            if (logThisStep) {
                printf("  GPU Picard completed: %d iterations, %s\n",
                       iterations, converged ? "converged" : "max iterations reached");
            }
            goto gpu_path_complete;
        }
        // Fall through to CPU on GPU failure
        if (logThisStep) {
            printf("  GPU Picard failed, falling back to CPU\n");
        }
    }
#endif

    // CPU Picard loop (fallback or when GPU disabled)
    while ( Steps < MaxTrials )
    {
        // --- execute a routing step & check for nodal convergence
        initNodeStates();
        if (logThisStep && g_gpuConfig.useCuda) {
            printf("  Picard iteration %d: calling findLinkFlows(dt=%.6f)\n", Steps, tStep);
        }
        findLinkFlows(tStep);
        if (logThisStep && g_gpuConfig.useCuda) {
            printf("  Picard iteration %d: calling findNodeDepths(dt=%.6f)\n", Steps, tStep);
        }
        converged = findNodeDepths(tStep);
        Steps++;
        if ( Steps > 1 )
        {
            if ( converged ) break;

            // --- check if link calculations can be skipped in next step
            findBypassedLinks();
        }
    }
    if ( !converged ) updateConvergenceStats();

#ifdef BUILD_GPU
gpu_path_complete:
#endif

#ifdef BUILD_GPU
    if (g_gpuConfig.useCuda) {
        gpu_flushConduitResults();

        // DEBUG: Log link flows after GPU flush (first 10 routing steps)
        static int flushDebugCount = 0;
        if (flushDebugCount < 10) {
            // Check a few key links for non-zero flows
            int nonZeroCount = 0;
            for (int i = 0; i < MIN(Nobjects[LINK], 100); i++) {
                if (fabs(Link[i].newFlow) > 0.01) nonZeroCount++;
            }
            printf("  gpu_flushConduitResults[step=%d]: %d/%d links have non-zero flow\n",
                   flushDebugCount, nonZeroCount, MIN(Nobjects[LINK], 100));
            // Log specific link for Session18
            if (Nobjects[LINK] > 895) {
                printf("    Link 895: newFlow=%.3f froude=%.3f newVolume=%.3f\n",
                       Link[895].newFlow, Link[895].froude, Link[895].newVolume);
            }
        }
        flushDebugCount++;
    }
#endif

    //  --- identify any capacity-limited conduits
    findLimitedLinks();

    // --- DEBUG: Log node states after routing step
    if (routingStepCount <= 20 && g_gpuConfig.useCuda && massBalanceLog) {
        for (int i = 0; i < Nobjects[NODE]; i++) {
            // Only log nodes with significant activity
            if (Node[i].newDepth > 0.01 || Node[i].inflow > 0.01 || Node[i].outflow > 0.01) {
                fprintf(massBalanceLog, "%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        routingStepCount-1, Node[i].ID,
                        Node[i].oldDepth, Node[i].newDepth,
                        Node[i].oldVolume, Node[i].newVolume,
                        Node[i].inflow, Node[i].outflow);
            }
        }
        fflush(massBalanceLog);

        if (routingStepCount == 20) {
            fprintf(massBalanceLog, "# Logging complete (20 routing steps)\n");
            fclose(massBalanceLog);
            massBalanceLog = NULL;
            printf("\n  ... GPU mass balance log written to /tmp/gpu_mass_balance.txt\n");
        }
    }

    return Steps;
}

//=============================================================================

void updateConvergenceStats()
{
    int i;
    NonConvergeCount++;
    for (i = 0; i < Nobjects[NODE]; i++)
        stats_updateConvergenceStats(i, Xnode[i].converged);
}

//=============================================================================

void   initRoutingStep()
{
    int i;
    for (i = 0; i < Nobjects[NODE]; i++)
    {
        Xnode[i].converged = FALSE;
        Xnode[i].dYdT = 0.0;
    }
    for (i = 0; i < Nobjects[LINK]; i++)
    {
        Link[i].bypassed = FALSE;
        Link[i].surfArea1 = 0.0;
        Link[i].surfArea2 = 0.0;
    }

    // --- a2 preserves conduit area from solution at last time step
    for ( i = 0; i < Nlinks[CONDUIT]; i++) Conduit[i].a2 = Conduit[i].a1;
}

//=============================================================================

void initNodeStates()
//
//  Input:   none
//  Output:  none
//  Purpose: initializes node's surface area, inflow & outflow
//
{
    int i;

    for (i = 0; i < Nobjects[NODE]; i++)
    {
        // --- initialize nodal surface area
        if ( AllowPonding )
        {
            Xnode[i].newSurfArea = node_getPondedArea(i, Node[i].newDepth);
        }
        else
        {
            Xnode[i].newSurfArea = node_getSurfArea(i, Node[i].newDepth);
        }

        // --- initialize nodal inflow & outflow
        Node[i].inflow = 0.0;
        Node[i].outflow = Node[i].losses;
        if ( Node[i].newLatFlow >= 0.0 )
        {    
            Node[i].inflow += Node[i].newLatFlow;
        }
        else
        {    
            Node[i].outflow -= Node[i].newLatFlow;
        }
        Xnode[i].sumdqdh = 0.0;
    }
}

//=============================================================================

void   findBypassedLinks()
{
    int i;
    for (i = 0; i < Nobjects[LINK]; i++)
    {
        if ( Xnode[Link[i].node1].converged &&
             Xnode[Link[i].node2].converged )
             Link[i].bypassed = TRUE;
        else Link[i].bypassed = FALSE;
    }
}

//=============================================================================

void  findLimitedLinks()
//
//  Input:   none
//  Output:  none
//  Purpose: determines if a conduit link is capacity limited.
//
{
    int    j, n1, n2, k;
    double h1, h2;

    for (j = 0; j < Nobjects[LINK]; j++)
    {
        // ---- check only non-dummy conduit links
        if ( !isTrueConduit(j) ) continue;

        // --- check that upstream end is full
        k = Link[j].subIndex;
        Conduit[k].capacityLimited = FALSE;
        if ( Conduit[k].a1 >= Link[j].xsect.aFull )
        {
            // --- check if HGL slope > conduit slope
            n1 = Link[j].node1;
            n2 = Link[j].node2;
            h1 = Node[n1].newDepth + Node[n1].invertElev;
            h2 = Node[n2].newDepth + Node[n2].invertElev;
            if ( (h1 - h2) > fabs(Conduit[k].slope) * Conduit[k].length )
                Conduit[k].capacityLimited = TRUE;
        }
    }
}

//=============================================================================

void findLinkFlows(double dt)
{
    int i;

#ifdef BUILD_GPU
    // --- try GPU path first if enabled
    if (g_gpuConfig.useCuda)
    {
        double crownCutoff = (SurchargeMethod == EXTRAN) ? EXTRAN_CROWN_CUTOFF : SLOT_CROWN_CUTOFF;

        int gpuResult = gpu_computeConduitFlows(
            &g_gpuLinks,
            &g_gpuConduits,
            &g_gpuXsects,
            &g_gpuNodes,
            dt,
            Steps,
            Omega,
            SurchargeMethod,
            crownCutoff,
            InertDamping);

        // If GPU succeeded, return (ALL links and node flows already updated by GPU)
        if (gpuResult == 0)
        {
            // GPU has processed ALL link types (conduits, pumps, orifices, weirs, outlets)
            // and updated node flows via sequential pump processing
            return;
        }

        // Otherwise fall through to CPU path (gpuResult < 0 indicates GPU error)
    }
#endif

    // --- find new flow in each non-dummy conduit (CPU path)
#pragma omp parallel num_threads(NumThreads)
{
    #pragma omp for
    for ( i = 0; i < Nobjects[LINK]; i++)
    {
        if ( isTrueConduit(i) && !Link[i].bypassed )
            dwflow_findConduitFlow(i, Steps, Omega, dt);
    }
}

    // --- update inflow/outflows for nodes attached to non-dummy conduits
    for ( i = 0; i < Nobjects[LINK]; i++)
    {
        if ( isTrueConduit(i) ) updateNodeFlows(i);
    }

    // --- find new flows for all dummy conduits, pumps & regulators
    for ( i = 0; i < Nobjects[LINK]; i++)
    {
        if ( !isTrueConduit(i) )
        {
            if ( !Link[i].bypassed ) findNonConduitFlow(i, dt);
            updateNodeFlows(i);
        }
    }
}

//=============================================================================

int isTrueConduit(int j)
{
    return ( Link[j].type == CONDUIT && Link[j].xsect.type != DUMMY );
}

//=============================================================================

void findNonConduitFlow(int i, double dt)
//
//  Input:   i = link index
//           dt = time step (sec)
//  Output:  none
//  Purpose: finds new flow in a non-conduit-type link
//
{
    double qLast;                      // previous link flow (cfs)
    double qNew;                       // new link flow (cfs)

    // --- get link flow from last iteration
    qLast = Link[i].newFlow;
    Link[i].dqdh = 0.0;

    // --- get new inflow to link from its upstream node
    //     (link_getInflow returns 0 if flap gate closed or pump is offline)
    qNew = link_getInflow(i);
    if ( Link[i].type == PUMP ) qNew = getModPumpFlow(i, qNew, dt);

    // --- find surface area at each end of link
    findNonConduitSurfArea(i);

    // --- apply under-relaxation with flow from previous iteration;
    // --- do not allow flow to change direction without first being 0
    if ( Steps > 0 && Link[i].type != PUMP ) 
    {
        qNew = (1.0 - Omega) * qLast + Omega * qNew;
        if ( qNew * qLast < 0.0 ) qNew = 0.001 * SGN(qNew);
    }
    Link[i].newFlow = qNew;
}

//=============================================================================

double getModPumpFlow(int i, double q, double dt)
//
//  Input:   i = link index
//           q = pump flow from pump curve (cfs)
//           dt = time step (sec)
//  Output:  returns modified pump flow rate (cfs)
//  Purpose: modifies pump curve pumping rate depending on amount of water
//           available at pump's inlet node.
//
{
    int    j = Link[i].node1;          // pump's inlet node index
    int    k = Link[i].subIndex;       // pump's index
    double newNetInflow;               // inflow - outflow rate (cfs)
    double netFlowVolume;              // inflow - outflow volume (ft3)
    double y;                          // node depth (ft)
    static int pumpDebugCounts[64];
    int debugSlot = -1;
    int logPump = (k == 1 || k == 15 || k == 20 || k == 22 || k == 24 || k == 25 || k == 26 ||
                   k == 32 || k == 35 || k == 38 || k == 41 || k == 43 || k == 45);

    if ( q == 0.0 ) return q;

    if (logPump) debugSlot = pumpDebugCounts[k]++;

    // --- case where inlet node is a storage node: 
    //     prevent node volume from going negative
    if ( Node[j].type == STORAGE ) {
        if (logPump && debugSlot < 200) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "CPU pump[%d] storage pre-mod (call=%d): dt=%.6f node=%d oldVol=%.6f inflow=%.6f outflow=%.6f oldNet=%.6f qCurve=%.6f",
                k, debugSlot, dt, j, Node[j].oldVolume, Node[j].inflow, Node[j].outflow,
                Node[j].oldNetInflow, q);
            printf("%s\n", msg);
        }
        double qMod = node_getMaxOutflow(j, q, dt);
        if (logPump && debugSlot < 200) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "CPU pump[%d] storage post-mod (call=%d): qFinal=%.6f", k, debugSlot, qMod);
            printf("%s\n", msg);
        }
        return qMod;
    }

    // --- case where inlet is a non-storage node
    switch ( Pump[k].type )
    {
      // --- for Type1 pump, a volume is computed for inlet node,
      //     so make sure it doesn't go negative
      case TYPE1_PUMP:
        return node_getMaxOutflow(j, q, dt);

      // --- for other types of pumps, if pumping rate would make depth
      //     at upstream node negative, then set pumping rate = inflow
      case TYPE2_PUMP:
      case TYPE4_PUMP:
      case TYPE3_PUMP:
         newNetInflow = Node[j].inflow - Node[j].outflow - q;
         netFlowVolume = 0.5 * (Node[j].oldNetInflow + newNetInflow ) * dt;
         y = Node[j].oldDepth + netFlowVolume / Xnode[j].newSurfArea;
         if (logPump && debugSlot < 200) {
             char msg[256];
             snprintf(msg, sizeof(msg),
                 "CPU pump[%d] junction pre-mod (call=%d): dt=%.6f node=%d oldDepth=%.6f oldVol=%.6f surf=%.6f inflow=%.6f outflow=%.6f qCurve=%.6f y=%.6f",
                 k, debugSlot, dt, j, Node[j].oldDepth, Node[j].oldVolume, Xnode[j].newSurfArea, Node[j].inflow, Node[j].outflow, q, y);
             printf("%s\n", msg);
         }
         if ( y <= 0.0 ) {
             if (logPump && debugSlot < 200) {
                 char msg[256];
                 snprintf(msg, sizeof(msg),
                     "CPU pump[%d] junction limited by depth (call=%d): returning inflow %.6f",
                     k, debugSlot, Node[j].inflow);
                 printf("%s\n", msg);
             }
             return Node[j].inflow;
         }
    }
    return q;
}

//=============================================================================

void  findNonConduitSurfArea(int i)
//
//  Input:   i = link index
//  Output:  none
//  Purpose: finds the surface area contributed by a non-conduit
//           link to its upstream and downstream nodes.
//
{
    if ( Link[i].type == ORIFICE )
    {
        Link[i].surfArea1 = Orifice[Link[i].subIndex].surfArea / 2.;
    }

    // --- no surface area for weirs to maintain SWMM 4 compatibility
    else Link[i].surfArea1 = 0.0;

    Link[i].surfArea2 = Link[i].surfArea1;
    if ( Link[i].flowClass == UP_CRITICAL ||
        Node[Link[i].node1].type == STORAGE ) Link[i].surfArea1 = 0.0;
    if ( Link[i].flowClass == DN_CRITICAL ||
        Node[Link[i].node2].type == STORAGE ) Link[i].surfArea2 = 0.0;
}

//=============================================================================

void updateNodeFlows(int i)
//
//  Input:   i = link index
//           q = link flow rate (cfs)
//  Output:  none
//  Purpose: updates cumulative inflow & outflow at link's end nodes.
//
{
    int    k;
    int    barrels = 1;
    int    n1 = Link[i].node1;
    int    n2 = Link[i].node2;
    double q = Link[i].newFlow;
    double conduitLossRate = 0.0;

    // --- update total inflow & outflow at upstream/downstream nodes
    if ( q >= 0.0 )
    {
        Node[n1].outflow += q;
        Node[n2].inflow  += q;
    }
    else
    {
        Node[n1].inflow   -= q;
        Node[n2].outflow  -= q;
    }
 
    // --- add any uniform evap & seepage loss from conduit link
    if ( Link[i].type == CONDUIT )
    {
        k = Link[i].subIndex;
        barrels = Conduit[k].barrels;
        conduitLossRate = (Conduit[k].evapLossRate + Conduit[k].seepLossRate) *
                          barrels;
        if (conduitLossRate > 0.0)
        {
            // --- outfall nodes do not share evap & seepage losses
            if (Node[n1].type != OUTFALL && Node[n2].type != OUTFALL)
                conduitLossRate /= 2.0;
            if (Node[n1].type != OUTFALL)
                Node[n1].outflow += conduitLossRate;
            if (Node[n2].type != OUTFALL)
                Node[n2].outflow += conduitLossRate;
        }
    }
    
    // --- add surf. area contributions to upstream/downstream nodes
    // DEBUG: Log surface area contributions to node 1 (J2) during routing step 1
    static int cpu_updateFlow_routingStep = 0;
    static int cpu_updateFlow_lastSteps = -1;
    if (Steps == 0 && cpu_updateFlow_lastSteps != 0) cpu_updateFlow_routingStep++;
    cpu_updateFlow_lastSteps = Steps;

    if ((Link[i].node1 == 1 || Link[i].node2 == 1) && cpu_updateFlow_routingStep == 1 && Steps == 0) {
        printf("  CPU_SURF_CONTRIB: link%d(n1=%d n2=%d) surfArea1=%.6f surfArea2=%.6f barrels=%d\n",
               i, Link[i].node1, Link[i].node2, Link[i].surfArea1, Link[i].surfArea2, barrels);
        if (Link[i].node1 == 1) {
            printf("    → node1=%d gets surfArea1*barrels = %.6f*%d = %.6f (before: %.6f after: %.6f)\n",
                   Link[i].node1, Link[i].surfArea1, barrels, Link[i].surfArea1 * barrels,
                   Xnode[Link[i].node1].newSurfArea, Xnode[Link[i].node1].newSurfArea + Link[i].surfArea1 * barrels);
        }
        if (Link[i].node2 == 1) {
            printf("    → node2=%d gets surfArea2*barrels = %.6f*%d = %.6f (before: %.6f after: %.6f)\n",
                   Link[i].node2, Link[i].surfArea2, barrels, Link[i].surfArea2 * barrels,
                   Xnode[Link[i].node2].newSurfArea, Xnode[Link[i].node2].newSurfArea + Link[i].surfArea2 * barrels);
        }
    }

    Xnode[Link[i].node1].newSurfArea += Link[i].surfArea1 * barrels;
    Xnode[Link[i].node2].newSurfArea += Link[i].surfArea2 * barrels;

    // --- update summed value of dqdh at each end node
    Xnode[Link[i].node1].sumdqdh += Link[i].dqdh;
    if ( Link[i].type == PUMP )
    {
        k = Link[i].subIndex;
        if ( Pump[k].type != TYPE4_PUMP )
        {
            Xnode[n2].sumdqdh += Link[i].dqdh;
        }
    }
    else Xnode[n2].sumdqdh += Link[i].dqdh;
}

//=============================================================================

int findNodeDepths(double dt)
//
//  Input:   dt = time step (sec)
//  Output:  returns TRUE if depth change at all non-Outfall nodes is
//           within the convergence tolerance and FALSE otherwise
//  Purpose: finds new depth at all nodes and checks if convergence achieved.
//
{
    int i;
    double yOld = 0.0;       // previous node depth (ft)

    // --- compute outfall depths based on flow in connecting link
    for ( i = 0; i < Nobjects[LINK]; i++ ) link_setOutfallDepth(i);

#ifdef BUILD_GPU
    // --- GPU-only path: no CPU fallback
    if (g_gpuConfig.useCuda)
    {
        int gpuConverged = gpu_runNodeDepthKernel(
            dt,
            AllowPonding,
            SurchargeMethod,
            MinSurfArea,
            Steps,
            Omega,
            HeadTol);

        if (gpuConverged < 0)
        {
            // GPU kernel failed - report error and abort
            report_writeErrorMsg(ERR_SYSTEM,
                "GPU node depth kernel failed. Aborting simulation.");
            ErrorCode = ERR_SYSTEM;
            return FALSE;
        }

        // GPU succeeded - check convergence
        for (i = 0; i < Nobjects[NODE]; i++)
        {
            if ( Node[i].type == OUTFALL ) continue;
            if (Xnode[i].converged == FALSE) return FALSE;
        }
        return TRUE;
    }
#endif

    // --- CPU path (only when GPU is disabled)
    // --- compute new depth for all non-outfall nodes and determine if
    //     depth change from previous iteration is below tolerance
#pragma omp parallel num_threads(NumThreads)
{
    #pragma omp for private(yOld)
    for ( i = 0; i < Nobjects[NODE]; i++ )
    {
        if ( Node[i].type == OUTFALL ) continue;
        yOld = Node[i].newDepth;
        setNodeDepth(i, dt);
        Xnode[i].converged = TRUE;
        if ( fabs(yOld - Node[i].newDepth) > HeadTol )
        {
            Xnode[i].converged = FALSE;

            // DEBUG: Log non-converging nodes for first 5 routing steps
            static int cpuRoutingStepCount = 0;
            if (cpuRoutingStepCount < 5 && Steps <= 8) {
                double depthChange = fabs(yOld - Node[i].newDepth);
                printf("  CPU_NODE_FAIL[step=%d iter=%d node=%d type=%d]: depthChange=%.6f > tol=%.6f (%.1fx) depth: %.6f→%.6f in=%.3f out=%.3f\n",
                       cpuRoutingStepCount, Steps, i, Node[i].type,
                       depthChange, HeadTol, depthChange / HeadTol,
                       yOld, Node[i].newDepth,
                       Node[i].inflow, Node[i].outflow);
            }
            if (Steps == 0) cpuRoutingStepCount++;  // Increment once per routing step
        }
    }
}

   // --- return FALSE if any non-Outfall node failed to converge
    // DEBUG: Track convergence for first 5 routing steps
    static int cpuResidualRoutingStep = 0;
    static int cpuResidualLastIter = -1;
    if (Steps == 0 && cpuResidualLastIter != 0) cpuResidualRoutingStep++;
    cpuResidualLastIter = Steps;

    if (cpuResidualRoutingStep >= 1 && cpuResidualRoutingStep <= 20) {
        int notConvergedCount = 0;
        int totalNodes = 0;

        for (i = 0; i < Nobjects[NODE]; i++) {
            if (Node[i].type == OUTFALL) continue;
            totalNodes++;
            if (!Xnode[i].converged) {
                notConvergedCount++;
            }
        }

        int convergedCount = totalNodes - notConvergedCount;
        printf("CPU_RESIDUAL[routeStep=%d iter=%d]: notConverged=%d/%d converged=%d/%d tol=%.6f\n",
               cpuResidualRoutingStep, Steps,
               notConvergedCount, totalNodes,
               convergedCount, totalNodes, HeadTol);
    }

    for (i = 0; i < Nobjects[NODE]; i++)
    {
        if ( Node[i].type == OUTFALL ) continue;
        if (Xnode[i].converged == FALSE) return FALSE;
    }
    return TRUE;
}

//=============================================================================

void setNodeDepth(int i, double dt)
//
//  Input:   i  = node index
//           dt = time step (sec)
//  Output:  none
//  Purpose: sets depth at non-outfall node after current time step.
//
{
    int     canPond;                   // TRUE if node can pond overflows
    int     isPonded;                  // TRUE if node is currently ponded 
    int     isSurcharged = FALSE;      // TRUE if node is surcharged
    double  dQ;                        // inflow minus outflow at node (cfs)
    double  dV;                        // change in node volume (ft3)
    double  dy;                        // change in node depth (ft)
    double  yMax;                      // max. depth at node (ft)
    double  yOld;                      // node depth at previous time step (ft)
    double  yLast;                     // previous node depth (ft)
    double  yNew;                      // new node depth (ft)
    double  yCrown;                    // depth to node crown (ft)
    double  surfArea;                  // node surface area (ft2)
    double  denom;                     // denominator term
    double  corr;                      // correction factor
    double  f;                         // relative surcharge depth

    // --- see if node can pond water above it
    canPond = (AllowPonding && Node[i].pondedArea > 0.0);
    isPonded = (canPond && Node[i].newDepth > Node[i].fullDepth);

    // --- initialize values
    yCrown = Node[i].crownElev - Node[i].invertElev;
    yOld = Node[i].oldDepth;
    yLast = Node[i].newDepth;
    Node[i].overflow = 0.0;
    surfArea = Xnode[i].newSurfArea;
    surfArea = MAX(surfArea, MinSurfArea);

    // DEBUG: Trace STOR-10 (node 926) surface area accumulation (first 3 routing steps)
    #ifdef GPU_DEBUG_SURF
    static int cpu_depth_steps = 0;
    if (i == 926 && Steps <= 3 && Node[i].type == STORAGE) {
        printf("CPU_STOR10_DEPTH[step=%d]: newSurfArea(conduits)=%.3f minSurfArea=%.3f surfArea(used)=%.3f\n",
               Steps, Xnode[i].newSurfArea, MinSurfArea, surfArea);
        printf("  inflow=%.6f outflow=%.6f dQ=%.6f oldNet=%.6f\n",
               Node[i].inflow, Node[i].outflow, Node[i].inflow - Node[i].outflow, Node[i].oldNetInflow);
        printf("  yOld=%.6f yLast=%.6f fullDepth=%.3f dt=%.6f\n",
               yOld, yLast, Node[i].fullDepth, dt);
    }
    #endif

    // --- determine average net flow volume into node over the time step
    dQ = Node[i].inflow - Node[i].outflow;
    dV = 0.5 * (Node[i].oldNetInflow + dQ) * dt;

    // DEBUG: Detailed logging for node 852 (storage being over-throttled)
    static int cpu_routingStepCounter = 0;
    static int cpu_lastSteps = -1;
    static int firstStorageIdxCPU = -1;
    if (Steps == 0 && cpu_lastSteps != 0) cpu_routingStepCounter++;
    cpu_lastSteps = Steps;

    // Log node 852 specifically
    if (i == 852 && cpu_routingStepCounter >= 1 && cpu_routingStepCounter <= 3 && Steps <= 2) {
        printf("CPU_NODE852_STOR[step=%d iter=%d]: oldVol=%.6f oldDepth=%.6f oldNetIn=%.6f\n",
               cpu_routingStepCounter, Steps, Node[i].oldVolume, yOld, Node[i].oldNetInflow);
        printf("  inflow=%.6f outflow=%.6f dQ=%.6f dV=%.6f dt=%.6f\n",
               Node[i].inflow, Node[i].outflow, dQ, dV, dt);
        printf("  surfArea=%.6f newSurfArea=%.6f fullVol=%.6f fullDepth=%.6f\n",
               surfArea, Xnode[i].newSurfArea, Node[i].fullVolume, Node[i].fullDepth);
    }

    if (Node[i].type == STORAGE && cpu_routingStepCounter <= 3 && Steps == 0) {
        if (firstStorageIdxCPU == -1) firstStorageIdxCPU = i;
        if (i == firstStorageIdxCPU) {
            printf("CPU_VOL_INT[step=%d iter=%d node=%d %s]: oldNetIn=%.6f in=%.6f out=%.6f dQ=%.6f dV=%.6f surfArea=%.6f oldVol=%.6f\n",
                   cpu_routingStepCounter, Steps, i, Node[i].ID, Node[i].oldNetInflow, Node[i].inflow, Node[i].outflow, dQ, dV, surfArea, Node[i].oldVolume);
        }
    }

    // DEBUG: Comprehensive Node 1 (J2, index 1) input logging for routing steps 150-151, iterations 0-1
    if (i == 1 && cpu_routingStepCounter >= 150 && cpu_routingStepCounter <= 151 && Steps <= 1) {
        printf("CPU_NODE1_INPUTS[routingStep=%d iter=%d]:\n", cpu_routingStepCounter, Steps);
        printf("  oldDepth=%.6f oldVolume=%.6f oldNetInflow=%.6f\n",
               Node[i].oldDepth, Node[i].oldVolume, Node[i].oldNetInflow);
        printf("  inflow=%.6f outflow=%.6f dQ=%.6f\n",
               Node[i].inflow, Node[i].outflow, dQ);
        printf("  dV=%.6f dt=%.6f surfArea=%.6f\n", dV, dt, surfArea);
        printf("  newDepth_last=%.6f fullDepth=%.6f\n", yLast, Node[i].fullDepth);
    }

    // --- determine if node is EXTRAN surcharged
    if (SurchargeMethod == EXTRAN)
    {
        // --- ponded nodes don't surcharge
        if (isPonded) isSurcharged = FALSE;

        // --- closed storage units that are full are in surcharge
        else if (Node[i].type == STORAGE)
        {
            isSurcharged = (Node[i].surDepth > 0.0 &&
                            yLast > Node[i].fullDepth);
        }

        // --- surcharge occurs when node depth exceeds top of its highest link
        else isSurcharged = (yCrown > 0.0 && yLast > yCrown);
    }

    // --- if node not surcharged, base depth change on surface area        
    if (!isSurcharged)
    {
        dy = dV / surfArea;
        yNew = yOld + dy;

        // --- save non-ponded surface area for use in surcharge algorithm
        if ( !isPonded ) Xnode[i].oldSurfArea = surfArea;

        // --- apply under-relaxation to new depth estimate
        if ( Steps > 0 )
        {
            yNew = (1.0 - Omega) * yLast + Omega * yNew;
        }

        // --- don't allow a ponded node to drop much below full depth
        if ( isPonded && yNew < Node[i].fullDepth )
            yNew = Node[i].fullDepth - FUDGE;
    }

    // --- if node surcharged, base depth change on dqdh
    //     NOTE: depth change is w.r.t depth from previous
    //     iteration; also, do not apply under-relaxation.
    else
    {
        // --- apply correction factor for upstream terminal nodes
        corr = 1.0;
        if ( Node[i].degree < 0 ) corr = 0.6;

        // --- allow surface area from last non-surcharged condition
        //     to influence dqdh if depth close to crown depth
        denom = Xnode[i].sumdqdh;
        if ( yLast < 1.25 * yCrown )
        {
            f = (yLast - yCrown) / yCrown;
            denom += (Xnode[i].oldSurfArea/dt -
                      Xnode[i].sumdqdh) * exp(-15.0 * f);
        }

        // --- compute new estimate of node depth
        if ( denom == 0.0 ) dy = 0.0;
        else dy = corr * dQ / denom;
        yNew = yLast + dy;
        if ( yNew < yCrown ) yNew = yCrown - FUDGE;

        // --- don't allow a newly ponded node to rise much above full depth
        if ( canPond && yNew > Node[i].fullDepth )
            yNew = Node[i].fullDepth + FUDGE;
    }

    // --- depth cannot be negative
    if ( yNew < 0 ) yNew = 0.0;

    // --- determine max. non-flooded depth
    yMax = Node[i].fullDepth;
    if ( canPond == FALSE ) yMax += Node[i].surDepth;

    // --- find flooded depth & volume
    if ( yNew > yMax )
    {
        yNew = getFloodedDepth(i, canPond, dV, yNew, yMax, dt);
    }
    else Node[i].newVolume = node_getVolume(i, yNew);

    // --- compute change in depth w.r.t. time
    Xnode[i].dYdT = fabs(yNew - yOld) / dt;

    // --- save new depth for node
    Node[i].newDepth = yNew;

    // DEBUG: Detailed trace for node 12 (J-229-OUT) at early steps to diagnose depth halving
    if (i == 12 && cpu_routingStepCounter <= 3 && Steps <= 2) {
        printf("CPU_NODE12[step=%d iter=%d]: oldDepth=%.9f yLast=%.9f inflow=%.6f outflow=%.6f\n",
               cpu_routingStepCounter, Steps, yOld, Node[i].newDepth, Node[i].inflow, Node[i].outflow);
        printf("  oldNetInflow=%.9f dQ=%.6f dV=%.9f surfArea=%.9f newSurfArea=%.9f\n",
               Node[i].oldNetInflow, dQ, dV, surfArea, Xnode[i].newSurfArea);
        printf("  dy_raw=%.9f yNew=%.9f omega=%.3f dYdT=%.6f dt=%.6f\n",
               dy, yNew, Omega, Xnode[i].dYdT, dt);
    }

    // DEBUG: Log outputs for node 852
    if (i == 852 && cpu_routingStepCounter >= 1 && cpu_routingStepCounter <= 3 && Steps <= 2) {
        printf("  OUTPUT: dy=%.6f yNew=%.6f newVol=%.6f overflow=%.6f\n",
               dV/surfArea, yNew, Node[i].newVolume, Node[i].overflow);
    }

    // DEBUG: Track divergence for J2 (node index 1) during routing steps 0-3
    static int cpu_diverge_routingStep = 0;
    static int cpu_diverge_lastIter = -1;
    if (Steps == 0 && cpu_diverge_lastIter != 0) cpu_diverge_routingStep++;
    cpu_diverge_lastIter = Steps;
    if (i == 1 && cpu_diverge_routingStep >= 0 && cpu_diverge_routingStep <= 3 && Steps <= 3) {
        printf("CPU_DIVERGE[routeStep=%d iter=%d node=J2]: inflow=%.6f outflow=%.6f oldDepth=%.6f newDepth=%.6f oldVol=%.6f newVol=%.6f\n",
               cpu_diverge_routingStep, Steps,
               Node[i].inflow, Node[i].outflow,
               Node[i].oldDepth, Node[i].newDepth,
               Node[i].oldVolume, Node[i].newVolume);
    }
}

//=============================================================================

double getFloodedDepth(int i, int canPond, double dV, double yNew,
                       double yMax, double dt)
//
//  Input:   i  = node index
//           canPond = TRUE if water can pond over node
//           isPonded = TRUE if water is currently ponded
//           dV = change in volume over time step (ft3)
//           yNew = current depth at node (ft)
//           yMax = max. depth at node before ponding (ft)
//           dt = time step (sec)
//  Output:  returns depth at node when flooded (ft)
//  Purpose: computes depth, volume and overflow for a flooded node.
//
{
    if ( canPond == FALSE )
    {
        Node[i].overflow = dV / dt;
        Node[i].newVolume = Node[i].fullVolume;
        yNew = yMax;
    }
    else
    {
        Node[i].newVolume = MAX((Node[i].oldVolume+dV), Node[i].fullVolume);
        Node[i].overflow = (Node[i].newVolume - 
            MAX(Node[i].oldVolume, Node[i].fullVolume)) / dt;
    }
    if ( Node[i].overflow < FUDGE ) Node[i].overflow = 0.0;
    return yNew;

}

//=============================================================================

#ifdef BUILD_GPU
void setNodeDepth_hostWrapper(int i, double dt)
{
    setNodeDepth(i, dt);
}
#endif

//=============================================================================

double getVariableStep(double maxStep)
//
//  Input:   maxStep = user-supplied max. time step (sec)
//  Output:  returns time step (sec)
//  Purpose: finds time step that satisfies stability criterion but
//           is no greater than the user-supplied max. time step.
//
{
    int    minLink = -1;                // index of link w/ min. time step
    int    minNode = -1;                // index of node w/ min. time step
    double tMin;                        // allowable time step (sec)
    double tMinLink;                    // allowable time step for links (sec)
    double tMinNode;                    // allowable time step for nodes (sec)

    // DEBUG: Log entry to getVariableStep (first 10 calls when using GPU)
    static int getVarStepCount = 0;
    if (g_gpuConfig.useCuda && getVarStepCount < 10) {
        printf("  getVariableStep[call=%d]: maxStep=%.6f\n", getVarStepCount, maxStep);
        // Sample a few links to see what data we're working with
        int sampleCount = 0;
        for (int i = 0; i < MIN(Nobjects[LINK], 100); i++) {
            if (Link[i].type == CONDUIT && fabs(Link[i].newFlow) > 0.01) {
                sampleCount++;
                if (sampleCount <= 3) {
                    printf("    Link %d: newFlow=%.3f froude=%.3f\n", i, Link[i].newFlow, Link[i].froude);
                }
            }
        }
        printf("    Total links with flow > 0.01: %d\n", sampleCount);
    }

    getVarStepCount++;

    // --- find stable time step for links & then nodes
    tMin = maxStep;
    tMinLink = getLinkStep(tMin, &minLink);
    tMinNode = getNodeStep(tMinLink, &minNode);

    // --- use smaller of the link and node time step
    tMin = tMinLink;
    if ( tMinNode < tMin )
    {
        tMin = tMinNode ;
        minLink = -1;
    }

    // DEBUG: Log computed timesteps (AFTER selection)
    if (g_gpuConfig.useCuda && getVarStepCount < 10) {
        printf("    getLinkStep returned: tMinLink=%.6f (minLink=%d)\n", tMinLink, minLink);
        printf("    getNodeStep returned: tMinNode=%.6f (minNode=%d)\n", tMinNode, minNode);
        printf("    Selected tMin=%.6f (before MinRouteStep check)\n", tMin);
    }

    // --- update count of times the minimum node or link was critical
    stats_updateCriticalTimeCount(minNode, minLink);

    // --- don't let time step go below an absolute minimum
    if ( tMin < MinRouteStep ) tMin = MinRouteStep;
    return tMin;
}

//=============================================================================

double getLinkStep(double tMin, int *minLink)
//
//  Input:   tMin = critical time step found so far (sec)
//  Output:  minLink = index of link with critical time step;
//           returns critical time step (sec)
//  Purpose: finds critical time step for conduits based on Courant criterion.
//
{
    int    i;                           // link index
    int    k;                           // conduit index
    double q;                           // conduit flow (cfs)
    double t;                           // time step (sec)
    double tLink = tMin;                // critical link time step (sec)

    // DEBUG: Log first call details
    static int linkStepCallCount = 0;
    int activeLinks = 0;
    int skippedByFlow = 0, skippedByArea = 0, skippedByFroude = 0;

    // --- examine each conduit link
    for ( i = 0; i < Nobjects[LINK]; i++ )
    {
        if ( Link[i].type == CONDUIT )
        {
            // --- skip conduits with negligible flow, area or Fr
            k = Link[i].subIndex;
            q = fabs(Link[i].newFlow) / Conduit[k].barrels;

            if (linkStepCallCount == 0 && i < 10) {
                printf("  CPU getLinkStep[link=%d]: newFlow=%.6f q=%.6f a1=%.6f froude=%.6f\n",
                       i, Link[i].newFlow, q, Conduit[k].a1, Link[i].froude);
            }

            if ( q <= FUDGE ) {
                skippedByFlow++;
                continue;
            }
            if ( Conduit[k].a1 <= FUDGE ) {
                skippedByArea++;
                continue;
            }
            if ( Link[i].froude <= 0.01 ) {
                skippedByFroude++;
                continue;
            }

            activeLinks++;

            // --- compute time step to satisfy Courant condition
            t = Link[i].newVolume / Conduit[k].barrels / q;
            t = t * Conduit[k].modLength / link_getLength(i);
            t = t * Link[i].froude / (1.0 + Link[i].froude) * CourantFactor;

            // --- update critical link time step
            if ( t < tLink )
            {
                tLink = t;
                *minLink = i;
                if (linkStepCallCount == 0) {
                    printf("  CPU getLinkStep: NEW MIN link=%d t=%.6f (vol=%.3f q=%.3f modLen=%.1f len=%.1f Fr=%.3f)\n",
                           i, t, Link[i].newVolume, q, Conduit[k].modLength, link_getLength(i), Link[i].froude);
                }
            }
        }
    }

    if (linkStepCallCount == 0) {
        printf("  CPU getLinkStep[call=0]: checked %d conduits, active=%d skipped(flow=%d area=%d froude=%d) → tLink=%.6f (minLink=%d)\n",
               Nobjects[LINK], activeLinks, skippedByFlow, skippedByArea, skippedByFroude, tLink, *minLink);
    }
    linkStepCallCount++;

    return tLink;
}

//=============================================================================

double getNodeStep(double tMin, int *minNode)
//
//  Input:   tMin = critical time step found so far (sec)
//  Output:  minNode = index of node with critical time step;
//           returns critical time step (sec)
//  Purpose: finds critical time step for nodes based on max. allowable
//           projected change in depth.
//
{
    int    i;                           // node index
    double maxDepth;                    // max. depth allowed at node (ft)
    double dYdT;                        // change in depth per unit time (ft/sec)
    double t1;                          // time needed to reach depth limit (sec)
    double tNode = tMin;                // critical node time step (sec)

    // DEBUG: Log first call details
    static int nodeStepCallCount = 0;
    int activeNodes = 0;
    int skippedByType = 0, skippedByDepth = 0, skippedByMaxDepth = 0, skippedByDydt = 0;

    // --- find smallest time so that estimated change in nodal depth
    //     does not exceed safety factor * maxdepth
    for ( i = 0; i < Nobjects[NODE]; i++ )
    {
        // --- see if node can be skipped
        if ( Node[i].type == OUTFALL ) {
            skippedByType++;
            continue;
        }
        if ( Node[i].newDepth <= FUDGE) {
            skippedByDepth++;
            continue;
        }
        if ( Node[i].newDepth  + FUDGE >=
             Node[i].crownElev - Node[i].invertElev ) {
            skippedByDepth++;
            continue;
        }

        // --- define max. allowable depth change using crown elevation
        maxDepth = (Node[i].crownElev - Node[i].invertElev) * 0.25;
        if ( maxDepth < FUDGE ) {
            skippedByMaxDepth++;
            continue;
        }
        dYdT = Xnode[i].dYdT;
        if (dYdT < FUDGE ) {
            skippedByDydt++;
            continue;
        }

        activeNodes++;

        // --- compute time to reach max. depth & compare with critical time
        t1 = maxDepth / dYdT;
        if ( t1 < tNode )
        {
            tNode = t1;
            *minNode = i;
            if (nodeStepCallCount == 0) {
                printf("  CPU getNodeStep: NEW MIN node=%d (%s) t1=%.6f (maxDepth=%.3f dYdT=%.6f newDepth=%.3f)\n",
                       i, Node[i].ID, t1, maxDepth, dYdT, Node[i].newDepth);
            }
        }
    }

    if (nodeStepCallCount == 0) {
        printf("  CPU getNodeStep[call=0]: checked %d nodes, active=%d skipped(type=%d depth=%d maxDepth=%d dydt=%d) → tNode=%.6f (minNode=%d)\n",
               Nobjects[NODE], activeNodes, skippedByType, skippedByDepth, skippedByMaxDepth, skippedByDydt, tNode, *minNode);
    }
    nodeStepCallCount++;

    return tNode;
}
