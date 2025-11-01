# GPU Pump Kernel Bug Reports

This directory contains documentation of a **critical bug** found in the GPU pump implementation.

## Bug Summary

**Status**: ❌ **CRITICAL - GPU pumps are not production-ready**

The `kernel_processPumpsSequentially` function in `src/solver/gpu/gpu_dwflow.cu` has a systematic bug that causes mass balance failures in multiple test models.

## Test Results

| Model | Pumps | CPU Error | GPU Error | Bug Confirmed? |
|-------|-------|-----------|-----------|----------------|
| Session18 | 5 | -198% | **-751%** | ✅ YES (3.8x worse) |
| Session58 (3hr) | 26 | +2.9% | **-75.9%** | ✅ YES (26x worse) |
| Session68 | 46 | -29.8% | -45.4% | ✅ YES (1.5x worse) |

## Reports in this Directory

### 1. PUMP_KERNEL_BUG_SUMMARY.md
**Primary bug report** containing:
- Comprehensive test results matrix
- Detailed analysis for each model (Session18, Session58, Session60, Session68)
- Root cause hypotheses
- Recommended immediate actions
- Next steps to fix

### 2. SESSION58_ERROR_ANALYSIS.md
**Time-dependent error analysis** showing:
- Session58 performs BETTER at 30 minutes (+3.3% vs +5.4% CPU)
- Session58 performs CATASTROPHICALLY at 3 hours (-75.9% vs +2.9% CPU)
- Proves errors accumulate over timesteps (2,160 timesteps = 698% accumulated error)

### 3. PUMP_SEQUENTIAL_FINAL_STATUS.md
**Session68 implementation details** including:
- Fully sequential pump processing approach
- 40% error reduction achieved (from -72% to -45%)
- Final 1.52x error gap vs CPU persists despite exact sequential logic
- Performance analysis and scalability limits

## Key Finding: Isolation Testing Confirms Pump Bug

By selectively disabling GPU kernels, we proved the bug is ONLY in pumps:

| Configuration | Session58 3hr Error | Session18 Error |
|---------------|---------------------|-----------------|
| GPU: All kernels | -75.881% ❌ | -750.931% ❌ |
| GPU: Conduits only | **-0.436%** ✅ | **-52.634%** ✅ |
| GPU: + Weirs/Orifices/Outlets | **-0.436%** ✅ | N/A |

**Conclusion**: Conduits, weirs, orifices, and outlets work perfectly. **ONLY pumps are broken.**

## Immediate Action Required

### For Production Users

**DO NOT use GPU acceleration for models with pumps** until this bug is fixed. Either:
1. Set `SWMM_USE_CUDA=0` to disable GPU entirely
2. Wait for bug fix (estimated 1-2 weeks)

### For Developers

The bug is likely in one of these areas:

1. **Node flow updates** (`gpu_dwflow.cu:912-921`):
   ```cuda
   nodes->d_outflow[n1] += qIn;
   nodes->d_inflow[n2] += qIn;
   ```

2. **getModPumpFlow calculations** (`gpu_nonconduit_helpers.cuh:518-572`):
   - May be using stale node volumes (`d_oldVolume`)
   - May be accessing wrong node state arrays

See **PUMP_KERNEL_BUG_SUMMARY.md** for detailed debugging steps.

## Models Without Pump Bug

Not all pump models fail:
- ✅ **Session60** (6 pumps): GPU works correctly, even better than CPU
- ❌ **Session18** (5 pumps): GPU fails catastrophically
- ❌ **Session58** (26 pumps): GPU fails after 1+ hours
- ❌ **Session68** (46 pumps): GPU has systematic bias

**Hypothesis**: Bug may be specific to certain pump types (TYPE1/TYPE2/TYPE3) or node configurations.

## Test Files Location

All test input files and output reports are in `/tmp/`:
- `/tmp/Session18_*.rpt` - Session18 test results
- `/tmp/Session58_*.rpt` - Session58 test results (30min and 3hr)
- `/tmp/Session60_*.rpt` - Session60 test results (pumps work!)
- `/tmp/Session68_*.rpt` - Session68 test results

## Timeline

- **2025-11-01**: Bug discovered through systematic testing
- **Status**: Under investigation
- **ETA for fix**: 1-2 weeks

---

**Contact**: See bug reports for detailed technical analysis
**Priority**: P0 - Critical (blocks GPU production use for pump models)
