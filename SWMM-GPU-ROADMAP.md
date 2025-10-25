# SWMM-GPU Development Roadmap

## Project Overview

**Goal:** Add CUDA GPU acceleration to SWMM's dynamic wave flow routing algorithm
**Target Hardware:** NVIDIA GPUs (Compute Capability 6.0+)
**Primary Development GPUs:**
  - RTX 4060 (Compute Capability 8.9) - Laptop (discrete GPU)
  - NVIDIA GB10 (Compute Capability 12.1) - DGX Spark (unified memory architecture)
**Expected Speedup:** 5-15x for large models (1000+ links)

**Note:** The DGX Spark uses unified memory, allowing CPU and GPU to share memory space with automatic migration. This simplifies memory management and may enable additional optimizations.

---

## Task List

### Phase 1: Infrastructure Setup (1-2 weeks)

| # | Task | Status | Estimated Time |
|---|------|--------|----------------|
| 1 | Set up CUDA development environment and verify installation | ✅ Complete | 1 day |
| 2 | Add CUDA detection and build options to CMake configuration | ✅ Complete | 1 day |
| 3 | Create gpu/ directory structure under src/solver/ | ✅ Complete | 0.5 day |

**Phase 1 Total:** 2.5 days (~1 week with testing)

---

### Phase 2: Data Structures & Memory Management (1-2 weeks)

| # | Task | Status | Estimated Time |
|---|------|--------|----------------|
| 4 | Design and implement GPU data structures (AoS to SoA conversion) | ✅ Complete | 3-4 days |
| 5 | Implement GPU memory management layer (cudaMallocManaged for unified memory on DGX, cudaMalloc fallback for discrete GPUs) | ✅ Complete | 2-3 days |
| 6 | Create simple test kernel to verify CUDA pipeline works | ✅ Complete | 1 day |

**Phase 2 Total:** 6-8 days (~1.5 weeks) - **COMPLETED**

**Key Deliverable:** ✅ Working CUDA compilation and basic GPU memory allocation
**Note:** Unified memory support on DGX simplifies implementation - using `cudaMallocManaged()` for automatic data migration

**Completed Files:**
- `src/solver/gpu/gpu_structures.h` - GPU SoA data structures
- `src/solver/gpu/gpu_memory.cu` - Memory allocation/deallocation with unified memory support
- `src/solver/gpu/gpu_test_kernels.cu` - Test kernels (vector add, node/link ops, mass balance)
- All tests passing on NVIDIA GB10 (Compute Capability 12.1)

---

### Phase 3: Kernel Implementation - Simple (1 week)

| # | Task | Status | Estimated Time |
|---|------|--------|----------------|
| 7 | Port helper functions (getArea, getHydRad, etc.) to __device__ functions | ⏳ Pending | 2-3 days |
| 8 | Implement findNodeDepths CUDA kernel (simpler, good starting point) | ⏳ Pending | 2-3 days |
| 9 | Add CPU/GPU result comparison tests for node depth calculations | ⏳ Pending | 1-2 days |

**Phase 3 Total:** 5-8 days (~1 week)

**Key Deliverable:** Working node depth kernel with validation

---

### Phase 4: Kernel Implementation - Complex (2-3 weeks)

| # | Task | Status | Estimated Time |
|---|------|--------|----------------|
| 10 | Implement findConduitFlows CUDA kernel (complex, main computation) | ⏳ Pending | 5-7 days |
| 11 | Add CPU/GPU result comparison tests for conduit flow calculations | ⏳ Pending | 2-3 days |
| 12 | Implement convergence check reduction kernel for GPU | ⏳ Pending | 2-3 days |

**Phase 4 Total:** 9-13 days (~2 weeks)

**Key Deliverable:** Complete GPU dynamic wave solver

---

### Phase 5: Integration & Runtime (1-2 weeks)

| # | Task | Status | Estimated Time |
|---|------|--------|----------------|
| 13 | Add runtime GPU detection and CPU/GPU fallback logic in dynwave.c | ⏳ Pending | 2-3 days |
| 14 | Optimize memory transfer patterns (prefetch hints for unified memory, minimize explicit copies for discrete GPUs) | ⏳ Pending | 2-3 days |
| 15 | Tune CUDA kernel launch parameters (block size, grid size) | ⏳ Pending | 2-3 days |

**Phase 5 Total:** 6-9 days (~1.5 weeks)

**Key Deliverable:** Production-ready hybrid CPU/GPU execution
**Note:** On DGX unified memory, use `cudaMemPrefetchAsync()` to hint data migration; on discrete GPUs, use explicit `cudaMemcpy()`

---

### Phase 6: Optimization & Profiling (2-3 weeks)

| # | Task | Status | Estimated Time |
|---|------|--------|----------------|
| 16 | Profile with NVIDIA Nsight and identify bottlenecks | ⏳ Pending | 2-3 days |
| 17 | Add CUDA-specific configuration options to SWMM input file format | ⏳ Pending | 1-2 days |
| 18 | Run full regression test suite comparing CPU vs GPU results | ⏳ Pending | 3-4 days |
| 19 | Create performance benchmarks for various model sizes | ⏳ Pending | 2-3 days |
| 20 | Update documentation with CUDA build instructions and usage | ⏳ Pending | 2-3 days |

**Phase 6 Total:** 10-15 days (~2.5 weeks)

**Key Deliverable:** Optimized, tested, documented SWMM-GPU release

---

## Timeline Summary

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Infrastructure | 1-2 weeks | 1-2 weeks |
| Phase 2: Data Structures | 1-2 weeks | 2-4 weeks |
| Phase 3: Simple Kernels | 1 week | 3-5 weeks |
| Phase 4: Complex Kernels | 2-3 weeks | 5-8 weeks |
| Phase 5: Integration | 1-2 weeks | 6-10 weeks |
| Phase 6: Optimization | 2-3 weeks | 8-13 weeks |

**Conservative Estimate:** 12-13 weeks (~3 months)
**Aggressive Estimate:** 8-10 weeks (~2 months)
**Realistic Target:** 10-12 weeks (2.5-3 months)

---

## Current Status (as of October 25, 2025)

**Phase:** 2 (Data Structures) - **COMPLETED ✅**
**Progress:** 6/20 tasks complete (30%)
**Branch:** `feature/swmm-gpu-acceleration`
**Next Milestone:** Phase 3 (Simple Kernel Implementation)

### Completed
- ✅ CUDA development environment verified (CUDA 12.4 on RTX 4060, CUDA 13.0 on GB10 DGX)
- ✅ CMake build system with BUILD_GPU option (multi-architecture support: 89, 121)
- ✅ GPU directory structure created
- ✅ Basic GPU manager with unified memory detection implemented
- ✅ GPU SoA data structures designed (Node, Link, Conduit, XSect)
- ✅ Memory management layer implemented (cudaMallocManaged + discrete GPU fallback)
- ✅ Test kernels implemented and verified on DGX Spark

### Next Phase
- Phase 3: Implement helper device functions (getArea, getHydRad, etc.)
- Port findNodeDepths to GPU kernel (simpler starting point)
- Create CPU/GPU validation tests

---

## Risk Factors

| Risk | Severity | Mitigation |
|------|----------|------------|
| Data structure conversion complexity | High | Start with simplified structures, iterate |
| Memory transfer overhead | Low | DGX unified memory auto-migrates; prefetch hints optimize further |
| Numerical accuracy differences | Medium | Rigorous testing, bit-exact comparisons |
| GPU memory limitations | Low | RTX 4060: 8GB, DGX GB10: 120GB - more than sufficient |
| CUDA learning curve | Medium | Start with simple kernels, build up complexity |

---

## Success Criteria

1. **Correctness:** GPU results match CPU within floating-point tolerance
2. **Performance:** 5-10x speedup for models with 1000+ links
3. **Reliability:** Passes all existing SWMM regression tests
4. **Usability:** Simple build flag, automatic CPU fallback
5. **Documentation:** Clear build/usage instructions

---

## Dependencies

- CUDA Toolkit 12.x or 13.x
- CMake 3.13+
- NVIDIA GPU with Compute Capability 6.0+ (tested on 8.9 and 12.1)
- NVIDIA driver 525+ (tested with 580.95.05)

---

## References

- [SWMM Hydraulics Manual](http://www.epa.gov/water-research/storm-water-management-model-swmm)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
