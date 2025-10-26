# SWMM-GPU Test Models

This directory contains test models for validating GPU-accelerated dynamic wave routing.

## Test Models

### simple_test.inp
**Description:** Simple 3-node network for basic validation
**Purpose:** Verify GPU node depth kernel correctness
**Network:** 1 subcatchment → J1 → J2 → OUT1
**Features:**
- 2 circular conduits (3 ft diameter)
- Dynamic wave routing
- Rainfall-runoff simulation
- Duration: 2 hours

**Validation Results:**
- ✅ **GPU vs CPU: IDENTICAL** (2024-10-26)
- Node depths match exactly (bit-exact)
- Binary output files match
- 2882 kernel launches, 54.6 ms GPU time

## Running Tests

### Single Model Test
```bash
./scripts/compare_runswmm_gpu_cpu.sh tests/test_models/simple_test.inp
```

### Batch Test (All Models)
```bash
./scripts/batch_compare_runswmm.sh tests/test_models/
```

### Python Summary
```bash
python3 scripts/compare_runs_summary.py tests/test_models/
```

## Expected Results

All test models should produce:
- **Reports match** - Text output is identical
- **Binary outputs match** - `.out` files are bit-exact

Any differences indicate a GPU kernel bug that must be fixed before merging.

## Adding New Test Models

1. Create `.inp` file in this directory
2. Run validation: `./scripts/compare_runswmm_gpu_cpu.sh tests/test_models/your_model.inp`
3. Verify reports and binary outputs match
4. Document in this README

## Test Coverage

| Feature | simple_test | Future Tests |
|---------|-------------|--------------|
| Node depth kernel | ✅ | |
| Conduit flow kernel | ⏳ | Pending Phase 4 |
| Circular pipes | ✅ | |
| Rectangular channels | ⏳ | TODO |
| Surcharging | ⏳ | TODO |
| Ponding | ⏳ | TODO |
| Multiple conduit types | ⏳ | TODO |

## Performance Benchmarks

Performance data from DGX Spark (NVIDIA GB10, Compute 12.1):

| Model | Nodes | Links | GPU Time (ms) | CPU Time (ms) | Speedup |
|-------|-------|-------|---------------|---------------|---------|
| simple_test | 3 | 2 | 54.6 | N/A | Baseline |

*Note: GPU overhead dominates small models. Speedup expected for 1000+ links.*
