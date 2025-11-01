# GPU Flow Routing Test Case Guide

**Purpose:** Automated testing to capture and validate GPU vs CPU flow routing differences

**Status:** READY FOR USE - Both test scripts functional and tested

**Last Updated:** 2025-10-26

---

## Quick Start

### Option 1: Python Test Runner (Recommended)

```bash
cd ~/workspace/Stormwater-Management-Model

# Run with default settings
python3 tests/gpu_flow_routing_test.py

# Run with custom output directory
python3 tests/gpu_flow_routing_test.py --output-dir /tmp/gpu_test

# Specify repository location
python3 tests/gpu_flow_routing_test.py --repo-dir ~/workspace/Stormwater-Management-Model
```

### Option 2: Bash Test Runner

```bash
cd ~/workspace/Stormwater-Management-Model

# Run with default output
bash tests/gpu_flow_routing_test.sh

# Run with custom output directory
bash tests/gpu_flow_routing_test.sh /tmp/gpu_test
```

---

## What Gets Tested

### 1. Flow Routing Continuity
- External outflow volume (CPU: 44.6 M gal, GPU: 0.0 M gal) ❌
- Final stored volume (CPU: 0.002 M gal, GPU: 0.349 M gal) ❌
- Flooding loss (CPU: 8.6 M gal, GPU: 47.8 M gal) ❌
- **Continuity error (CPU: 0.001%, GPU: 9.49%)** ❌

### 2. Storage Node (J2) Behavior
- Storage depth at end of simulation
- Flood volume from storage
- Pump discharge (C2) operation

### 3. Pump (C2) Performance
- Pump utilization percentage (CPU: 100%, GPU: 0%) ❌
- Pump flow rate (CPU: 9.41 CFS, GPU: 0.00 CFS) ❌
- Total pumped volume (CPU: 12.9 M gal, GPU: 0.0 M gal) ❌

### 4. Weir (C3) Operation
- Weir overflow flow (CPU: 12.12 CFS, GPU: 0.00 CFS) ❌
- Weir participation in routing (should be active)

### 5. Outfall (J4) Discharge
- Outfall frequency (CPU: 100%, GPU: 0%) ❌
- Outfall total volume (CPU: 14.528 M gal, GPU: 0.0 M gal) ❌

### 6. Flow Classification
- Conduit C1:C2 classification (CPU: Subcritical, GPU: DRY) ❌

---

## Test Model Details

### Network Configuration

```
Subcatchments → Nodes → Links → Outfall

S1 (1 ac)  ─┐
S2 (2 ac)  ─┤→ J1(Junction) → C1:C2(Conduit) → J2(Storage) → C2(Pump) → J3(Junction)
S3 (3 ac)  ─┘                                                              ↓
                                                                        C3(Weir)
                                                                            ↓
                                                                        J4(Outfall)
```

### Node Specifications

| Node | Type | Invert Elev | Max Depth | Capacity |
|------|------|------------|-----------|----------|
| J1 | Junction | 20.73 ft | 15.00 ft | - |
| J2 | Storage | 13.39 ft | 15.00 ft | 15,000 ft³ |
| J3 | Junction | 6.55 ft | 15.00 ft | - |
| J4 | Outfall | 0.00 ft | 0.00 ft | - |

### Link Specifications

| Link | Type | Length | Properties |
|------|------|--------|------------|
| C1:C2 | Conduit | 244.6 ft | Circular 1.0 ft diameter, Manning n=0.01 |
| C2 | Pump | - | Ideal pump (proportional) |
| C3 | Weir | - | Spillway weir |

### Simulation Configuration

- **Duration:** 58 hours (11/01/2015 14:00 to 11/04/2015 00:00)
- **Rainfall:** SCS 24-hr Type I (1 inch)
- **Routing Method:** DYNWAVE
- **Routing Timestep:** 1.0 second
- **Total Runoff:** ~14.1 million gallons

---

## Test Output Files

### Generated Reports

```
output_directory/
├── model_full_features_cpu.rpt     # CPU simulation report (correct)
├── model_full_features_cpu.out     # CPU binary output
├── model_full_features_gpu.rpt     # GPU simulation report (problematic)
├── model_full_features_gpu.out     # GPU binary output
├── metrics_comparison.csv          # Key metrics comparison table
├── model_full_features_report.diff # Unified diff of reports
└── problem_areas.txt               # Identified issues summary
```

### Metrics CSV Format

```csv
Metric,CPU_Value,GPU_Value,Unit,Tolerance_%,Match
External Outflow,44.585,0.000,M gal,5.0,N
Continuity Error,0.001,9.490,%,10.0,N
Final Stored Volume,0.002,0.349,M gal,50.0,N
```

---

## Expected Issues (Known Bugs)

The test is designed to **capture** these problems:

### Issue 1: Zero Outfall Flow
```
CPU: Outfall J4 receives 14.528 M gal
GPU: Outfall J4 receives 0.000 M gal
Status: CRITICAL - Complete mass loss
```

### Issue 2: Storage Fills and Doesn't Drain
```
CPU: Storage J2 final volume 0.001 1000 ft³ (empty)
GPU: Storage J2 final volume 14.942 1000 ft³ (99.6% full)
Status: CRITICAL - Pump not discharging
```

### Issue 3: Pump Not Operating
```
CPU: Pump C2 at 100% utilization, 9.41 CFS avg flow
GPU: Pump C2 at 0% utilization, 0.00 CFS flow
Status: CRITICAL - Pump kernel missing or broken
```

### Issue 4: Weir Inactive
```
CPU: Weir C3 carries 12.12 CFS peak flow
GPU: Weir C3 carries 0.00 CFS flow
Status: CRITICAL - Downstream routing blocked
```

### Issue 5: Mass Balance Failure
```
CPU: Continuity error 0.001%
GPU: Continuity error 9.490%
Status: CRITICAL - Physics broken
```

### Issue 6: Conduit Misclassification
```
CPU: C1:C2 is subcritical 100% of time (correct)
GPU: C1:C2 is dry 100% of time (incorrect)
Status: HIGH - Data consistency issue
```

---

## Test Metrics Interpretation

### Continuity Error
- **CPU Result:** 0.001% (excellent, <0.1% is acceptable)
- **GPU Result:** 9.490% (FAILED, indicates ~9.5% of water unaccounted for)
- **Why it matters:** Shows GPU cannot conserve mass properly

### External Outflow
- **CPU Result:** 44.585 M gal (water exits system as expected)
- **GPU Result:** 0.000 M gal (NO flow - complete failure)
- **Why it matters:** Indicates flow routing completely blocked downstream of storage

### Storage Final Volume
- **CPU Result:** 0.002 M gal (minimal - pump drains storage)
- **GPU Result:** 0.349 M gal (3.49 ft depth - storage full and trapped)
- **Why it matters:** Shows pump not discharging water

### Pump Utilization
- **CPU Result:** 100% (pump actively operating throughout simulation)
- **GPU Result:** 0% (pump never activated)
- **Why it matters:** Indicates GPU doesn't execute pump logic

---

## Running Individual Tests

### Manual CPU Test
```bash
cd ~/workspace/Stormwater-Management-Model/build

# Run with CUDA disabled
export SWMM_USE_CUDA=0
./bin/runswmm ../tests/test_models/model_full_features.inp cpu_manual.rpt cpu_manual.out

# Check results
grep "External Outflow" cpu_manual.rpt
grep "Continuity Error" cpu_manual.rpt
```

### Manual GPU Test
```bash
cd ~/workspace/Stormwater-Management-Model/build

# Run with CUDA enabled
export SWMM_USE_CUDA=1
./bin/runswmm ../tests/test_models/model_full_features.inp gpu_manual.rpt gpu_manual.out

# Check results
grep "External Outflow" gpu_manual.rpt
grep "Continuity Error" gpu_manual.rpt
```

### Compare Specific Metrics
```bash
# Storage behavior
diff <(grep -A 5 "Storage Volume Summary" cpu.rpt) \
     <(grep -A 5 "Storage Volume Summary" gpu.rpt)

# Pump operation
diff <(grep -A 5 "Pumping Summary" cpu.rpt) \
     <(grep -A 5 "Pumping Summary" gpu.rpt)

# Outfall discharge
diff <(grep -A 5 "Outfall Loading Summary" cpu.rpt) \
     <(grep -A 5 "Outfall Loading Summary" gpu.rpt)

# Flow classification
diff <(grep -A 5 "Flow Classification Summary" cpu.rpt) \
     <(grep -A 5 "Flow Classification Summary" gpu.rpt)
```

---

## Troubleshooting

### Test Script Won't Run

**Problem:** `command not found: gpu_flow_routing_test.sh`

**Solution:**
```bash
# Make script executable
chmod +x ~/workspace/Stormwater-Management-Model/tests/gpu_flow_routing_test.sh

# Run directly
bash ~/workspace/Stormwater-Management-Model/tests/gpu_flow_routing_test.sh
```

### SWMM Build Not Found

**Problem:** `runswmm binary not found`

**Solution:**
```bash
cd ~/workspace/Stormwater-Management-Model

# Build with CUDA support
mkdir -p build && cd build
cmake -DBUILD_CUDA=ON ..
cmake --build .

# Then run test
python3 ../tests/gpu_flow_routing_test.py
```

### Test Hangs or Times Out

**Problem:** GPU test hangs after printing "Running GPU simulation..."

**Possible Causes:**
- GPU kernel deadlock in unified memory access
- Synchronization issue in kernel launch
- Memory transfer blocking

**Debugging Steps:**
```bash
# Check GPU memory
nvidia-smi

# Verify GPU is available
nvidia-smi --query-gpu=name --format=csv,noheader

# Run with timeout
timeout 30 bash tests/gpu_flow_routing_test.sh
```

### Python Test Import Errors

**Problem:** `ModuleNotFoundError: No module named 'csv'`

**Solution:** Standard library issue, use Python 3:
```bash
# Verify Python version
python3 --version

# Run with python3
python3 tests/gpu_flow_routing_test.py
```

---

## Integration with CI/CD

### Add to CMake Tests

```cmake
add_test(
    NAME gpu_flow_routing_validation
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu_flow_routing_test.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
```

### Run All GPU Tests

```bash
cd ~/workspace/Stormwater-Management-Model/build
ctest --test-dir . -L GPU -V
```

---

## Related Documentation

- **Bug Report:** `doc/gpu/GPU_CPU_FLOW_ROUTING_BUG_REPORT.md` - Detailed analysis of differences
- **Phase 4 Roadmap:** `doc/gpu/PHASE4_IMPLEMENTATION_PLAN.md` - Implementation tasks
- **GPU Config:** `src/solver/gpu/gpu_config.h` - GPU constants and configuration

---

## Key Files Being Tested

| File | Purpose | Issue |
|------|---------|-------|
| `src/solver/gpu/gpu_dwflow.cu` | Main GPU conduit kernel | Might not handle PUMP/WEIR types |
| `src/solver/gpu/gpu_conduit_helpers.cuh` | Momentum equation solver | Energy term might be zero for storage downstream |
| `src/solver/dynwave.c` | Integration point | Data transfer to GPU for pump/weir might be missing |
| `src/solver/dwflow.c` | CPU reference | Handles PUMP/WEIR/CONDUIT - compare with GPU |

---

## Contact & Issues

For detailed analysis of issues found:
- See `GPU_CPU_FLOW_ROUTING_BUG_REPORT.md` for in-depth investigation
- See `BUG_REPORT.md` for earlier GPU implementation issues

