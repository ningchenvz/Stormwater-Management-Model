# GPU Test Suite - Individual Test Cases

**Location:** `/tests/gpu/`

**Purpose:** Automated test cases to isolate and verify each GPU flow routing issue

**Status:** All 6 test cases ready for execution

---

## Quick Start

```bash
cd ~/workspace/Stormwater-Management-Model

# Run all GPU tests
for test in tests/gpu/test_gpu_*.sh; do
    echo "Running $(basename $test)..."
    bash "$test" || echo "FAILED: $test"
done

# Run specific test
bash tests/gpu/test_gpu_pump_not_operating.sh
```

---

## Test Cases

### 1. Zero Outfall Flow

**File:** `test_gpu_zero_outfall_flow.sh`

**Issue:** GPU produces 0.000 M gal outfall discharge vs CPU 14.528 M gal

**Metrics:**
- Outfall J4 frequency: Should be ~100%
- Outfall J4 volume: Should be ~14.5 M gal

**Pass Criteria:**
- Frequency >= 99%
- Volume >= 14.0 M gal

**Run:**
```bash
bash tests/gpu/test_gpu_zero_outfall_flow.sh
```

---

### 2. Storage Not Draining

**File:** `test_gpu_storage_not_draining.sh`

**Issue:** GPU storage J2 final volume 14.942 1000 ft³ vs CPU 0.001 1000 ft³

**Metrics:**
- Storage J2 max depth: Should be ~0 ft
- Storage J2 final volume: Should be ~0 1000 ft³
- Storage J2 flooding: Should be 0 hours

**Pass Criteria:**
- Max depth < 1.0 ft
- Final volume < 0.1 1000 ft³
- No flooding

**Run:**
```bash
bash tests/gpu/test_gpu_storage_not_draining.sh
```

---

### 3. Pump Not Operating

**File:** `test_gpu_pump_not_operating.sh`

**Issue:** GPU pump C2 at 0% utilization vs CPU 100% utilization

**Metrics:**
- Pump C2 utilization: Should be ~100%
- Pump C2 avg flow: Should be ~8.25 CFS
- Pump C2 total volume: Should be ~12.9 M gal

**Pass Criteria:**
- Utilization >= 95%
- Avg flow >= 7.5 CFS
- Volume >= 11.0 M gal

**Run:**
```bash
bash tests/gpu/test_gpu_pump_not_operating.sh
```

---

### 4. Weir Flow Blocked

**File:** `test_gpu_weir_flow_blocked.sh`

**Issue:** GPU weir C3 at 0.00 CFS vs CPU 12.12 CFS

**Metrics:**
- Weir C3 max flow: Should be ~12.12 CFS
- Weir C3 active: Should carry positive flow

**Pass Criteria:**
- Max flow >= 11.0 CFS
- Occurs at expected time (around 10:00)

**Run:**
```bash
bash tests/gpu/test_gpu_weir_flow_blocked.sh
```

---

### 5. Mass Balance Failure

**File:** `test_gpu_mass_balance_failure.sh`

**Issue:** GPU continuity error 9.490% vs CPU 0.001%

**Metrics:**
- Continuity error: Should be <0.1%
- External outflow: Should be ~44.6 M gal
- Flooding loss: Should be ~8.6 M gal

**Pass Criteria:**
- Error < 0.1% (excellent)
- Error < 0.5% (acceptable)

**Run:**
```bash
bash tests/gpu/test_gpu_mass_balance_failure.sh
```

---

### 6. Performance Degradation

**File:** `test_gpu_performance_degradation.sh`

**Issue:** GPU 147x slower (146 sec) vs CPU (1 sec)

**Metrics:**
- CPU time: ~1 second
- GPU time: ~146 seconds
- Speedup: ~0.0068x (should be >= 1.0x, ideally >= 5.0x)

**Pass Criteria:**
- Speedup >= 1.0x (break even)
- Preferably >= 5.0x (expected on GPU)

**Run:**
```bash
bash tests/gpu/test_gpu_performance_degradation.sh
```

---

## Full Test Execution

### Run All Tests

```bash
cd ~/workspace/Stormwater-Management-Model

echo "Running full GPU test suite..."
echo ""

for test in tests/gpu/test_gpu_*.sh; do
    TEST_NAME=$(basename "$test" .sh)
    echo "================================================================"
    echo "Running: $TEST_NAME"
    echo "================================================================"

    if bash "$test" test_output_dir; then
        echo "✓ PASSED: $TEST_NAME"
    else
        echo "✗ FAILED: $TEST_NAME"
    fi
    echo ""
done

echo "================================================================"
echo "Test suite execution complete"
echo "Check test_output_dir/ for detailed reports"
```

### Run with Custom Output Directory

```bash
OUTPUT_DIR="/tmp/gpu_test_results"
mkdir -p "$OUTPUT_DIR"

for test in tests/gpu/test_gpu_*.sh; do
    bash "$test" "$OUTPUT_DIR"
done

# Review results
ls -la "$OUTPUT_DIR"
```

---

## Output Files

Each test generates comparison reports:

```
output_directory/
├── test_gpu_zero_outfall_flow_cpu.rpt     # CPU baseline
├── test_gpu_zero_outfall_flow_gpu.rpt     # GPU results (broken)
├── test_gpu_storage_not_draining_cpu.rpt
├── test_gpu_storage_not_draining_gpu.rpt
├── ... (similar for other tests)
```

---

## Understanding Test Output

### Passing Test
```
✓ TEST PASSED: GPU [feature] working correctly
  Expected: [description of expected behavior]
  Got:      [actual GPU behavior matches]
```

### Failing Test
```
✗ TEST FAILED: GPU [feature] broken
  Expected: [description of expected behavior]
  Got:      [actual GPU behavior differs]

Detailed comparison:
[diff output showing differences]
```

---

## Key Metrics to Watch

### Test 1: Outfall Flow
- Look for J4 row in "Outfall Loading Summary"
- CPU shows 100% frequency, GPU should also
- CPU shows ~14.5 M gal, GPU should match

### Test 2: Storage Draining
- Look for J2 row in "Storage Volume Summary"
- CPU shows ~0 1000 ft³, GPU should also
- Watch for flooding hours - should be 0

### Test 3: Pump Operating
- Look for C2 row in "Pumping Summary"
- CPU shows ~100% utilization, GPU should match
- Watch flow rates and total volume

### Test 4: Weir Flow
- Look for C3 row in "Link Flow Summary"
- CPU shows ~12 CFS, GPU should match
- Check time of max occurrence

### Test 5: Mass Balance
- Find "Continuity Error" line
- CPU shows ~0.001%, GPU should match
- Check "Highest Continuity Errors" section for problem nodes

### Test 6: Performance
- Compare wall-clock execution time
- GPU should be >= 1.0x speedup vs CPU
- Watch for 147x slowdown issue

---

## Troubleshooting

### Test Won't Run
```bash
# Make sure script is executable
chmod +x tests/gpu/test_gpu_*.sh

# Check for build artifacts
ls -la build/bin/runswmm
```

### SWMM Build Issues
```bash
cd build
cmake -DBUILD_CUDA=ON ..
cmake --build .
```

### Test Hangs
```bash
# Set a timeout
timeout 60 bash tests/gpu/test_gpu_pump_not_operating.sh

# Check GPU availability
nvidia-smi
```

---

## Integration with CI/CD

### Add to CMake
```cmake
add_test(
    NAME gpu_flow_routing_tests
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu/run_all_tests.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
```

### Run Specific Tests
```bash
ctest --test-dir build -L gpu -V
ctest --test-dir build -R pump_not_operating -V
```

---

## Divide & Conquer Development

Use these tests to isolate fixes:

1. **Fix Pump Issue** → Run `test_gpu_pump_not_operating.sh` ✓
2. **Fix Weir Issue** → Run `test_gpu_weir_flow_blocked.sh` ✓
3. **Fix Storage Drainage** → Run `test_gpu_storage_not_draining.sh` ✓
4. **Verify Outfall** → Run `test_gpu_zero_outfall_flow.sh` ✓
5. **Check Mass Balance** → Run `test_gpu_mass_balance_failure.sh` ✓
6. **Optimize Performance** → Run `test_gpu_performance_degradation.sh` ✓

---

## Related Documentation

- **Bug Report:** `doc/gpu/GPU_FLOW_ROUTING_BUG_REPORT.md` - Complete analysis
- **Test Guide:** `doc/gpu/GPU_FLOW_ROUTING_TEST_GUIDE.md` - How to use tests

---

## Status Summary

| Test | File | Issue | Status |
|------|------|-------|--------|
| 1 | `test_gpu_zero_outfall_flow.sh` | 0.000 M gal vs 14.528 M gal | ❌ FAIL |
| 2 | `test_gpu_storage_not_draining.sh` | Storage at 100% vs empty | ❌ FAIL |
| 3 | `test_gpu_pump_not_operating.sh` | 0% util vs 100% util | ❌ FAIL |
| 4 | `test_gpu_weir_flow_blocked.sh` | 0 CFS vs 12.12 CFS | ❌ FAIL |
| 5 | `test_gpu_mass_balance_failure.sh` | 9.49% error vs 0.001% | ❌ FAIL |
| 6 | `test_gpu_performance_degradation.sh` | 147x slower | ❌ FAIL |

All tests fail, indicating GPU flow routing is completely non-functional for realistic networks with storage, pump, and weir elements.

