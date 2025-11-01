# GPU/CPU Comparison Scripts

## Overview

This directory contains scripts for comparing GPU-accelerated and CPU-only SWMM simulations to verify correctness and measure performance.

## File Extension Support

**All scripts support case-insensitive file matching:**
- Uppercase: `.INP`
- Lowercase: `.inp`
- Mixed case: `.Inp`, `.iNp`, `.InP`, etc.

This ensures compatibility with files from different sources (Windows typically uses `.INP`, Linux often uses `.inp`).

## Scripts

### compare_runswmm_gpu_cpu.sh

Compares GPU vs CPU runs for a single INP file.

**Usage:**
```bash
scripts/compare_runswmm_gpu_cpu.sh <path/to/model.inp>
```

**Environment Variables:**
- `RUNSWMM` - Path to runswmm executable (default: `build/bin/runswmm`)
- `OUT_DIR` - Directory to store run artifacts (default: `build/gpu_cpu_compare`)

**Output:**
- Creates timestamped directory with comparison artifacts
- Generates CPU and GPU report/output files
- Creates diff files showing differences

**Difference Classification:**

The script classifies report differences into three categories:

1. **match** - Reports are identical (after filtering metadata)
2. **minor** - Reports differ but with ≤10 lines of difference
   - Typical causes: Small numeric differences, rounding errors
   - Usually acceptable for GPU/CPU comparisons
3. **major** - Reports differ with >10 lines of difference
   - Indicates significant structural or value differences
   - Requires investigation

**Example Output:**
```
Reports match (after filtering metadata; diff stored at ...)
Reports differ - MINOR differences (5 lines; see ...)
Reports differ - MAJOR differences (25 lines; see ...)
```

### compare_runs_summary.py

Runs comparison script on multiple INP files and generates summary table.

**Usage:**
```bash
python3 scripts/compare_runs_summary.py <directory>
```

**Arguments:**
- `directory` - Directory containing .inp/.INP files (searched recursively by default)
- `--compare-script PATH` - Path to compare script (default: auto-detected)
- `--runswmm PATH` - Optional runswmm executable path
- `--out-dir DIR` - Optional output directory
- `--non-recursive` - Only process INP files directly in directory

**File Matching:**
- Handles both uppercase `.INP` and lowercase `.inp` extensions (case-insensitive)
- Also matches mixed case like `.Inp`, `.iNp`, etc.
- Uses suffix matching rather than pattern matching for maximum compatibility

**Example Output:**
```
Input           | Report Diff | Binary Diff | Links | Duration | GPU Time (ms) | CPU Time (ms)
----------------+-------------+-------------+-------+----------+---------------+--------------
model_1.inp     | match       | match       | 150   | 24.0h    | 1234.56       | 2456.78
model_2.inp     | minor       | match       | 50    | 6.0h     | 567.89        | 789.01
model_3.inp     | major       | differ      | 500   | 7.0d     | 3210.12       | 4567.34
```

## Performance Metrics

### Timing Display

All execution times are displayed in **milliseconds (ms)** for better precision:
- Typical values: 100-5000 ms for small models
- Large models: 5000+ ms
- Precision: 2 decimal places (e.g., 1234.56 ms)

### Model Information

The scripts automatically extract and display:

**Links Count**: Total number of hydraulic links in the model
- Includes: CONDUITS + PUMPS + ORIFICES + WEIRS + OUTLETS
- Helps identify model complexity
- Useful for performance analysis (GPU benefits scale with link count)

**Simulation Duration**: Total simulation time span from INP file
- Format:
  - Hours: `2.0h`, `24.0h` (for < 24 hours)
  - Days: `7.0d`, `30.0d` (for ≥ 24 hours)
- Extracted from OPTIONS section: `START_DATE/TIME` to `END_DATE/TIME`
- Shows "N/A" if dates not found or unparseable

## Difference Classification Heuristics

The comparison uses the following heuristics to classify differences:

### Minor Differences (≤10 diff lines)
- Small numeric variations due to GPU floating-point operations
- Acceptable rounding differences
- Minor timing variations in report timestamps
- Generally indicates correct GPU implementation with acceptable precision

### Major Differences (>10 diff lines)
- Significant structural changes in output
- Large numeric discrepancies
- Missing or extra output sections
- Indicates potential GPU implementation issues requiring investigation

### Threshold Tuning

The 10-line threshold is configurable in `compare_runswmm_gpu_cpu.sh`:

```bash
# Heuristic thresholds:
# - Minor: 1-10 lines of diff (small numeric differences, rounding errors)
# - Major: >10 lines of diff (significant structural or value differences)
if [ "$DIFF_LINE_COUNT" -le 10 ]; then
    echo "Reports differ - MINOR differences ..."
else
    echo "Reports differ - MAJOR differences ..."
fi
```

Adjust the threshold based on your accuracy requirements and typical difference patterns.

## Filtering

The comparison automatically filters out non-semantic differences:

- Metadata (timestamps, file paths)
- Diff formatting (`@@` markers)
- Analysis begin/end timestamps
- Total elapsed time
- Trailing newlines

This ensures only meaningful differences are counted in the classification.

## Best Practices

1. **Regular Testing**: Run comparisons on representative test suite
2. **Investigate Majors**: Always investigate MAJOR differences
3. **Monitor Minors**: Track MINOR differences over time to detect drift
4. **Binary Checks**: Always check binary output matching (stricter than reports)
5. **Performance**: Compare GPU vs CPU execution times

## Example Workflow

```bash
# Single model comparison
scripts/compare_runswmm_gpu_cpu.sh tests/model.inp

# Batch comparison with summary
python3 scripts/compare_runs_summary.py tests/models/

# With custom runswmm binary
RUNSWMM=./my_build/runswmm scripts/compare_runswmm_gpu_cpu.sh model.inp

# Parallel batch comparison
scripts/batch_compare_runswmm.sh tests/models/
```

## Troubleshooting

**All comparisons show MAJOR differences:**
- Check GPU implementation for correctness bugs
- Verify data transfer completeness (CPU ↔ GPU)
- Review kernel logic for numerical precision issues

**Reports differ but binary outputs match:**
- Likely formatting differences in report text
- May indicate report generation differences between GPU/CPU paths
- Binary match is more reliable indicator of correctness

**GPU slower than CPU:**
- Model may be too small to benefit from GPU (check heuristic thresholds)
- Check for excessive CPU-GPU transfers
- Profile with `nsight-systems` to identify bottlenecks
