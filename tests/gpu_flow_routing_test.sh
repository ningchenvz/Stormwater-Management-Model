#!/bin/bash

################################################################################
# GPU Flow Routing Test Suite
#
# Purpose: Automated test to capture GPU vs CPU flow routing differences
# Captures: Flow continuity, storage behavior, pump/weir/outfall performance
#
# Usage: ./gpu_flow_routing_test.sh [output_dir]
#
# Output:
#   - CPU/GPU .rpt files
#   - Detailed comparison report
#   - Metrics CSV for tracking
#   - Visual diff highlighting key differences
#
################################################################################

set -e

# Configuration
REPO_ROOT="/home/ningchenspark/workspace/Stormwater-Management-Model"
BUILD_DIR="${REPO_ROOT}/build"
TEST_MODEL="${REPO_ROOT}/tests/test_models/model_full_features.inp"
RUNSWMM_BIN="${BUILD_DIR}/bin/runswmm"

# Output directory (allow override)
OUTPUT_DIR="${1:-${BUILD_DIR}/gpu_flow_routing_test_$(date +%Y%m%d-%H%M%S)}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_failure() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

extract_metric() {
    local file=$1
    local metric_name=$2
    local pattern=$3

    grep "$pattern" "$file" 2>/dev/null | head -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9.-]+$/) print $i; exit}' | head -1
}

extract_node_depth() {
    local file=$1
    local node=$2

    awk -v node="$node" '
    /Node Depth Summary/,/^$/ {
        if ($1 == node) print $5
    }' "$file" | head -1
}

extract_link_flow() {
    local file=$1
    local link=$2

    awk -v link="$link" '
    /Link Flow Summary/,/^$/ {
        if ($1 == link) print $4
    }' "$file" | head -1
}

compare_values() {
    local name=$1
    local cpu_val=$2
    local gpu_val=$3
    local tolerance=${4:-0.01}  # Default 1% tolerance

    if [ -z "$cpu_val" ] || [ -z "$gpu_val" ]; then
        print_warning "$name: Missing value (CPU: $cpu_val, GPU: $gpu_val)"
        return 1
    fi

    # Check if values are approximately equal
    local diff=$(echo "$cpu_val - $gpu_val" | bc -l 2>/dev/null | sed 's/-//')
    local abs_cpu=$(echo "$cpu_val" | sed 's/-//')

    if [ "$abs_cpu" != "0" ]; then
        local percent_diff=$(echo "scale=4; ($diff / $abs_cpu) * 100" | bc -l 2>/dev/null || echo "999")
    else
        local percent_diff=$(echo "$gpu_val" | sed 's/-//')
    fi

    if (( $(echo "$percent_diff < $tolerance" | bc -l 2>/dev/null || echo "0") )); then
        print_success "$name: CPU=$cpu_val GPU=$gpu_val (Diff: ${percent_diff}%)"
        return 0
    else
        print_failure "$name: CPU=$cpu_val GPU=$gpu_val (Diff: ${percent_diff}%) EXCEEDS TOLERANCE"
        return 1
    fi
}

################################################################################
# Main Test Flow
################################################################################

main() {
    print_header "GPU Flow Routing Test Suite"

    # Check prerequisites
    if [ ! -f "$RUNSWMM_BIN" ]; then
        print_failure "runswmm binary not found: $RUNSWMM_BIN"
        echo "Please build the project first:"
        echo "  cd $BUILD_DIR && cmake -DBUILD_CUDA=ON .. && cmake --build ."
        exit 1
    fi

    if [ ! -f "$TEST_MODEL" ]; then
        print_failure "Test model not found: $TEST_MODEL"
        exit 1
    fi

    print_success "Prerequisites verified"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    print_success "Output directory: $OUTPUT_DIR"
    echo ""

    # Test 1: CPU baseline run
    print_header "Test 1: CPU Baseline Run"
    local cpu_rpt="${OUTPUT_DIR}/model_full_features_cpu.rpt"
    local cpu_out="${OUTPUT_DIR}/model_full_features_cpu.out"

    echo "Running CPU simulation..."
    export SWMM_USE_CUDA=0
    if timeout 30 "$RUNSWMM_BIN" "$TEST_MODEL" "$cpu_rpt" "$cpu_out" > /dev/null 2>&1; then
        print_success "CPU simulation completed"
    else
        print_failure "CPU simulation failed"
        exit 1
    fi

    # Verify output
    if [ -f "$cpu_rpt" ] && [ -s "$cpu_rpt" ]; then
        print_success "CPU report generated ($(wc -l < "$cpu_rpt") lines)"
    else
        print_failure "CPU report generation failed"
        exit 1
    fi
    echo ""

    # Test 2: GPU run
    print_header "Test 2: GPU Simulation Run"
    local gpu_rpt="${OUTPUT_DIR}/model_full_features_gpu.rpt"
    local gpu_out="${OUTPUT_DIR}/model_full_features_gpu.out"

    echo "Running GPU simulation (SWMM_USE_CUDA=1)..."
    export SWMM_USE_CUDA=1
    local gpu_start_time=$(date +%s%N)
    if timeout 30 "$RUNSWMM_BIN" "$TEST_MODEL" "$gpu_rpt" "$gpu_out" > /dev/null 2>&1; then
        local gpu_end_time=$(date +%s%N)
        print_success "GPU simulation completed"
    else
        print_failure "GPU simulation failed"
        exit 1
    fi

    # Verify output
    if [ -f "$gpu_rpt" ] && [ -s "$gpu_rpt" ]; then
        print_success "GPU report generated ($(wc -l < "$gpu_rpt") lines)"
    else
        print_failure "GPU report generation failed"
        exit 1
    fi
    echo ""

    # Test 3: Extract and compare critical metrics
    print_header "Test 3: Critical Metrics Comparison"

    local metrics_file="${OUTPUT_DIR}/metrics_comparison.csv"
    echo "Metric,CPU_Value,GPU_Value,Unit,Match" > "$metrics_file"

    # Flow Routing Continuity
    echo ""
    echo "Flow Routing Continuity Metrics:"

    local cpu_ext_outflow=$(extract_metric "$cpu_rpt" "External Outflow" "External Outflow")
    local gpu_ext_outflow=$(extract_metric "$gpu_rpt" "External Outflow" "External Outflow")
    echo "External_Outflow,$cpu_ext_outflow,$gpu_ext_outflow,M gal,$(compare_values 'External Outflow' "$cpu_ext_outflow" "$gpu_ext_outflow" 0.5 && echo 'Y' || echo 'N')" >> "$metrics_file"

    local cpu_continuity=$(extract_metric "$cpu_rpt" "Continuity Error" "Continuity Error")
    local gpu_continuity=$(extract_metric "$gpu_rpt" "Continuity Error" "Continuity Error")
    echo "Continuity_Error,$cpu_continuity,$gpu_continuity,%,$(compare_values 'Continuity Error' "$cpu_continuity" "$gpu_continuity" 1.0 && echo 'Y' || echo 'N')" >> "$metrics_file"

    local cpu_final_stored=$(extract_metric "$cpu_rpt" "Final Stored" "Final Stored Volume")
    local gpu_final_stored=$(extract_metric "$gpu_rpt" "Final Stored" "Final Stored Volume")
    echo "Final_Stored_Volume,$cpu_final_stored,$gpu_final_stored,M gal,$(compare_values 'Final Stored Volume' "$cpu_final_stored" "$gpu_final_stored" 50.0 && echo 'Y' || echo 'N')" >> "$metrics_file"

    # Storage Behavior
    echo ""
    echo "Storage Node (J2) Behavior:"

    local cpu_j2_depth=$(extract_node_depth "$cpu_rpt" "J2")
    local gpu_j2_depth=$(extract_node_depth "$gpu_rpt" "J2")
    echo "J2_Max_Depth,$cpu_j2_depth,$gpu_j2_depth,ft,$(compare_values 'J2 Max Depth' "$cpu_j2_depth" "$gpu_j2_depth" 100.0 && echo 'Y' || echo 'N')" >> "$metrics_file"

    # Pump Performance
    echo ""
    echo "Pump (C2) Performance:"

    local cpu_pump_util=$(grep -A 5 "Pumping Summary" "$cpu_rpt" | grep "C2" | awk '{print $2}')
    local gpu_pump_util=$(grep -A 5 "Pumping Summary" "$gpu_rpt" | grep "C2" | awk '{print $2}')
    echo "Pump_Utilization,$cpu_pump_util,$gpu_pump_util,%,$(compare_values 'Pump Utilization' "$cpu_pump_util" "$gpu_pump_util" 50.0 && echo 'Y' || echo 'N')" >> "$metrics_file"

    # Outfall Performance
    echo ""
    echo "Outfall (J4) Performance:"

    local cpu_outfall_freq=$(grep -A 5 "Outfall Loading Summary" "$cpu_rpt" | grep "J4" | awk '{print $2}')
    local gpu_outfall_freq=$(grep -A 5 "Outfall Loading Summary" "$gpu_rpt" | grep "J4" | awk '{print $2}')
    echo "Outfall_Frequency,$cpu_outfall_freq,$gpu_outfall_freq,%,$(compare_values 'Outfall Frequency' "$cpu_outfall_freq" "$gpu_outfall_freq" 50.0 && echo 'Y' || echo 'N')" >> "$metrics_file"

    print_success "Metrics comparison complete"
    echo "Saved to: $metrics_file"
    echo ""

    # Test 4: Generate detailed diff
    print_header "Test 4: Detailed Difference Report"

    local diff_file="${OUTPUT_DIR}/model_full_features_report.diff"
    diff -u "$cpu_rpt" "$gpu_rpt" > "$diff_file" || true

    print_success "Diff report generated"
    echo "File: $diff_file"

    # Count significant differences
    local num_diffs=$(grep "^[+-]" "$diff_file" | grep -v "^[+-][+-][+-]" | wc -l)
    echo "Number of different lines: $num_diffs"
    echo ""

    # Test 5: Extract key problem areas
    print_header "Test 5: Problem Area Analysis"

    local problems_file="${OUTPUT_DIR}/problem_areas.txt"
    cat > "$problems_file" << 'EOF'
CRITICAL ISSUES FOUND
====================

External Outflow (Flow Routing Continuity):
EOF

    echo "CPU: $(extract_metric "$cpu_rpt" "External Outflow" "External Outflow") M gal" >> "$problems_file"
    echo "GPU: $(extract_metric "$gpu_rpt" "External Outflow" "External Outflow") M gal" >> "$problems_file"
    echo "" >> "$problems_file"

    cat >> "$problems_file" << 'EOF'
Storage Node (J2) Depth:
EOF

    echo "CPU: $(extract_node_depth "$cpu_rpt" "J2") ft (should be near 0)" >> "$problems_file"
    echo "GPU: $(extract_node_depth "$gpu_rpt" "J2") ft (ISSUE: should be near 0, not 15)" >> "$problems_file"
    echo "" >> "$problems_file"

    cat >> "$problems_file" << 'EOF'
Pump (C2) Operation:
EOF

    grep -A 5 "Pumping Summary" "$cpu_rpt" | grep "C2" | awk '{print "CPU: Utilization=" $2 "%, Flow=" $5 " CFS"}' >> "$problems_file"
    grep -A 5 "Pumping Summary" "$gpu_rpt" | grep "C2" | awk '{print "GPU: Utilization=" $2 "%, Flow=" $5 " CFS (ISSUE: should be operating)"}' >> "$problems_file"
    echo "" >> "$problems_file"

    cat >> "$problems_file" << 'EOF'
Outfall (J4) Discharge:
EOF

    grep -A 5 "Outfall Loading Summary" "$cpu_rpt" | grep "J4" | awk '{print "CPU: Frequency=" $2 "%, Total Volume=" $5 " M gal"}' >> "$problems_file"
    grep -A 5 "Outfall Loading Summary" "$gpu_rpt" | grep "J4" | awk '{print "GPU: Frequency=" $2 "%, Total Volume=" $5 " M gal (ISSUE: should be 100%)"}' >> "$problems_file"
    echo "" >> "$problems_file"

    cat >> "$problems_file" << 'EOF'
Continuity Error:
EOF

    grep "Continuity Error" "$cpu_rpt" | tail -1 | awk '{print "CPU: " $0}' >> "$problems_file"
    grep "Continuity Error" "$gpu_rpt" | tail -1 | awk '{print "GPU: " $0 " (ISSUE: should be <0.1%)"}' >> "$problems_file"

    print_success "Problem analysis complete"
    cat "$problems_file"
    echo ""

    # Test 6: Summary report
    print_header "Test Summary"

    echo "Test Results:"
    echo "  CPU run: SUCCESS"
    echo "  GPU run: SUCCESS (but results differ)"
    echo ""
    echo "Output Files:"
    echo "  CPU Report:     $cpu_rpt"
    echo "  GPU Report:     $gpu_rpt"
    echo "  Diff Report:    $diff_file"
    echo "  Metrics CSV:    $metrics_file"
    echo "  Analysis:       $problems_file"
    echo ""

    # Key findings
    echo "KEY FINDINGS:"

    if [ "$(extract_metric "$gpu_rpt" "External Outflow" "External Outflow")" == "0.000" ]; then
        print_failure "GPU: Zero outfall discharge (complete flow routing failure)"
    fi

    if [ "$(extract_node_depth "$gpu_rpt" "J2")" != "0.00" ] && [ "$(extract_node_depth "$gpu_rpt" "J2")" != "" ]; then
        print_failure "GPU: Storage node J2 not draining (pump failure detected)"
    fi

    if grep -q "100.00" <(grep -A 5 "Outfall Loading Summary" "$cpu_rpt" | grep "J4") && \
       grep -q "0.00" <(grep -A 5 "Outfall Loading Summary" "$gpu_rpt" | grep "J4"); then
        print_failure "GPU: Outfall not operating (CPU 100% vs GPU 0%)"
    fi

    local cpu_cont=$(extract_metric "$cpu_rpt" "Continuity Error" "Continuity Error")
    local gpu_cont=$(extract_metric "$gpu_rpt" "Continuity Error" "Continuity Error")

    if (( $(echo "$gpu_cont > 1.0" | bc -l 2>/dev/null) )); then
        print_failure "GPU: Continuity error exceeds 1% (CPU: $cpu_cont%, GPU: $gpu_cont%)"
    fi

    echo ""
    print_header "Test Complete"
    echo "All results saved to: $OUTPUT_DIR"
}

################################################################################
# Entry Point
################################################################################

main "$@"
