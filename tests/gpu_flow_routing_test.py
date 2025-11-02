#!/usr/bin/env python3

"""
GPU Flow Routing Test Suite

Automated test to compare GPU vs CPU flow routing behavior on the model_full_features
test case. Captures flow continuity, storage behavior, pump/weir performance, and
generates comprehensive comparison reports with intelligent numerical tolerance.

Features:
    - Smart diff comparison with configurable tolerance for numerical values
    - Lines differing only by minor numerical values (within tolerance) are considered equivalent
    - Default 5% tolerance for all numerical comparisons
    - Only significant differences reported in output

Usage:
    python3 gpu_flow_routing_test.py [--output-dir OUTPUT_DIR] [--tolerance PERCENT]

Output:
    - CPU/GPU .rpt files
    - Detailed comparison metrics CSV
    - Problem areas summary
    - Smart diff report showing only significant differences
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import csv

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color

class GPUFlowRoutingTest:
    """Main test class for GPU flow routing validation"""

    def __init__(self, repo_root: str = None, output_dir: str = None, tolerance_pct: float = 5.0):
        """Initialize test suite

        Args:
            repo_root: Path to SWMM repository root
            output_dir: Directory for test outputs
            tolerance_pct: Percentage tolerance for numerical comparisons (default: 5.0%)
        """
        if repo_root is None:
            repo_root = "/home/ningchenspark/workspace/Stormwater-Management-Model"

        self.repo_root = Path(repo_root)
        self.build_dir = self.repo_root / "build"
        self.test_model = self.repo_root / "tests/test_models/model_full_features.inp"
        self.runswmm_bin = self.build_dir / "bin/runswmm"
        self.tolerance_pct = tolerance_pct

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = self.build_dir / f"gpu_flow_routing_test_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Result files
        self.cpu_rpt = self.output_dir / "model_full_features_cpu.rpt"
        self.gpu_rpt = self.output_dir / "model_full_features_gpu.rpt"
        self.cpu_out = self.output_dir / "model_full_features_cpu.out"
        self.gpu_out = self.output_dir / "model_full_features_gpu.out"
        self.metrics_csv = self.output_dir / "metrics_comparison.csv"
        self.diff_file = self.output_dir / "model_full_features_report.diff"
        self.problems_file = self.output_dir / "problem_areas.txt"

        self.metrics = {}
        self.passed_tests = 0
        self.failed_tests = 0

    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.BLUE}{'='*50}{Colors.NC}")
        print(f"{Colors.BLUE}{text}{Colors.NC}")
        print(f"{Colors.BLUE}{'='*50}{Colors.NC}")

    def print_success(self, text: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓ {text}{Colors.NC}")
        self.passed_tests += 1

    def print_failure(self, text: str):
        """Print failure message"""
        print(f"{Colors.RED}✗ {text}{Colors.NC}")
        self.failed_tests += 1

    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

    def verify_prerequisites(self) -> bool:
        """Verify all prerequisites are met"""
        self.print_header("Verifying Prerequisites")

        checks = [
            (self.runswmm_bin.exists(), f"runswmm binary exists: {self.runswmm_bin}"),
            (self.test_model.exists(), f"test model exists: {self.test_model}"),
        ]

        all_ok = True
        for check, message in checks:
            if check:
                self.print_success(message)
            else:
                self.print_failure(message)
                all_ok = False

        if not all_ok:
            print("\nPlease build the project first:")
            print(f"  cd {self.build_dir}")
            print("  cmake -DBUILD_CUDA=ON ..")
            print("  cmake --build .")
            return False

        return True

    def run_simulation(self, use_gpu: bool) -> bool:
        """Run SWMM simulation"""
        test_type = "GPU" if use_gpu else "CPU"
        self.print_header(f"Running {test_type} Simulation")

        rpt_file = self.gpu_rpt if use_gpu else self.cpu_rpt
        out_file = self.gpu_out if use_gpu else self.cpu_out

        env = os.environ.copy()
        env['SWMM_USE_CUDA'] = '1' if use_gpu else '0'

        try:
            print(f"Executing: {self.runswmm_bin} {self.test_model} {rpt_file} {out_file}")
            result = subprocess.run(
                [str(self.runswmm_bin), str(self.test_model), str(rpt_file), str(out_file)],
                env=env,
                timeout=30,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.print_failure(f"{test_type} simulation failed")
                if result.stderr:
                    print(f"Error output:\n{result.stderr}")
                return False

            # Verify output file
            if not rpt_file.exists() or rpt_file.stat().st_size == 0:
                self.print_failure(f"{test_type} report generation failed")
                return False

            num_lines = len(rpt_file.read_text().splitlines())
            self.print_success(f"{test_type} simulation completed ({num_lines} lines)")
            return True

        except subprocess.TimeoutExpired:
            self.print_failure(f"{test_type} simulation timeout (>30s)")
            return False
        except Exception as e:
            self.print_failure(f"{test_type} simulation error: {e}")
            return False

    def extract_metric(self, file_path: Path, pattern: str) -> Optional[str]:
        """Extract metric value from report file"""
        try:
            content = file_path.read_text()
            match = re.search(pattern, content)
            if match:
                # Try to find numeric value after the pattern
                line_match = re.search(pattern + r'.*?([0-9.+-]+)', content)
                if line_match:
                    return line_match.group(1)
            return None
        except Exception:
            return None

    def extract_node_metric(self, file_path: Path, node: str, metric_col: int = 4) -> Optional[str]:
        """Extract node-specific metric from report"""
        try:
            content = file_path.read_text()
            # Find section and extract metric
            in_section = False
            for line in content.split('\n'):
                if 'Node Depth Summary' in line:
                    in_section = True
                    continue

                if in_section:
                    if line.strip() == '':
                        break

                    parts = line.split()
                    if parts and parts[0] == node:
                        if len(parts) > metric_col:
                            return parts[metric_col]

            return None
        except Exception:
            return None

    def compare_metrics(self):
        """Extract and compare key metrics"""
        self.print_header("Comparing Critical Metrics")

        # Define metrics to compare with tolerance (%)
        metrics_specs = [
            ("External Outflow", "External Outflow.*?([0-9.+-]+)", 5.0),
            ("Continuity Error", "Continuity Error.*?([0-9.+-]+)%", 10.0),
            ("Final Stored Volume", "Final Stored Volume.*?([0-9.+-]+)", 50.0),
        ]

        # Create CSV
        with open(self.metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'CPU_Value', 'GPU_Value', 'Unit', 'Tolerance_%', 'Match'])

            for metric_name, pattern, tolerance in metrics_specs:
                cpu_val = self.extract_metric(self.cpu_rpt, pattern)
                gpu_val = self.extract_metric(self.gpu_rpt, pattern)

                try:
                    cpu_float = float(cpu_val) if cpu_val else 0.0
                    gpu_float = float(gpu_val) if gpu_val else 0.0

                    match = 'Y'
                    if cpu_float != 0:
                        diff_pct = abs((gpu_float - cpu_float) / cpu_float) * 100
                        match = 'Y' if diff_pct <= tolerance else 'N'
                        diff_str = f"({diff_pct:.1f}%)"
                    else:
                        diff_str = ""

                    writer.writerow([metric_name, cpu_val or "N/A", gpu_val or "N/A", "", tolerance, match])

                    status = self.print_success if match == 'Y' else self.print_failure
                    status(f"{metric_name}: CPU={cpu_val} GPU={gpu_val} {diff_str}")

                except (ValueError, TypeError):
                    self.print_warning(f"{metric_name}: Could not parse values")
                    writer.writerow([metric_name, cpu_val or "N/A", gpu_val or "N/A", "", tolerance, "?"])

        # Node-specific comparisons
        print("\nStorage Node (J2) Analysis:")
        cpu_j2_depth = self.extract_node_metric(self.cpu_rpt, "J2", 4)
        gpu_j2_depth = self.extract_node_metric(self.gpu_rpt, "J2", 4)

        if cpu_j2_depth and gpu_j2_depth:
            self.print_failure(f"J2 Depth: CPU={cpu_j2_depth} ft GPU={gpu_j2_depth} ft (Storage behavior differs)")

        print("\nMetrics saved to: {self.metrics_csv}")

    def lines_are_similar(self, line1: str, line2: str, tolerance_pct: float = 5.0) -> bool:
        """
        Compare two lines considering minor numerical differences.

        Args:
            line1: First line to compare
            line2: Second line to compare
            tolerance_pct: Percentage tolerance for numerical differences

        Returns:
            True if lines are identical or differ only by minor numerical values
        """
        # If exactly the same, return True
        if line1 == line2:
            return True

        # Extract all numbers from both lines
        import re
        num_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        nums1 = re.findall(num_pattern, line1)
        nums2 = re.findall(num_pattern, line2)

        # If different number of numeric values, not similar
        if len(nums1) != len(nums2):
            return False

        # If no numbers found, compare as strings
        if len(nums1) == 0:
            return line1.strip() == line2.strip()

        # Replace numbers with placeholders and compare structure
        text1 = re.sub(num_pattern, '{NUM}', line1)
        text2 = re.sub(num_pattern, '{NUM}', line2)

        # If structure differs, not similar
        if text1 != text2:
            return False

        # Compare all numeric values with tolerance
        for n1_str, n2_str in zip(nums1, nums2):
            try:
                n1 = float(n1_str)
                n2 = float(n2_str)

                # Handle zero case
                if n1 == 0 and n2 == 0:
                    continue
                if n1 == 0 or n2 == 0:
                    # If one is zero and the other is very small, consider similar
                    if abs(n1 - n2) < 0.001:
                        continue
                    return False

                # Calculate percentage difference
                diff_pct = abs((n2 - n1) / n1) * 100
                if diff_pct > tolerance_pct:
                    return False

            except (ValueError, ZeroDivisionError):
                # If can't convert to float, must match exactly
                if n1_str != n2_str:
                    return False

        return True

    def generate_diff(self):
        """Generate detailed diff report with tolerance for minor numerical differences"""
        self.print_header("Generating Difference Report")

        try:
            cpu_text = self.cpu_rpt.read_text()
            gpu_text = self.gpu_rpt.read_text()

            cpu_lines = cpu_text.splitlines()
            gpu_lines = gpu_text.splitlines()

            # Write smart diff (only showing significant differences)
            with open(self.diff_file, 'w') as f:
                f.write(f"Smart Diff Report (tolerance: {self.tolerance_pct}% for numerical values)\n")
                f.write(f"CPU Report: {self.cpu_rpt}\n")
                f.write(f"GPU Report: {self.gpu_rpt}\n")
                f.write("="*80 + "\n\n")

                significant_diffs = 0
                minor_diffs = 0

                max_lines = max(len(cpu_lines), len(gpu_lines))

                for i in range(max_lines):
                    cpu_line = cpu_lines[i] if i < len(cpu_lines) else "[MISSING]"
                    gpu_line = gpu_lines[i] if i < len(gpu_lines) else "[MISSING]"

                    if cpu_line == gpu_line:
                        continue  # Identical, skip

                    # Check if similar with tolerance
                    if self.lines_are_similar(cpu_line, gpu_line, tolerance_pct=self.tolerance_pct):
                        minor_diffs += 1
                        # Don't write minor differences to report
                        continue

                    # Significant difference - write to report
                    significant_diffs += 1
                    f.write(f"Line {i+1}:\n")
                    f.write(f"  CPU: {cpu_line}\n")
                    f.write(f"  GPU: {gpu_line}\n")
                    f.write("\n")

                # Summary at end of file
                f.write("\n" + "="*80 + "\n")
                f.write("SUMMARY:\n")
                f.write(f"  Total lines compared: {max_lines}\n")
                f.write(f"  Identical lines: {max_lines - significant_diffs - minor_diffs}\n")
                f.write(f"  Minor differences (within {self.tolerance_pct}% tolerance): {minor_diffs}\n")
                f.write(f"  Significant differences: {significant_diffs}\n")

            if significant_diffs == 0:
                self.print_success(f"No significant differences found ({minor_diffs} minor numerical variations)")
            else:
                self.print_warning(f"Found {significant_diffs} significant differences ({minor_diffs} minor)")

            print(f"File: {self.diff_file}")

        except Exception as e:
            self.print_failure(f"Diff generation failed: {e}")

    def analyze_problems(self):
        """Analyze and report problem areas"""
        self.print_header("Problem Area Analysis")

        problems = []

        # Check outfall performance
        try:
            content = self.gpu_rpt.read_text()
            if "J4                     0.00" in content:
                problems.append("GPU: Outfall (J4) showing 0% frequency - flow not reaching outfall")
        except:
            pass

        # Check storage level
        try:
            gpu_j2 = self.extract_node_metric(self.gpu_rpt, "J2", 4)
            if gpu_j2 and float(gpu_j2) > 1.0:
                problems.append(f"GPU: Storage node (J2) depth {gpu_j2} ft - not draining properly")
        except:
            pass

        # Check pump operation
        try:
            content = self.gpu_rpt.read_text()
            if "C2                       0.00" in content:
                problems.append("GPU: Pump (C2) at 0% utilization - not operating")
        except:
            pass

        # Check weir operation
        try:
            content = self.gpu_rpt.read_text()
            weir_match = re.search(r'C3\s+WEIR\s+(\d+\.\d+)', content)
            if weir_match and float(weir_match.group(1)) == 0.0:
                problems.append("GPU: Weir (C3) carrying zero flow - flow routing stopped")
        except:
            pass

        # Check continuity error
        try:
            gpu_cont = self.extract_metric(self.gpu_rpt, r"Continuity Error.*?([0-9.]+)%")
            if gpu_cont and float(gpu_cont) > 1.0:
                problems.append(f"GPU: Continuity error {gpu_cont}% - exceeds acceptable tolerance")
        except:
            pass

        # Write problems file
        with open(self.problems_file, 'w') as f:
            f.write("GPU FLOW ROUTING ISSUES DETECTED\n")
            f.write("="*60 + "\n\n")

            if problems:
                for problem in problems:
                    f.write(f"• {problem}\n")
                    self.print_failure(problem)
            else:
                f.write("No critical issues detected.\n")
                self.print_success("No critical issues detected")

        print(f"\nAnalysis saved to: {self.problems_file}")

    def print_summary(self):
        """Print test summary"""
        self.print_header("Test Summary")

        print(f"\nResults:")
        print(f"  Passed: {Colors.GREEN}{self.passed_tests}{Colors.NC}")
        print(f"  Failed: {Colors.RED}{self.failed_tests}{Colors.NC}")

        print(f"\nOutput Files:")
        print(f"  CPU Report:    {self.cpu_rpt}")
        print(f"  GPU Report:    {self.gpu_rpt}")
        print(f"  Metrics CSV:   {self.metrics_csv}")
        print(f"  Diff Report:   {self.diff_file}")
        print(f"  Analysis:      {self.problems_file}")

        print(f"\nAll results saved to: {Colors.BLUE}{self.output_dir}{Colors.NC}")

        if self.failed_tests > 0:
            print(f"\n{Colors.RED}CRITICAL ISSUES DETECTED - GPU results differ significantly from CPU{Colors.NC}")
            return False
        else:
            print(f"\n{Colors.GREEN}All tests passed{Colors.NC}")
            return True

    def run_all(self) -> bool:
        """Run complete test suite"""
        print(f"{Colors.BLUE}GPU Flow Routing Test Suite{Colors.NC}")
        print(f"Repository: {self.repo_root}")
        print(f"Output: {self.output_dir}\n")

        # Run tests in sequence
        if not self.verify_prerequisites():
            return False

        if not self.run_simulation(use_gpu=False):
            return False

        if not self.run_simulation(use_gpu=True):
            return False

        self.compare_metrics()
        self.generate_diff()
        self.analyze_problems()

        return self.print_summary()

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU Flow Routing Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default output directory and 5% tolerance
  python3 gpu_flow_routing_test.py

  # Run with custom output directory
  python3 gpu_flow_routing_test.py --output-dir /tmp/gpu_test

  # Specify repository location
  python3 gpu_flow_routing_test.py --repo-dir ~/workspace/SWMM

  # Use stricter tolerance (2% for numerical differences)
  python3 gpu_flow_routing_test.py --tolerance 2.0

  # Use more lenient tolerance (10% for numerical differences)
  python3 gpu_flow_routing_test.py --tolerance 10.0
        """
    )

    parser.add_argument(
        '--output-dir',
        help='Output directory for test results',
        default=None
    )

    parser.add_argument(
        '--repo-dir',
        help='Repository root directory',
        default=None
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        help='Percentage tolerance for numerical differences (default: 5.0)',
        default=5.0
    )

    args = parser.parse_args()

    test = GPUFlowRoutingTest(
        repo_root=args.repo_dir,
        output_dir=args.output_dir,
        tolerance_pct=args.tolerance
    )
    success = test.run_all()

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
