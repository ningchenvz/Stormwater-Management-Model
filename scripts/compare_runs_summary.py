#!/usr/bin/env python3
"""Run GPU/CPU comparisons for every INP file in a directory and summarize results."""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ANSI color codes
class Colors:
    RED = '\033[1;31m'
    YELLOW = '\033[1;33m'
    GREEN = '\033[1;32m'
    RESET = '\033[0m'

    @staticmethod
    def colorize_status(status: str) -> str:
        """Add color and symbol based on status."""
        status_lower = status.lower()
        if status_lower == "match":
            return f"{Colors.GREEN}✓ {status}{Colors.RESET}"
        elif status_lower == "minor":
            return f"{Colors.YELLOW}⚠ {status}{Colors.RESET}"
        elif status_lower == "major":
            return f"{Colors.RED}! {status}{Colors.RESET}"
        elif status_lower == "differ":
            return f"{Colors.RED}! {status}{Colors.RESET}"
        elif status_lower == "error":
            return f"{Colors.RED}✗ {status}{Colors.RESET}"
        else:
            return status


OUTPUT_DIR_RE = re.compile(r"^Output folder\s*:\s*(.+)$")
ARTIFACT_RE = re.compile(r"^Artifacts written to:\s*(.+)$")
REPORT_RE = re.compile(r"^Reports\s+(match|differ(?:\s*-\s*(MINOR|MAJOR)\s+differences)?)(?:.*?(?:see|at)\s+([^)]*))?", re.IGNORECASE)
COMPLETED_RE = re.compile(r"completed in\s+([0-9.]+)\s+ms", re.IGNORECASE)
BINARY_RE = re.compile(r"^Binary outputs\s+(match|differ)", re.IGNORECASE)
MODEL_INFO_RE = re.compile(r"^Model info\s*:\s*(\d+)\s+links,\s*(.+?)\s+simulation", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compare_runswmm_gpu_cpu.sh on every .inp under a directory "
        "and print a summary table."
    )
    parser.add_argument("directory", help="Directory containing .inp files (searched recursively).")
    parser.add_argument(
        "--compare-script",
        default=str(Path(__file__).resolve().with_name("compare_runswmm_gpu_cpu.sh")),
        help="Path to compare_runswmm_gpu_cpu.sh (default: %(default)s)",
    )
    parser.add_argument(
        "--runswmm",
        help="Optional runswmm executable path (forwarded to compare script via RUNSWMM env var).",
    )
    parser.add_argument(
        "--out-dir",
        help="Optional output directory (forwarded to compare script via OUT_DIR env var).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only process INP files directly inside the given directory.",
    )
    return parser.parse_args()


def collect_inp_files(root: Path, recursive: bool) -> List[Path]:
    """Collect all .inp files (case-insensitive) from the given directory."""
    # Use glob to get all files, then filter by extension (case-insensitive)
    if recursive:
        all_files = root.rglob("*")
    else:
        all_files = root.glob("*")

    # Filter for .inp extension (case-insensitive)
    inp_files = [f for f in all_files if f.is_file() and f.suffix.lower() == ".inp"]

    return sorted(inp_files)


def run_compare_script(
    compare_script: Path,
    inp_path: Path,
    env: dict,
    cwd: Path,
) -> Tuple[int, str]:
    result = subprocess.run(
        [str(compare_script), str(inp_path)],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
    )
    output = result.stdout + ("\n" + result.stderr if result.stderr else "")
    return result.returncode, output.strip()


def extract_summary(output: str):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    out_folder = None
    artifacts = None
    report_status = "n/a"
    report_path = ""
    binary_status = "n/a"
    gpu_time = "n/a"
    cpu_time = "n/a"
    total_links = "n/a"
    sim_duration = "n/a"
    current_mode = None

    for line in lines:
        if out_folder is None:
            match = OUTPUT_DIR_RE.match(line)
            if match:
                out_folder = match.group(1).strip()
        if artifacts is None:
            match = ARTIFACT_RE.match(line)
            if match:
                artifacts = match.group(1).strip()
        if report_status == "n/a":
            match = REPORT_RE.match(line)
            if match:
                base_status = match.group(1).lower()
                severity = match.group(2)  # MINOR or MAJOR
                path_group = match.group(3)

                if "differ" in base_status and severity:
                    report_status = severity.lower()  # "minor" or "major"
                elif "match" in base_status:
                    report_status = "match"
                else:
                    report_status = "differ"

                if path_group:
                    report_path = path_group.strip()
        if binary_status == "n/a":
            match = BINARY_RE.match(line)
            if match:
                binary_status = match.group(1).lower()
        if total_links == "n/a":
            match = MODEL_INFO_RE.match(line)
            if match:
                total_links = match.group(1)
                sim_duration = match.group(2).strip()
        if line.startswith("==> Running "):
            if "GPU" in line.upper():
                current_mode = "gpu"
            elif "CPU" in line.upper():
                current_mode = "cpu"
            else:
                current_mode = None
            continue
        match = COMPLETED_RE.search(line)
        if match and current_mode:
            elapsed = match.group(1)
            if current_mode == "gpu" and gpu_time == "n/a":
                gpu_time = elapsed
            elif current_mode == "cpu" and cpu_time == "n/a":
                cpu_time = elapsed
            current_mode = None

    if artifacts is None:
        artifacts = out_folder
    return {
        "output": output,
        "output_folder": out_folder or "",
        "artifacts": artifacts or "",
        "report_status": report_status,
        "report_path": report_path,
        "binary_status": binary_status,
        "gpu_time": gpu_time,
        "cpu_time": cpu_time,
        "total_links": total_links,
        "sim_duration": sim_duration,
    }


def format_table(headers: Tuple[str, ...], rows: List[Tuple[str, ...]]) -> str:
    """Format a table with proper column alignment, handling ANSI color codes."""
    # ANSI color code pattern
    ansi_pattern = re.compile(r'\033\[[0-9;]+m')

    def strip_ansi(text: str) -> str:
        """Remove ANSI color codes for length calculation."""
        return ansi_pattern.sub('', text)

    all_rows = [headers] + rows
    widths = [0] * len(headers)
    for row in all_rows:
        for idx, cell in enumerate(row):
            # Calculate width without ANSI codes
            widths[idx] = max(widths[idx], len(strip_ansi(cell)))

    def fmt(row):
        formatted_cells = []
        for idx, cell in enumerate(row):
            # Calculate padding needed (accounting for ANSI codes)
            visible_len = len(strip_ansi(cell))
            padding = widths[idx] - visible_len
            formatted_cells.append(cell + ' ' * padding)
        return " | ".join(formatted_cells)

    sep = "-+-".join("-" * w for w in widths)
    parts = [fmt(headers), sep]
    for row in rows:
        parts.append(fmt(row))
    return "\n".join(parts)


def main():
    args = parse_args()
    target_dir = Path(args.directory).resolve()
    if not target_dir.is_dir():
        sys.exit(f"Directory not found: {target_dir}")

    compare_script = Path(args.compare_script).resolve()
    if not compare_script.exists():
        sys.exit(f"Compare script not found: {compare_script}")

    repo_root = compare_script.parent.parent

    env = os.environ.copy()
    if args.runswmm:
        env["RUNSWMM"] = args.runswmm
    if args.out_dir:
        env["OUT_DIR"] = args.out_dir

    inp_files = collect_inp_files(target_dir, recursive=not args.non_recursive)
    if not inp_files:
        sys.exit(f"No .inp/.INP files found under {target_dir}")

    summary_rows: List[Tuple[str, str, str, str, str, str, str]] = []
    diff_rows: List[Tuple[str, str]] = []
    any_failures = False
    total = len(inp_files)

    for idx, inp in enumerate(inp_files, 1):
        rel = os.path.relpath(inp, start=target_dir)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{idx}/{total}] {rel} @ {timestamp}", flush=True)
        code, output = run_compare_script(compare_script, inp, env, repo_root)
        if code != 0:
            any_failures = True
            print(output, file=sys.stderr)
            error_cell = Colors.colorize_status("error")
            summary_rows.append((inp.name, error_cell, error_cell, "n/a", "n/a", "n/a", "n/a"))
            continue

        summary = extract_summary(output)
        report_cell = Colors.colorize_status(summary["report_status"])
        if summary["report_path"]:
            diff_rows.append((Path(summary["report_path"]).name, summary["report_path"]))
        binary_cell = Colors.colorize_status(summary["binary_status"])
        summary_rows.append((inp.name, report_cell, binary_cell,
                           summary["total_links"], summary["sim_duration"],
                           summary["gpu_time"], summary["cpu_time"]))

        print("\nCurrent Summary:\n")
        print(
            format_table(
                ("Input", "Report Diff", "Binary Diff", "Links", "Duration", "GPU Time (ms)", "CPU Time (ms)"),
                summary_rows,
            )
        )

    print("\nFinal Summary:\n")
    print(
        format_table(
            ("Input", "Report Diff", "Binary Diff", "Links", "Duration", "GPU Time (ms)", "CPU Time (ms)"),
            summary_rows,
        )
    )
    print("\nReport Diff Files:\n")
    if diff_rows:
        print(format_table(("Diff Name", "Diff Path"), diff_rows))
    else:
        print("No diff files generated.")
    print()

    if any_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
