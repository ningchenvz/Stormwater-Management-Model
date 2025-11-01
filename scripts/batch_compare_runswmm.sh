#!/usr/bin/env bash
# Run GPU/CPU comparisons for every .inp/.INP file within a directory.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/batch_compare_runswmm.sh <directory-with-inp-files>

Environment variables:
  COMPARE_SCRIPT  Path to compare_runswmm_gpu_cpu.sh
                  (default: scripts/compare_runswmm_gpu_cpu.sh)
  RUNSWMM         Passed through to the compare script (optional)
  OUT_DIR         Passed through to the compare script (optional)
EOF
    exit 1
}

die() {
    echo "Error: $*" >&2
    exit 1
}

[ $# -eq 1 ] || usage
TARGET_DIR=$1
[[ -d "$TARGET_DIR" ]] || die "directory not found: $TARGET_DIR"
TARGET_DIR=$(realpath "$TARGET_DIR")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMPARE_SCRIPT=${COMPARE_SCRIPT:-${SCRIPT_DIR}/compare_runswmm_gpu_cpu.sh}
[[ -x "$COMPARE_SCRIPT" ]] || die "compare script not found or not executable: $COMPARE_SCRIPT"

echo "Batch compare directory : $TARGET_DIR"
echo "Compare script          : $COMPARE_SCRIPT"

mapfile -t INP_FILES < <(find "$TARGET_DIR" -type f \( -iname '*.inp' \) | sort)

if [ ${#INP_FILES[@]} -eq 0 ]; then
    die "no .inp/.INP files found under $TARGET_DIR"
fi

failed=0
for inp in "${INP_FILES[@]}"; do
    echo
    echo ">>> Processing $(realpath --relative-to="$TARGET_DIR" "$inp")"
    if "$COMPARE_SCRIPT" "$inp"; then
        echo ">>> SUCCESS: $inp"
    else
        echo ">>> FAILED: $inp"
        failed=1
    fi
done

if [ $failed -eq 0 ]; then
    echo
    echo "All comparisons completed successfully."
else
    echo
    echo "One or more comparisons failed."
fi

exit $failed
