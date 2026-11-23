#!/usr/bin/env bash
# Run the NusaLearn test suite.
#
# Usage: ./test.sh [venv-dir]
# Expects the package to be installed already (see build.sh). The
# suite is run twice: once against the installed package (compiled
# kernels active) and once from a neutral directory with
# NUSALEARN_NO_KERNEL=1 to verify the pure Python fallback path.
set -euo pipefail

VENV_DIR="${1:-.venv}"
cd "$(dirname "$0")"

if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

WORKDIR="$(mktemp -d)"
cp -r tests "$WORKDIR/"
cd "$WORKDIR"

echo "== tests with compiled kernels (if built) =="
python -m pytest tests -q

echo "== tests with pure Python fallback (NUSALEARN_NO_KERNEL=1) =="
NUSALEARN_NO_KERNEL=1 python -m pytest tests -q

rm -rf "$WORKDIR"
echo "All test runs passed."
