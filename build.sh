#!/usr/bin/env bash
# Build and install NusaLearn, including the compiled kernels.
#
# Usage: ./build.sh [venv-dir]
# Creates a virtual environment (default .venv), installs the build
# toolchain into it, then installs this project in editable mode.
set -euo pipefail

VENV_DIR="${1:-.venv}"
cd "$(dirname "$0")"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install "meson-python" "meson>=1.1" ninja "Cython>=3.0" "numpy>=1.26" pytest
python -m pip install --no-build-isolation -e .

python - <<'PY'
import nusalearn
from nusalearn._pairwise import KERNEL_AVAILABLE
from nusalearn import _native

print("nusalearn", nusalearn.__version__)
print("Cython kernel available:", KERNEL_AVAILABLE)
print("C dot available:", _native.DOT_AVAILABLE)
print("C++ matmul available:", _native.MATOPS_AVAILABLE)
PY
