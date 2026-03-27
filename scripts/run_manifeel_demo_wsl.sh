#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEEL_ROOT="${MANIFEEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniforge3}
ENV_NAME=${ENV_NAME:-tacsl}
ISAAC_GYM_ROOT=${ISAAC_GYM_ROOT:-$HOME/IsaacGym_Preview_TacSL_Package/isaacgym}
DEMO_SCRIPT=${DEMO_SCRIPT:-${MANIFEEL_ROOT}/examples/demo.py}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${ISAAC_GYM_ROOT}/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"

cd "${MANIFEEL_ROOT}/examples"
python "${DEMO_SCRIPT}" "$@"
