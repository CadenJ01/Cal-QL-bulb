#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEEL_ROOT="${MANIFEEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CALQL_ROOT="${CALQL_ROOT:-$HOME/calql-wsl}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tacsl}"
ISAAC_GYM_ROOT="${ISAAC_GYM_ROOT:-$HOME/IsaacGym_Preview_TacSL_Package/isaacgym}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_ROOT/envs/$CONDA_ENV_NAME/lib:$ISAAC_GYM_ROOT/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export CALQL_ROOT

cd "$MANIFEEL_ROOT/examples"
python demo_bulb_policy_playback.py task=TacSLTaskBulb train=TacSLTaskBulbInsertionPPO_LSTM_dict_AAC
