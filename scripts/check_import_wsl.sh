#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:?env name required}"
MODULE_NAME="${2:?module name required}"

source /home/caden/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/caden/miniforge3/envs/${ENV_NAME}/lib:/home/caden/IsaacGym_Preview_TacSL_Package/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/caden/manifeel-isaacgymenvs:${PYTHONPATH:-}"
python - "$ENV_NAME" "$MODULE_NAME" <<'PY'
import importlib
import sys

env_name = sys.argv[1]
module_name = sys.argv[2]

try:
    importlib.import_module(module_name)
    print(f"{env_name} {module_name} OK")
except Exception as exc:
    print(f"{env_name} {module_name} FAIL: {type(exc).__name__}: {exc}")
PY
