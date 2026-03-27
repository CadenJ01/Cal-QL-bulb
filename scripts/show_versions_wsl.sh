#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:?env name required}"

source /home/caden/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
python - <<'PY'
import importlib

mods = ["jax", "flax", "optax", "distrax", "ml_collections", "wandb", "torch", "hydra", "isaacgymenvs"]
for mod_name in mods:
    try:
        mod = importlib.import_module(mod_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{mod_name} {version}")
    except Exception as exc:
        print(f"{mod_name} FAIL: {type(exc).__name__}: {exc}")
PY
