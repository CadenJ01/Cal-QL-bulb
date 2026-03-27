#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-calql}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/bulb_40eps_lowdim_calql.npz}"
EXPORT_DIR="${EXPORT_DIR:-$REPO_ROOT/exports}"
EXPORT_PATH="${EXPORT_PATH:-$EXPORT_DIR/bulb_policy_smoke.npz}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME/lib:$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

mkdir -p "$EXPORT_DIR"
cd "$REPO_ROOT"
PYTHONPATH=. python -m JaxCQL.conservative_sac_main \
  --dataset_type=custom_npz \
  --dataset_path="$DATASET_PATH" \
  --offline_only=True \
  --logging.online=False \
  --seed=0 \
  --batch_size=256 \
  --n_train_step_per_epoch_offline=200 \
  --n_pretrain_epochs=1 \
  --policy_arch=256-256 \
  --qf_arch=256-256-256-256 \
  --reward_scale=1.0 \
  --reward_bias=0.0 \
  --enable_calql=True \
  --policy_export_path="$EXPORT_PATH"
