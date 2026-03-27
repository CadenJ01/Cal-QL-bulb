#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-calql}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/bulb_40eps_lowdim_calql.npz}"
EXPORT_DIR="${EXPORT_DIR:-$REPO_ROOT/exports}"
EXPORT_PATH="${EXPORT_PATH:-$EXPORT_DIR/bulb_policy_offline_200ep.npz}"
PROJECT_NAME="${PROJECT_NAME:-bulb-calql}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-1000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
POLICY_ARCH="${POLICY_ARCH:-256-256}"
QF_ARCH="${QF_ARCH:-256-256-256-256}"
REWARD_SCALE="${REWARD_SCALE:-1.0}"
REWARD_BIAS="${REWARD_BIAS:-0.0}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME/lib:$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$EXPORT_DIR"
cd "$REPO_ROOT"
PYTHONPATH=. python -m JaxCQL.conservative_sac_main \
  --dataset_type=custom_npz \
  --dataset_path="$DATASET_PATH" \
  --offline_only=True \
  --logging.online=True \
  --logging.project="$PROJECT_NAME" \
  --seed="$SEED" \
  --batch_size="$BATCH_SIZE" \
  --n_train_step_per_epoch_offline="$TRAIN_STEPS_PER_EPOCH" \
  --n_pretrain_epochs="$PRETRAIN_EPOCHS" \
  --policy_arch="$POLICY_ARCH" \
  --qf_arch="$QF_ARCH" \
  --reward_scale="$REWARD_SCALE" \
  --reward_bias="$REWARD_BIAS" \
  --enable_calql=True \
  --policy_export_path="$EXPORT_PATH"
