#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/caden/miniforge3/etc/profile.d/conda.sh
conda activate calql

export PATH=/home/caden/miniforge3/envs/calql/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/home/caden/miniforge3/envs/calql/lib:/home/caden/.mujoco/mujoco210/bin
export XLA_PYTHON_CLIENT_PREALLOCATE=false

DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/bulb_40eps_lowdim_calql.npz}"
PROJECT_NAME="${PROJECT_NAME:-bulb-calql}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-1000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
POLICY_ARCH="${POLICY_ARCH:-256-256}"
QF_ARCH="${QF_ARCH:-256-256-256-256}"
REWARD_SCALE="${REWARD_SCALE:-1.0}"
REWARD_BIAS="${REWARD_BIAS:-0.0}"

cd "$REPO_ROOT"

python -m JaxCQL.conservative_sac_main \
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
  --enable_calql=True
