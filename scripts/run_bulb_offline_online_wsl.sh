#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-calql}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/bulb_40eps_lowdim_calql.npz}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints}"
EXPORT_DIR="${EXPORT_DIR:-$REPO_ROOT/exports}"
FINAL_EXPORT_PATH="${FINAL_EXPORT_PATH:-$EXPORT_DIR/bulb_policy_offline_online_final.npz}"
PROJECT_NAME="${PROJECT_NAME:-bulb-calql-online}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-1000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
POLICY_ARCH="${POLICY_ARCH:-256-256}"
QF_ARCH="${QF_ARCH:-256-256-256-256}"
REWARD_SCALE="${REWARD_SCALE:-1.0}"
REWARD_BIAS="${REWARD_BIAS:-0.0}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-120,160,180,200}"
MAX_ONLINE_ENV_STEPS="${MAX_ONLINE_ENV_STEPS:-50000}"
ONLINE_NUM_ENVS="${ONLINE_NUM_ENVS:-50}"
ONLINE_MAX_EPISODE_STEPS="${ONLINE_MAX_EPISODE_STEPS:-1000}"
ONLINE_TRAJS_PER_EPOCH="${ONLINE_TRAJS_PER_EPOCH:-50}"
ONLINE_UTD_RATIO="${ONLINE_UTD_RATIO:-1}"
ONLINE_OBS_MODE="${ONLINE_OBS_MODE:-legacy_7d}"
MANIFEEL_CFG_DIR="${MANIFEEL_CFG_DIR:-$HOME/manifeel-isaacgymenvs/isaacgymenvs/cfg}"
ISAAC_GYM_ROOT="${ISAAC_GYM_ROOT:-$HOME/IsaacGym_Preview_TacSL_Package/isaacgym/python}"
MANIFEEL_ROOT="${MANIFEEL_ROOT:-$HOME/manifeel-isaacgymenvs}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_ROOT/envs/$CONDA_ENV_NAME/lib:$ISAAC_GYM_ROOT/isaacgym/_bindings/linux-x86_64:$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:$MANIFEEL_ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MANIFEEL_CFG_DIR

mkdir -p "$CHECKPOINT_DIR" "$EXPORT_DIR"
cd "$REPO_ROOT"

python -m JaxCQL.conservative_sac_main \
  --dataset_type=custom_npz \
  --dataset_path="$DATASET_PATH" \
  --offline_only=False \
  --online_env_type=manifeel_bulb \
  --checkpoint_dir="$CHECKPOINT_DIR" \
  --checkpoint_epochs="$CHECKPOINT_EPOCHS" \
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
  --max_online_env_steps="$MAX_ONLINE_ENV_STEPS" \
  --online_num_envs="$ONLINE_NUM_ENVS" \
  --online_max_episode_steps="$ONLINE_MAX_EPISODE_STEPS" \
  --n_online_traj_per_epoch="$ONLINE_TRAJS_PER_EPOCH" \
  --online_utd_ratio="$ONLINE_UTD_RATIO" \
  --online_obs_mode="$ONLINE_OBS_MODE" \
  --policy_export_path="$FINAL_EXPORT_PATH"
