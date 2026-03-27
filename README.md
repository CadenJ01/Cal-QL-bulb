# Cal-QL Bulb Offline-to-Online Pipeline

This repository adapts the original Cal-QL codebase for a custom bulb manipulation workflow based on:

- a low-dimensional offline bulb dataset,
- Cal-QL / CQL offline pretraining in JAX,
- ManiFeel TacSL bulb simulation in Isaac Gym,
- optional online fine-tuning from the offline checkpoint.

The original upstream Cal-QL README is preserved in `UPSTREAM_README.md`.

## Current Scope

This repository now supports four connected stages:

1. preprocess a custom bulb dataset into Cal-QL format,
2. run offline Cal-QL training on the processed `.npz`,
3. export checkpointed or final policies as portable `.npz` actor files,
4. run a unified offline-to-online pipeline with ManiFeel / Isaac Gym rollout collection and W&B logging.

The unified pipeline has been smoke-tested with:

- offline pretraining,
- checkpoint saving,
- ManiFeel online rollout collection,
- W&B success-rate logging,
- final policy export.

This verifies the code path and deployment flow. It does not imply that the current bulb policy is task-successful.

## Main Additions Over Upstream Cal-QL

- custom `custom_npz` dataset support,
- preprocessing for bulb episodes, done flags, action normalization, and Monte Carlo returns,
- policy export from JAX actor parameters to a portable NumPy format,
- ManiFeel bulb online sampler for offline-to-online training,
- checkpoint saving at selected epochs,
- a unified WSL runner for offline pretraining followed by online rollout and continued updates,
- Isaac Gym replay and visualization scripts for smoke testing.

## Repository Layout

- `JaxCQL/`
  - core Cal-QL training code
  - custom dataset support
  - checkpoint export
  - ManiFeel online sampler
- `scripts/preprocess_custom_npz.py`
  - converts raw bulb `.npz` data into Cal-QL-ready format
- `scripts/run_bulb_offline.sh`
  - offline-only training entry
- `scripts/run_bulb_policy_export_full.sh`
  - full offline training with final actor export
- `scripts/run_bulb_offline_online_wsl.sh`
  - unified offline-to-online WSL runner
- `scripts/demo_bulb_policy_playback.py`
  - replay an exported actor in the TacSL bulb task
- `scripts/demo_bulb_visual.py`
  - random-action viewer smoke test
- `viskit/`
  - Cal-QL logging utilities

## Environment Strategy

This project was originally split across two WSL environments:

- `calql` for JAX / offline training
- `tacsl` for Isaac Gym / ManiFeel visualization

The unified offline-to-online pipeline is now designed to run in a single WSL Conda environment: `calql`.

That unified `calql` environment must contain:

- JAX / Flax / Optax / Distrax / ml-collections / W&B,
- PyTorch,
- Hydra,
- Isaac Gym,
- `manifeel-isaacgymenvs`,
- `opencv-python-headless`.

The separate `tacsl` environment can still be useful for isolated viewer tests, but it is no longer required for the unified trainer.

## System Requirements

- Windows host
- WSL2 Ubuntu
- NVIDIA GPU
- Miniforge or Conda inside WSL
- local TacSL-compatible Isaac Gym package
- local `manifeel-isaacgymenvs` checkout

Known runtime note:

- Isaac Gym viewer sessions under WSL can still be unstable because of native graphics / Vulkan behavior.
- Headless training is the intended path for unified offline-to-online training.

## Expected WSL Layout

The scripts assume a layout similar to:

```text
~/calql-wsl
~/manifeel-isaacgymenvs
~/IsaacGym_Preview_TacSL_Package
~/miniforge3
```

These are defaults only. All important paths can be overridden with environment variables.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/CadenJ01/Cal-QL-bulb.git
cd Cal-QL-bulb
```

### 2. Create the unified `calql` environment

Use Python 3.8:

```bash
conda create -n calql python=3.8 -y
conda activate calql
pip install -r requirements.txt
```

Then install the additional unified-training dependencies:

```bash
pip install torch==2.2.0 torchvision==0.17.0 hydra-core==1.3.2
pip install opencv-python-headless==4.10.0.84
pip install -e /path/to/IsaacGym_Preview_TacSL_Package/isaacgym/python
pip install -e /path/to/manifeel-isaacgymenvs
```

Important:

- Isaac Gym and ManiFeel must be installed in the same environment used for the unified trainer.
- If you use different install locations, update:
  - `ISAAC_GYM_ROOT`
  - `MANIFEEL_ROOT`
  - `MANIFEEL_CFG_DIR`

### 3. Configure W&B

Create:

```text
JaxCQL/wandb_config.py
```

using `JaxCQL/wandb_config_example.py` as the template.

That file is intentionally excluded from version control.

## Dataset Preprocessing

The training pipeline expects a processed bulb dataset in Cal-QL format.

Example raw input:

- `bulb_40eps_lowdim.npz`

Run preprocessing:

```bash
python scripts/preprocess_custom_npz.py \
  --input /path/to/bulb_40eps_lowdim.npz \
  --output /path/to/output/bulb_40eps_lowdim_calql.npz
```

Preprocessing does the following:

- reconstructs episodes from `episode_starts` and `episode_lengths`,
- rewrites `done` so only the last step of each episode is terminal,
- normalizes actions into `[-1, 1]`,
- computes dense Monte Carlo returns.

## Offline-Only Training

Run standard offline training:

```bash
bash scripts/run_bulb_offline.sh
```

Run the full offline export path:

```bash
bash scripts/run_bulb_policy_export_full.sh
```

Typical output:

```text
exports/bulb_policy_offline_200ep.npz
```

## Unified Offline-to-Online Training

The main runner is:

```bash
bash scripts/run_bulb_offline_online_wsl.sh
```

This runner now supports:

- offline pretraining,
- selected checkpoint saves,
- automatic transition into ManiFeel online rollout,
- replay-buffer updates from online trajectories,
- W&B logging for rollout statistics and success rate,
- final actor export.

### Important environment variables

- `CONDA_ROOT`
- `CONDA_ENV_NAME`
- `DATASET_PATH`
- `CHECKPOINT_DIR`
- `EXPORT_DIR`
- `FINAL_EXPORT_PATH`
- `PROJECT_NAME`
- `CHECKPOINT_EPOCHS`
- `MAX_ONLINE_ENV_STEPS`
- `ONLINE_NUM_ENVS`
- `ONLINE_MAX_EPISODE_STEPS`
- `ONLINE_TRAJS_PER_EPOCH`
- `ONLINE_UTD_RATIO`
- `ONLINE_OBS_MODE`
- `MANIFEEL_CFG_DIR`
- `MANIFEEL_ROOT`
- `ISAAC_GYM_ROOT`

### Default unified settings

The script defaults to:

- `CHECKPOINT_EPOCHS=120,160,180,200`
- `ONLINE_NUM_ENVS=50`
- `ONLINE_MAX_EPISODE_STEPS=1000`

These defaults are intended for the planned bulb offline-to-online workflow, but should be scaled down for smoke tests.

### Example smoke test

```bash
CONDA_ROOT=$HOME/miniforge3 \
CONDA_ENV_NAME=calql \
PRETRAIN_EPOCHS=2 \
TRAIN_STEPS_PER_EPOCH=10 \
CHECKPOINT_EPOCHS=1,2 \
ONLINE_NUM_ENVS=2 \
ONLINE_MAX_EPISODE_STEPS=20 \
ONLINE_TRAJS_PER_EPOCH=2 \
MAX_ONLINE_ENV_STEPS=40 \
ONLINE_UTD_RATIO=1 \
PROJECT_NAME=bulb-calql-online-smoke \
bash scripts/run_bulb_offline_online_wsl.sh
```

## Checkpointing

Checkpoint saving is built into `JaxCQL/checkpointing.py`.

For every selected epoch, the unified trainer writes:

- `checkpoint_epoch_<N>.pkl`
- `policy_epoch_<N>.npz`

The final export path is controlled separately by:

- `FINAL_EXPORT_PATH`

This makes it possible to compare intermediate policies such as epochs `120`, `160`, `180`, and `200`.

## Online Observation Modes

The ManiFeel online wrapper currently supports:

- `legacy_7d`
- `relative_7d`

### `legacy_7d`

Uses:

- `ee_pos`
- `ee_quat`

This preserves the original 7D input size used by the existing offline actor.

### `relative_7d`

Uses:

- `ee_pos - socket_pos`
- relative quaternion between socket and end-effector

This is intended to improve alignment information without changing the actor input dimension.

Important constraint:

- the current offline actor is a 7D-input network,
- so online fine-tuning can change the semantics of the 7D observation,
- but it cannot arbitrarily change the input dimensionality without retraining the architecture.

## W&B Metrics Added for Online Training

The unified trainer now logs:

- `evaluation/average_return`
- `evaluation/average_traj_length`
- `evaluation/success_rate`
- `exploration/average_return`
- `exploration/average_traj_length`
- `exploration/success_rate`
- `exploration/num_envs`
- `exploration/max_episode_steps`

These are the main rollout-side metrics for validating the online phase.

## Policy Export Format

Exported policy files store actor parameters, not trajectories.

Current actor format:

- observation dimension: `7`
- action dimension: `7`
- MLP architecture: `256-256`
- final layer output: `14`

Those 14 outputs correspond to:

- 7 action means
- 7 action log standard deviations

Deterministic replay uses:

```text
action = tanh(mean)
```

## Isaac Gym Replay

### Viewer smoke test

```bash
bash scripts/run_demo_bulb_visual_wsl.sh
```

### Exported policy replay

```bash
bash scripts/run_demo_bulb_policy_playback_wsl.sh
```

These scripts are for validation and visualization only. The unified trainer itself is intended to run headless.

## Verified Status

At the current repository state:

- custom bulb `.npz` preprocessing works,
- offline Cal-QL training works,
- final policy export works,
- checkpoint export works,
- ManiFeel random-action visualization works,
- exported policy playback works,
- the unified offline-to-online pipeline has been smoke-tested end-to-end.

The remaining work is algorithmic, not plumbing:

- improve observation design for alignment,
- compare intermediate checkpoints,
- tune online rollout and update schedules,
- improve task success rather than only pipeline correctness.

## Known Limitations

- The current bulb policy is still low-dimensional and state-only.
- The online wrapper can switch to `relative_7d`, but this still preserves the same 7D actor input size.
- Long viewer sessions under WSL may still segfault.
- Some Isaac Gym / ManiFeel dependencies rely on older libraries and need compatibility handling under newer NumPy environments.

## Files Most Likely To Change Next

- `JaxCQL/conservative_sac_main.py`
- `JaxCQL/manifeel_sampler.py`
- `JaxCQL/replay_buffer.py`
- `scripts/run_bulb_offline_online_wsl.sh`
- `scripts/preprocess_custom_npz.py`

## Credits

- Original Cal-QL implementation: [Cal-QL](https://github.com/nakamotoo/Cal-QL)
- Offline RL method: Cal-QL / CQL
- Simulation environment: Isaac Gym + ManiFeel TacSL bulb task
