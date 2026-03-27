# Cal-QL Bulb Policy Transfer for Isaac Gym

This repository extends the original Cal-QL codebase to support a custom low-dimensional bulb manipulation dataset and replay the resulting offline policy inside an Isaac Gym / ManiFeel environment.

It is intended for a team workflow: a new contributor should be able to clone the repository, set up the required WSL environments, preprocess the dataset, train an offline policy, export that policy, and replay it inside Isaac Gym.

The original upstream Cal-QL README is preserved in `UPSTREAM_README.md`.

## Project Goals

This project currently supports three stages:

1. preprocessing a custom bulb manipulation dataset into Cal-QL format,
2. offline training of a Cal-QL policy on that dataset,
3. policy export and replay inside the TacSL bulb task in Isaac Gym.

This repository is based on the original [Cal-QL](https://github.com/nakamotoo/Cal-QL) implementation and has been adapted for a Purdue bulb manipulation workflow.

## What Was Added

Compared with the upstream Cal-QL repository, this project adds:

- custom `.npz` dataset support for offline training,
- preprocessing utilities for reconstructing episodes and computing Monte Carlo returns,
- WSL-based setup scripts for Cal-QL and Isaac Gym / ManiFeel,
- policy export utilities that convert the trained JAX actor into a portable NumPy policy,
- playback scripts that load the exported policy and run it in the TacSL bulb environment.

## Repository Structure

- `JaxCQL/`: Cal-QL training code, replay buffer, and policy export logic
- `scripts/preprocess_custom_npz.py`: converts a bulb dataset into Cal-QL format
- `scripts/run_bulb_offline.sh`: runs offline bulb training
- `scripts/run_bulb_policy_export_full.sh`: runs the full 200-epoch offline training and exports the final actor
- `scripts/demo_bulb_policy_playback.py`: loads an exported policy and replays it in Isaac Gym
- `scripts/demo_bulb_visual.py`: random-action visual smoke test for the bulb task
- `scripts/demo_bulb_smoke.py`: minimal smoke test for the bulb environment
- `viskit/`: logging and visualization utilities used by Cal-QL

## System Requirements

This project currently assumes:

- Windows host machine,
- WSL2 Ubuntu,
- NVIDIA GPU with CUDA support,
- Miniforge or Conda installed inside WSL,
- a TacSL-compatible Isaac Gym package available locally.

The current playback and viewer scripts were tested in WSL. Long-running Isaac Gym viewer sessions may still be unstable because of native WSL graphics / Vulkan issues.

## Environment Layout

This project uses two separate WSL Conda environments:

### 1. `calql`

Used for:

- dataset preprocessing,
- offline training,
- actor export.

### 2. `tacsl`

Used for:

- Isaac Gym / ManiFeel environment execution,
- bulb policy playback,
- environment visualization.

## Deployment Guide

### 1. Clone this repository

On Windows or WSL, clone the repository normally:

```bash
git clone https://github.com/CadenJ01/Cal-QL-bulb.git
cd Cal-QL-bulb
```

### 2. Prepare WSL and Conda

Inside WSL, make sure Conda or Miniforge is installed and available. The scripts in this repository assume you can source:

```bash
source <your-conda-root>/etc/profile.d/conda.sh
```

If your Conda installation path is not standard, set:

```bash
export CONDA_ROOT=/path/to/your/conda
```

before running the helper scripts.

### 3. Create the `calql` environment

The training environment should use Python 3.8 and the dependencies listed in `requirements.txt`.

Example:

```bash
conda create -n calql python=3.8 -y
conda activate calql
pip install -r requirements.txt
```

### 4. Create the `tacsl` environment

Create a separate environment for Isaac Gym / ManiFeel:

```bash
conda create -n tacsl python=3.8 -y
conda activate tacsl
```

Then install:

- the TacSL Isaac Gym package,
- the ManiFeel Isaac Gym environments,
- TacSL sensor dependencies.

This repository includes helper scripts for checking the WSL Isaac Gym setup:

- `scripts/check_isaacgym_wsl.sh`
- `scripts/check_tacsl_imports.py`

### 5. Configure Weights & Biases

Create `JaxCQL/wandb_config.py` using `JaxCQL/wandb_config_example.py` as the template.

That file is intentionally excluded from version control.

## Dataset Workflow

The project expects a low-dimensional bulb dataset in `.npz` format.

### Input dataset

Example input:

- `bulb_40eps_lowdim.npz`

### Preprocess the dataset

Run:

```bash
python scripts/preprocess_custom_npz.py \
  --input /path/to/bulb_40eps_lowdim.npz \
  --output /path/to/output/bulb_40eps_lowdim_calql.npz
```

The preprocessing stage:

- reconstructs episode boundaries from `episode_starts` and `episode_lengths`,
- rewrites terminal flags so only the last step of each episode is marked done,
- normalizes actions into `[-1, 1]`,
- computes Monte Carlo returns for Cal-QL.

## Offline Training Workflow

### Run offline bulb training

```bash
bash scripts/run_bulb_offline.sh
```

This script supports environment-variable overrides, including:

- `CONDA_ROOT`
- `CONDA_ENV_NAME`
- `DATASET_PATH`
- `PROJECT_NAME`
- `SEED`
- `BATCH_SIZE`
- `TRAIN_STEPS_PER_EPOCH`
- `PRETRAIN_EPOCHS`
- `POLICY_ARCH`
- `QF_ARCH`
- `REWARD_SCALE`
- `REWARD_BIAS`

### Export the final offline policy

To run the full 200-epoch offline export:

```bash
bash scripts/run_bulb_policy_export_full.sh
```

This script also supports path overrides through environment variables and writes an exported actor file such as:

```text
<repo>/exports/bulb_policy_offline_200ep.npz
```

## Policy Format

The exported file stores a trained actor network, not a trajectory.

Its structure is:

- observation dimension: `7`
- action dimension: `7`
- MLP architecture: `256-256`
- output layer size: `14`

The 14 output values correspond to:

- 7 action means,
- 7 action log standard deviations.

For deterministic playback, the action is computed as:

```text
action = tanh(mean)
```

The exported file also stores metadata such as:

- dataset type,
- dataset path,
- random seed.

## Isaac Gym Replay Workflow

### Random-action visual smoke test

```bash
bash scripts/run_demo_bulb_visual_wsl.sh
```

### Offline policy playback

```bash
bash scripts/run_demo_bulb_policy_playback_wsl.sh
```

The playback path is configurable with:

- `POLICY_PATH`
- `ACTION_GAIN`
- `CONDA_ROOT`
- `CONDA_ENV_NAME`
- `MANIFEEL_ROOT`

By default, the playback script loads the exported offline policy and runs it inside the TacSL bulb environment.

## Local Development Notes

Contributors should update the following areas when modifying the project:

- `JaxCQL/replay_buffer.py` if the dataset format changes,
- `scripts/preprocess_custom_npz.py` if the raw bulb dataset schema changes,
- `scripts/demo_bulb_policy_playback.py` if the ManiFeel observation mapping changes,
- `scripts/run_*` WSL scripts if environment paths or deployment assumptions change.

## Known Limitations

- The observation bridge from ManiFeel to the 7D offline policy input is still a temporary compatibility layer.
- Long-running viewer sessions in WSL can occasionally crash because of native graphics / Vulkan instability.
- The final offline model is functional for playback, but it still needs stronger task-aligned evaluation and better observation matching before online fine-tuning.

## Recommended Next Steps

1. define the exact semantic mapping between the 7D offline observation and the ManiFeel bulb state,
2. export intermediate checkpoints, not only the final epoch,
3. compare multiple offline checkpoints in Isaac Gym,
4. connect the environment for online fine-tuning after the observation interface is finalized.

## Credits

- Original Cal-QL implementation: [Cal-QL](https://github.com/nakamotoo/Cal-QL)
- Underlying offline RL method: Cal-QL / CQL
- Simulation environment: Isaac Gym + ManiFeel TacSL bulb task
