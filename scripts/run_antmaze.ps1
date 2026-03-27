param(
    [string]$EnvName = "antmaze-medium-diverse-v2",
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at $python"
}

$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"
$env:WANDB_MODE = "offline"
$env:XLA_PYTHON_CLIENT_PREALLOCATE = "false"

& $python -m JaxCQL.conservative_sac_main `
    --env $EnvName `
    --logging.online=False `
    --seed $Seed `
    --logging.project=Cal-QL-example `
    --cql_min_q_weight=5.0 `
    --cql.cql_target_action_gap=0.8 `
    --cql.cql_lagrange=True `
    --policy_arch=256-256 `
    --qf_arch=256-256-256-256 `
    --offline_eval_every_n_epoch=50 `
    --online_eval_every_n_env_steps=2000 `
    --eval_n_trajs=20 `
    --n_train_step_per_epoch_offline=1000 `
    --n_pretrain_epochs=1000 `
    --max_online_env_steps=1e6 `
    --mixing_ratio=0.5 `
    --reward_scale=10.0 `
    --reward_bias=-5 `
    --enable_calql=True
