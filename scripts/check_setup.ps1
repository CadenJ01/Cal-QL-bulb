$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at $python"
}

$checkScript = @'
import importlib.util
import platform
import sys

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())

for module in [
    "jax",
    "flax",
    "optax",
    "gym",
    "wandb",
    "plotly",
    "flask",
    "d4rl",
    "dm_control",
    "mujoco_py",
]:
    print(f"{module}: {'ok' if importlib.util.find_spec(module) else 'missing'}")
'@

$checkScript | & $python -
