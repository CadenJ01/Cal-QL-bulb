$ErrorActionPreference = "Stop"

$checks = @(
  @{ Env = "calql"; Module = "jax" },
  @{ Env = "calql"; Module = "flax" },
  @{ Env = "calql"; Module = "optax" },
  @{ Env = "calql"; Module = "distrax" },
  @{ Env = "calql"; Module = "ml_collections" },
  @{ Env = "calql"; Module = "wandb" },
  @{ Env = "calql"; Module = "torch" },
  @{ Env = "calql"; Module = "hydra" },
  @{ Env = "calql"; Module = "isaacgym" },
  @{ Env = "calql"; Module = "isaacgymenvs" },
  @{ Env = "tacsl"; Module = "jax" },
  @{ Env = "tacsl"; Module = "flax" },
  @{ Env = "tacsl"; Module = "optax" },
  @{ Env = "tacsl"; Module = "distrax" },
  @{ Env = "tacsl"; Module = "ml_collections" },
  @{ Env = "tacsl"; Module = "wandb" },
  @{ Env = "tacsl"; Module = "torch" },
  @{ Env = "tacsl"; Module = "hydra" },
  @{ Env = "tacsl"; Module = "isaacgym" },
  @{ Env = "tacsl"; Module = "isaacgymenvs" }
)

foreach ($check in $checks) {
  $envName = $check.Env
  $moduleName = $check.Module
  Write-Host "Checking $envName / $moduleName"
  wsl -d Ubuntu bash /mnt/c/Users/26972/Desktop/Spring\ 2026/cs441/calql/scripts/check_import_wsl.sh $envName $moduleName
}
