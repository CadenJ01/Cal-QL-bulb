from pathlib import Path

import cloudpickle as pickle

from .policy_export import save_policy_numpy_export


def parse_checkpoint_epochs(value):
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def save_training_checkpoint(
    checkpoint_dir,
    epoch,
    sac,
    observation_dim,
    action_dim,
    policy_arch,
    dataset_type,
    dataset_path,
    seed,
    action_scale=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "epoch": epoch,
        "train_params": sac.train_params,
        "target_qf_params": sac._target_qf_params,
        "total_steps": sac.total_steps,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "policy_arch": policy_arch,
        "dataset_type": dataset_type,
        "dataset_path": dataset_path,
        "seed": seed,
    }

    snapshot_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
    with open(snapshot_path, "wb") as fout:
        pickle.dump(snapshot, fout)

    policy_export_path = checkpoint_dir / f"policy_epoch_{epoch}.npz"
    save_policy_numpy_export(
        policy_export_path,
        sac.train_params["policy"],
        observation_dim=observation_dim,
        action_dim=action_dim,
        arch=policy_arch,
        action_scale=action_scale,
        metadata={
            "dataset_type": dataset_type,
            "dataset_path": dataset_path,
            "seed": seed,
            "epoch": epoch,
        },
    )

    return str(snapshot_path), str(policy_export_path)
