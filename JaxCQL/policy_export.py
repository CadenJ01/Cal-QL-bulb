from pathlib import Path
import re

import numpy as np


def _dense_sort_key(name):
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else name


def extract_policy_arrays(policy_params):
    params = policy_params["params"]
    base_network = params["base_network"]
    dense_keys = sorted(
        [key for key in base_network.keys() if key.startswith("Dense_")],
        key=_dense_sort_key,
    )

    arrays = {
        "n_layers": np.asarray(len(dense_keys), dtype=np.int32),
        "log_std_multiplier": np.asarray(params["log_std_multiplier_module"]["value"]),
        "log_std_offset": np.asarray(params["log_std_offset_module"]["value"]),
    }

    for idx, key in enumerate(dense_keys):
        arrays[f"layer_{idx}_kernel"] = np.asarray(base_network[key]["kernel"])
        arrays[f"layer_{idx}_bias"] = np.asarray(base_network[key]["bias"])

    return arrays


def save_policy_numpy_export(
    export_path,
    policy_params,
    observation_dim,
    action_dim,
    arch,
    action_scale=None,
    metadata=None,
):
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = extract_policy_arrays(policy_params)
    arrays["observation_dim"] = np.asarray(observation_dim, dtype=np.int32)
    arrays["action_dim"] = np.asarray(action_dim, dtype=np.int32)
    arrays["arch"] = np.asarray(str(arch))

    if action_scale is not None:
        arrays["action_scale"] = np.asarray(action_scale, dtype=np.float32)

    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, bytes)):
                arrays[key] = np.asarray(value)
            else:
                arrays[key] = np.asarray(value)

    np.savez(export_path, **arrays)
    return str(export_path)
