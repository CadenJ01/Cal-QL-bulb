import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python dataset_stats.py <path>")
        return 1

    path = Path(sys.argv[1])
    data = np.load(path, allow_pickle=True)

    rewards = data["rewards"].reshape(-1)
    done_key = "terminals" if "terminals" in data else "dones"
    terminals = data[done_key].reshape(-1)
    actions = data["actions"]
    observations = data["observations"]

    print(f"path: {path}")
    print(f"num_transitions: {len(rewards)}")
    if "episode_lengths" in data:
        lengths = data["episode_lengths"]
        print(f"num_episodes: {len(lengths)}")
        print(
            "episode_lengths(min/max/mean):",
            int(lengths.min()),
            int(lengths.max()),
            float(lengths.mean()),
        )
    print("reward(min/max):", float(rewards.min()), float(rewards.max()))
    print("reward(unique_sample):", np.unique(rewards)[:20])
    print("terminals_sum:", int(terminals.sum()))
    if "success" in data:
        print("success_sum:", int(data["success"].reshape(-1).sum()))
    if "timeout" in data:
        print("timeout_sum:", int(data["timeout"].reshape(-1).sum()))
    print("actions(min/max):", float(actions.min()), float(actions.max()))
    print("observations(min/max):", float(observations.min()), float(observations.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
