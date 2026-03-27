import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python dataset_episode_check.py <path>")
        return 1

    path = Path(sys.argv[1])
    data = np.load(path, allow_pickle=True)

    episode_id = data["episode_id"].reshape(-1)
    terminals = data["terminals"].reshape(-1)
    rewards = data["rewards"].reshape(-1)

    unique_eps = np.unique(episode_id)
    print("num_episodes_from_ids:", len(unique_eps))
    for eid in unique_eps[:10]:
        idx = np.where(episode_id == eid)[0]
        term_idx = idx[terminals[idx] > 0]
        print(
            f"episode {int(eid)} len={len(idx)} terminal_count={len(term_idx)} "
            f"last_idx={int(idx[-1])} terminal_positions={[int(i - idx[0]) for i in term_idx[:10]]}"
        )
        print(
            "  reward(min/max):",
            float(rewards[idx].min()),
            float(rewards[idx].max()),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
