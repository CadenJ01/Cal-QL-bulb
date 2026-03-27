import argparse
from pathlib import Path

import numpy as np


def compute_mc_returns(rewards, dones, gamma):
    mc_returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running * (1.0 - float(dones[idx]))
        mc_returns[idx] = running
    return mc_returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--action-scale-mode",
        choices=("max_abs", "none"),
        default="max_abs",
        help="Normalize actions into [-1, 1] with per-dimension max abs scaling.",
    )
    args = parser.parse_args()

    src = np.load(args.input, allow_pickle=True)
    observations = src["observations"].astype(np.float32)
    next_observations = src["next_observations"].astype(np.float32)
    actions = src["actions"].astype(np.float32)
    rewards = src["rewards"].reshape(-1).astype(np.float32)

    if "episode_starts" not in src or "episode_lengths" not in src:
        raise ValueError("Expected episode_starts and episode_lengths in input npz.")

    episode_starts = src["episode_starts"].astype(np.int64)
    episode_lengths = src["episode_lengths"].astype(np.int32)

    if episode_starts.shape[0] != episode_lengths.shape[0]:
        raise ValueError("episode_starts and episode_lengths must have the same length.")

    dones = np.zeros(rewards.shape[0], dtype=np.float32)
    for start, length in zip(episode_starts, episode_lengths):
        if length <= 0:
            continue
        dones[start + length - 1] = 1.0

    if args.action_scale_mode == "max_abs":
        action_scale = np.max(np.abs(actions), axis=0).astype(np.float32)
        action_scale[action_scale < 1e-6] = 1.0
        normalized_actions = np.clip(actions / action_scale, -1.0, 1.0).astype(np.float32)
    else:
        action_scale = np.ones(actions.shape[1], dtype=np.float32)
        normalized_actions = actions

    mc_returns = np.zeros_like(rewards, dtype=np.float32)
    for start, length in zip(episode_starts, episode_lengths):
        end = start + length
        mc_returns[start:end] = compute_mc_returns(rewards[start:end], dones[start:end], args.gamma)

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        observations=observations,
        actions=normalized_actions,
        next_observations=next_observations,
        rewards=rewards.astype(np.float32),
        dones=dones.astype(np.float32),
        mc_returns=mc_returns.astype(np.float32),
        raw_actions=actions.astype(np.float32),
        action_scale=action_scale.astype(np.float32),
        episode_starts=episode_starts,
        episode_lengths=episode_lengths,
    )

    print(f"wrote: {dst}")
    print(f"observations: {observations.shape}")
    print(f"actions: {normalized_actions.shape}")
    print(f"rewards: {rewards.shape}")
    print(f"dones_sum: {int(dones.sum())}")
    print(f"action_scale: {action_scale}")


if __name__ == "__main__":
    main()
