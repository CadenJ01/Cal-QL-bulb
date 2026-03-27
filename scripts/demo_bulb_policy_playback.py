import os
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

import isaacgym  # noqa: F401
import isaacgymenvs  # noqa: F401
import torch

from isaacgymenvs.tasks.tacsl.tacsl_task_bulb import TacSLTaskBulb
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed


ACTION_GAIN = float(os.environ.get("ACTION_GAIN", "5.0"))


def resolve_policy_path():
    if os.environ.get("POLICY_PATH"):
        return os.environ["POLICY_PATH"]

    calql_root = os.environ.get("CALQL_ROOT")
    if calql_root:
        return str(Path(calql_root) / "exports" / "bulb_policy_offline_200ep.npz")

    return str(Path.home() / "calql-wsl" / "exports" / "bulb_policy_offline_200ep.npz")


class NumpyTanhGaussianMeanPolicy:
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.observation_dim = int(data["observation_dim"])
        self.action_dim = int(data["action_dim"])
        self.n_layers = int(data["n_layers"])
        self.layers = []
        for idx in range(self.n_layers):
            self.layers.append(
                (
                    data[f"layer_{idx}_kernel"].astype(np.float32),
                    data[f"layer_{idx}_bias"].astype(np.float32),
                )
            )

    def __call__(self, obs):
        x = obs.astype(np.float32)
        for idx, (kernel, bias) in enumerate(self.layers):
            x = x @ kernel + bias
            if idx < self.n_layers - 1:
                x = np.maximum(x, 0.0)
        mean = x[..., : self.action_dim]
        return np.tanh(mean)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_policy_obs(obs, expected_dim):
    payload = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs

    if isinstance(payload, dict):
        if expected_dim == 7 and "ee_pos" in payload and "ee_quat" in payload:
            vec = np.concatenate(
                [
                    to_numpy(payload["ee_pos"]).reshape(1, -1),
                    to_numpy(payload["ee_quat"]).reshape(1, -1),
                ],
                axis=-1,
            )
            return vec.astype(np.float32)

        flat_parts = []
        for key in sorted(payload.keys()):
            flat_parts.append(to_numpy(payload[key]).reshape(1, -1))
        vec = np.concatenate(flat_parts, axis=-1).astype(np.float32)
    else:
        vec = to_numpy(payload).reshape(1, -1).astype(np.float32)

    if vec.shape[-1] >= expected_dim:
        return vec[:, :expected_dim]

    padded = np.zeros((vec.shape[0], expected_dim), dtype=np.float32)
    padded[:, : vec.shape[-1]] = vec
    return padded


class BulbViewerEnv:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._prepare_cfg()
        self._start_task()

    def _prepare_cfg(self):
        self.cfg.task.name = "TacSLTaskBulb"
        self.cfg.num_envs = 1
        self.cfg.headless = False
        self.cfg.force_render = True
        self.cfg.capture_video = False
        self.cfg.task.env.numEnvs = 1
        self.cfg.task.env.use_camera = False
        self.cfg.task.env.use_camera_obs = False
        self.cfg.task.env.use_tactile_field_obs = False
        self.cfg.task.env.use_isaac_gym_tactile = False
        self.cfg.task.env.use_gelsight = False
        self.cfg.task.env.enableCameraSensors = False

    def _start_task(self):
        cfg_dict = omegaconf_to_dict(self.cfg["task"])
        self.env = TacSLTaskBulb(
            cfg=cfg_dict,
            rl_device=self.cfg.rl_device,
            sim_device=self.cfg.sim_device,
            graphics_device_id=self.cfg.graphics_device_id,
            headless=self.cfg.headless,
            virtual_screen_capture=self.cfg.capture_video,
            force_render=self.cfg.force_render,
        )

    def seed(self, seed=0):
        set_seed(seed, torch_deterministic=self.cfg.torch_deterministic, rank=0)

    def reset(self):
        self.env.reset_idx(torch.arange(self.env.num_environments, device=self.env.device))
        self.env.compute_observations()
        return self.env.reset()

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
        return self.env.step(action)


@hydra.main(version_base="1.1", config_path="../isaacgymenvs/cfg", config_name="config")
def main(cfg: DictConfig):
    env = BulbViewerEnv(cfg)
    policy_path = resolve_policy_path()
    policy = NumpyTanhGaussianMeanPolicy(policy_path)
    env.seed(0)
    obs = env.reset()
    print(f"Loaded policy: {policy_path}")
    print(f"Policy obs_dim={policy.observation_dim}, act_dim={policy.action_dim}")

    step_idx = 0
    while True:
        obs_vec = extract_policy_obs(obs, policy.observation_dim)
        action = policy(obs_vec)
        action = np.clip(action * ACTION_GAIN, -1.0, 1.0)
        obs, reward, done, _ = env.step(action)
        if step_idx % 30 == 0:
            print(
                f"step={step_idx} "
                f"reward={reward.detach().cpu().numpy()} "
                f"done={done.detach().cpu().numpy()} "
                f"action={np.array2string(action, precision=4, suppress_small=False)}"
            )
        step_idx += 1
        time.sleep(0.03)


if __name__ == "__main__":
    main()
