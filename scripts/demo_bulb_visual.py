import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import isaacgym  # noqa: F401
import isaacgymenvs  # noqa: F401
import torch

from isaacgymenvs.tasks.tacsl.tacsl_task_bulb import TacSLTaskBulb
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed


isaacgym_task_map = {
    "TacSLTaskBulb": TacSLTaskBulb,
}


class ManifeelEnvWrapper:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._prepare_cfg()
        self._start_task(self.cfg)

    def _prepare_cfg(self):
        self.cfg.task.name = "TacSLTaskBulb"
        self.cfg.num_envs = 1
        self.cfg.headless = False
        self.cfg.force_render = True
        self.cfg.capture_video = False
        self.cfg.task.env.numEnvs = 1

        # Keep the simulator viewer, but disable image/tactile pipelines that
        # currently fail under WSL GPU external-memory import.
        self.cfg.task.env.use_camera = False
        self.cfg.task.env.use_camera_obs = False
        self.cfg.task.env.use_tactile_field_obs = False
        self.cfg.task.env.use_isaac_gym_tactile = False
        self.cfg.task.env.use_gelsight = False
        self.cfg.task.env.enableCameraSensors = False

    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def num_envs(self):
        return self.envs.num_environments

    def _start_task(self, cfg):
        cfg_task = cfg["task"]
        cfg_dict = omegaconf_to_dict(cfg_task)
        self.envs = isaacgym_task_map[cfg_dict["name"]](
            cfg=cfg_dict,
            rl_device=cfg.rl_device,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=cfg.capture_video,
            force_render=cfg.force_render,
        )

    def seed(self, seed=0):
        set_seed(seed, torch_deterministic=self.cfg.torch_deterministic, rank=0)

    def reset(self):
        self.envs.reset_idx(torch.arange(self.num_envs, device=self.envs.device))
        self.envs.compute_observations()
        return self.envs.reset()

    def step(self, action):
        actions = torch.as_tensor(action, dtype=torch.float32, device=self.envs.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        return self.envs.step(actions)


@hydra.main(version_base="1.1", config_path="../isaacgymenvs/cfg", config_name="config")
def main(cfg: DictConfig):
    wrapper = ManifeelEnvWrapper(cfg)
    wrapper.seed(0)
    obs = wrapper.reset()
    print("Viewer launched with task config:")
    print(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "task": cfg.task.name,
                    "num_envs": cfg.task.env.numEnvs,
                    "headless": cfg.headless,
                    "use_camera": cfg.task.env.use_camera,
                    "use_tactile_field_obs": cfg.task.env.use_tactile_field_obs,
                    "use_isaac_gym_tactile": cfg.task.env.use_isaac_gym_tactile,
                }
            )
        )
    )
    print("Reset keys:", list(obs.keys()))

    action_dim = wrapper.action_space.shape[0]
    step_idx = 0
    while True:
        action = np.random.uniform(-1.0, 1.0, size=(wrapper.num_envs, action_dim)).astype(np.float32)
        _, reward, done, _ = wrapper.step(action)
        if step_idx % 30 == 0:
            print(f"step={step_idx} reward={reward.detach().cpu().numpy()} done={done.detach().cpu().numpy()}")
        step_idx += 1
        time.sleep(0.03)


if __name__ == "__main__":
    main()
