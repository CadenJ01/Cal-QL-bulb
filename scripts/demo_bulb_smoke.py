import numpy as np
import hydra
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


def shape_of(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    return type(value).__name__


def summarize_obs(obs):
    if isinstance(obs, dict):
        print("obs keys:", sorted(obs.keys()))
        for key, value in obs.items():
            print(f"  {key}: {shape_of(value)}")
    else:
        print("obs type:", type(obs).__name__)
        print("obs shape:", shape_of(obs))


class ManifeelEnvWrapper:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._prepare_cfg()
        self._start_task(self.cfg)

    def _prepare_cfg(self):
        self.cfg.task.name = "TacSLTaskBulb"
        self.cfg.num_envs = 1
        self.cfg.headless = True
        self.cfg.force_render = False
        self.cfg.capture_video = False
        self.cfg.task.env.numEnvs = 1

        # Disable image/tactile sensors for a minimal smoke test in WSL.
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
        task_name = cfg_dict["name"]
        self.envs = isaacgym_task_map[task_name](
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
        if isinstance(action, torch.Tensor):
            actions = action.to(dtype=torch.float32, device=self.envs.device)
        else:
            actions = torch.as_tensor(action, dtype=torch.float32, device=self.envs.device)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        return self.envs.step(actions)


@hydra.main(version_base="1.1", config_path="../isaacgymenvs/cfg", config_name="config")
def main(cfg: DictConfig):
    print("Hydra task before patch:", cfg.task.name)
    wrapper = ManifeelEnvWrapper(cfg)
    print("Hydra task after patch:", cfg.task.name)
    print("Effective env config:")
    print(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "task": cfg.task.name,
                    "num_envs": cfg.task.env.numEnvs,
                    "use_camera": cfg.task.env.use_camera,
                    "use_camera_obs": cfg.task.env.use_camera_obs,
                    "use_tactile_field_obs": cfg.task.env.use_tactile_field_obs,
                    "use_isaac_gym_tactile": cfg.task.env.use_isaac_gym_tactile,
                }
            )
        )
    )

    wrapper.seed(0)
    obs = wrapper.reset()
    print("reset observation:")
    summarize_obs(obs)

    action_dim = wrapper.action_space.shape[0]
    for step_idx in range(5):
        random_actions = np.random.uniform(-1.0, 1.0, size=(wrapper.num_envs, action_dim)).astype(np.float32)
        obs, reward, done, info = wrapper.step(random_actions)
        print(f"step={step_idx}")
        summarize_obs(obs)
        print("reward shape:", shape_of(reward), "done shape:", shape_of(done), "info type:", type(info).__name__)


if __name__ == "__main__":
    main()
