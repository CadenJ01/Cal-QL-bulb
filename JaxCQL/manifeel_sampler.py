from types import SimpleNamespace

import numpy as np
import torch


def _torch_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_conjugate(q):
    xyz = -q[..., :3]
    w = q[..., 3:]
    return torch.cat([xyz, w], dim=-1)


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
    x2, y2, z2, w2 = torch.unbind(q2, dim=-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


class ManiFeelBulbWrapper:
    def __init__(
        self,
        cfg,
        task_class,
        num_envs=50,
        max_episode_steps=1000,
        headless=True,
        force_render=False,
        obs_mode="legacy_7d",
    ):
        self.cfg = cfg
        self.task_class = task_class
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.obs_mode = obs_mode
        self._prepare_cfg(headless=headless, force_render=force_render)
        self._start_task()
        self.spec = SimpleNamespace(name="manifeel-bulb", max_episode_steps=max_episode_steps)

    def _prepare_cfg(self, headless, force_render):
        self.cfg.task.name = "TacSLTaskBulb"
        self.cfg.num_envs = self.num_envs
        self.cfg.headless = headless
        self.cfg.force_render = force_render
        self.cfg.capture_video = False
        self.cfg.task.env.numEnvs = self.num_envs
        self.cfg.task.rl.max_episode_length = self.max_episode_steps
        self.cfg.task.env.use_camera = False
        self.cfg.task.env.use_camera_obs = False
        self.cfg.task.env.use_tactile_field_obs = False
        self.cfg.task.env.use_isaac_gym_tactile = False
        self.cfg.task.env.use_gelsight = False
        self.cfg.task.env.enableCameraSensors = False

    def _start_task(self):
        from isaacgymenvs.utils.reformat import omegaconf_to_dict

        cfg_dict = omegaconf_to_dict(self.cfg["task"])
        self.env = self.task_class(
            cfg=cfg_dict,
            rl_device=self.cfg.rl_device,
            sim_device=self.cfg.sim_device,
            graphics_device_id=self.cfg.graphics_device_id,
            headless=self.cfg.headless,
            virtual_screen_capture=self.cfg.capture_video,
            force_render=self.cfg.force_render,
        )

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        self.env.reset_idx(torch.arange(self.env.num_environments, device=self.env.device))
        self.env.compute_observations()
        raw_obs = self.env.reset()
        return self.extract_observation(raw_obs)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.extract_observation(raw_obs)
        success = _torch_to_numpy(self.env._check_success()).astype(np.float32)
        info = dict(info)
        info["success"] = success
        return obs, _torch_to_numpy(reward).astype(np.float32), _torch_to_numpy(done).astype(np.float32), info

    def extract_observation(self, raw_obs):
        payload = raw_obs["obs"] if isinstance(raw_obs, dict) and "obs" in raw_obs else raw_obs
        if not isinstance(payload, dict):
            return _torch_to_numpy(payload).astype(np.float32)

        if self.obs_mode == "legacy_7d":
            obs = torch.cat([payload["ee_pos"], payload["ee_quat"]], dim=-1)
            return _torch_to_numpy(obs).astype(np.float32)

        if self.obs_mode == "relative_7d":
            rel_pos = payload["ee_pos"] - payload["socket_pos"]
            rel_quat = _quat_multiply(_quat_conjugate(payload["socket_quat"]), payload["ee_quat"])
            obs = torch.cat([rel_pos, rel_quat], dim=-1)
            return _torch_to_numpy(obs).astype(np.float32)

        raise ValueError(f"Unsupported obs_mode: {self.obs_mode}")


class ManiFeelVecTrajSampler:
    def __init__(self, env, gamma=0.99, reward_scale=1.0, reward_bias=0.0):
        self.env = env
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias

    def _sample_batch(self, policy, deterministic=False, replay_buffer=None):
        obs = self.env.reset()
        trajectories = [
            dict(observations=[], actions=[], rewards=[], next_observations=[], dones=[], successes=[])
            for _ in range(self.env.num_envs)
        ]

        for _ in range(self.env.max_episode_steps):
            actions = policy(obs, deterministic=deterministic)
            next_obs, rewards, dones, info = self.env.step(actions)
            rewards = rewards * self.reward_scale + self.reward_bias
            successes = info.get("success", np.zeros(self.env.num_envs, dtype=np.float32))

            for env_idx in range(self.env.num_envs):
                trajectories[env_idx]["observations"].append(obs[env_idx].copy())
                trajectories[env_idx]["actions"].append(actions[env_idx].copy())
                trajectories[env_idx]["rewards"].append(float(rewards[env_idx]))
                trajectories[env_idx]["next_observations"].append(next_obs[env_idx].copy())
                trajectories[env_idx]["dones"].append(float(dones[env_idx]))
                trajectories[env_idx]["successes"].append(float(successes[env_idx]))

            obs = next_obs

        final_trajs = []
        for traj in trajectories:
            rewards = np.asarray(traj["rewards"], dtype=np.float32)
            dones = np.asarray(traj["dones"], dtype=np.float32)
            mc_returns = np.zeros_like(rewards, dtype=np.float32)
            running = 0.0
            for idx in range(len(rewards) - 1, -1, -1):
                running = rewards[idx] + self.gamma * running * (1.0 - dones[idx])
                mc_returns[idx] = running

            traj_dict = {
                "observations": np.asarray(traj["observations"], dtype=np.float32),
                "actions": np.asarray(traj["actions"], dtype=np.float32),
                "rewards": rewards,
                "next_observations": np.asarray(traj["next_observations"], dtype=np.float32),
                "dones": dones,
                "mc_returns": mc_returns,
                "successes": np.asarray(traj["successes"], dtype=np.float32),
            }
            final_trajs.append(traj_dict)

            if replay_buffer is not None:
                for i in range(len(rewards)):
                    replay_buffer.add_sample(
                        traj_dict["observations"][i],
                        traj_dict["actions"][i],
                        traj_dict["rewards"][i],
                        traj_dict["next_observations"][i],
                        traj_dict["dones"][i],
                        traj_dict["mc_returns"][i],
                    )

        return final_trajs

    def sample(self, policy, n_trajs=None, deterministic=False, replay_buffer=None):
        target = self.env.num_envs if n_trajs is None else int(n_trajs)
        all_trajs = []
        while len(all_trajs) < target:
            all_trajs.extend(self._sample_batch(policy, deterministic=deterministic, replay_buffer=replay_buffer))
        return all_trajs[:target]
