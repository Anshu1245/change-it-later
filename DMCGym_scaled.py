# inspired from https://github.com/denisyarats/dmc2gym

import logging
import os
from gymnasium.spaces import Box
from gymnasium.core import Env
import numpy as np
from dm_control import suite, manipulation
from dm_env import specs


def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, velocity_scale, dtype=np.float32):
    obs_pieces = []
    for k, v in obs.items():
        if "velocity" in k:
            obs[k] = v / velocity_scale    
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class DMCGym(Env):
    def __init__(
        self,
        domain,
        task,
        task_kwargs={},
        environment_kwargs={},
        rendering="egl",
    ):
        """TODO comment up"""

        # for details see https://github.com/deepmind/dm_control
        assert rendering in ["glfw", "egl", "osmesa"]
        os.environ["MUJOCO_GL"] = rendering

        self.env_name = domain+"_"+task
        if self.env_name == "cheetah_run":
            print("scaling velocity for", self.env_name)
            self.velocity_scale = 20
        elif self.env_name == "walker_walk":
            print("scaling velocity for", self.env_name)
            self.velocity_scale = 60
        elif self.env_name == "humanoid_walk":
            print("scaling velocity for", self.env_name)
            self.velocity_scale = 80
        else:
            print("Unknown env_name: {}".format(self.env_name), "Scaling factor set to 1")
            self.velocity_scale = 1
        
        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs)

        # self._env = manipulation.load(domain, seed=1000)

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])

        # set seed if provided with task_kwargs
        if "random" in task_kwargs:
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        """DMC always has a per-step reward range of (0, 1)"""
        return 0, 1

    def step(self, action):
        if action.dtype.kind == "f":
            # print("before converting action to np.float32 in dmcgym", action.dtype, action)
            action = action.astype(np.float32)
            # action = np.clip(action, -1, 1)
        # if action.any() > 1.:
        # print("after", action)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation, self.velocity_scale)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed:
            logging.warn(
                "Currently DMC has no way of seeding episodes. It only allows to seed experiments on environment initialization"
            )

        if options:
            logging.warn("Currently doing nothing with options={:}".format(options))
        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation, self.velocity_scale)
        info = {}
        return observation, info

    def render(self, height=480, width=480, camera_id=0):
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
