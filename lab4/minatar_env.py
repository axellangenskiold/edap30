import gymnasium as gym
from minatar.gym import register_envs
import numpy as np
import numpy.typing as npt
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.utils.passive_env_checker")


class MinatarEnv(object):
    """
    A wrapper class for the MinAtar environment, based on the gymnasium interface.
    """
    def __init__(self, game_name: str, horizon: int=250, gamma: float=0.99, use_cuda=False):
        """
        Constructor.

        Args:
            game_name: the name of the game, using gymnasium convention;
            horizon: the maximum number of steps allowed per episode;
            gamma: the discount factor;
            use_cuda: whether to use cuda or not.

        """
        all_envs = gym.envs.registry.keys()
        minatar_envs = [env_id for env_id in all_envs if "MinAtar" in env_id]

        if len(minatar_envs) == 0:
            register_envs()

        self._gym_env = gym.make('MinAtar/'+ game_name, render_mode='rgb_array')

        self._use_cuda = use_cuda

        self.horizon = horizon
        self.gamma = gamma

    @property
    def state_shape(self):
        """
        The observation shape of the images.
        """
        shape = self._gym_env.observation_space.shape

        new_shape = (shape[2],) + shape[:2]
        return new_shape

    @property
    def num_actions(self):
        """
        The number of possible actions.
        """
        return self._gym_env.action_space.n

    def reset(self) -> torch.Tensor:
        """
        Resets the environment to its initial state.

        Returns:
            The initial state of the environment.
        """
        state, _ = self._gym_env.reset()

        return self._convert_state(state)

    def step(self, action: int):
        """
        Performs a step in the environment.

        Args:
            action: the action to perform.

        Returns:
            The next state, the reward, and the absorbing flag.
        """
        ret = self._gym_env.step(action)

        state = ret[0]
        reward = ret[1]
        absorbing = ret[2]

        return self._convert_state(state), reward, absorbing

    def render(self) -> npt.NDArray:
        """
        Calls the render functionality of the environment and returns an RGB image to be displayed.

        Notice that, while the image is the same as the one used by the policy, the format is different.

        Returns:
            an RGB image for display.
        """
        return self._gym_env.render()

    def _convert_state(self, state: npt.NDArray) -> torch.Tensor:
        """
        Function converting the state into a torch tensor.

        Args:
            state: the state to be converted

        Returns:
            a torch array of the state, with the input channels as first dimension.

        """
        state_reordered = np.moveaxis(state, -1, 0).astype("float32")

        state_torch = torch.from_numpy(state_reordered)

        if self._use_cuda:
            state_torch = state_torch.cuda()

        return state_torch



