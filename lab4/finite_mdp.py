import numpy as np
from numpy.typing import NDArray


class FiniteMDP(object):
    """
    Finite Markov Decision Process.

    """
    def __init__(
            self,
            p: NDArray,
            rew: NDArray,
            mu: NDArray | None = None,
            gamma: float = .9,
            horizon: float = np.inf
    ):
        """
        Constructor.

        Args:
            p: transition probability matrix;
            rew: reward matrix;
            mu: initial state probability distribution;
            gamma: discount factor;
            horizon: the horizon.

        """
        assert p.shape == rew.shape
        assert mu is None or p.shape[0] == mu.size

        # MDP parameters
        self.p = p
        self.r = rew
        self.mu = mu

        # MDP properties
        self.n_states = p.shape[0]
        self.n_actions = p.shape[1]
        self.horizon = horizon
        self.gamma = gamma

        # State of the MDP
        self._state = None


    def reset(self, state: int | None = None) -> int:
        """
        Resets the state of the MDP.

        Returns:
            The new state of the MDP.
        """
        if state is None:
            if self.mu is not None:
                self._state = np.random.choice(self.mu.size, p=self.mu)
            else:
                self._state = np.random.choice(self.p.shape[0])
        else:
            self._state = state

        return self._state

    def step(self, action: int):
        """
        Performs a step in the MDP.

        Args:
            action: action taken by the MDP.

        Returns:
            The new state, the reward, and the absorbing flag.
        """
        p = self.p[self._state, action, :]
        assert len(p.shape)==1, f'wrong shape of p matrix for transition. Current shape: {p.shape}'

        next_state = np.random.choice(p.size, p=p)
        absorbing = np.all(self.p[next_state, :, next_state] == 1.0)
        reward = self.r[self._state, action, next_state]

        self._state = next_state

        return self._state, reward, absorbing
