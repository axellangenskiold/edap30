import numpy as np

class TD(object):
    """
    A generic TD algorithm class.

    """
    def __init__(self, n_states: int, n_actions: int, gamma: float, policy):
        """
        Constructor.

        Args:
            n_states: number of states of the MDP;
            n_actions: number of actions of the MDP;
            gamma: the discount factor;
            policy: policy to be used.

        """
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        self.policy = policy

        self.policy.set_q(self.Q)

        self.next_action = None

    def reset(self):
        """
        Resets the internal state of the TD algorithm.

        """
        self.next_action = None

    def draw_action(self, state: int):
        """
        Draws an action from the policy

        Args:
            state: the current state.

        Returns:
            The sampled action

        """
        if self.next_action is None:
            action = self.policy.draw_action(state)
        else:
            action = self.next_action
            self.next_action = None

        return action

    def update(self, state: int, action: int, reward: float, next_state: int, absorbing: bool):
        """
        Updates the Q-function

        Args:
            state: the starting state;
            action: the current action;
            reward: the reward;
            next_state: the state reached.

        """
        raise NotImplementedError





