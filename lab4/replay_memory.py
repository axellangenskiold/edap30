import numpy as np
import torch

class ReplayMemory(object):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al.

    """
    def __init__(self, initial_size: int, max_size: int, use_cuda: bool):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory can contain.

        """
        self._initial_size = initial_size
        self._max_size = max_size
        self._use_cuda = use_cuda

        self.reset()

    def add(self, dataset: list):
        """
        Add elements to the replay memory.

        Args:
            dataset: list of transitions to add to the replay memory.

        """
        i = 0
        while i < len(dataset):
            self._states[self._idx] = dataset[i][0]
            self._actions[self._idx] = dataset[i][1]
            self._rewards[self._idx] = dataset[i][2]

            self._next_states[self._idx] = dataset[i][3]
            self._absorbing[self._idx] = dataset[i][4]
            self._last[self._idx] = dataset[i][5]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

            i += 1

    def get(self, n_samples: int):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(self._states[i])
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(self._next_states[i])
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        s_t = torch.stack(s)
        a_t = torch.tensor(a, dtype=torch.long)
        r_t = torch.tensor(r)
        ss_t = torch.stack(ss)
        ab_t = torch.tensor(ab, dtype=torch.uint8)
        last_t = torch.tensor(last, dtype=torch.uint8)

        if self._use_cuda:
            s_t = s_t.cuda()
            a_t = a_t.cuda()
            r_t = r_t.cuda()
            ss_t = ss_t.cuda()
            ab_t = ab_t.cuda()
            last_t = last_t.cuda()

        return s_t, a_t, r_t, ss_t, ab_t, last_t

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = [None for _ in range(self._max_size)]
        self._actions = [None for _ in range(self._max_size)]
        self._rewards = [None for _ in range(self._max_size)]
        self._next_states = [None for _ in range(self._max_size)]
        self._absorbing = [None for _ in range(self._max_size)]
        self._last = [None for _ in range(self._max_size)]

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size >= self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size
