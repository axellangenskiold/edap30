import numpy as np
import numpy.typing as npt

def parse_grid(grid: str) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Parse the grid file

    Args:
        grid: the path of the file containing the grid structure

    Returns:
        A list containing the grid structure

    """
    grid_map = list()
    cell_list = list()
    with open(grid, 'r') as f:
        m = f.read()

        assert 'S' in m and 'G' in m

        row = list()
        row_idx = 0
        col_idx = 0

        for c in m:
            if c in ['#', '.', 'S', 'G', 'X']:
                row.append(c)
                if c != '#':
                    cell_list.append([row_idx, col_idx])
                col_idx += 1
            elif c == '\n':
                grid_map.append(row)
                row = list()
                row_idx += 1
                col_idx = 0
            else:
                raise ValueError('Unknown marker.')

    return np.array(grid_map), np.array(cell_list, dtype=int)

def compute_probabilities(grid_map: npt.NDArray, cell_list: npt.NDArray) -> npt.NDArray:
    """
    Compute the transition probability matrix.

    Args:
        grid_map: list containing the grid structure;
        cell_list: list of valid cells.

    Returns:
        The transition probability matrix;

    """
    cell_array = np.array(cell_list)
    n_states = len(cell_list)
    p = np.zeros((n_states, 4, n_states))
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    for i in range(len(cell_array)):
        state = cell_array[i]

        if grid_map[tuple(state)] in ['S', '.']:
            for a in range(len(directions)):
                new_state = state + directions[a]
                j = np.where((cell_array == new_state).all(axis=1))[0]
                if j.size > 0:
                    assert j.size == 1
                    p[i, a, j] = 1.
                else:
                    p[i, a, i] = 1.
        else:
            for a in range(len(directions)):
                p[i, a, i] = 1.

    return p


def compute_reward(grid_map: npt.NDArray, cell_list: npt.NDArray) -> npt.NDArray:
    """
    Compute the reward matrix.

    Args:
        grid_map: list containing the grid structure;
        cell_list: list of non-wall cells.

    Returns:
        The reward matrix.

    """
    n_states = len(cell_list)
    r = np.zeros((n_states, 4, n_states))

    def give_reward(t, rew):
        for x in np.argwhere(grid_map == t):
            j = np.where((cell_list == x).all(axis=1))[0]
            r[:, :, j] = rew

    give_reward('G', 1)

    return r
