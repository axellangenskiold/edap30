import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import warnings

import numpy.typing as npt


def get_mean_and_confidence(data: npt.NDArray):
    """
    Compute the mean and 95% confidence interval

    Args:
        data: Array of experiment data of shape (n_runs, n_epochs).

    Returns:
        The mean of the dataset at each epoch along with the confidence interval.

    """
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
        _, interval = st.t.interval(0.95, n-1, scale=se)
        return mean, interval


def plot_mean_conf(data: npt.NDArray, ax: list, color=None, line='-', facecolor=None, alpha=0.4, label=None):
    """
    Method to plot mean and confidence interval for data on matplotlib axes.

    Args:
        data: Array of experiment data of shape (n_runs, n_epochs);
        ax: list of matplotlib axes where to create the curve;
        color (str, None): matplotlib color identifier for the mean curve;
        line (str, '-'): matplotlib line type to be used for the mean curve;
        facecolor (str, None): matplotlib color identifier for the confidence interval;
        alpha (float, 0.4): transparency of the confidence interval;
        label (str, one): legend label for the plotted curve.

    """

    mean, conf = get_mean_and_confidence(np.array(data))
    upper_bound = mean + conf
    lower_bound = mean - conf

    p = ax.plot(mean, color=color, linestyle=line, label=label)

    facecolor = color if facecolor is None else facecolor
    facecolor = p[-1].get_color() if facecolor is None else facecolor
    ax.fill_between(np.arange(np.size(mean)), upper_bound, lower_bound, facecolor=facecolor, alpha=alpha)


def visualize_value(value: npt.NDArray, grid_map: npt.NDArray, cell_list: list[tuple], label: str):
  """
  This function visualizes a specific value in a grid.

  Args:
      value: value matrix
      grid_map: np array containing the grid structure;
      cell_list: list of non-wall cells.
      label: label of plot

  """
  _, H, W = grid_map.shape
  # initialize value function
  V = -100*np.ones((2, H, W))

  # iterate over all valid states
  for i, c in enumerate(cell_list):
    V[tuple(c)] = value[i]

  # create plot of value matrix
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  ax[0].set_title("Inactive")
  ax[1].set_title("Active")
  ax[0].imshow(V[0], cmap='RdBu')
  ax[1].imshow(V[1], cmap='RdBu')

  # write the values in the cells
  for i in range(H):
      for j in range(W):
          for k in range(2):
              if grid_map[k,i,j] not in ['#', '/', 'X', 'G']:
                ax[k].text(j, i, round(V[k, i, j], 2), ha="center", va="center", color="w")

  # set title and show
  fig.suptitle(label)


def draw_map(grid_map: npt.NDArray, ax: list):
    """
    This function paints the map in greyscale using matplotlib axis

    Args:
        grid_map: array containing the ascii map;
        ax: list of two axis used to paint the map.

    """
    _, H, W = grid_map.shape

    map_colors = np.ones((2, H, W))
    map_colors[grid_map == '#'] = 0.0
    map_colors[grid_map == '/'] = 0.0
    map_colors[grid_map == 'X'] = 0.5
    map_colors[grid_map == 'G'] = 0.75
    map_colors[grid_map == '='] = 0.9

    ax[0].set_title("Inactive")
    ax[1].set_title("Active")
    ax[0].imshow(map_colors[0], cmap='gray')
    ax[1].imshow(map_colors[1], cmap='gray')


def visualize_policy(pi_input: npt.NDArray,  grid_map: npt.NDArray, cell_list: list[tuple], label: str):
    """
    This function visualizes the policy matrices.

    Args:
        pi_input: policy matrix;
        grid_map: np array containing the grid structure;
        cell_list: list of non-wall cells;
        label: label of plot.

    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    draw_map(grid_map, ax)

    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    # Compute the policy
    _, H, W = grid_map.shape
    pi = np.ones((2, H, W), dtype=int) * (-1)

    # iterate over all valid states
    for i, c in enumerate(cell_list):
        cell_type = grid_map[tuple(c)]

        # disregard absorbing states
        if cell_type not in ['X', 'G']:
            pi[tuple(c)] = pi_input[i]

    # draw the policy arrows
    for i in range(H):
        for j in range(W):
          if pi[0,i,j]!=-1:
            ax[0].arrow(j, i, 0.25*directions[pi[0, i, j]][1], 0.25*directions[pi[0, i, j]][0],
                        head_width=0.2, head_length=0.1, color='k')
          if pi[1,i,j]!=-1:
            ax[1].arrow(j, i, 0.25*directions[pi[1, i, j]][1], 0.25*directions[pi[1, i, j]][0],
                        head_width=0.2, head_length=0.1, color='k')

    # set title and show
    fig.suptitle(label)


def visualize_state_numbers(grid_map: npt.NDArray, cell_list: list[tuple]):
    """
    This function visualizes the state numbers in the policy map.

    Args:
        grid_map (np.ndarray): array containing the ascii map;
        cell_list (list): list of cell coordinates to be used.

    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    draw_map(grid_map, ax)

    _, H, W = grid_map.shape

    state_numbers = -np.ones((2, H, W), dtype=int)

    for i, c in enumerate(cell_list):
        state_numbers[tuple(c)] = i

    for i in range(H):
        for j in range(W):
            for k in range(2):
                if state_numbers[k, i, j] >= 0:
                    ax[k].text(j, i, str(state_numbers[k, i, j]),
                               ha='center', va='center', color='blue', fontsize=9, fontweight='bold')

