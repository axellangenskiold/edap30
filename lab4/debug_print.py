import numpy.typing as npt

def print_map(grid_map: npt.NDArray):
    """
    A small function to print the map nicely.

    Args:
        grid_map: the map to print

    """
    for i in range(grid_map.shape[1]):
        for s in range(grid_map.shape[0]):
            for j in range(grid_map.shape[2]):
                cell = grid_map[s, i, j]
                print(cell, end='')
            print('  ', end='')
        print()


def print_agent_action(action: int):
    """
    A small function to print the agent action nicely.

    Args:
        action: the chosen action.
    """
    action_names = ['up', 'down', 'left', 'right']

    print(f'Action sampled: {action_names[action]}')


def print_agent_state(
        grid_map: npt.NDArray,
        cell_list: npt.NDArray,
        state: int,
):
    """
   A small function to print the agent in the map nicely

    Args:
        grid_map: the map to print
        cell_list: the list of cells to print
        state: the state to print
    """
    agent_coordinates = tuple(cell_list[state][1:])

    active_index = cell_list[state][0]

    for i, line in enumerate(grid_map[active_index]):
        for j, cell in enumerate(line):
            if (i, j) == agent_coordinates:
                print('A', end='')
            else:
                print(cell, end='')
        print()