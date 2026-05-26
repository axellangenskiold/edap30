import numpy.typing as npt

import time
import matplotlib.pyplot as plt
from IPython import display


class Visualizer(object):
    """
    This class can be used to visualize an image in a Jupyter notebook or Python file using matplotlib.

    """
    def __init__(self, dt: float=0.1, is_notebook: bool=True, figsize: tuple=(3,3)):
        """
        Constructor

        Args:
            dt: the time step to visualize.
            is_notebook: boolean indicating whether you are using a Jupyter notebook or not;
            figsize: the size of the figure to be displayed.

        """
        self._dt = dt
        self._is_notebook = is_notebook
        self._figsize = figsize

        self._fig = None
        self._img = None

    def display(self, rgb: npt.NDArray):
        """
        Displays an RGB image.

        It also activates matplotlib interactive mode for the first image visualized.

        Args:
            rgb: the RGB image to display.

        """
        if self._img is None:
            self._fig = plt.figure(figsize=self._figsize)
            self._img = plt.imshow(rgb)
            plt.axis('off')
            plt.ion()
            plt.show()
        else:
            self._img.set_data(rgb)

        if self._is_notebook:
            display.clear_output(wait=True)
            display.display(self._fig)
            time.sleep(self._dt)
        else:
            plt.pause(self._dt)
            plt.draw()

    def close(self):
        """
        Closes the figure. Also makes sure matplotlib interactive mode is off.
        """
        if self._is_notebook:
            display.clear_output(wait=True)
            display.display(self._fig)
        else:
            plt.close(self._fig)

        plt.ioff()

        self._fig = None
        self._img = None

