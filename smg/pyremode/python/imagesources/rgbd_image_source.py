import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple


class RGBDImageSource(ABC):
    """A source of RGB-D images."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get_colour_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get the colour camera intrinsics.

        :return:    The colour camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        pass

    @abstractmethod
    def get_colour_size(self) -> Tuple[int, int]:
        """
        Get the size of the colour images.

        :return:    The size of the colour images, as a (width, height) tuple.
        """
        pass

    @abstractmethod
    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get colour and depth images from the image source.

        :return:    A tuple consisting of a colour image and a depth image from the image source (in that order).
        """
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Tell the image source to terminate."""
        pass
