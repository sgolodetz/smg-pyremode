import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple


class RGBImageSource(ABC):
    """A source of RGB images."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """
        Get an image from the image source.

        :return:    The image.
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """
        Get the size of the images.

        :return:    The size of the images, as a (width, height) tuple.
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get the camera intrinsics.

        :return:    The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Tell the image source to terminate."""
        pass
