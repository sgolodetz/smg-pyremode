import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple


class RGBDImageSource(ABC):
    """A source of RGB-D images."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get colour and depth images from the image source.

        :return:    A tuple consisting of a colour image and a depth image from the image source (in that order).
        """
        pass
