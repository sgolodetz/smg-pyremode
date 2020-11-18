import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple


class RGBDImageSource(ABC):
    """A source of RGB-D images."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO

        :return:    TODO
        """
        pass
