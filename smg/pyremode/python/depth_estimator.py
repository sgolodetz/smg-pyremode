import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class DepthEstimator(ABC):
    """TODO"""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        TODO

        :return:    TODO
        """
        pass

    @abstractmethod
    def put(self, input_image: np.ndarray, input_pose: np.ndarray) -> None:
        """
        TODO

        :param input_image:     TODO
        :param input_pose:      TODO
        """
        pass

    @abstractmethod
    def terminate(self) -> None:
        """TODO"""
        pass
