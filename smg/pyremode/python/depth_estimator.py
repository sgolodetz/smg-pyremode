import numpy as np

from abc import ABC, abstractmethod


class DepthEstimator(ABC):
    """TODO"""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def put(self, input_image: np.ndarray, input_pose: np.ndarray) -> None:
        """
        TODO

        :param input_image:     TODO
        :param input_pose:      TODO
        """
        pass
