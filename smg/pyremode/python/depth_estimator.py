import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class DepthEstimator(ABC):
    """A depth estimator that assembles and yields RGB-D keyframes."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        TODO

        :return:    TODO
        """
        pass

    @abstractmethod
    def put(self, input_colour_image: np.ndarray, input_pose: np.ndarray) -> None:
        """
        Try to add a colour image with a known pose to the depth estimator.

        :param input_colour_image:  The input colour image.
        :param input_pose:          The input camera pose (denoting a transformation from camera space to world space).
        """
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Tell the depth estimator to terminate."""
        pass
