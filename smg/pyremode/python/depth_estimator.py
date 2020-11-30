import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class DepthEstimator(ABC):
    """A multi-view depth estimator that assembles and yields RGB-D keyframes."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def add_posed_image(self, input_colour_image: np.ndarray, input_pose: np.ndarray) -> None:
        """
        Try to add a colour image with a known pose to the depth estimator.

        :param input_colour_image:  The input colour image.
        :param input_pose:          The input camera pose (denoting a transformation from camera space to world space).
        """
        pass

    @abstractmethod
    def get_keyframe(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        Try to get the images, pose and convergence % / map for an assembled keyframe.

        .. note::
            This is intended to be blocking, and should only return None when the estimator has been told to terminate.

        :return:    The images, pose and ocnvergence % / map for an assembled keyframe, if possible, or None otherwise.
        """
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Tell the depth estimator to terminate."""
        pass
