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

        .. note::
            This is intended to be non-blocking, for performance reasons, and so any particular image passed in
            may in practice be silently dropped rather than being used as part of the depth estimation process.
            The intended use case is to pass in a stream of images from a camera, and let the depth estimator
            use the ones it can. It is of course possible to write a depth estimator that uses every image that's
            passed to it, if desired, but that would necessarily involve blocking if the estimator can't keep up,
            which would slow down the calling thread. An alternative approach of buffering the images that are
            passed in until the estimator's ready for them is also possible, but with finite memory, there's
            eventually going to come a point where an estimator that can't keep up has to drop some images.
            Given this, a non-blocking put function with no buffering seems best.

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
