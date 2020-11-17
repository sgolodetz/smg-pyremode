import numpy as np

from queue import Queue
from typing import Tuple

from smg.pyremode import DepthAssembler, DepthEstimator


class TemporalKeyframeDepthEstimator(DepthEstimator):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float], *,
                 min_depth: float = 0.1, max_depth: float = 4.0):
        """
        TODO

        :param image_size:  TODO
        :param intrinsics:  TODO
        :param min_depth:   TODO
        :param max_depth:   TODO
        """
        self.__depth_assemblers: Queue[DepthAssembler] = Queue()
        self.__frame_idx: int = 0

    # PUBLIC METHODS

    def put(self, input_image: np.ndarray, input_pose: np.ndarray) -> None:
        """
        TODO

        :param input_image:     TODO
        :param input_pose:      TODO
        """
        pass
