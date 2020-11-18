import numpy as np
import threading

from typing import Optional, Tuple

from smg.pyremode import DepthAssembler, DepthEstimator


class TemporalKeyframeDepthEstimator(DepthEstimator):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float], *,
                 images_per_keyframe: int = 100, min_depth: float = 0.1, max_depth: float = 4.0):
        """
        TODO

        :param image_size:  TODO
        :param intrinsics:  TODO
        :param min_depth:   TODO
        :param max_depth:   TODO
        """
        self.__back_assembler: Optional[DepthAssembler] = None
        self.__cyclic_frame_idx: int = 0
        self.__front_assembler: Optional[DepthAssembler] = None
        self.__image_size: Tuple[int, int] = image_size
        self.__images_per_keyframe: int = images_per_keyframe
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__keyframe_is_ready: bool = False
        self.__max_depth: float = max_depth
        self.__min_depth: float = min_depth
        self.__should_terminate: bool = False

        # Set up the locks and conditions.
        self.__lock = threading.Lock()
        self.__keyframe_ready = threading.Condition(self.__lock)

    # SPECIAL METHODS

    def __enter__(self):
        """TODO"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """TODO"""
        self.terminate()

    # PUBLIC METHODS

    def get(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        TODO

        :return:    TODO
        """
        with self.__lock:
            while not self.__keyframe_is_ready:
                self.__keyframe_ready.wait(0.1)
                if self.__should_terminate:
                    return None

            self.__keyframe_is_ready = False
            return self.__front_assembler.get(blocking=True)

    def put(self, input_image: np.ndarray, input_pose: np.ndarray) -> None:
        """
        TODO

        :param input_image:     TODO
        :param input_pose:      TODO
        """
        if self.__cyclic_frame_idx == 0:
            acquired: bool = self.__lock.acquire(blocking=False)
            if acquired:
                try:
                    if self.__should_terminate:
                        return

                    if self.__front_assembler is not None:
                        self.__front_assembler.terminate()
                    self.__front_assembler = self.__back_assembler
                    self.__back_assembler = DepthAssembler(
                        self.__image_size, self.__intrinsics, min_depth=self.__min_depth, max_depth=self.__max_depth
                    )
                    if self.__front_assembler is not None:
                        self.__keyframe_is_ready = True
                        self.__keyframe_ready.notify()
                finally:
                    self.__lock.release()
            else:
                return

        self.__back_assembler.put(input_image, input_pose, blocking=False)
        self.__cyclic_frame_idx = (self.__cyclic_frame_idx + 1) % self.__images_per_keyframe

    def terminate(self) -> None:
        """TODO"""
        with self.__lock:
            if not self.__should_terminate:
                self.__should_terminate = True
                if self.__front_assembler is not None:
                    self.__front_assembler.terminate()
                if self.__back_assembler is not None:
                    self.__back_assembler.terminate()
