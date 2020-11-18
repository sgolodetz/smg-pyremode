import numpy as np
import threading

from typing import Optional, Tuple

from smg.pyremode import DepthAssembler, DepthEstimator


class TemporalKeyframeDepthEstimator(DepthEstimator):
    """
    A depth estimator that assembles RGB-D keyframes by passing a fixed number of input RGB images in order
    to each depth assembler before moving on to the next one.
    """

    # CONSTRUCTOR

    def __init__(self, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float], *,
                 images_per_keyframe: int = 100, min_depth: float = 0.1, max_depth: float = 4.0):
        """
        Construct a temporal keyframe depth estimator.

        :param image_size:  The image size, as a (width, height) tuple.
        :param intrinsics:  The camera intrinsics.
        :param min_depth:   An estimate of the lower bound of the depths present in the scene.
        :param max_depth:   An estimate of the upper bound of the depths present in the scene.
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

        # Set up the lock and condition.
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

    def put(self, input_colour_image: np.ndarray, input_pose: np.ndarray) -> None:
        """
        Try to add a colour image with a known pose to the depth estimator.

        :param input_colour_image:  The input colour image.
        :param input_pose:          The input camera pose (denoting a transformation from camera space to world space).
        """
        # If it's time for the next keyframe:
        if self.__cyclic_frame_idx == 0:
            # Try to acquire the lock.
            acquired: bool = self.__lock.acquire(blocking=False)
            if acquired:
                # If we managed to acquire the lock:
                try:
                    # Check whether the depth estimator has been told to terminate, and early out if so.
                    if self.__should_terminate:
                        return

                    # Terminate the front assembler (if it exists).
                    if self.__front_assembler is not None:
                        self.__front_assembler.terminate()

                    # Make the front assembler point to the current back assembler (if it exists).
                    self.__front_assembler = self.__back_assembler

                    # Make a new back assembler.
                    self.__back_assembler = DepthAssembler(
                        self.__image_size, self.__intrinsics, min_depth=self.__min_depth, max_depth=self.__max_depth
                    )

                    # If the front assembler now exists (i.e. the previous back assembler was not None),
                    # signal that a keyframe is ready.
                    if self.__front_assembler is not None:
                        self.__keyframe_is_ready = True
                        self.__keyframe_ready.notify()
                finally:
                    # Whatever happens, make sure to release the lock again before continuing.
                    self.__lock.release()
            else:
                # If the lock cannot be acquired, early out without updating the frame index.
                # This will ensure that we will try to create a new keyframe again next time.
                return

        # Add the input colour image and pose to the back assembler, and update the cyclic frame index
        # that is used to keep track of when to move on to the next keyframe.
        self.__back_assembler.put(input_colour_image, input_pose, blocking=False)
        self.__cyclic_frame_idx = (self.__cyclic_frame_idx + 1) % self.__images_per_keyframe

    def terminate(self) -> None:
        """Tell the depth estimator to terminate."""
        with self.__lock:
            # If the depth estimator hasn't already been told to terminate:
            if not self.__should_terminate:
                # Set its own termination flag.
                self.__should_terminate = True

                # Tell any depth assemblers it owns to terminate.
                if self.__front_assembler is not None:
                    self.__front_assembler.terminate()
                if self.__back_assembler is not None:
                    self.__back_assembler.terminate()
