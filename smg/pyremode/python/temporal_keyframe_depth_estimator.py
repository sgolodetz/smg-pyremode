import numpy as np
import threading

from typing import Optional, Tuple

from smg.pyremode import DepthAssembler, DepthEstimator


class TemporalKeyframeDepthEstimator(DepthEstimator):
    """
    A depth estimator that assembles RGB-D keyframes by passing input RGB images to each depth assembler in time order.
    """

    # CONSTRUCTOR

    def __init__(self, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float], *,
                 min_converged_percentage: float = 40.0, min_depth: float = 0.1, max_depth: float = 4.0,
                 min_images_per_keyframe: int = 30, max_images_per_keyframe: int = 30):
        """
        Construct a temporal keyframe depth estimator.

        .. note::
            When at least the minimum but fewer than the maximum number of images have been supplied, the decision
            about whether to yield a keyframe early is based on how well the REMODE depthmap has converged.

        :param image_size:                  The image size, as a (width, height) tuple.
        :param intrinsics:                  The camera intrinsics.
        :param min_converged_percentage:    The minimum percentage of the REMODE depthmap that needs to have
                                            converged for a keyframe to be yielded early.
        :param min_images_per_keyframe:     The minimum number of images to require before yielding a keyframe.
        :param max_images_per_keyframe:     The maximum number of images to require before yielding a keyframe.
        :param min_depth:                   An estimate of the lower bound of the depths present in the scene.
        :param max_depth:                   An estimate of the upper bound of the depths present in the scene.
        """
        self.__back_assembler: Optional[DepthAssembler] = None
        self.__front_assembler: Optional[DepthAssembler] = None
        self.__image_size: Tuple[int, int] = image_size
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__keyframe_image_count: int = 0
        self.__keyframe_is_ready: bool = False
        self.__max_depth: float = max_depth
        self.__max_images_per_keyframe: int = max_images_per_keyframe
        self.__min_converged_percentage: float = min_converged_percentage
        self.__min_depth: float = min_depth
        self.__min_images_per_keyframe: int = min_images_per_keyframe
        self.__should_terminate: bool = False

        # Set up the lock and condition.
        self.__lock = threading.Lock()
        self.__keyframe_ready = threading.Condition(self.__lock)

    # PUBLIC METHODS

    def add_posed_image(self, input_colour_image: np.ndarray, input_pose: np.ndarray) -> None:
        """
        Try to add a colour image with a known pose to the depth estimator.

        :param input_colour_image:  The input colour image.
        :param input_pose:          The input camera pose (denoting a transformation from camera space to world space).
        """
        # Decide whether or not it's time for the next keyframe.
        time_for_next_keyframe: bool = \
            self.__back_assembler is None or self.__keyframe_image_count >= self.__max_images_per_keyframe

        if self.__keyframe_image_count >= self.__min_images_per_keyframe and not time_for_next_keyframe:
            result = self.__back_assembler.get(blocking=True)
            if result is not None:
                _, _, _, converged_percentage, _ = result
                if converged_percentage >= self.__min_converged_percentage:
                    time_for_next_keyframe = True

        # If it's time for the next keyframe:
        if time_for_next_keyframe:
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

                    # Make a new back assembler and reset the image count.
                    self.__back_assembler = DepthAssembler(
                        self.__image_size, self.__intrinsics, min_depth=self.__min_depth, max_depth=self.__max_depth
                    )
                    self.__keyframe_image_count = 0

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

        # Add the input colour image and pose to the back assembler, and update the image count
        # that is used to keep track of when to move on to the next keyframe.
        self.__back_assembler.put(input_colour_image, input_pose)
        self.__keyframe_image_count += 1

    def get_keyframe(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        Try to get the images, pose and convergence % / map for an assembled keyframe.

        .. note::
            This is intended to be blocking, and should only return None when the estimator has been told to terminate.

        :return:    The images, pose and ocnvergence % / map for an assembled keyframe, if possible, or None otherwise.
        """
        with self.__lock:
            # Wait for a keyframe to be ready.
            while not self.__keyframe_is_ready:
                self.__keyframe_ready.wait(0.1)
                if self.__should_terminate:
                    return None

            # Reset the flag so that we only get the keyframe once.
            self.__keyframe_is_ready = False

            # Return the keyframe. Note that we block here because whilst the depth assembler may have seen
            # enough images to make the keyframe at this point, it's possible that it still needs to finish
            # assembling the depth image for the keyframe internally, so we have to wait for it.
            return self.__front_assembler.get(blocking=True)

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
