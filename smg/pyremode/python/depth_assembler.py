import cv2
import numpy as np
import threading

import smg.pyopencv as pyopencv
import smg.pyremode as pyremode

from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from smg.geometry import GeometryUtil


class DepthAssembler:
    """Used to assemble a depth image over time from multiple colour images with known poses."""

    # CONSTRUCTOR

    def __init__(self, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float], *,
                 min_depth: float = 0.1, max_depth: float = 4.0):
        """
        Construct a depth assembler.

        :param image_size:  The image size, as a (width, height) tuple.
        :param intrinsics:  The camera intrinsics.
        :param min_depth:   An estimate of the lower bound of the depths present in the scene.
        :param max_depth:   An estimate of the upper bound of the depths present in the scene.
        """
        self.__convergence_map: Optional[np.ndarray] = None
        self.__estimated_depth_image: Optional[np.ndarray] = None
        self.__image_size: Tuple[int, int] = image_size
        self.__input_image: Optional[np.ndarray] = None
        self.__input_is_ready: bool = False
        self.__input_is_keyframe: bool = True
        self.__input_pose: Optional[np.ndarray] = None
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__output_is_ready: bool = False
        self.__max_depth: float = max_depth
        self.__min_depth: float = min_depth
        self.__reference_image: Optional[np.ndarray] = None
        self.__reference_pose: Optional[np.ndarray] = None
        self.__should_terminate = False

        # Set up the locks and conditions.
        self.__get_lock = threading.Lock()
        self.__put_lock = threading.Lock()

        self.__input_ready = threading.Condition(self.__put_lock)
        self.__output_ready = threading.Condition(self.__get_lock)

        # Start the assembly thread.
        self.__assembly_thread = threading.Thread(target=self.__assemble_depth_image)
        self.__assembly_thread.start()

    # PUBLIC METHODS

    def get(self, *, blocking: bool) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        TODO

        :param blocking:    TODO
        :return:            TODO
        """
        acquired: bool = self.__get_lock.acquire(blocking=blocking)
        if acquired:
            try:
                # TODO
                if blocking:
                    while not self.__output_is_ready:
                        self.__output_ready.wait(0.1)
                        if self.__should_terminate:
                            return None
                else:
                    if not self.__output_is_ready:
                        return None

                # TODO
                self.__output_is_ready = False

                # TODO
                return self.__reference_image, self.__reference_pose, \
                    self.__estimated_depth_image.copy(), self.__convergence_map.copy()
            finally:
                self.__get_lock.release()
        else:
            return None

    def put(self, input_image: np.ndarray, input_pose: np.ndarray, *, blocking: bool) -> None:
        """
        Try to add an image with a known pose to the depth assembler.

        .. note::
            It normally makes sense to set blocking to False, since the rate at which images
            arrive from the camera will generally be higher than the rate at which they can
            be processed. If blocking is set to False, the depth assembler will naturally be
            fed images at the rate at which it can process them, rather than having to keep
            a queue of images to process in the future. However, there may be times when we
            want to ensure that every single image passed to the assembler is processed, in
            which blocking can be set to True to arrange this.

        :param input_image:     The image.
        :param input_pose:      The camera pose when the image was captured.
        :param blocking:        Whether or not to block until the image is successfully added.
        """
        # FIXME: If blocking is True, this can block forever, even if the assembler has been told to terminate.
        acquired: bool = self.__put_lock.acquire(blocking=blocking)
        if acquired:
            self.__input_image = input_image
            self.__input_pose = input_pose

            self.__input_is_ready = True
            self.__input_ready.notify()
            self.__put_lock.release()

    def terminate(self) -> None:
        """Tell the depth assembler to terminate."""
        self.__should_terminate = True

    # PRIVATE METHODS

    def __assemble_depth_image(self) -> None:
        """Assemble the depth image from any input images and poses that are passed to the assembler."""
        depthmap: Optional[pyremode.Depthmap] = None

        # Until the assembler is told to terminate:
        while not self.__should_terminate:
            with self.__put_lock:
                # Wait for a new input image and pose.
                while not self.__input_is_ready:
                    self.__input_ready.wait(0.1)

                    # If the assembler is told to terminate whilst waiting for inputs, early out.
                    if self.__should_terminate:
                        return

                # Make local references to the inputs so that we can release the lock as soon as possible.
                input_image: np.ndarray = self.__input_image
                input_pose: np.ndarray = self.__input_pose
                self.__input_is_ready = False

            # Convert the input image to greyscale.
            grey_image: np.ndarray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            cv_grey_image: pyopencv.CVMat1b = pyopencv.CVMat1b.zeros(*grey_image.shape[:2])
            np.copyto(np.array(cv_grey_image, copy=False), grey_image)

            # Convert the input pose into a quaternion + vector form that can be passed to REMODE.
            r: Rotation = Rotation.from_matrix(input_pose[0:3, 0:3])
            t: np.ndarray = input_pose[0:3, 3]
            qx, qy, qz, qw = r.as_quat()
            se3: pyremode.SE3f = pyremode.SE3f(qw, qx, qy, qz, *t)

            # If this input image and pose are the keyframe:
            if self.__input_is_keyframe:
                # Store them for later.
                self.__reference_image = input_image
                self.__reference_pose = input_pose

                # Ensure that no future input is treated as the keyframe.
                self.__input_is_keyframe = False

                # Make the initial REMODE depthmap, setting the input image as its reference image.
                width, height = self.__image_size
                fx, fy, cx, cy = self.__intrinsics
                depthmap = pyremode.Depthmap(width, height, fx, cx, fy, cy)
                depthmap.set_reference_image(cv_grey_image, se3, self.__min_depth, self.__max_depth)
            else:
                # Otherwise, use the inputs to update the existing REMODE depthmap.
                depthmap.update(cv_grey_image, se3)

            # Update the output convergence map and estimated depth image, and signal that they're ready.
            with self.__get_lock:
                self.__convergence_map = np.array(depthmap.get_convergence_map(), copy=False)
                self.__estimated_depth_image = np.array(depthmap.get_denoised_depthmap(), copy=False)
                GeometryUtil.make_depths_orthogonal(self.__estimated_depth_image, self.__intrinsics)

                self.__output_is_ready = True
                self.__output_ready.notify()
