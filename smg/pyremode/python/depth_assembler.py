import cv2
import numpy as np
import threading

import smg.pyopencv as pyopencv
import smg.pyremode as pyremode

from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from smg.geometry import GeometryUtil


class DepthAssembler:
    """Used to assemble an RGB-D keyframe over time from multiple RGB images with known poses."""

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
        self.__converged_percentage: float = 0.0
        self.__convergence_map: Optional[np.ndarray] = None
        self.__image_size: Tuple[int, int] = image_size
        self.__input_image: Optional[np.ndarray] = None
        self.__input_is_ready: bool = False
        self.__input_is_keyframe: bool = True
        self.__input_pose: Optional[np.ndarray] = None
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__keyframe_colour_image: Optional[np.ndarray] = None
        self.__keyframe_depth_image: Optional[np.ndarray] = None
        self.__keyframe_pose: Optional[np.ndarray] = None
        self.__output_is_ready: bool = False
        self.__max_depth: float = max_depth
        self.__min_depth: float = min_depth
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

    def get(self, *, blocking: bool) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        Try to get the images, pose and convergence % / map for the assembled keyframe.

        .. note::
            For clarity, there are three different scenarios in which this function can return None:
              (i) The lock cannot be acquired, and we're unwilling to wait for it.
             (ii) The lock can be acquired, but the output is not yet ready, and we're unwilling to wait for it.
            (iii) The lock can be acquired, but the output is not yet ready, and the depth assembler is told to
                  terminate whilst we're waiting for it.

        :param blocking:    Whether or not to block until the images, pose and convergence map are available.
        :return:            A tuple consisting of the colour image, depth image, pose, converged percentage
                            and convergence map for the keyframe, if successful, or None otherwise.
        """
        # Try to acquire the lock.
        acquired: bool = self.__get_lock.acquire(blocking=blocking)
        if acquired:
            try:
                # If the output is not yet ready, wait for it if we're willing to do so. Otherwise, early out.
                if blocking:
                    while not self.__output_is_ready:
                        self.__output_ready.wait(0.1)
                        if self.__should_terminate:
                            return None
                else:
                    if not self.__output_is_ready:
                        return None

                # Mark the output as no longer ready so that we only get it once, and return it.
                self.__output_is_ready = False

                return self.__keyframe_colour_image, self.__keyframe_depth_image.copy(), \
                    self.__keyframe_pose, self.__converged_percentage, self.__convergence_map.copy()
            finally:
                # Regardless of what happens, make sure we release the lock again at the end of the function.
                self.__get_lock.release()
        else:
            # If the lock can't be acquired right now, and we don't want to wait for it, early out.
            return None

    def put(self, input_image: np.ndarray, input_pose: np.ndarray, *, blocking: bool) -> None:
        """
        Try to add a colour image with a known pose to the depth assembler.

        .. note::
            It normally makes sense to set blocking to False, since the rate at which images
            arrive from the camera will generally be higher than the rate at which they can
            be processed. If blocking is set to False, the depth assembler will naturally be
            fed images at the rate at which it can process them, rather than having to keep
            a queue of images to process in the future. However, there may be times when we
            want to ensure that every single image passed to the assembler is processed, in
            which case blocking can be set to True to arrange this.

        :param input_image:     The input colour image.
        :param input_pose:      The input camera pose (denoting a transformation from camera space to world space).
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
                self.__keyframe_colour_image = input_image
                self.__keyframe_pose = input_pose

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

            # Update the output convergence % and map, and keyframe depth image, and signal that they're ready.
            with self.__get_lock:
                self.__converged_percentage = depthmap.get_converged_percentage()
                self.__convergence_map = np.array(depthmap.get_convergence_map(), copy=False)
                self.__keyframe_depth_image = np.array(depthmap.get_denoised_depthmap(), copy=False)
                GeometryUtil.make_depths_orthogonal(self.__keyframe_depth_image, self.__intrinsics)

                self.__output_is_ready = True
                self.__output_ready.notify()
