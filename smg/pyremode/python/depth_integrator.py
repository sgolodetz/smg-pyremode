import cv2
import numpy as np
import threading

import smg.pyopencv as pyopencv
import smg.pyremode as pyremode

from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from smg.geometry import GeometryUtil


class DepthIntegrator:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float], *,
                 min_depth: float, max_depth: float):
        """
        TODO

        :param image_size:  TODO
        :param intrinsics:  TODO
        :param min_depth:   TODO
        :param max_depth:   TODO
        """
        self.__convergence_map: Optional[np.ndarray] = None
        self.__estimated_depth_image: Optional[np.ndarray] = None
        self.__image_size: Tuple[int, int] = image_size
        self.__input_image: Optional[np.ndarray] = None
        self.__input_is_ready: bool = False
        self.__input_is_reference: bool = False
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

        # Start the threads.
        self.__integration_thread = threading.Thread(target=self.__integrate_images)
        self.__integration_thread.start()

    # PUBLIC METHODS

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO

        :return:    TODO
        """
        with self.__get_lock:
            # TODO
            while not self.__output_is_ready:
                self.__output_ready.wait(0.1)

            # TODO
            self.__output_is_ready = False

            # TODO
            return self.__reference_image, self.__reference_pose, self.__estimated_depth_image, self.__convergence_map

    def put(self, input_image: np.ndarray, input_pose: np.ndarray, input_is_reference: bool) -> None:
        """
        TODO

        :param input_image:         TODO
        :param input_pose:          TODO
        :param input_is_reference:  TODO
        """
        acquired: bool = self.__put_lock.acquire(blocking=False)
        if acquired:
            self.__input_image = input_image
            self.__input_is_reference = input_is_reference
            self.__input_pose = input_pose

            self.__input_is_ready = True
            self.__input_ready.notify()
            self.__put_lock.release()

    # PRIVATE METHODS

    def __integrate_images(self) -> None:
        """TODO"""
        while not self.__should_terminate:
            depthmap: Optional[pyremode.Depthmap] = None

            with self.__put_lock:
                # TODO
                while not self.__input_is_ready:
                    self.__input_ready.wait(0.1)

                # TODO
                if self.__should_terminate:
                    return

                # TODO
                integration_image: np.ndarray = self.__input_image
                integration_is_reference: bool = self.__input_is_reference
                integration_pose: np.ndarray = self.__input_pose
                self.__input_is_ready = False

            # TODO
            grey_image: np.ndarray = cv2.cvtColor(integration_image, cv2.COLOR_BGR2GRAY)
            cv_grey_image: pyopencv.CVMat1b = pyopencv.CVMat1b.zeros(*grey_image.shape[:2])
            np.copyto(np.array(cv_grey_image, copy=False), grey_image)

            # TODO
            r: Rotation = Rotation.from_matrix(integration_pose[0:3, 0:3])
            t: np.ndarray = integration_pose[0:3, 3]
            qx, qy, qz, qw = r.as_quat()
            se3: pyremode.SE3f = pyremode.SE3f(qw, qx, qy, qz, *t)

            # TODO
            if integration_is_reference:
                self.__reference_image = integration_image
                self.__reference_pose = integration_pose
                width, height = self.__image_size
                fx, fy, cx, cy = self.__intrinsics
                depthmap = pyremode.Depthmap(width, height, fx, cx, fy, cy)
                depthmap.set_reference_image(cv_grey_image, se3, self.__min_depth, self.__max_depth)
            else:
                depthmap.update(cv_grey_image, se3)

            # TODO
            with self.__get_lock:
                self.__convergence_map = np.array(depthmap.get_convergence_map(), copy=False)
                self.__estimated_depth_image = np.array(depthmap.get_denoised_depthmap(), copy=False)
                GeometryUtil.make_depths_orthogonal(self.__estimated_depth_image, self.__intrinsics)

                self.__output_is_ready = True
                self.__output_ready.notify()
