import cv2
import numpy as np
import threading

import smg.pyopencv as pyopencv
import smg.pyremode as pyremode

from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from smg.geometry import GeometryUtil


class DepthAssembler:
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
        TODO

        :param input_image:     TODO
        :param input_pose:      TODO
        :param blocking:        TODO
        """
        acquired: bool = self.__put_lock.acquire(blocking=blocking)
        if acquired:
            self.__input_image = input_image
            self.__input_pose = input_pose

            self.__input_is_ready = True
            self.__input_ready.notify()
            self.__put_lock.release()

    def terminate(self) -> None:
        """TODO"""
        self.__should_terminate = True

    # PRIVATE METHODS

    def __assemble_depth_image(self) -> None:
        """TODO"""
        depthmap: Optional[pyremode.Depthmap] = None

        while not self.__should_terminate:
            with self.__put_lock:
                # TODO
                while not self.__input_is_ready:
                    self.__input_ready.wait(0.1)
                    if self.__should_terminate:
                        return

                # TODO
                assembly_image: np.ndarray = self.__input_image
                assembly_pose: np.ndarray = self.__input_pose
                self.__input_is_ready = False

            # TODO
            grey_image: np.ndarray = cv2.cvtColor(assembly_image, cv2.COLOR_BGR2GRAY)
            cv_grey_image: pyopencv.CVMat1b = pyopencv.CVMat1b.zeros(*grey_image.shape[:2])
            np.copyto(np.array(cv_grey_image, copy=False), grey_image)

            # TODO
            r: Rotation = Rotation.from_matrix(assembly_pose[0:3, 0:3])
            t: np.ndarray = assembly_pose[0:3, 3]
            qx, qy, qz, qw = r.as_quat()
            se3: pyremode.SE3f = pyremode.SE3f(qw, qx, qy, qz, *t)

            # TODO
            if self.__input_is_keyframe:
                self.__reference_image = assembly_image
                self.__reference_pose = assembly_pose
                self.__input_is_keyframe = False
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
