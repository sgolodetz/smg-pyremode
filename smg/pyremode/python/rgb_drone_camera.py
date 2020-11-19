import cv2
import numpy as np

from typing import Tuple

from smg.pyremode import RGBImageSource
from smg.rotory.drones.drone import Drone


class RGBDroneCamera(RGBImageSource):
    """An RGB image source that wraps a drone."""

    # CONSTRUCTOR

    def __init__(self, drone: Drone):
        """
        Construct an RGB image source that wraps a drone.

        :param drone:   The drone to wrap.
        """
        self.__camera_matrix: np.ndarray = np.array([
            [946.60441222, 0., 460.29254907],
            [0., 941.38386885, 357.08431882],
            [0., 0., 1.]
        ])
        self.__dist_coeffs: np.ndarray = np.array([[0.04968041, -0.59998154, -0.00377696, -0.00863985, 2.14472665]])
        self.__drone: Drone = drone

    # PUBLIC METHODS

    def get_image(self) -> np.ndarray:
        """
        Get an image from the image source.

        :return:    The image.
        """
        distorted_image: np.ndarray = self.__drone.get_image()
        return cv2.undistort(distorted_image, self.__camera_matrix, self.__dist_coeffs)

    def get_image_dims(self) -> Tuple[int, int]:
        """
        Get the dimensions of the images.

        :return:    The dimensions of the images, as a (width, height) tuple.
        """
        # FIXME: These parameters are for the Tello.
        return 960, 720
        # return self.__drone

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get the camera intrinsics.

        :return:    The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        # FIXME: These parameters are for the Tello.
        # return 935.33218901, 931.73412273, 480.0, 360.0
        return self.__camera_matrix[0][0], self.__camera_matrix[1][1], self.__camera_matrix[0][2], self.__camera_matrix[1][2]
        # return self.__drone.get_intrinsics()
