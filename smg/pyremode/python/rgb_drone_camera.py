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
        self.__drone: Drone = drone

    # PUBLIC METHODS

    def get_image(self) -> np.ndarray:
        """
        Get an image from the image source.

        :return:    The image.
        """
        return self.__drone.get_image()

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
        return 921.0, 921.0, 480.0, 360.0
        # return self.__drone.get_intrinsics()
