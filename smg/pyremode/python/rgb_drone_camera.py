import numpy as np

from typing import cast, Optional, Tuple

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
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = drone.get_intrinsics()
        if self.__intrinsics is None:
            raise RuntimeError("Cannot get drone camera intrinsics")

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
        return self.__drone.get_image_size()

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get the camera intrinsics.

        :return:    The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        return cast(Tuple[float, float, float, float], self.__intrinsics)
