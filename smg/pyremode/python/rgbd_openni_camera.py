import numpy as np

from typing import Tuple

from smg.openni.openni_camera import OpenNICamera
from smg.pyremode.python.rgbd_image_source import RGBDImageSource


class RGBDOpenNICamera(RGBDImageSource):
    """An RGB-D image source that wraps an OpenNI camera."""

    # CONSTRUCTOR

    def __init__(self, camera: OpenNICamera):
        """
        Construct an RGB-D image source that wraps an OpenNI camera.

        :param camera:  The OpenNI camera.
        """
        self.__camera: OpenNICamera = camera

    # PUBLIC METHODS

    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get colour and depth images from the image source.

        :return:    A tuple consisting of a colour image and a depth image from the image source (in that order).
        """
        return self.__camera.get_images()
