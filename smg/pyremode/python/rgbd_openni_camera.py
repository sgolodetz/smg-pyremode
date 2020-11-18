import numpy as np

from typing import Tuple

from smg.openni.openni_camera import OpenNICamera
from smg.pyremode.python.rgbd_image_source import RGBDImageSource


class RGBDOpenNICamera(RGBDImageSource):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, camera: OpenNICamera):
        """
        TODO

        :param camera:  TODO
        """
        self.__camera: OpenNICamera = camera

    # PUBLIC METHODS

    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO

        :return:    TODO
        """
        return self.__camera.get_images()
