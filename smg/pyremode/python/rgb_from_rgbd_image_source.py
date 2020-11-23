import numpy as np

from typing import Tuple

from smg.pyremode import RGBImageSource, RGBDImageSource


class RGBFromRGBDImageSource(RGBImageSource):
    """An RGB image source that wraps an RGB-D image source."""

    # CONSTRUCTOR

    def __init__(self, image_source: RGBDImageSource):
        """
        Construct an RGB image source that wraps an RGB-D image source.

        :param image_source:    The RGB-D image source.
        """
        self.__image_source: RGBDImageSource = image_source

    # PUBLIC METHODS

    def get_image(self) -> np.ndarray:
        """
        Get an image from the image source.

        :return:    The image.
        """
        colour_image, _ = self.__image_source.get_images()
        return colour_image

    def get_image_dims(self) -> Tuple[int, int]:
        """
        Get the dimensions of the images.

        :return:    The dimensions of the images, as a (width, height) tuple.
        """
        return self.__image_source.get_colour_dims()

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Get the camera intrinsics.

        :return:    The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        """
        return self.__image_source.get_colour_intrinsics()

    def terminate(self) -> None:
        """Tell the image source to terminate."""
        self.__image_source.terminate()
