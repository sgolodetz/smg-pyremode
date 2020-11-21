import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from argparse import ArgumentParser
from typing import Tuple


def densify_depth_image(input_depth_image: np.ndarray) -> Tuple[np.ndarray, mtri.Triangulation]:
    iy, ix = np.nonzero(input_depth_image)
    iz = input_depth_image[(iy, ix)]
    triangulation: mtri.Triangulation = mtri.Triangulation(ix, iy)

    # See: https://stackoverflow.com/questions/52457964/how-to-deal-with-the-undesired-triangles-that-form-between-the-edges-of-my-geo
    max_radius = 5
    triangles = triangulation.triangles
    xtri = ix[triangles] - np.roll(ix[triangles], 1, axis=1)
    ytri = iy[triangles] - np.roll(iy[triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri ** 2 + ytri ** 2), axis=1)
    triangulation.set_mask(maxi > max_radius)

    oy, ox = np.nonzero(np.ones_like(input_depth_image))
    interpolator: mtri.LinearTriInterpolator = mtri.LinearTriInterpolator(triangulation, iz)
    result: np.ma.core.MaskedArray = interpolator(ox, oy)
    output_depth_image: np.ndarray = np.where(result.mask, 0.0, result.data).astype(np.float32)
    output_depth_image = output_depth_image.reshape(input_depth_image.shape)
    return output_depth_image, triangulation


def load_depth_image(filename: str, *, depth_scale_factor: float = 1000.0) -> np.ndarray:
    """
    TODO

    :param filename:            TODO
    :param depth_scale_factor:  TODO
    :return:                    TODO
    """
    # TODO: Move this to ImageUtil.
    return cv2.imread(filename, cv2.IMREAD_UNCHANGED) / depth_scale_factor


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, required=True, help="the input file")
    parser.add_argument("--output_file", "-o", type=str, required=True, help="the output file")
    args: dict = vars(parser.parse_args())

    input_depth_image: np.ndarray = load_depth_image(args["input_file"])

    # iy, ix = np.nonzero(input_depth_image)
    # iz = input_depth_image[(iy, ix)]
    # triangulation: mtri.Triangulation = mtri.Triangulation(ix, iy)
    #
    # # See: https://stackoverflow.com/questions/52457964/how-to-deal-with-the-undesired-triangles-that-form-between-the-edges-of-my-geo
    # max_radius = 5
    # triangles = triangulation.triangles
    # xtri = ix[triangles] - np.roll(ix[triangles], 1, axis=1)
    # ytri = iy[triangles] - np.roll(iy[triangles], 1, axis=1)
    # maxi = np.max(np.sqrt(xtri ** 2 + ytri ** 2), axis=1)
    # triangulation.set_mask(maxi > max_radius)
    #
    # oy, ox = np.nonzero(np.ones_like(input_depth_image))
    # interpolator: mtri.LinearTriInterpolator = mtri.LinearTriInterpolator(triangulation, iz)
    # result: np.ma.core.MaskedArray = interpolator(ox, oy)
    # output_depth_image: np.ndarray = np.where(result.mask, 0.0, result.data).astype(np.float32)
    # output_depth_image = output_depth_image.reshape(input_depth_image.shape)
    output_depth_image, triangulation = densify_depth_image(input_depth_image)

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(input_depth_image, vmin=0.0, vmax=4.0)
    ax[1].imshow(output_depth_image, vmin=0.0, vmax=4.0)
    plt.gca().invert_yaxis()
    ax[2].triplot(triangulation)
    ax[2].set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main()
