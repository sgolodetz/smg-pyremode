import cv2
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

from smg.pyremode import DepthDenoiser


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
    output_depth_image, triangulation = DepthDenoiser.densify_depth_image(input_depth_image)

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(input_depth_image, vmin=0.0, vmax=4.0)
    ax[1].imshow(output_depth_image, vmin=0.0, vmax=4.0)
    plt.gca().invert_yaxis()
    ax[2].triplot(triangulation)
    ax[2].set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main()
