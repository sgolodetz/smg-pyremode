import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from argparse import ArgumentParser
from typing import Optional, Tuple

from smg.open3d import VisualisationUtil
from smg.pyremode import DepthDenoiser
from smg.utility import GeometryUtil, ImageUtil


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument("--colour_file", type=str, required=False, help="the colour image file")
    parser.add_argument("--convergence_file", "-c", type=str, required=True, help="the convergence map file")
    parser.add_argument("--input_depth_file", "-i", type=str, required=True, help="the input depth image file")
    parser.add_argument("--output_depth_file", "-o", type=str, required=True, help="the output depth image file")
    args: dict = vars(parser.parse_args())

    convergence_map: np.ndarray = cv2.imread(args["convergence_file"], cv2.IMREAD_UNCHANGED)
    input_depth_image: np.ndarray = ImageUtil.load_depth_image(args["input_depth_file"])
    # intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)
    intrinsics: Tuple[float, float, float, float] = (946.60441222, 941.38386885, 460.29254907, 357.08431882)
    output_depth_image: np.ndarray = DepthDenoiser.denoise_depth(input_depth_image, convergence_map, intrinsics)

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(input_depth_image, vmin=0.0, vmax=4.0)
    ax[1].imshow(convergence_map)
    ax[2].imshow(output_depth_image, vmin=0.0, vmax=4.0)
    plt.draw()
    plt.waitforbuttonpress()

    # TODO
    colour_file: Optional[str] = args.get("colour_file")
    if colour_file is not None:
        colour_image: np.ndarray = cv2.imread(colour_file)
        depth_mask: np.ndarray = np.where(output_depth_image != 0, 255, 0).astype(np.uint8)

        # TODO
        pcd_points, pcd_colours = GeometryUtil.make_point_cloud(
            colour_image, output_depth_image, depth_mask, intrinsics
        )

        # Convert the point cloud to Open3D format.
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

        # Visualise the point cloud.
        VisualisationUtil.visualise_geometry(pcd)

    # TODO
    # ImageUtil.save_depth_image(args["output_depth_file"], output_depth_image)


if __name__ == "__main__":
    main()
