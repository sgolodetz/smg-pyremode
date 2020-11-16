import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from smg.geometry import GeometryUtil
from smg.openni.openni_camera import OpenNICamera
from smg.pyopencv import CVMat1b
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import CONVERGED, DepthIntegrator, Depthmap, SE3f


def add_axis(vis: o3d.visualization.Visualizer, pose: np.ndarray, *,
             colour: Optional[Tuple[float, float, float]] = None, size: float = 1.0) -> None:
    """
    Add to the specified Open3D visualisation a set of axes for the specified pose.

    :param vis:     The Open3D visualisation.
    :param pose:    The pose (specified in camera space).
    :param colour:  An optional colour with which to paint the axes.
    :param size:    The size to give the axes (defaults to 1).
    """
    # noinspection PyArgumentList
    axes: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if colour is not None:
        axes.paint_uniform_color(colour)
    axes.transform(pose)
    # noinspection PyTypeChecker
    vis.add_geometry(axes)


def print_se3(se3: SE3f) -> None:
    print()
    for row in range(3):
        print([se3.data(row, col) for col in range(4)])
    print()


def main():
    with OpenNICamera(mirror_images=True) as camera:
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            intrinsics: Tuple[float, float, float, float] = camera.get_colour_intrinsics()
            integrator: DepthIntegrator = DepthIntegrator(camera.get_colour_dims(), intrinsics)
            is_keyframe: bool = True
            reference_depth_image: Optional[np.ndarray] = None

            _, ax = plt.subplots(2, 2)

            while True:
                colour_image, depth_image = camera.get_images()
                cv2.imshow("Image", colour_image)
                cv2.waitKey(1)

                if not tracker.is_ready():
                    continue
                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image, depth_image)
                if pose is None:
                    continue

                integrator.put(colour_image, pose, is_keyframe=is_keyframe)
                if is_keyframe:
                    reference_depth_image = depth_image
                    is_keyframe = False

                reference_colour_image, reference_pose, estimated_depth_image, convergence_map = integrator.get()

                ax[0, 0].clear()
                ax[0, 1].clear()
                ax[1, 0].clear()
                ax[1, 1].clear()
                ax[0, 0].imshow(reference_colour_image[:, :, [2, 1, 0]])
                ax[0, 1].imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
                ax[1, 0].imshow(colour_image[:, :, [2, 1, 0]])
                ax[1, 1].imshow(reference_depth_image, vmin=0.0, vmax=4.0)

                plt.draw()
                if plt.waitforbuttonpress(0.001):
                    break

            cv2.destroyAllWindows()

            depth_mask: np.ndarray = np.where(estimated_depth_image != 0, 255, 0).astype(np.uint8)

            # convergence_map: np.ndarray = np.array(depthmap.get_convergence_map(), copy=False)
            # depth_mask: np.ndarray = np.where(convergence_map == CONVERGED, 255, 0).astype(np.uint8)

            pcd_points, pcd_colours = GeometryUtil.make_point_cloud(
                reference_colour_image, estimated_depth_image, depth_mask, intrinsics
            )

            # Convert the point cloud to Open3D format.
            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

            # Denoise the point cloud (slow).
            # pcd = pcd.uniform_down_sample(every_k_points=5)
            # pcd, _ = pcd.remove_radius_outlier(64, 0.05)

            # Set up the visualisation.
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            render_option: o3d.visualization.RenderOption = vis.get_render_option()
            render_option.line_width = 10

            # noinspection PyTypeChecker
            vis.add_geometry(pcd)
            add_axis(vis, np.eye(4), size=0.1)

            # Set the initial pose for the visualiser.
            params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            m = np.eye(4)
            params.extrinsic = m
            vis.get_view_control().convert_from_pinhole_camera_parameters(params)

            # Run the visualiser.
            vis.run()


if __name__ == "__main__":
    main()
