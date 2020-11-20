import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from smg.open3d import VisualisationUtil
from smg.pyopencv import CVMat1b
from smg.pyorbslam2 import MonocularTracker
from smg.utility import GeometryUtil

from smg.pyremode import *
from smg.rotory.drone_factory import DroneFactory


def print_se3(se3: SE3f) -> None:
    print()
    for row in range(3):
        print([se3.data(row, col) for col in range(4)])
    print()


def main():
    with DroneFactory.make_drone("tello", local_ip="192.168.10.3") as drone:
        with MonocularTracker(
            settings_file=f"settings-tello.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            intrinsics: Optional[Tuple[float, float, float, float]] = drone.get_intrinsics()
            if intrinsics is None:
                raise RuntimeError("Cannot get drone camera intrinsics")

            width, height = drone.get_image_size()
            fx, fy, cx, cy = intrinsics
            depthmap: Depthmap = Depthmap(width, height, fx, cx, fy, cy)

            reference_colour_image: Optional[np.ndarray] = None

            _, ax = plt.subplots(2, 2)

            while True:
                distorted_image = drone.get_image()

                # camera_matrix: np.ndarray = np.array([
                #     [946.60441222, 0., 460.29254907],
                #     [0., 941.38386885, 357.08431882],
                #     [0., 0., 1.]
                # ])
                # dist_coeffs: np.ndarray = np.array([[0.04968041, -0.59998154, -0.00377696, -0.00863985, 2.14472665]])
                camera_matrix: np.ndarray = np.array([
                    [938.55289501, 0., 480.],
                    [0., 932.86950291, 360.],
                    [0., 0., 1.]
                ])
                dist_coeffs: np.ndarray = np.array([[0.03306774, -0.40497806, -0.00216106, -0.00294729, 1.31711308]])
                colour_image = cv2.undistort(distorted_image, camera_matrix, dist_coeffs)

                cv2.imshow("Image", colour_image)
                cv2.waitKey(1)

                if not tracker.is_ready():
                    continue
                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)
                if pose is None:
                    continue

                r: Rotation = Rotation.from_matrix(pose[0:3, 0:3])
                t: np.ndarray = pose[0:3, 3]
                qx, qy, qz, qw = r.as_quat()
                se3: SE3f = SE3f(qw, qx, qy, qz, *t)

                # print_se3(se3)

                grey_image: np.ndarray = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
                cv_grey_image: CVMat1b = CVMat1b.zeros(*grey_image.shape[:2])
                np.copyto(np.array(cv_grey_image, copy=False), grey_image)

                if reference_colour_image is None:
                    reference_colour_image = colour_image
                    depthmap.set_reference_image(cv_grey_image, se3, 0.1, 4.0)
                else:
                    depthmap.update(cv_grey_image, se3)

                estimated_depth_image: np.ndarray = np.array(depthmap.get_denoised_depthmap())
                convergence_map: np.ndarray = np.array(depthmap.get_convergence_map())

                print(depthmap.get_converged_percentage())

                ax[0, 0].clear()
                ax[0, 1].clear()
                ax[1, 0].clear()
                ax[1, 1].clear()
                ax[0, 0].imshow(reference_colour_image)
                ax[0, 1].imshow(estimated_depth_image)  # , vmin=0.0, vmax=4.0)
                ax[1, 0].imshow(colour_image)
                # ax[1, 1].imshow(reference_depth_image, vmin=0.0, vmax=4.0)

                plt.draw()
                if plt.waitforbuttonpress(0.001):
                    break

            # TODO
            cv2.destroyAllWindows()

            # TODO
            depth_mask: np.ndarray = np.where(convergence_map == CONVERGED, 255, 0).astype(np.uint8)
            pcd_points, pcd_colours = GeometryUtil.make_point_cloud(
                reference_colour_image, estimated_depth_image, depth_mask, (fx, fy, cx, cy)
            )

            # Convert the point cloud to Open3D format.
            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

            # Denoise the point cloud (slow).
            pcd = pcd.uniform_down_sample(every_k_points=5)
            pcd, _ = pcd.remove_statistical_outlier(20, 2.0)

            # Visualise the point cloud.
            VisualisationUtil.visualise_geometry(pcd)


if __name__ == "__main__":
    main()
