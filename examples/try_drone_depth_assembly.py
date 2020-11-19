import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from typing import Optional, Tuple

from smg.open3d import VisualisationUtil
from smg.pyorbslam2 import MonocularTracker
from smg.rotory.drone_factory import DroneFactory

from smg.pyremode import CONVERGED, DepthAssembler
from smg.utility import GeometryUtil


def main():
    with DroneFactory.make_drone("tello", local_ip="192.168.10.3") as drone:
        with MonocularTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            colour_dims: Tuple[int, int] = (960, 720)
            intrinsics: Tuple[float, float, float, float] = (938.55289501, 932.86950291, 480.0, 360.0)
            depth_assembler: DepthAssembler = DepthAssembler(colour_dims, intrinsics)
            is_keyframe: bool = True

            reference_colour_image: Optional[np.ndarray] = None
            estimated_depth_image: Optional[np.ndarray] = None
            convergence_map: Optional[np.ndarray] = None

            _, ax = plt.subplots(2, 2)

            while True:
                # TODO
                distorted_image = drone.get_image()
                camera_matrix: np.ndarray = np.array([
                    [938.55289501, 0., 480.],
                    [0., 932.86950291, 360.],
                    [0., 0., 1.]
                ])
                dist_coeffs: np.ndarray = np.array([[0.03306774, -0.40497806, -0.00216106, -0.00294729, 1.31711308]])
                colour_image = cv2.undistort(distorted_image, camera_matrix, dist_coeffs)

                cv2.imshow("Image", colour_image)
                cv2.waitKey(1)

                # TODO
                if not tracker.is_ready():
                    continue
                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)
                if pose is None:
                    continue

                # TODO
                depth_assembler.put(colour_image, pose, blocking=False)

                # TODO
                if is_keyframe:
                    reference_colour_image = colour_image
                    is_keyframe = False

                # TODO
                result = depth_assembler.get(blocking=False)
                if result is not None:
                    _, estimated_depth_image, _, converged_percentage, convergence_map = result
                    print(f"Converged %: {converged_percentage}")

                # TODO
                ax[0, 0].clear()
                ax[0, 1].clear()
                ax[1, 0].clear()
                ax[1, 1].clear()
                ax[0, 0].imshow(reference_colour_image[:, :, [2, 1, 0]])
                if estimated_depth_image is not None:
                    ax[0, 1].imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
                ax[1, 0].imshow(colour_image[:, :, [2, 1, 0]])
                # ax[1, 1].imshow(reference_depth_image, vmin=0.0, vmax=4.0)

                plt.draw()
                if plt.waitforbuttonpress(0.001):
                    break

            # TODO
            cv2.destroyAllWindows()

            # TODO
            depth_mask: np.ndarray = np.where(convergence_map == CONVERGED, 255, 0).astype(np.uint8)
            pcd_points, pcd_colours = GeometryUtil.make_point_cloud(
                reference_colour_image, estimated_depth_image, depth_mask, intrinsics
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
