import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from typing import Optional, Tuple

from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import CONVERGED, DepthAssembler
from smg.utility import GeometryUtil


def main():
    with OpenNICamera(mirror_images=True) as camera:
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            intrinsics: Tuple[float, float, float, float] = camera.get_colour_intrinsics()
            depth_assembler: DepthAssembler = DepthAssembler(camera.get_colour_dims(), intrinsics)
            is_keyframe: bool = True

            reference_colour_image: Optional[np.ndarray] = None
            reference_depth_image: Optional[np.ndarray] = None
            estimated_depth_image: Optional[np.ndarray] = None
            convergence_map: Optional[np.ndarray] = None

            _, ax = plt.subplots(2, 2)

            while True:
                # TODO
                colour_image, depth_image = camera.get_images()
                cv2.imshow("Image", colour_image)
                cv2.waitKey(1)

                # TODO
                if not tracker.is_ready():
                    continue
                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image, depth_image)
                if pose is None:
                    continue

                # TODO
                depth_assembler.put(colour_image, pose, blocking=False)

                # TODO
                if is_keyframe:
                    reference_colour_image = colour_image
                    reference_depth_image = depth_image
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
                ax[1, 1].imshow(reference_depth_image, vmin=0.0, vmax=4.0)

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

            rng_state = np.random.get_state()
            shuffle_indices = np.array(range(len(pcd_points)))
            np.random.shuffle(shuffle_indices)
            np.random.set_state(rng_state)
            np.random.shuffle(pcd_points)
            np.random.set_state(rng_state)
            np.random.shuffle(pcd_colours)

            # Convert the point cloud to Open3D format.
            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

            # Denoise the point cloud (slow).
            factor: int = 10
            pcd = pcd.uniform_down_sample(every_k_points=factor)
            pcd, ind = pcd.remove_statistical_outlier(20, 2.0)

            ind = list(map(lambda x: shuffle_indices[x * factor], ind))

            ind_mask: np.ndarray = np.zeros(depth_image.shape, dtype=np.uint8)
            ind_mask = ind_mask.flatten()
            ind_mask[ind] = 255
            ind_mask = ind_mask.reshape(depth_image.shape, order='C')

            estimated_depth_image = np.where(ind_mask != 0, estimated_depth_image, 0.0).astype(np.float32)

            depth_scale_factor: float = 1000.0
            scaled_depth_image: np.ndarray = (estimated_depth_image * depth_scale_factor).astype(np.uint16)
            kernel: np.ndarray = np.ones((5, 5), np.uint8)
            scaled_depth_image = cv2.dilate(scaled_depth_image, kernel)

            estimated_depth_image = scaled_depth_image.astype(np.float32) / depth_scale_factor

            plt.imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
            plt.waitforbuttonpress()

            depth_mask = np.where(scaled_depth_image != 0, 255, 0).astype(np.uint8)
            pcd_points, pcd_colours = GeometryUtil.make_point_cloud(
                reference_colour_image, estimated_depth_image, depth_mask, intrinsics
            )

            # Convert the point cloud to Open3D format.
            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

            # Visualise the point cloud.
            VisualisationUtil.visualise_geometry(pcd)


if __name__ == "__main__":
    main()
