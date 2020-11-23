import cv2
import numpy as np
import open3d as o3d
import os

from typing import Tuple

from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.pyremode import CONVERGED, DepthProcessor
from smg.utility import GeometryUtil, ImageUtil, PoseUtil


def main():
    # sequence_dir: str = "C:/spaint/build/bin/apps/spaintgui/sequences/remode-kinect"
    # intrinsics: Tuple[float, float, float, float] = (532.5694641250893, 531.5410880910171, 320.0, 240.0)
    # o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     640, 480, 532.5694641250893, 531.5410880910171, 320.0, 240.0
    # )
    sequence_dir: str = "C:/spaint/build/bin/apps/spaintgui/sequences/remode-mono"
    intrinsics: Tuple[float, float, float, float] = (946.60441222, 941.38386885, 460.29254907, 357.08431882)
    o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
        960, 720, 946.60441222, 941.38386885, 460.29254907, 357.08431882
    )

    tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.005,
        sdf_trunc=0.2,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    frame_idx: int = 0
    while True:
        colour_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.color.png")
        convergence_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.convergence.png")
        depth_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.depth.png")
        pose_filename: str = os.path.join(sequence_dir, f"frame-{frame_idx:06d}.pose.txt")
        if not os.path.exists(colour_filename):
            break

        print(f"Processing frame {frame_idx}...")

        colour_image: np.ndarray = cv2.imread(colour_filename)
        convergence_map: np.ndarray = cv2.imread(convergence_filename, cv2.IMREAD_UNCHANGED)
        depth_image: np.ndarray = ImageUtil.load_depth_image(depth_filename)
        pose: np.ndarray = np.linalg.inv(PoseUtil.load_pose(pose_filename))

        # depth_mask: np.ndarray = np.where(convergence_map == CONVERGED, 255, 0).astype(np.uint8)
        # depth_image = np.where(depth_mask != 0, depth_image, 0.0).astype(np.float32)
        depth_image = DepthProcessor.denoise_depth(depth_image, convergence_map, intrinsics)

        # TODO
        # depth_mask: np.ndarray = np.where(depth_image != 0, 255, 0).astype(np.uint8)
        # pcd_points, pcd_colours = GeometryUtil.make_point_cloud(colour_image, depth_image, depth_mask, intrinsics)
        # pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcd_points)
        # pcd.colors = o3d.utility.Vector3dVector(pcd_colours)
        # VisualisationUtil.visualise_geometry(pcd)

        # TODO
        ReconstructionUtil.integrate_frame(
            ImageUtil.flip_channels(colour_image), depth_image, pose, o3d_intrinsics, tsdf
        )
        mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(tsdf)
        VisualisationUtil.visualise_geometry(mesh)

        frame_idx += 1


if __name__ == "__main__":
    main()
