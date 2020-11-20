import cv2
import numpy as np
import open3d as o3d

from typing import Tuple

from smg.pyremode import CONVERGED
from smg.utility import GeometryUtil


class DepthDenoiser:
    """Utility functions for denoising depth images."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def denoise_depth(depth_image: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        # TODO: Update all of this.
        """
        Denoise the specified depth image.

        .. note::
            The technique currently used is to:

              (i) Make a point cloud.
             (ii) Downsample the point cloud and remove statistical outliers using Open3D.
            (iii) Use the inliers found by Open3D to make a new depth image.
             (iv) Dilate the new depth image to restore some of its original density.

            Various improvements may be made in the future, e.g. restoring the estimated depths of the pixels
            surrounding the known inliers if they're close enough to the inlier depth to be trusted.

        :param depth_image: The depth image.
        :param intrinsics:  The camera intrinsics.
        :return:            The denoised depth image.
        """
        height, width = depth_image.shape
        colour_image: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        depth_mask: np.ndarray = np.where(depth_image != 0, 255, 0).astype(np.uint8)

        # Make the original point cloud.
        pcd_points, pcd_colours = GeometryUtil.make_point_cloud(colour_image, depth_image, depth_mask, intrinsics)

        # Shuffle the points to avoid artifacts when the point cloud is uniformly downsampled.
        rng_state = np.random.get_state()
        original_indices: np.ndarray = np.array(range(len(pcd_points)))
        np.random.shuffle(original_indices)
        np.random.set_state(rng_state)
        np.random.shuffle(pcd_points)
        np.random.set_state(rng_state)
        np.random.shuffle(pcd_colours)

        # Convert the original point cloud to Open3D format.
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

        # Downsample the original point cloud and denoise the downsampled point cloud.
        factor: int = 10
        pcd = pcd.uniform_down_sample(every_k_points=factor)
        pcd, ind = pcd.remove_statistical_outlier(20, 2.0)

        # Remap the indices of the retained points to correspond to the points in the original point cloud.
        ind = list(map(lambda x: original_indices[x * factor], ind))

        # Convert the retained point indices into a mask.
        ind_mask: np.ndarray = np.zeros(depth_image.shape, dtype=np.uint8)
        ind_mask = ind_mask.flatten()
        ind_mask[ind] = 255
        ind_mask = ind_mask.reshape(depth_image.shape, order='C')

        # Mask out the outliers in the depth image.
        filtered_depth_image: np.ndarray = np.where(ind_mask != 0, depth_image, 0.0).astype(np.float32)

        # Convert the floating-point depth image to a short depth image so that it can be dilated.
        depth_scale_factor: float = 1000.0
        short_depth_image: np.ndarray = (filtered_depth_image * depth_scale_factor).astype(np.uint16)

        # Dilate the short depth image.
        kernel: np.ndarray = np.ones((5, 5), np.uint8)
        short_depth_image = cv2.dilate(short_depth_image, kernel)

        # Convert the dilated short depth image back to a floating-point depth image and return it.
        dilated_depth_image: np.ndarray = short_depth_image.astype(np.float32) / depth_scale_factor

        return np.where(np.fabs(dilated_depth_image - depth_image) < 0.02, depth_image, 0.0).astype(np.float32)

    @staticmethod
    def denoise_depth_ex(raw_depth_image: np.ndarray, convergence_map: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        # TODO: Update all of this.
        """
        Denoise the specified depth image.

        .. note::
            The technique currently used is to:

              (i) Make a point cloud.
             (ii) Downsample the point cloud and remove statistical outliers using Open3D.
            (iii) Use the inliers found by Open3D to make a new depth image.
             (iv) Dilate the new depth image to restore some of its original density.

            Various improvements may be made in the future, e.g. restoring the estimated depths of the pixels
            surrounding the known inliers if they're close enough to the inlier depth to be trusted.

        :param depth_image: The depth image.
        :param intrinsics:  The camera intrinsics.
        :return:            The denoised depth image.
        """
        converged_depth_image: np.ndarray = np.where(convergence_map == CONVERGED, raw_depth_image, 0.0).astype(np.float32)

        # Make the converged point cloud.
        height, width = converged_depth_image.shape
        colour_image: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        depth_mask: np.ndarray = np.where(converged_depth_image != 0, 255, 0).astype(np.uint8)
        pcd_points, pcd_colours = GeometryUtil.make_point_cloud(colour_image, converged_depth_image, depth_mask, intrinsics)

        # Shuffle the points to avoid artifacts when the point cloud is uniformly downsampled.
        rng_state = np.random.get_state()
        original_indices: np.ndarray = np.array(range(len(pcd_points)))
        np.random.shuffle(original_indices)
        np.random.set_state(rng_state)
        np.random.shuffle(pcd_points)
        np.random.set_state(rng_state)
        np.random.shuffle(pcd_colours)

        # Convert the converged point cloud to Open3D format.
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

        # Downsample the converged point cloud and denoise it.
        factor: int = 10
        pcd = pcd.uniform_down_sample(every_k_points=factor)
        pcd, ind = pcd.remove_statistical_outlier(20, 2.0)

        # Remap the indices of the retained points to correspond to the points in the converged point cloud.
        ind = list(map(lambda x: original_indices[x * factor], ind))

        # Convert the retained point indices into a mask.
        ind_mask: np.ndarray = np.zeros(converged_depth_image.shape, dtype=np.uint8)
        ind_mask = ind_mask.flatten()
        ind_mask[ind] = 255
        ind_mask = ind_mask.reshape(converged_depth_image.shape, order='C')

        # Mask out the outliers in the depth image.
        trusted_depth_image: np.ndarray = np.where(ind_mask != 0, converged_depth_image, 0.0).astype(np.float32)

        # Convert the floating-point depth image to a short depth image so that it can be dilated.
        depth_scale_factor: float = 1000.0
        short_depth_image: np.ndarray = (trusted_depth_image * depth_scale_factor).astype(np.uint16)

        # Dilate the short depth image.
        kernel: np.ndarray = np.ones((5, 5), np.uint8)
        short_depth_image = cv2.dilate(short_depth_image, kernel)

        # Convert the dilated short depth image back to a floating-point depth image and return it.
        dilated_depth_image: np.ndarray = short_depth_image.astype(np.float32) / depth_scale_factor

        return np.where(np.fabs(dilated_depth_image - raw_depth_image) < 0.02, raw_depth_image, 0.0).astype(np.float32)
