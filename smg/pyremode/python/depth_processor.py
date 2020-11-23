import cv2
import matplotlib.tri as mtri
import numpy as np
import open3d as o3d

from typing import Tuple

from smg.pyremode import CONVERGED
from smg.utility import GeometryUtil


class DepthProcessor:
    """Utility functions for post-processing depth images."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def denoise_depth(raw_depth_image: np.ndarray, convergence_map: np.ndarray,
                      intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
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

        :param raw_depth_image: The raw depth image.
        :param convergence_map: TODO
        :param intrinsics:      The camera intrinsics.
        :return:                The denoised depth image.
        """
        converged_depth_image: np.ndarray = np.where(convergence_map == CONVERGED, raw_depth_image, 0.0).astype(np.float32)

        if True:
            converged_depth_image = cv2.medianBlur(converged_depth_image, 7)

            # Convert the converged depth image to a short depth image so that it can be dilated.
            depth_scale_factor: float = 1000.0
            short_depth_image: np.ndarray = (converged_depth_image * depth_scale_factor).astype(np.uint16)

            # Dilate the short depth image.
            dilation_kernel: np.ndarray = np.ones((9, 9), np.uint8)
            erosion_kernel: np.ndarray = np.ones((3, 3), np.uint8)
            short_dilated_depth_image: np.ndarray = cv2.dilate(short_depth_image, dilation_kernel)
            short_eroded_depth_image: np.ndarray = cv2.erode(short_depth_image, erosion_kernel)

            # Convert the dilated short depth image back to a floating-point depth image.
            dilated_depth_image: np.ndarray = short_dilated_depth_image.astype(np.float32) / depth_scale_factor
            eroded_depth_image: np.ndarray = short_eroded_depth_image.astype(np.float32) / depth_scale_factor

            converged_depth_image = np.where(
                (np.fabs(dilated_depth_image - converged_depth_image) <= 0.05) & (eroded_depth_image != 0.0),
                converged_depth_image, 0.0
            ).astype(np.float32)

            # output_mask = np.where(output_image != 0, 255, 0).astype(np.uint8)
            # new_kernel = np.ones((9, 9), np.uint8)
            # output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_OPEN, new_kernel)
            # # output_image = np.where(output_mask != 0, output_image, 0.0).astype(np.float32)
            # return output_image  # DepthDenoiser.densify_depth_image(output_image)[0]
            # return DepthDenoiser.densify_depth_image(converged_depth_image)[0]
            return converged_depth_image

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
        dilation_kernel: np.ndarray = np.ones((9, 9), np.uint8)
        short_depth_image = cv2.dilate(short_depth_image, dilation_kernel)

        # Convert the dilated short depth image back to a floating-point depth image.
        dilated_depth_image: np.ndarray = short_depth_image.astype(np.float32) / depth_scale_factor

        return np.where(np.fabs(dilated_depth_image - converged_depth_image) < 0.02, converged_depth_image, 0.0).astype(np.float32)

    @staticmethod
    def densify_depth_image(input_depth_image: np.ndarray) -> Tuple[np.ndarray, mtri.Triangulation]:
        iy, ix = np.nonzero(input_depth_image)
        iz = input_depth_image[(iy, ix)]
        triangulation: mtri.Triangulation = mtri.Triangulation(ix, iy)

        # See: https://stackoverflow.com/questions/52457964/how-to-deal-with-the-undesired-triangles-that-form-between-the-edges-of-my-geo
        max_radius = 10
        triangles = triangulation.triangles
        xtri = ix[triangles] - np.roll(ix[triangles], 1, axis=1)
        ytri = iy[triangles] - np.roll(iy[triangles], 1, axis=1)
        maxi = np.max(np.sqrt(xtri ** 2 + ytri ** 2), axis=1)
        ztri = np.fabs(iz[triangles] - np.roll(iz[triangles], 1, axis=1))
        triangulation.set_mask((maxi > max_radius) | (np.max(ztri, axis=1) > 0.02))

        oy, ox = np.nonzero(np.ones_like(input_depth_image))
        interpolator: mtri.LinearTriInterpolator = mtri.LinearTriInterpolator(triangulation, iz)
        result: np.ma.core.MaskedArray = interpolator(ox, oy)
        output_depth_image: np.ndarray = np.where(result.mask, 0.0, result.data).astype(np.float32)
        output_depth_image = output_depth_image.reshape(input_depth_image.shape)
        return output_depth_image, triangulation
