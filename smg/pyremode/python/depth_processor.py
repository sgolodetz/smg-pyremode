import cv2
import matplotlib.tri as mtri
import numpy as np
import open3d as o3d

from typing import Optional, Tuple

from smg.pyremode import CONVERGED
from smg.utility import GeometryUtil


class DepthProcessor:
    """Utility functions for post-processing depth images."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def denoise_depth(raw_depth_image: np.ndarray, convergence_map: np.ndarray,
                      intrinsics: Tuple[float, float, float, float], *,
                      extended_denoising: bool = False) -> np.ndarray:
        """
        Denoise a REMODE depth image.

        .. note::
            The extended denoising sometimes helps, but always takes longer. It may or may not be worth using.

        :param raw_depth_image:     The raw depth image produced by REMODE (after doing its own denoising).
        :param convergence_map:     The convergence map produced by REMODE.
        :param intrinsics:          The camera intrinsics.
        :param extended_denoising:  Whether to perform extended denoising.
        :return:                    The denoised depth image.
        """
        # Filter the raw depth image to keep only those pixels whose REMODE-produced depth has converged.
        converged_depth_image: np.ndarray = np.where(
            convergence_map == CONVERGED, raw_depth_image, 0.0
        ).astype(np.float32)

        # Median filter the converged depth image to help reduce impulsive noise.
        converged_depth_image = cv2.medianBlur(converged_depth_image, 7)

        # Make a short version of the converged depth image so that it can be dilated.
        depth_scale_factor: float = 1000.0
        converged_depth_image_s: np.ndarray = (converged_depth_image * depth_scale_factor).astype(np.uint16)

        # Construct dilated and eroded versions of the short converged depth image.
        dilation_kernel: np.ndarray = np.ones((9, 9), np.uint8)
        erosion_kernel: np.ndarray = np.ones((3, 3), np.uint8)
        dilated_depth_image_s: np.ndarray = cv2.dilate(converged_depth_image_s, dilation_kernel)
        eroded_depth_image_s: np.ndarray = cv2.erode(converged_depth_image_s, erosion_kernel)

        # Convert the short dilated and eroded depth images back to floating point.
        dilated_depth_image: np.ndarray = dilated_depth_image_s.astype(np.float32) / depth_scale_factor
        eroded_depth_image: np.ndarray = eroded_depth_image_s.astype(np.float32) / depth_scale_factor

        # Filter the converged depth image to remove pixels that either (i) have a nearby pixel with a much
        # larger depth, or (ii) have a missing pixel in their close neighbourhood. The idea is that these
        # are much more likely to be true of noisy points floating in space than points that form part of
        # a surface. We'll lose some good points as well, but so be it.
        converged_depth_image = np.where(
            (np.fabs(dilated_depth_image - converged_depth_image) <= 0.05) & (eroded_depth_image != 0.0),
            converged_depth_image, 0.0
        ).astype(np.float32)

        # If we're not going to perform extended denoising, return the converged depth image at this point.
        if not extended_denoising:
            return converged_depth_image

        # Make the converged point cloud.
        height, width = converged_depth_image.shape
        colour_image: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        depth_mask: np.ndarray = np.where(converged_depth_image != 0, 255, 0).astype(np.uint8)
        pcd_points, pcd_colours = GeometryUtil.make_point_cloud(
            colour_image, converged_depth_image, depth_mask, intrinsics
        )

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

        # Downsample the converged point cloud and denoise it using a built-in (and quite slow) approach from Open3D.
        factor: int = 10
        pcd = pcd.uniform_down_sample(every_k_points=factor)
        pcd, inliers = pcd.remove_statistical_outlier(20, 2.0)

        # Remap the indices of the inliers to correspond to the points in the converged point cloud.
        inliers = list(map(lambda x: original_indices[x * factor], inliers))

        # Convert the inlier indices into a mask.
        inlier_mask: np.ndarray = np.zeros(converged_depth_image.shape, dtype=np.uint8)
        inlier_mask = inlier_mask.flatten()
        inlier_mask[inliers] = 255
        inlier_mask = inlier_mask.reshape(converged_depth_image.shape, order='C')

        # Make a new depth image that contains only the inliers from the downsampled point cloud (i.e. quite sparse).
        inlier_depth_image: np.ndarray = np.where(inlier_mask != 0, converged_depth_image, 0.0).astype(np.float32)

        # Make a short version of the inlier depth image so that it can be dilated.
        depth_scale_factor: float = 1000.0
        inlier_depth_image_s: np.ndarray = (inlier_depth_image * depth_scale_factor).astype(np.uint16)

        # Dilate the short inlier depth image.
        dilation_kernel = np.ones((9, 9), np.uint8)
        inlier_depth_image_s = cv2.dilate(inlier_depth_image_s, dilation_kernel)

        # Convert the dilated short inlier depth image back to floating point.
        dilated_inlier_depth_image: np.ndarray = inlier_depth_image_s.astype(np.float32) / depth_scale_factor

        # TODO: Comment here.
        return np.where(
            np.fabs(dilated_inlier_depth_image - converged_depth_image) < 0.02, converged_depth_image, 0.0
        ).astype(np.float32)

    @staticmethod
    def densify_depth_image(input_depth_image: np.ndarray) -> Tuple[np.ndarray, Optional[mtri.Triangulation]]:
        """
        Densify the specified depth image.

        .. note::
            The approach used is to construct a Delaunay triangulation of the pixels with known depths and then
            linearly interpolate the depth values at the triangles' vertices.

        :param input_depth_image:   TODO
        :return:                    TODO
        """
        iy, ix = np.nonzero(input_depth_image)
        iz = input_depth_image[(iy, ix)]
        if len(iz) < 3:
            return input_depth_image, None
        triangulation: mtri.Triangulation = mtri.Triangulation(ix, iy)

        # See: https://stackoverflow.com/questions/52457964/how-to-deal-with-the-undesired-triangles-that-form-between-the-edges-of-my-geo
        max_radius = 10
        triangles = triangulation.triangles
        xtri = ix[triangles] - np.roll(ix[triangles], 1, axis=1)
        ytri = iy[triangles] - np.roll(iy[triangles], 1, axis=1)
        maxi = np.max(np.sqrt(xtri ** 2 + ytri ** 2), axis=1)
        ztri = np.fabs(iz[triangles] - np.roll(iz[triangles], 1, axis=1))
        # triangulation.set_mask((maxi > max_radius) | (np.max(ztri, axis=1) > 0.02))
        triangulation.set_mask(np.max(ztri, axis=1) > 0.05)

        oy, ox = np.nonzero(np.ones_like(input_depth_image))
        interpolator: mtri.LinearTriInterpolator = mtri.LinearTriInterpolator(triangulation, iz)
        result: np.ma.core.MaskedArray = interpolator(ox, oy)
        output_depth_image: np.ndarray = np.where(result.mask, 0.0, result.data).astype(np.float32)
        output_depth_image = output_depth_image.reshape(input_depth_image.shape)
        return output_depth_image, triangulation
