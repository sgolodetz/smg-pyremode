import cv2
import matplotlib.tri as mtri
import numpy as np
import open3d as o3d

from typing import Optional, Tuple

from smg.utility import GeometryUtil

from ..cpp.pyremode import CONVERGED


class DepthProcessor:
    """Utility functions for post-processing depth images."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def denoise_depth_fast(depth_image: np.ndarray) -> np.ndarray:
        """
        Denoise a REMODE depth image using a simple, fast technique.

        .. note::
            This is intended to be called on a depth image from which unconverged pixels have already been removed.
        .. note::
            The technique used:
             (i) Applies a median filter to help reduce impulsive noise.
            (ii) Removes pixels that either have a nearby pixel with a much larger depth,
                 or have a missing pixel in their close neighbourhood.

        :param depth_image: A REMODE depth image (after filtering out unconverged pixels).
        :return:            The denoised depth image.
        """
        # Median filter the depth image to help reduce impulsive noise.
        depth_image = cv2.medianBlur(depth_image, 7)

        # Make dilated and eroded versions of the depth image. Note that dilation on a non-binary image performs
        # a windowed maximum, and erosion on a non-binary image performs a windowed minimum.
        dilated_depth_image: np.ndarray = DepthProcessor.__dilate_depth_image(depth_image, 9)
        eroded_depth_image: np.ndarray = DepthProcessor.__erode_depth_image(depth_image, 3)

        # Filter the depth image to remove pixels that either (i) have a nearby pixel with a much larger depth,
        # or (ii) have a missing pixel in their close neighbourhood. The idea is that these are much more likely
        # to be true of noisy points floating in space than points that form part of a surface. We'll lose some
        # good points as well, but so be it.
        return np.where(
            (np.fabs(dilated_depth_image - depth_image) <= 0.05) & (eroded_depth_image != 0.0), depth_image, 0.0
        ).astype(np.float32)

    @staticmethod
    def denoise_depth_statistical(depth_image: np.ndarray, intrinsics: Tuple[float, float, float, float]) \
            -> np.ndarray:
        """
        Denoise a REMODE depth image using a statistical outlier removal method from Open3D.

        .. note::
            This is intended to be called on a depth image from which unconverged pixels have already been removed.

        :param depth_image: A REMODE depth image (after filtering out unconverged pixels).
        :param intrinsics:  The camera intrinsics.
        :return:            The denoised depth image.
        """
        # Make a point cloud from the depth image.
        height, width = depth_image.shape
        colour_image: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        depth_mask: np.ndarray = np.where(depth_image != 0, 255, 0).astype(np.uint8)
        pcd_points, pcd_colours = GeometryUtil.make_point_cloud(colour_image, depth_image, depth_mask, intrinsics)

        # Shuffle the points to avoid artifacts when the point cloud is uniformly downsampled.
        rng_state = np.random.get_state()
        original_indices: np.ndarray = np.array(range(len(pcd_points)))
        np.random.shuffle(original_indices)
        np.random.set_state(rng_state)
        np.random.shuffle(pcd_points)
        np.random.set_state(rng_state)
        np.random.shuffle(pcd_colours)

        # Convert the point cloud to Open3D format.
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

        # Downsample the point cloud and denoise it using a built-in (and quite slow) approach from Open3D.
        factor: int = 10
        pcd = pcd.uniform_down_sample(every_k_points=factor)
        pcd, inliers = pcd.remove_statistical_outlier(20, 2.0)

        # Remap the indices of the inliers to correspond to the points in the original point cloud.
        inliers = list(map(lambda x: original_indices[x * factor], inliers))

        # Convert the inlier indices into a mask.
        inlier_mask: np.ndarray = np.zeros(depth_image.shape, dtype=np.uint8)
        inlier_mask = inlier_mask.flatten()
        inlier_mask[inliers] = 255
        inlier_mask = inlier_mask.reshape(depth_image.shape, order='C')

        # Return a new depth image that contains only the inliers from the downsampled point cloud (i.e. quite sparse).
        return np.where(inlier_mask != 0, depth_image, 0.0).astype(np.float32)

    @staticmethod
    def densify_depth_delaunay(input_depth_image: np.ndarray) -> Tuple[np.ndarray, Optional[mtri.Triangulation]]:
        """
        Try to densify a depth image by constructing a Delaunay triangulation of the pixels with known depths,
        and then linearly interpolating from the depth values at the triangles' vertices.

        .. note::
            No densification will be performed if the triangulation cannot be constructed,
            e.g. if there are fewer than three non-zero points in the input depth image.

        :param input_depth_image:   The input depth image.
        :return:                    A tuple consisting of the densified depth image and the triangulation that was used
                                    to make it, if possible, else a tuple consisting of the input depth image and None.
        """
        # Try to triangulate the input depth image. If this isn't possible, early out.
        iy, ix = np.nonzero(input_depth_image)
        iz = input_depth_image[(iy, ix)]
        if len(iz) < 3:
            return input_depth_image, None

        triangulation: mtri.Triangulation = mtri.Triangulation(ix, iy)

        # Filter out any unsuitable triangles (see https://stackoverflow.com/questions/52457964/
        # how-to-deal-with-the-undesired-triangles-that-form-between-the-edges-of-my-geo).
        triangles = triangulation.triangles
        # xtri = ix[triangles] - np.roll(ix[triangles], 1, axis=1)
        # ytri = iy[triangles] - np.roll(iy[triangles], 1, axis=1)
        # maxi = np.max(np.sqrt(xtri ** 2 + ytri ** 2), axis=1)
        # max_radius = 10
        # triangulation.set_mask(maxi > max_radius)
        ztri = np.fabs(iz[triangles] - np.roll(iz[triangles], 1, axis=1))
        triangulation.set_mask(np.max(ztri, axis=1) > 0.05)

        # Use linear interpolation to fill in the depth values within each triangle.
        oy, ox = np.nonzero(np.ones_like(input_depth_image))
        interpolator: mtri.LinearTriInterpolator = mtri.LinearTriInterpolator(triangulation, iz)
        result: np.ma.core.MaskedArray = interpolator(ox, oy)
        output_depth_image: np.ndarray = np.where(result.mask, 0.0, result.data).astype(np.float32)
        output_depth_image = output_depth_image.reshape(input_depth_image.shape)
        return output_depth_image, triangulation

    @staticmethod
    def densify_depth_simple(trusted_depth_image: np.ndarray, untrusted_depth_image: np.ndarray) -> np.ndarray:
        """
        Densify a trusted depth image by looking up suitable additional depth values from a less-trusted depth image.

        .. note::
            The idea is to trust additional points from the untrusted image that are close to points we already trust.

        :param trusted_depth_image:     The trusted depth image.
        :param untrusted_depth_image:   The less-trusted depth image.
        :return:                        A densified version of the trusted depth image.
        """
        # Make a dilated version of the trusted depth image.
        dilated_trusted_depth_image: np.ndarray = DepthProcessor.__dilate_depth_image(trusted_depth_image, 9)

        # Filter the untrusted depth image for those pixels that have a nearby trusted pixel with a very similar depth.
        return np.where(
            np.fabs(dilated_trusted_depth_image - untrusted_depth_image) <= 0.02, untrusted_depth_image, 0.0
        ).astype(np.float32)

    @staticmethod
    def postprocess_depth(raw_depth_image: np.ndarray, convergence_map: np.ndarray,
                          intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Post-process a REMODE depth image to reduce noise.

        :param raw_depth_image:     A raw depth image produced by REMODE (after REMODE's own denoising).
        :param convergence_map:     The convergence map produced by REMODE.
        :param intrinsics:          The camera intrinsics.
        :return:                    The post-processed depth image.
        """
        depth_image: np.ndarray = DepthProcessor.remove_unconverged_pixels(raw_depth_image, convergence_map)

        # Approach #1 (Slow, best so far)
        depth_image = DepthProcessor.denoise_depth_fast(depth_image)
        depth_image = DepthProcessor.denoise_depth_statistical(depth_image, intrinsics)
        depth_image, _ = DepthProcessor.densify_depth_delaunay(depth_image)

        # Approach #2 (Slow, reasonable)
        # trusted_depth_image: np.ndarray = DepthProcessor.denoise_depth_statistical(depth_image, intrinsics)
        # depth_image = DepthProcessor.densify_depth_simple(trusted_depth_image, depth_image)

        # Approach #3 (Fast but less good)
        # depth_image = DepthProcessor.denoise_depth_fast(depth_image)

        return depth_image

    @staticmethod
    def remove_unconverged_pixels(raw_depth_image, convergence_map: np.ndarray) -> np.ndarray:
        """
        Filter a raw REMODE depth image to keep only those pixels whose depth has converged.

        :param raw_depth_image:     A raw depth image produced by REMODE (after REMODE's own denoising).
        :param convergence_map:     The convergence map produced by REMODE.
        :return:                    The filtered depth image.
        """
        return np.where(convergence_map == CONVERGED, raw_depth_image, 0.0).astype(np.float32)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __dilate_depth_image(depth_image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Dilate a floating-point depth image.

        :param depth_image:     The depth image.
        :param kernel_size:     The size of kernel to use.
        :return:                The dilated depth image.
        """
        # Make a short version of the depth image so that it can be dilated.
        depth_scale_factor: float = 1000.0
        depth_image_s: np.ndarray = (depth_image * depth_scale_factor).astype(np.uint16)

        # Dilate the short depth image.
        kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_depth_image_s: np.ndarray = cv2.dilate(depth_image_s, kernel)

        # Convert the dilated short depth image back to floating point.
        return dilated_depth_image_s.astype(np.float32) / depth_scale_factor

    @staticmethod
    def __erode_depth_image(depth_image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Erode a floating-point depth image.

        :param depth_image:     The depth image.
        :param kernel_size:     The size of kernel to use.
        :return:                The eroded depth image.
        """
        # Make a short version of the depth image so that it can be eroded.
        depth_scale_factor: float = 1000.0
        depth_image_s: np.ndarray = (depth_image * depth_scale_factor).astype(np.uint16)

        # Erode the short depth image.
        kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_depth_image_s: np.ndarray = cv2.erode(depth_image_s, kernel)

        # Convert the eroded short depth image back to floating point.
        return eroded_depth_image_s.astype(np.float32) / depth_scale_factor
