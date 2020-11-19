import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import threading

from typing import Optional

from smg.open3d import ReconstructionUtil
from smg.pyorbslam2 import MonocularTracker
from smg.pyremode import CONVERGED, DepthEstimator, RGBImageSource
from smg.utility import ImageUtil


class MonocularMappingSystem:
    """
    A REMODE-based multi-view mapping system that uses a monocular tracker to estimate the camera poses,
    and fuses keyframes into an Open3D TSDF.
    """

    # CONSTRUCTOR

    def __init__(self, image_source: RGBImageSource, tracker: MonocularTracker, depth_estimator: DepthEstimator):
        """
        Construct an RGB-D mapping system.

        :param image_source:    A source of RGB-D images.
        :param tracker:         The RGB-D tracker to use.
        :param depth_estimator: The depth estimator to use.
        """
        self.__depth_estimator: DepthEstimator = depth_estimator
        self.__image_source: RGBImageSource = image_source
        self.__should_terminate: bool = False
        self.__tracker: MonocularTracker = tracker

        self.__tsdf: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        self.__mapping_thread = threading.Thread(target=self.__run_mapping)

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the mapping system's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the mapping system at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> o3d.pipelines.integration.ScalableTSDFVolume:
        """
        Run the mapping system.

        :return:    The final TSDF.
        """
        # Start the mapping thread.
        self.__mapping_thread.start()

        # Until the mapping system should terminate:
        while not self.__should_terminate:
            # Get the latest images from the image source.
            colour_image = self.__image_source.get_image()

            # Show the colour image so that the user can see what's going on. If the user presses 'q',
            # tell the mapping system to terminate, and early out.
            cv2.imshow("Tracking Image", colour_image)
            c: int = cv2.waitKey(1)
            if c == ord('q'):
                return self.terminate()

            # If the tracker's ready:
            if self.__tracker.is_ready():
                # Try to estimate the pose of the camera.
                pose: Optional[np.ndarray] = self.__tracker.estimate_pose(colour_image)

                # If this succeeds, pass the colour image and pose to the depth estimator.
                if pose is not None:
                    self.__depth_estimator.add_posed_image(colour_image, pose)

    def terminate(self) -> o3d.pipelines.integration.ScalableTSDFVolume:
        """
        Tell the mapping system to terminate.

        .. note::
            We return the final TSDF from this function because we can guarantee that the mapping thread
            will have terminated by then, and so there won't be any race conditions.

        :return:    The final TSDF.
        """
        # If the mapping system hasn't already been told to terminate.
        if not self.__should_terminate:
            # Set its own termination flag.
            self.__should_terminate = True

            # Tell the depth estimator it owns to terminate.
            self.__depth_estimator.terminate()

            # Wait for the mapping thread to terminate.
            self.__mapping_thread.join()

        return self.__tsdf

    # PRIVATE METHODS

    def __run_mapping(self) -> None:
        """Make a map of the scene based on the keyframes yielded by the depth estimator."""
        width, height = self.__image_source.get_image_dims()
        fx, fy, cx, cy = self.__image_source.get_intrinsics()
        intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        _, ax = plt.subplots(1, 3)

        # Until the mapping system should terminate:
        while not self.__should_terminate:
            # Try to get the next keyframe from the depth estimator.
            keyframe = self.__depth_estimator.get_keyframe()
            if keyframe is not None:
                colour_image, depth_image, pose, converged_percentage, convergence_map = keyframe

                # If the keyframe hasn't sufficiently converged, don't fuse it.
                # if converged_percentage < 30.0:
                #     continue
                print(converged_percentage)

                # Post-process the depth image to keep only those pixels whose depth has converged.
                depth_mask: np.ndarray = np.where(convergence_map == CONVERGED, 255, 0).astype(np.uint8)
                # depth_mask: np.ndarray = np.where(depth_image != 0, 255, 0).astype(np.uint8)
                depth_image = np.where(depth_mask != 0, depth_image, 0).astype(np.float32)

                # Integrate the keyframe into the map.
                ReconstructionUtil.integrate_frame(
                    ImageUtil.flip_channels(colour_image), depth_image, pose, intrinsics, self.__tsdf
                )

                # Show the keyframe images for debugging purposes.
                ax[0].clear()
                ax[1].clear()
                ax[2].clear()
                ax[0].imshow(colour_image[:, :, [2, 1, 0]])
                ax[1].imshow(depth_image, vmin=0.0, vmax=4.0)
                ax[2].imshow(convergence_map)

                plt.draw()
                plt.waitforbuttonpress(0.001)
