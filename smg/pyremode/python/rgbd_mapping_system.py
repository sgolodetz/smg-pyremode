import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading

from typing import Optional

from smg.openni.openni_camera import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import CONVERGED, DepthEstimator, RGBDImageSource


class RGBDMappingSystem:
    """A REMODE-based multi-view mapping system that uses an RGB-D tracker to estimate metric camera poses."""

    # CONSTRUCTOR

    def __init__(self, image_source: RGBDImageSource, tracker: RGBDTracker, depth_estimator: DepthEstimator):
        """
        Construct an RGB-D mapping system.

        :param image_source:    A source of RGB-D images.
        :param tracker:         The RGB-D tracker to use.
        :param depth_estimator: The depth estimator to use.
        """
        self.__depth_estimator: DepthEstimator = depth_estimator
        self.__image_source: OpenNICamera = image_source
        self.__should_terminate: bool = False
        self.__tracker: RGBDTracker = tracker

        self.__mapping_thread = threading.Thread(target=self.__run_mapping)

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the mapping system's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the mapping system at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the mapping system."""
        # Start the mapping thread.
        self.__mapping_thread.start()

        # Until the mapping system should terminate:
        while not self.__should_terminate:
            # Get the latest images from the image source.
            colour_image, depth_image = self.__image_source.get_images()

            # Show the colour image so that the user can see what's going on. If the user presses 'q',
            # tell the mapping system to terminate, and early out.
            cv2.imshow("Tracking Image", colour_image)
            c: int = cv2.waitKey(1)
            if c == ord('q'):
                self.terminate()
                return

            # If the tracker's ready:
            if self.__tracker.is_ready():
                # Try to estimate the pose of the camera.
                pose: Optional[np.ndarray] = self.__tracker.estimate_pose(colour_image, depth_image)

                # If this succeeds, pass the colour image and pose to the depth estimator.
                if pose is not None:
                    self.__depth_estimator.add_posed_image(colour_image, pose)

    def terminate(self) -> None:
        """Tell the mapping system to terminate."""
        # If the mapping system hasn't already been told to terminate.
        if not self.__should_terminate:
            # Set its own termination flag.
            self.__should_terminate = True

            # Tell the depth estimator it owns to terminate.
            self.__depth_estimator.terminate()

            # Wait for the mapping thread to terminate.
            self.__mapping_thread.join()

    # PRIVATE METHODS

    def __run_mapping(self) -> None:
        """Make a map of the scene based on the keyframes yielded by the depth estimator."""
        _, ax = plt.subplots(1, 3)

        # Until the mapping system should terminate:
        while not self.__should_terminate:
            # Try to get the next keyframe from the depth estimator.
            keyframe = self.__depth_estimator.get_keyframe()
            if keyframe is not None:
                colour_image, depth_image, pose, converged_percentage, convergence_map = keyframe

                # If this succeeds, post-process the depth image to keep only those pixels whose depth has converged.
                depth_mask: np.ndarray = np.where(convergence_map == CONVERGED, 255, 0).astype(np.uint8)
                depth_image = np.where(depth_mask != 0, depth_image, 0).astype(np.float32)

                # TEMPORARY
                # Show the keyframe images for debugging purposes.
                ax[0].clear()
                ax[1].clear()
                ax[2].clear()
                ax[0].imshow(colour_image[:, :, [2, 1, 0]])
                ax[1].imshow(depth_image, vmin=0.0, vmax=4.0)
                ax[2].imshow(convergence_map)

                plt.draw()
                plt.waitforbuttonpress(0.001)
