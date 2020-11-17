import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading

from typing import Optional

from smg.openni.openni_camera import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import CONVERGED, DepthEstimator


class KinectMappingSystem:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, camera: OpenNICamera, depth_estimator: DepthEstimator, tracker: RGBDTracker):
        """
        TODO

        :param depth_estimator:     TODO
        """
        self.__camera: OpenNICamera = camera
        self.__depth_estimator: DepthEstimator = depth_estimator
        self.__should_terminate: bool = False
        self.__tracker: RGBDTracker = tracker

        self.__mapping_thread = threading.Thread(target=self.__run_mapping)

    # PUBLIC METHODS

    def run(self) -> None:
        """TODO"""
        self.__mapping_thread.start()

        while not self.__should_terminate:
            # TODO
            colour_image, depth_image = self.__camera.get_images()
            cv2.imshow("Tracking Image", colour_image)
            c: int = cv2.waitKey(1)
            if c == ord('q'):
                self.terminate()
                return

            # TODO
            if not self.__tracker.is_ready():
                continue
            pose: Optional[np.ndarray] = self.__tracker.estimate_pose(colour_image, depth_image)
            if pose is None:
                continue

            # TODO
            self.__depth_estimator.put(colour_image, pose)

    def terminate(self) -> None:
        """TODO"""
        self.__should_terminate = True
        self.__depth_estimator.terminate()
        self.__mapping_thread.join()

    # PRIVATE METHODS

    def __run_mapping(self) -> None:
        """TODO"""
        _, ax = plt.subplots(1, 3)
        while not self.__should_terminate:
            # TODO
            result = self.__depth_estimator.get()
            if result is not None:
                reference_image, reference_pose, estimated_depth_image, convergence_map = result

                depth_mask: np.ndarray = np.where(convergence_map == CONVERGED, 255, 0).astype(np.uint8)
                estimated_depth_image = np.where(depth_mask != 0, estimated_depth_image, 0).astype(np.float32)

                # TEMPORARY
                ax[0].clear()
                ax[1].clear()
                ax[2].clear()
                ax[0].imshow(reference_image[:, :, [2, 1, 0]])
                ax[1].imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
                ax[2].imshow(convergence_map)

                plt.draw()
                plt.waitforbuttonpress(0.001)
