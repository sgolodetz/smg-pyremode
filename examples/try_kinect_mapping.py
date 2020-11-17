import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import threading

from typing import Optional, Tuple

from smg.geometry import GeometryUtil
from smg.openni.openni_camera import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import KinectMappingSystem, TemporalKeyframeDepthEstimator, DepthEstimator


def main():
    with OpenNICamera(mirror_images=True) as camera:
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            depth_estimator: DepthEstimator = TemporalKeyframeDepthEstimator(
                camera.get_colour_dims(), camera.get_colour_intrinsics()
            )
            with KinectMappingSystem(camera, depth_estimator, tracker) as system:
                system.run()


if __name__ == "__main__":
    main()
