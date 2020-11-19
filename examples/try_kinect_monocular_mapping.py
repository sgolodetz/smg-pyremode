import cv2
import open3d as o3d
import os

from typing import Optional

from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.openni import OpenNICamera
from smg.pyorbslam2 import MonocularTracker
from smg.pyremode import DepthEstimator, MonocularMappingSystem, RGBDroneCamera, RGBDOpenNICamera, \
    RGBFromRGBDImageSource, TemporalKeyframeDepthEstimator
from smg.rotory.drones.tello import Tello


def main():
    # noinspection PyUnusedLocal
    tsdf: Optional[o3d.pipelines.integration.ScalableTSDFVolume] = None

    # with OpenNICamera(mirror_images=True) as camera:
    with Tello(local_ip="192.168.10.3", print_commands=False, print_responses=False, print_state_messages=False) as drone:
        with MonocularTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            image_dims = (960, 720)
            intrinsics = 946.60441222, 941.38386885, 460.29254907, 357.08431882
            depth_estimator: DepthEstimator = TemporalKeyframeDepthEstimator(
                image_dims, intrinsics
            )
            with MonocularMappingSystem(
                RGBDroneCamera(drone), tracker, depth_estimator
            ) as system:
                tsdf = system.run()

            # If ORB-SLAM's not ready yet, forcibly terminate the whole process (this isn't graceful, but
            # if we don't do it then we may have to wait a very long time for it to finish initialising).
            if not tracker.is_ready():
                # noinspection PyProtectedMember
                os._exit(0)

    cv2.destroyAllWindows()
    mesh: o3d.geometry.TriangleMesh = ReconstructionUtil.make_mesh(tsdf, print_progress=True)
    VisualisationUtil.visualise_geometry(mesh)


if __name__ == "__main__":
    main()
