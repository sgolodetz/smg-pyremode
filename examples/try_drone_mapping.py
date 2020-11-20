import cv2
import open3d as o3d
import os

from typing import Optional

from smg.open3d import ReconstructionUtil, VisualisationUtil
from smg.pyorbslam2 import MonocularTracker
from smg.pyremode import DepthEstimator, MonocularMappingSystem, RGBDroneCamera, RGBImageSource, \
    TemporalKeyframeDepthEstimator
from smg.rotory.drones.tello import Tello


def main():
    # noinspection PyUnusedLocal
    tsdf: Optional[o3d.pipelines.integration.ScalableTSDFVolume] = None

    with Tello(print_commands=False, print_responses=False, print_state_messages=False) as drone:
        with MonocularTracker(
            settings_file=f"settings-tello.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            image_source: RGBImageSource = RGBDroneCamera(drone)
            depth_estimator: DepthEstimator = TemporalKeyframeDepthEstimator(
                image_source.get_image_dims(), image_source.get_intrinsics()
            )
            with MonocularMappingSystem(image_source, tracker, depth_estimator) as system:
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
