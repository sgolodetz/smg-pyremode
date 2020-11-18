import os

from smg.openni.openni_camera import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import DepthEstimator, RGBDMappingSystem, RGBDOpenNICamera, TemporalKeyframeDepthEstimator


def main():
    with OpenNICamera(mirror_images=True) as camera:
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            depth_estimator: DepthEstimator = TemporalKeyframeDepthEstimator(
                camera.get_colour_dims(), camera.get_colour_intrinsics()
            )
            with RGBDMappingSystem(RGBDOpenNICamera(camera), tracker, depth_estimator) as system:
                system.run()

            # If ORB-SLAM's not ready yet, forcibly terminate the whole process (this isn't graceful, but
            # if we don't do it then we may have to wait a very long time for it to finish initialising).
            if not tracker.is_ready():
                # noinspection PyProtectedMember
                os._exit(0)


if __name__ == "__main__":
    main()
