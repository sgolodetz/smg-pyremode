import os

from openni import openni2

from smg.openni.openni_camera import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import DepthEstimator, RGBDMappingSystem, RGBDOpenNICamera, TemporalKeyframeDepthEstimator


def main():
    openni2.initialize()

    with OpenNICamera(manage_openni=False, mirror_images=True) as camera:
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            depth_estimator: DepthEstimator = TemporalKeyframeDepthEstimator(
                camera.get_colour_dims(), camera.get_colour_intrinsics()
            )
            with RGBDMappingSystem(RGBDOpenNICamera(camera), tracker, depth_estimator) as system:
                system.run()

            # noinspection PyProtectedMember
            os._exit(0)


if __name__ == "__main__":
    main()
