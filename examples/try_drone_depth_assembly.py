import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from typing import Optional, Tuple

from smg.open3d import VisualisationUtil
from smg.pyorbslam2 import MonocularTracker
from smg.pyremode import CONVERGED, DepthAssembler, DepthProcessor
from smg.rotory.drone_factory import DroneFactory
from smg.utility import GeometryUtil


def main():
    with DroneFactory.make_drone("tello") as drone:
        with MonocularTracker(
            settings_file=f"settings-tello.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            intrinsics: Optional[Tuple[float, float, float, float]] = drone.get_intrinsics()
            if intrinsics is None:
                raise RuntimeError("Cannot get drone camera intrinsics")

            image_size: Tuple[int, int] = drone.get_image_size()
            depth_assembler: DepthAssembler = DepthAssembler(image_size, intrinsics)
            is_keyframe: bool = True

            reference_colour_image: Optional[np.ndarray] = None
            estimated_depth_image: Optional[np.ndarray] = None
            convergence_map: Optional[np.ndarray] = None

            _, ax = plt.subplots(2, 2)

            while True:
                # TODO
                colour_image: np.ndarray = drone.get_image()
                cv2.imshow("Image", colour_image)
                cv2.waitKey(1)

                # TODO
                if not tracker.is_ready():
                    continue
                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)
                if pose is None:
                    continue

                # TODO
                depth_assembler.put(colour_image, pose)

                # TODO
                if is_keyframe:
                    reference_colour_image = colour_image
                    is_keyframe = False

                # TODO
                result = depth_assembler.get(blocking=False)
                if result is not None:
                    _, estimated_depth_image, _, converged_percentage, convergence_map = result
                    print(f"Converged %: {converged_percentage}")

                # TODO
                ax[0, 0].clear()
                ax[0, 1].clear()
                ax[1, 0].clear()
                ax[1, 1].clear()
                ax[0, 0].imshow(reference_colour_image[:, :, [2, 1, 0]])
                if estimated_depth_image is not None:
                    ax[0, 1].imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
                ax[1, 0].imshow(colour_image[:, :, [2, 1, 0]])
                # ax[1, 1].imshow(reference_depth_image, vmin=0.0, vmax=4.0)

                plt.draw()
                if plt.waitforbuttonpress(0.001):
                    break

            # TODO
            cv2.destroyAllWindows()

            # TODO
            estimated_depth_image = DepthProcessor.denoise_depth(estimated_depth_image, convergence_map, intrinsics)
            estimated_depth_image, _ = DepthProcessor.densify_depth_image(estimated_depth_image)

            plt.imshow(estimated_depth_image, vmin=0.0, vmax=4.0)
            plt.waitforbuttonpress()

            # TODO
            VisualisationUtil.visualise_rgbd_image(reference_colour_image, estimated_depth_image, intrinsics)


if __name__ == "__main__":
    main()
