import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation
from typing import Optional

from smg.pyopencv import CVMat1b
from smg.pyorbslam2 import MonocularTracker
from smg.pyremode import *
from smg.rotory.drone_factory import DroneFactory


def print_se3(se3: SE3f) -> None:
    print()
    for row in range(3):
        print([se3.data(row, col) for col in range(4)])
    print()


def main():
    with DroneFactory.make_drone("tello", local_ip="192.168.10.3") as drone:
        with MonocularTracker(
            settings_file=f"settings-tello.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            # fx, fy, cx, cy = 946.60441222, 941.38386885, 460.29254907, 357.08431882
            fx, fy, cx, cy = 938.55289501, 932.86950291, 480.0, 360.0
            depthmap: Depthmap = Depthmap(960, 720, fx, cx, fy, cy)
            reference_colour_image: Optional[np.ndarray] = None

            _, ax = plt.subplots(2, 2)

            while True:
                distorted_image = drone.get_image()

                # camera_matrix: np.ndarray = np.array([
                #     [946.60441222, 0., 460.29254907],
                #     [0., 941.38386885, 357.08431882],
                #     [0., 0., 1.]
                # ])
                # dist_coeffs: np.ndarray = np.array([[0.04968041, -0.59998154, -0.00377696, -0.00863985, 2.14472665]])
                camera_matrix: np.ndarray = np.array([
                    [938.55289501, 0., 480.],
                    [0., 932.86950291, 360.],
                    [0., 0., 1.]
                ])
                dist_coeffs: np.ndarray = np.array([[0.03306774, -0.40497806, -0.00216106, -0.00294729, 1.31711308]])
                colour_image = cv2.undistort(distorted_image, camera_matrix, dist_coeffs)

                cv2.imshow("Image", colour_image)
                cv2.waitKey(1)

                if not tracker.is_ready():
                    continue
                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)
                if pose is None:
                    continue

                r: Rotation = Rotation.from_matrix(pose[0:3, 0:3])
                t: np.ndarray = pose[0:3, 3]
                qx, qy, qz, qw = r.as_quat()
                se3: SE3f = SE3f(qw, qx, qy, qz, *t)

                # print_se3(se3)

                grey_image: np.ndarray = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
                cv_grey_image: CVMat1b = CVMat1b.zeros(*grey_image.shape[:2])
                np.copyto(np.array(cv_grey_image, copy=False), grey_image)

                if reference_colour_image is None:
                    reference_colour_image = colour_image
                    depthmap.set_reference_image(cv_grey_image, se3, 0.1, 4.0)
                else:
                    depthmap.update(cv_grey_image, se3)

                estimated_depth_image: np.ndarray = np.array(depthmap.get_denoised_depthmap())

                print(depthmap.get_converged_percentage())

                ax[0, 0].clear()
                ax[0, 1].clear()
                ax[1, 0].clear()
                ax[1, 1].clear()
                ax[0, 0].imshow(reference_colour_image)
                ax[0, 1].imshow(estimated_depth_image)  # , vmin=0.0, vmax=4.0)
                ax[1, 0].imshow(colour_image)
                # ax[1, 1].imshow(reference_depth_image, vmin=0.0, vmax=4.0)

                plt.draw()
                plt.waitforbuttonpress(0.001)


if __name__ == "__main__":
    main()
