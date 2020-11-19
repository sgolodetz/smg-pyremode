import numpy as np
import open3d as o3d
import os

from typing import Optional, Tuple

from smg.openni import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.pyremode import DepthEstimator, RGBDMappingSystem, RGBDOpenNICamera, TemporalKeyframeDepthEstimator


def add_axis(vis: o3d.visualization.Visualizer, pose: np.ndarray, *,
             colour: Optional[Tuple[float, float, float]] = None, size: float = 1.0) -> None:
    """
    Add to the specified Open3D visualisation a set of axes for the specified pose.

    :param vis:     The Open3D visualisation.
    :param pose:    The pose (specified in camera space).
    :param colour:  An optional colour with which to paint the axes.
    :param size:    The size to give the axes (defaults to 1).
    """
    # FIXME: This function is duplicated in several places.
    # noinspection PyArgumentList
    axes: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if colour is not None:
        axes.paint_uniform_color(colour)
    axes.transform(pose)
    # noinspection PyTypeChecker
    vis.add_geometry(axes)


def make_mesh(tsdf: o3d.pipelines.integration.ScalableTSDFVolume, *, print_progress: bool = False) \
        -> o3d.geometry.TriangleMesh:
    """
    Make a triangle mesh from the specified TSDF.

    :param tsdf:            The TSDF.
    :param print_progress:  Whether or not to print out progress messages.
    :return:                The triangle mesh.
    """
    if print_progress:
        print("Extracting a triangle mesh from the TSDF")

    mesh: o3d.geometry.TriangleMesh = tsdf.extract_triangle_mesh()

    if print_progress:
        print("Computing vertex normals for the mesh")

    mesh.compute_vertex_normals()

    return mesh


def visualise_mesh(mesh: o3d.geometry.TriangleMesh):
    # Set up the visualisation.
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    render_option: o3d.visualization.RenderOption = vis.get_render_option()
    render_option.line_width = 10

    # noinspection PyTypeChecker
    vis.add_geometry(mesh)
    add_axis(vis, np.eye(4), size=0.1)

    # Set the initial pose for the visualiser.
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    m = np.eye(4)
    params.extrinsic = m
    vis.get_view_control().convert_from_pinhole_camera_parameters(params)

    # Run the visualiser.
    vis.run()


def main():
    # noinspection PyUnusedLocal
    tsdf: Optional[o3d.pipelines.integration.ScalableTSDFVolume] = None

    with OpenNICamera(mirror_images=True) as camera:
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            depth_estimator: DepthEstimator = TemporalKeyframeDepthEstimator(
                camera.get_colour_dims(), camera.get_colour_intrinsics()
            )
            with RGBDMappingSystem(RGBDOpenNICamera(camera), tracker, depth_estimator) as system:
                tsdf = system.run()

            # If ORB-SLAM's not ready yet, forcibly terminate the whole process (this isn't graceful, but
            # if we don't do it then we may have to wait a very long time for it to finish initialising).
            if not tracker.is_ready():
                # noinspection PyProtectedMember
                os._exit(0)

    visualise_mesh(make_mesh(tsdf, print_progress=True))


if __name__ == "__main__":
    main()
