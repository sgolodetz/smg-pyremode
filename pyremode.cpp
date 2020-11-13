#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include <Python.h>

#include <rmd/depthmap.h>
#include <rmd/pinhole_camera.cuh>

PYBIND11_MODULE(pyremode, m)
{
  // CLASSES

  py::class_<cv::Mat1b>(m, "CVMat1b", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1b& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(unsigned char),
        pybind11::format_descriptor<unsigned char>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(unsigned char) * img.cols, sizeof(unsigned char) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1b {
      return cv::Mat1b::zeros(rows, cols);
    }, py::call_guard<py::gil_scoped_release>())
  ;

  py::class_<cv::Mat1f>(m, "CVMat1f", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1f& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(float),
        pybind11::format_descriptor<float>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(float) * img.cols, sizeof(float) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1f { return cv::Mat1f::zeros(rows, cols); }, "")
  ;

  py::class_<cv::Mat1i>(m, "CVMat1i", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1i& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(int),
        pybind11::format_descriptor<int>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(int) * img.cols, sizeof(int) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1i { return cv::Mat1i::zeros(rows, cols); }, "")
  ;

  py::class_<cv::Mat3b>(m, "CVMat3b", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat3b& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(unsigned char),
        pybind11::format_descriptor<unsigned char>::format(),
        3,
        { img.rows, img.cols, 3 },
        { sizeof(unsigned char) * 3 * img.cols, sizeof(unsigned char) * 3, sizeof(unsigned char) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat3b { return cv::Mat3b::zeros(rows, cols); }, "")
  ;

  py::class_<rmd::Depthmap>(m, "Depthmap")
    .def(py::init<size_t, size_t, float, float, float, float>(), py::call_guard<py::gil_scoped_release>())
    .def("get_converged_count", &rmd::Depthmap::getConvergedCount, py::call_guard<py::gil_scoped_release>())
    .def("get_converged_percentage", &rmd::Depthmap::getConvergedPercentage, py::call_guard<py::gil_scoped_release>())
    .def("get_convergence_map", [](rmd::Depthmap& self) -> cv::Mat1i {
      self.downloadConvergenceMap();
      return self.getConvergenceMap();
    }, py::call_guard<py::gil_scoped_release>())
    .def("get_denoised_depthmap", [](rmd::Depthmap& self, float lambda, int iterations) -> cv::Mat1f {
      self.downloadDenoisedDepthmap(lambda, iterations);
      return self.getDepthmap();
    }, py::arg("lambda") = 0.5f, py::arg("iterations") = 200, py::call_guard<py::gil_scoped_release>())
    .def("get_depthmap", [](rmd::Depthmap& self) -> cv::Mat1f {
      self.downloadDepthmap();
      return self.getDepthmap();
    }, py::call_guard<py::gil_scoped_release>())
    .def("set_reference_image", [](rmd::Depthmap& self, const cv::Mat1b& img_curr, const rmd::SE3<float>& T_curr_world, float min_depth, float max_depth) -> bool {
      return self.setReferenceImage(img_curr, T_curr_world, min_depth, max_depth);
    }, py::call_guard<py::gil_scoped_release>())
    .def("update", [](rmd::Depthmap& self, const cv::Mat1b& img_curr, const rmd::SE3<float>& T_curr_world) {
      self.update(img_curr, T_curr_world);
    }, py::call_guard<py::gil_scoped_release>())
    .def_static("scale_mat", [](const cv::Mat1f& depthmap) -> cv::Mat3b {
      return rmd::Depthmap::scaleMat(depthmap);
    }, py::call_guard<py::gil_scoped_release>())
  ;

  py::class_<rmd::PinholeCamera>(m, "PinholeCamera")
    .def(
      py::init<float, float, float, float>(),
      py::arg("fx") = 0.0f, py::arg("fy") = 0.0f, py::arg("cx") = 0.0f, py::arg("cy") = 0.0f,
      py::call_guard<py::gil_scoped_release>()
    )
    .def_readonly("fx", &rmd::PinholeCamera::fx)
    .def_readonly("fy", &rmd::PinholeCamera::fy)
    .def_readonly("cx", &rmd::PinholeCamera::cx)
    .def_readonly("cy", &rmd::PinholeCamera::cy)
  ;

  py::class_<rmd::SE3<float>>(m, "SE3f")
    .def(
      py::init<float, float, float, float, float, float, float>(),
      py::call_guard<py::gil_scoped_release>()
    )
    .def("data", [](const rmd::SE3<float>& self, int row, int col) -> float { return self.data(row, col); })
    .def("inv", &rmd::SE3<float>::inv, py::call_guard<py::gil_scoped_release>())
  ;

  // ENUMERATIONS

  py::enum_<rmd::ConvergenceState>(m, "EConvergenceState")
    .value("UPDATE", rmd::ConvergenceState::UPDATE)
    .value("CONVERGED", rmd::ConvergenceState::CONVERGED)
    .value("BORDER", rmd::ConvergenceState::BORDER)
    .value("DIVERGED", rmd::ConvergenceState::DIVERGED)
    .value("NO_MATCH", rmd::ConvergenceState::NO_MATCH)
    .value("NOT_VISIBLE", rmd::ConvergenceState::NOT_VISIBLE)
    .export_values()
  ;
}

#if 0
#pragma warning(disable:4244 4267 4996)
#include "System.h"
#pragma warning(default:4244 4267 4996)
using namespace ORB_SLAM2;

PYBIND11_MODULE(pyorbslam2, m)
{
  // CLASSES

  py::class_<cv::Mat1d>(m, "CVMat1d", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1d& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(double),
        pybind11::format_descriptor<double>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(double) * img.cols, sizeof(double) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1d { return cv::Mat1d::zeros(rows, cols); }, "")
  ;

  py::class_<cv::Mat1f>(m, "CVMat1f", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat1f& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(float),
        pybind11::format_descriptor<float>::format(),
        2,
        { img.rows, img.cols },
        { sizeof(float) * img.cols, sizeof(float) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat1f { return cv::Mat1f::zeros(rows, cols); }, "")
  ;

  py::class_<cv::Mat3b>(m, "CVMat3b", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat3b& img) -> pybind11::buffer_info {
      return pybind11::buffer_info(
        img.data,
        sizeof(unsigned char),
        pybind11::format_descriptor<unsigned char>::format(),
        3,
        { img.rows, img.cols, 3 },
        { sizeof(unsigned char) * 3 * img.cols, sizeof(unsigned char) * 3, sizeof(unsigned char) }
      );
    })
    .def_static("zeros", [](int rows, int cols) -> cv::Mat3b { return cv::Mat3b::zeros(rows, cols); }, "")
  ;

  py::class_<System>(m, "System")
    .def(py::init<std::string, std::string, System::eSensor, bool>(), py::call_guard<py::gil_scoped_release>())
    .def("track_monocular", [](System& self, const cv::Mat3b& im, float timestamp) -> cv::Mat1d {
      return self.TrackMonocular(im, timestamp);
    }, py::call_guard<py::gil_scoped_release>())
    .def("track_rgbd", [](System& self, const cv::Mat3b& im, const cv::Mat1f& depthmap, float timestamp) -> cv::Mat1d {
      return self.TrackRGBD(im, depthmap, timestamp);
    }, py::call_guard<py::gil_scoped_release>())
  ;

  // ENUMERATIONS

  py::enum_<System::eSensor>(m, "ESensor")
    .value("MONOCULAR", System::MONOCULAR)
    .value("STEREO", System::STEREO)
    .value("RGBD", System::RGBD)
    .export_values()
  ;
}
#endif
