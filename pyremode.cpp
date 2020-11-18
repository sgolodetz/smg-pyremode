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
