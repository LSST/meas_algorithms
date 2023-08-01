/*
 * LSST Data Management System
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 * See the COPYRIGHT file
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */
#include "pybind11/pybind11.h"
#include "lsst/cpputils/python.h"

#include "lsst/meas/algorithms/WarpedPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace algorithms {
namespace {

void declareWarpedPsf(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyWarpedPsf = py::class_<WarpedPsf, ImagePsf>;
    wrappers.wrapType(PyWarpedPsf(wrappers.module, "WarpedPsf", py::is_final()), [](auto &mod, auto &cls) {
        /* Constructors */
        cls.def(py::init<std::shared_ptr<afw::detection::Psf const>,
                                 std::shared_ptr<afw::geom::TransformPoint2ToPoint2 const>,
                                 std::shared_ptr<afw::math::WarpingControl const>>(),
                         "undistortedPsf"_a, "distortion"_a, "control"_a);
        cls.def(py::init<std::shared_ptr<afw::detection::Psf const>,
                                 std::shared_ptr<afw::geom::TransformPoint2ToPoint2 const>, std::string const &,
                                 unsigned int>(),
                         "undistortedPsf"_a, "distortion"_a, "kernelName"_a = "lanczos3", "cache"_a = 10000);

        /* Members */
        cls.def("getAveragePosition", &WarpedPsf::getAveragePosition);
        cls.def("clone", &WarpedPsf::clone);
    });
}
}  // namespace
void wrapWarpedPsf(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareWarpedPsf(wrappers);
}
}  // namespace algorithms
}  // namespace meas
}  // namespace lsst
