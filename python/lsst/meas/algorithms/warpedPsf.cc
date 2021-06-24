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

#include "lsst/utils/python/PySharedPtr.h"

#include "lsst/meas/algorithms/WarpedPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

using lsst::utils::python::PySharedPtr;

namespace lsst {
namespace meas {
namespace algorithms {
namespace {

PYBIND11_MODULE(warpedPsf, mod) {
    py::class_<WarpedPsf, PySharedPtr<WarpedPsf>, ImagePsf> clsWarpedPsf(mod, "WarpedPsf", py::is_final());

    /* Constructors */
    clsWarpedPsf.def(py::init<std::shared_ptr<afw::detection::Psf const>,
                              std::shared_ptr<afw::geom::TransformPoint2ToPoint2 const>,
                              std::shared_ptr<afw::math::WarpingControl const>>(),
                     "undistortedPsf"_a, "distortion"_a, "control"_a);
    clsWarpedPsf.def(py::init<std::shared_ptr<afw::detection::Psf const>,
                              std::shared_ptr<afw::geom::TransformPoint2ToPoint2 const>, std::string const &,
                              unsigned int>(),
                     "undistortedPsf"_a, "distortion"_a, "kernelName"_a = "lanczos3", "cache"_a = 10000);

    /* Members */
    clsWarpedPsf.def("getAveragePosition", &WarpedPsf::getAveragePosition);
    clsWarpedPsf.def("clone", &WarpedPsf::clone);
}

}  // namespace
}  // namespace algorithms
}  // namespace meas
}  // namespace lsst
