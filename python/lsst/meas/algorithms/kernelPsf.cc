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

#include "lsst/geom/Point.h"
#include "lsst/afw/table/io/python.h"
#include "lsst/meas/algorithms/KernelPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace algorithms {
namespace {
void declareKernelPsf(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyKernelPsf = py::class_<KernelPsf, ImagePsf>;

    auto clsKernelPsf = wrappers.wrapType(PyKernelPsf(wrappers.module, "KernelPsf"), [](auto &mod, auto &cls) {
        cls.def(py::init<afw::math::Kernel const &, geom::Point2D const &>(), "kernel"_a,
                "averagePosition"_a = geom::Point2D());

        cls.def("getKernel", &KernelPsf::getKernel);
        cls.def("getAveragePosition", &KernelPsf::getAveragePosition);
        cls.def("clone", &KernelPsf::clone);
    });
    afw::table::io::python::addPersistableMethods<KernelPsf>(clsKernelPsf);
}
}  // namespace

void wrapKernelPsf(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareKernelPsf(wrappers);
}
}  // namespace algorithms
}  // namespace meas
}  // namespace lsst
