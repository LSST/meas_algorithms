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

#include "lsst/afw/table/io/python.h"
#include "lsst/meas/algorithms/SingleGaussianPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace algorithms {
namespace {

PYBIND11_MODULE(singleGaussianPsf, mod) {
    py::class_<SingleGaussianPsf, std::shared_ptr<SingleGaussianPsf>, KernelPsf> clsSingleGaussianPsf(
            mod, "SingleGaussianPsf");
    afw::table::io::python::addPersistableMethods<SingleGaussianPsf>(clsSingleGaussianPsf);

    clsSingleGaussianPsf.def(py::init<int, int, double>(), "width"_a, "height"_a, "sigma"_a);

    clsSingleGaussianPsf.def("clone", &SingleGaussianPsf::clone);
    clsSingleGaussianPsf.def("resized", &SingleGaussianPsf::resized, "width"_a, "height"_a);
    clsSingleGaussianPsf.def("getSigma", &SingleGaussianPsf::getSigma);
}

}  // namespace
}  // namespace algorithms
}  // namespace meas
}  // namespace lsst
