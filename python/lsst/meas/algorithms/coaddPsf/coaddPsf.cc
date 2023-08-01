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

#include "lsst/afw/table/io/python.h"
#include "lsst/meas/algorithms/CoaddPsf.h"
#include "lsst/pex/config/python.h"  // for LSST_DECLARE_CONTROL_FIELD

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace algorithms {
namespace {

void declareCoaddPsf(lsst::cpputils::python::WrapperCollection &wrappers) {
    /* CoaddPsfControl */
    using PyCoaddPsfControl = py::class_<CoaddPsfControl>;

    wrappers.wrapType(PyCoaddPsfControl(wrappers.module, "CoaddPsfControl"), [](auto &mod, auto &cls) {
        cls.def(py::init<std::string, int>(), "warpingKernelName"_a = "lanczos3", "cacheSize"_a = 10000);
        LSST_DECLARE_CONTROL_FIELD(cls, CoaddPsfControl, warpingKernelName);
        LSST_DECLARE_CONTROL_FIELD(cls, CoaddPsfControl, cacheSize);
    });

    /* CoaddPsf */
    using PyCoaddPsf = py::class_<CoaddPsf, ImagePsf>;

    auto clsCoaddPsf = wrappers.wrapType(PyCoaddPsf(wrappers.module, "CoaddPsf"), [](auto &mod, auto &cls) {
        /* Constructors */
        cls.def(py::init<afw::table::ExposureCatalog const &, afw::geom::SkyWcs const &,
                                std::string const &, std::string const &, int>(),
                        "catalog"_a, "coaddWcs"_a, "weightFieldName"_a = "weight",
                        "warpingKernelName"_a = "lanczos3", "cacheSize"_a = 10000);
        cls.def(py::init<afw::table::ExposureCatalog const &, afw::geom::SkyWcs const &,
                                geom::Point2D const &, std::string const &, int>(),
                        "catalog"_a, "coaddWcs"_a, "averagePosition"_a,
                        "warpingKernelName"_a = "lanczos3", "cacheSize"_a = 10000);
        cls.def(py::init<afw::table::ExposureCatalog const &, afw::geom::SkyWcs const &,
                                CoaddPsfControl const &, std::string const &>(),
                        "catalog"_a, "coaddWcs"_a, "ctrl"_a, "weightFieldName"_a = "weight");

        /* Members */
        cls.def("clone", &CoaddPsf::clone);
        cls.def("getAveragePosition", &CoaddPsf::getAveragePosition);
        cls.def("getCoaddWcs", &CoaddPsf::getCoaddWcs);
        cls.def("getComponentCount", &CoaddPsf::getComponentCount);
        cls.def("getPsf", &CoaddPsf::getPsf);
        cls.def("getWcs", &CoaddPsf::getWcs);
        cls.def("getWeight", &CoaddPsf::getWeight);
        cls.def("getId", &CoaddPsf::getId);
        cls.def("getBBox", &CoaddPsf::getBBox);
        cls.def("getValidPolygon", &CoaddPsf::getValidPolygon);
        cls.def("isPersistable", &CoaddPsf::isPersistable);
    });

    afw::table::io::python::addPersistableMethods<CoaddPsf>(clsCoaddPsf);
}

}  // namespace
void wrapCoaddPsf(lsst::cpputils::python::WrapperCollection &wrappers)
{
    declareCoaddPsf(wrappers);
}
}  // namespace algorithms
}  // namespace meas
}  // namespace lsst
