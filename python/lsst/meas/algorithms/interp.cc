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
#include "pybind11/stl.h"

#include "lsst/geom/Box.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/meas/algorithms/Interp.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace algorithms {
namespace {

template <typename PixelT>
void declareInterpolateOverDefects(py::module& mod) {
    mod.def("interpolateOverDefects",
            interpolateOverDefects<
                    afw::image::MaskedImage<PixelT, afw::image::MaskPixel, afw::image::VariancePixel>>,
            "image"_a, "psf"_a, "badList"_a, "fallBackValue"_a = 0.0, "useFallbackValueAtEdge"_a = false);
}

void declareInterp(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyDefect =  py::class_<Defect, afw::image::DefectBase>;
    auto clsDefect = wrappers.wrapType(PyDefect(wrappers.module, "Defect"), [](auto &mod, auto &cls) {
        cls.def(py::init<const geom::BoxI &>(), "bbox"_a = geom::BoxI());

        cls.def("classify", &Defect::classify);
        cls.def("getType", &Defect::getType);
        cls.def("getPos", &Defect::getPos);
    });
    wrappers.wrapType(py::enum_<Defect::DefectPosition>(clsDefect, "DefectPosition"), [](auto &mod, auto &enm) {
        enm.value("LEFT", Defect::DefectPosition::LEFT);
        enm.value("NEAR_LEFT", Defect::DefectPosition::NEAR_LEFT);
        enm.value("WIDE_LEFT", Defect::DefectPosition::WIDE_LEFT);
        enm.value("MIDDLE", Defect::DefectPosition::MIDDLE);
        enm.value("WIDE_NEAR_LEFT", Defect::DefectPosition::WIDE_NEAR_LEFT);
        enm.value("WIDE", Defect::DefectPosition::WIDE);
        enm.value("WIDE_NEAR_RIGHT", Defect::DefectPosition::WIDE_NEAR_RIGHT);
        enm.value("NEAR_RIGHT", Defect::DefectPosition::NEAR_RIGHT);
        enm.value("WIDE_RIGHT", Defect::DefectPosition::WIDE_RIGHT);
        enm.value("RIGHT", Defect::DefectPosition::RIGHT);
        enm.export_values();
    });
    declareInterpolateOverDefects<float>(wrappers.module);
}

}  // namespace
void wrapInterp(lsst::cpputils::python::WrapperCollection &wrappers)
{
    declareInterp(wrappers);
}
}  // namespace algorithms
}  // namespace meas
}  // namespace lsst
