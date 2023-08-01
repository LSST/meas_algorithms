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

#include "lsst/meas/algorithms/PsfCandidate.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace algorithms {
namespace {

template <typename PixelT>
void declarePsfCandidate(lsst::cpputils::python::WrapperCollection &wrappers, std::string const& suffix) {
    using Class = PsfCandidate<PixelT>;
    using PyClass = py::class_<Class, afw::math::SpatialCellImageCandidate>;
    std::string name = "PsfCandidate" + suffix;

    wrappers.wrapType(PyClass(wrappers.module, name.c_str()), [](auto &mod, auto &cls) {
        cls.def(py::init<std::shared_ptr<afw::table::SourceRecord> const &,
                        std::shared_ptr<afw::image::Exposure<PixelT> const>>(),
                "source"_a, "parentExposure"_a);
        cls.def(py::init<std::shared_ptr<afw::table::SourceRecord> const &,
                        std::shared_ptr<afw::image::Exposure<PixelT> const>, double, double>(),
                "source"_a, "parentExposure"_a, "xCenter"_a, "yCenter"_a);

        /* SpatialCellCandidate.getCandidateRating is defined in Python.
         * Therefore we cannot override it from the C++ wrapper.
         * Instead we give it a temporary name here, and assign it to the
         * class from Python. */
        cls.def("_getCandidateRating", &Class::getCandidateRating);
        cls.def("getSource", &Class::getSource);
        cls.def("getAmplitude", &Class::getAmplitude);
        cls.def("setAmplitude", &Class::setAmplitude);
        cls.def("getVar", &Class::getVar);
        cls.def("setVar", &Class::setVar);
        cls.def("getMaskedImage", (std::shared_ptr<afw::image::MaskedImage<PixelT> const>(Class::*)() const) &
                Class::getMaskedImage);
        cls.def("getMaskedImage",
                (std::shared_ptr<afw::image::MaskedImage<PixelT> const>(Class::*)(int, int) const) &
                        Class::getMaskedImage,
                "width"_a, "height"_a);
        cls.def("getOffsetImage", &Class::getOffsetImage);
        cls.def_static("getBorderWidth", &Class::getBorderWidth);
        cls.def_static("setBorderWidth", &Class::setBorderWidth);
        cls.def_static("setPixelThreshold", &Class::setPixelThreshold);
        cls.def_static("getPixelThreshold", &Class::getPixelThreshold);
        cls.def_static("setMaskBlends", &Class::setMaskBlends);
        cls.def_static("getMaskBlends", &Class::getMaskBlends);

        mod.def("makePsfCandidate", makePsfCandidate<PixelT>, "source"_a, "image"_a);
    });
}

}  // namespace

void wapPsfCandidate(lsst::cpputils::python::WrapperCollection &wrappers) {
    declarePsfCandidate<float>(wrappers, "F");
}

}  // namespace algorithms
}  // namespace meas
}  // namespace lsst
