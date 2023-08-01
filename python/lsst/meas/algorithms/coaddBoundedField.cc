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
#include "lsst/afw/table/io/python.h"
#include "lsst/meas/algorithms/CoaddBoundedField.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace algorithms {
namespace {

void declareCoaddBoundedField(lsst::cpputils::python::WrapperCollection &wrappers) {
    using PyElement = py::class_<CoaddBoundedFieldElement>;
    static auto clsElement = wrappers.wrapType(PyElement(wrappers.module, "CoaddBoundedFieldElement"), [](auto &mod, auto &cls) {
        cls.def(py::init([](std::shared_ptr<afw::math::BoundedField> field,
                            std::shared_ptr<afw::geom::SkyWcs const> wcs, py::object polygon, double weight) {
                    if (polygon.is(py::none())) {
                        return new CoaddBoundedFieldElement(field, wcs, nullptr, weight);
                    } else {
                        auto pgon = py::cast<std::shared_ptr<afw::geom::polygon::Polygon const>>(polygon);
                        return new CoaddBoundedFieldElement(field, wcs, pgon, weight);
                    }
                }),
                "field"_a, "wcs"_a, "validPolygon"_a, "weight"_a = 1.0);
        cls.def_readwrite("field", &CoaddBoundedFieldElement::field);
        cls.def_readwrite("wcs", &CoaddBoundedFieldElement::wcs);
        cls.def_readwrite("validPolygon", &CoaddBoundedFieldElement::validPolygon);
        cls.def_readwrite("weight", &CoaddBoundedFieldElement::weight);

        cls.def("__eq__", &CoaddBoundedFieldElement::operator==, py::is_operator());
        cls.def("__ne__", &CoaddBoundedFieldElement::operator!=, py::is_operator());
    });
    using PyClass = py::class_<CoaddBoundedField, afw::math::BoundedField>;
    auto clsField = wrappers.wrapType(PyClass(wrappers.module, "CoaddBoundedField"), [](auto &mod, auto &cls) {

        cls.attr("Element") = clsElement;

        /* Constructors */
        cls.def(py::init<geom::Box2I const &, std::shared_ptr<afw::geom::SkyWcs const>,
                                         typename CoaddBoundedField::ElementVector const &>(),
                                 "bbox"_a, "coaddWcs"_a, "elements"_a);
        cls.def(py::init<geom::Box2I const &, std::shared_ptr<afw::geom::SkyWcs const>,
                                         typename CoaddBoundedField::ElementVector const &, double>(),
                                 "bbox"_a, "coaddWcs"_a, "elements"_a, "default"_a);

        /* Operators */
        cls.def("__eq__", &CoaddBoundedField::operator==, py::is_operator());
        cls.def("__ne__", &CoaddBoundedField::operator!=, py::is_operator());
        cls.def("__imul__", &CoaddBoundedField::operator*);

        /* Members */
        cls.def("evaluate", &CoaddBoundedField::evaluate);
        cls.def("getCoaddWcs", &CoaddBoundedField::getCoaddWcs);
        cls.def("getDefault", &CoaddBoundedField::getDefault);
        cls.def("getElements", &CoaddBoundedField::getElements);
    });
    afw::table::io::python::addPersistableMethods<CoaddBoundedField>(clsField);
}

}  // namespace
void wrapCoaddBoundedField(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareCoaddBoundedField(wrappers);
}
}  // namespace algorithms
}  // namespace meas
}  // namespace lsst
