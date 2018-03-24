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

#include "lsst/utils/Cache.h"
#include "lsst/afw/table/io/python.h"
#include "lsst/meas/algorithms/ImagePsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace meas {
namespace algorithms {

PYBIND11_PLUGIN(imagePsf) {
    py::module::import("lsst.afw.detection");
    py::module mod("imagePsf");

    afw::table::io::python::declarePersistableFacade<ImagePsf>(mod, "ImagePsf");

    py::class_<ImagePsf, std::shared_ptr<ImagePsf>, afw::table::io::PersistableFacade<ImagePsf>,
               afw::detection::Psf>
            clsImagePsf(mod, "ImagePsf");

    return mod.ptr();
}

}  // algorithms
}  // meas
}  // lsst
