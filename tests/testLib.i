// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
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
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
%define testLib_DOCSTRING
"
Various swigged-up C++ classes for testing
"
%enddef

%feature("autodoc", "1");
%module(package="testLib", docstring=testLib_DOCSTRING) testLib

%pythonnondynamic;
%naturalvar;  // use const reference typemaps

%include "lsst/p_lsstSwig.i"

%lsst_exceptions()

%{
#include "lsst/pex/logging.h"
#include "lsst/afw.h"
#include "lsst/meas/algorithms/ShapeletKernel.h"
#include "lsst/meas/algorithms/ShapeletPsf.h"
#include "lsst/meas/algorithms/ShapeletPsfCandidate.h"
#include "lsst/meas/algorithms/PsfCandidate.h"
#include "lsst/meas/algorithms/Interp.h"
#include "lsst/meas/algorithms/CentroidControl.h"
#include "lsst/meas/algorithms/ShapeControl.h"
#include "lsst/meas/algorithms/FluxControl.h"
#include "lsst/meas/algorithms/Classification.h"
#include "lsst/meas/algorithms/PixelFlags.h"
%}

%import "lsst/meas/algorithms/algorithmsLib.i"

%shared_ptr(lsst::meas::algorithms::SillyCentroidControl)

%inline %{
#include "sillyCentroid.h"
%}

namespace lsst { namespace meas { namespace algorithms {
class SillyCentroidControl : public CentroidControl {
public:
    SillyCentroidControl();
private:
    virtual PTR(AlgorithmControl) _clone() const;
    virtual PTR(Algorithm) _makeAlgorithm(
        afw::table::Schema & schema,         
        PTR(daf::base::PropertyList) const & metadata
    ) const;
};
}}}
