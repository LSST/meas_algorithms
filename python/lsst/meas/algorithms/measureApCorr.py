#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import lsst.pex.config
import lsst.afw.image
import lsst.pipe.base

class KeyTuple(object):

    __slots__ = "flux", "err", "flag"

    def __init__(self, name, schema, index=None):
        self.flux = schema.find(name).key
        self.err = schema.find(name + ".err").key
        self.flag = schema.find(name + ".flags").key

class MeasureApCorrConfig(lsst.pex.config.Config):
    reference = lsst.pex.config.Field(
        dtype=str, default="flux.sinc",
        doc="Name of the flux field other measurements should be corrected to match"
    )
    toCorrect = lsst.pex.config.ListField(
        dtype=str, default=[],
        doc="Names of flux fields to correct to match the reference flux"
    )
    inputFilterFlag = lsst.pex.config.Field(
        dtype=str, default="calib.psf.used",
        doc=("Name of a flag field that indicates that a source should be used to constrain the"
             " aperture corrections")
    )
    minDegreesOfFreedom = lsst.pex.config.Field(
        dtype=int, default=1,
        doc=("Minimum number of degrees of freedom (# of valid data points - # of parameters);"
             " if this is exceeded, the order of the fit is decreased (in both dimensions), and"
             " if we can't decrease it enough, we'll raise ValueError.")
        )
    fit = lsst.pex.config.ConfigField(
        dtype=lsst.afw.math.ChebyshevBoundedFieldConfig,
        doc="Configuration used in fitting the aperture correction fields"
    )

class MeasureApCorrTask(lsst.pipe.base.Task):

    ConfigClass = MeasureApCorrConfig
    _DefaultName = "measureApCorr"

    def __init__(self, schema, **kwds):
        lsst.pipe.base.Task.__init__(self, **kwds):
        self.reference = KeyTuple(self.config.referenceFlux, schema)
        self.toCorrect = {name: KeyTuple(name, schema) for name in self.config.toCorrect}
        self.inputFilterFlag = schema.find(self.config.inputFilterFlag).key

    def run(self, bbox, catalog):

        # First, create a subset of the catalog that contains only objects with inputFilterFlag set
        # and non-flagged reference fluxes.
        subset1 = [record for record in catalog
                   if record.get(self.inputFilterFlag) and not record.get(self.reference.flag)]

        apCorrMap = lsst.afw.image.ApCorrMap()

        # Outer loop over the fields we want to correct
        for name, keys in self.toCorrect.iteritems():

            # Create a more restricted subset with only the objects where the to-be-correct flux
            # is not flagged.
            subset2 = [record for record in subset1 if not record.get(keys.flag)]

            # Check that we have enough data points to at least fit a 0th-order (constant) model
            if len(subset2) - 1 < self.config.minDegreesOfFreedom:
                raise ValueError("Only %d sources for calculation of aperture correction for '%s'"
                                 % (len(subset2), name,))

            # If we don't have enough data points to constrain the fit, reduce the order until we do
            ctrl = self.config.fit.makeControl()
            while n - ctrl.computeSize() < self.config.minDegreesOfFreedom:
                if ctrl.orderX > 0:
                    ctrl.orderX -= 1
                if ctrl.orderY > 0:
                    ctrl.orderY -= 1

            # Fill numpy arrays with positions and the ratio of the reference flux to the to-correct flux
            x = numpy.zeros(subset2.size, dtype=float)
            y = numpy.zeros(subset2.size, dtype=float)
            apCorrData = numpy.zeros(subset2.size, dtype=float)
            for n, record in enumerate(subset2):
                x[n] = record.getX()
                y[n] = record.getY()
                apCorrData[n] = record.get(self.reference.flux)/record.get(keys.flux)

            # Do the fit, save it in the output map
            apCorrField = lsst.afw.math.ChebyshevBoundedField.fit(bbox, x, y, apCorrData, ctrl)
            apCorrMap[name] = apCorrField

            # Compute errors empirically, using the RMS difference between the true reference flux and the
            # corrected to-be-corrected flux.  The error is constant spatially (we could imagine being
            # more clever, but we're not yet sure if it's worth the effort).
            # We save the errors as a 0th-order ChebyshevBoundedField
            apCorrDiffs = numpy.zeros(subset2.size, dtype=float)
            apCorrField.evaluate(x, y, apCorrDiffs)
            apCorrDiffs -= apCorrData
            apCorrErr = numpy.mean(apCorrDiffs**2)**0.5
            apCorrErrCoefficients = numpy.array([[apCorrErr]], dtype=float)
            apCorrMap[name + ".err"] = lsst.afw.math.ChebyshevBoundedField(bbox, apCorrErrCoefficients)

        return apCorrMap
