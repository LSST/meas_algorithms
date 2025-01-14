#
# LSST Data Management System
#
# Copyright 2008-2016  AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import unittest
import numpy as np
import logging

import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.afw.math import ChebyshevBoundedField
import lsst.pex.config
import lsst.meas.algorithms.measureApCorr as measureApCorr
from lsst.meas.base.apCorrRegistry import addApCorrName
import lsst.meas.base.tests
import lsst.utils.tests


def apCorrDefaultMap(value=None, bbox=None):
    default_coefficients = np.ones((1, 1), dtype=float)
    default_coefficients /= value
    default_apCorrMap = ChebyshevBoundedField(bbox, default_coefficients)
    default_fill = afwImage.ImageF(bbox)
    default_apCorrMap.fillImage(default_fill)
    return default_fill


class MeasureApCorrTestCase(lsst.meas.base.tests.AlgorithmTestCase, lsst.utils.tests.TestCase):

    def makeCatalog(self, apCorrScale=1.0, numSources=5):
        sourceCat = afwTable.SourceCatalog(self.schema)

        centroidKey = afwTable.Point2DKey(self.schema["slot_Centroid"])
        x = np.random.rand(numSources)*self.exposure.getWidth() + self.exposure.getX0()
        y = np.random.rand(numSources)*self.exposure.getHeight() + self.exposure.getY0()
        for _i in range(numSources):
            source_test_centroid = lsst.geom.Point2D(x[_i], y[_i])
            source = sourceCat.addNew()
            source.set(centroidKey, source_test_centroid)
            # All sources are unresolved.
            source[self.unresolvedName] = 0.0

        source_test_instFlux = 5.1
        source_test_instFluxErr = 1e-3

        for name in self.names:
            sourceCat[name + "_instFlux"] = source_test_instFlux
            sourceCat[name + "_instFluxErr"] = source_test_instFluxErr
            sourceCat[name + "_flag"] = np.zeros(len(sourceCat), dtype=bool)
            sourceCat[name + self.apNameStr + "_instFlux"] = source_test_instFlux * apCorrScale
            sourceCat[name + self.apNameStr + "_instFluxErr"] = source_test_instFluxErr * apCorrScale
            sourceCat[name + self.apNameStr + "_flag"] = np.zeros(len(sourceCat), dtype=bool)

        return sourceCat

    def setUp(self):
        schema = afwTable.SourceTable.makeMinimalSchema()
        apNameStr = "Ap"
        config = measureApCorr.MeasureApCorrTask.ConfigClass()
        unresolvedName = config.sourceSelector.active.unresolved.name
        # Add fields in anti-sorted order to try to impose a need for sorting
        # in the addition of the apCorr fields (may happen by fluke, but this
        # is the best we can do to test this here.
        names = ["test2", "test1"]
        for name in names:
            apName = name + apNameStr
            addApCorrName(apName)
            schema.addField(name + "_instFlux", type=float)
            schema.addField(name + "_instFluxErr", type=float)
            schema.addField(name + "_flag", type="Flag")
            schema.addField(apName + "_instFlux", type=float)
            schema.addField(apName + "_instFluxErr", type=float)
            schema.addField(apName + "_flag", type="Flag")
        schema.addField(names[0] + "_Centroid_x", type=float)
        schema.addField(names[0] + "_Centroid_y", type=float)
        schema.getAliasMap().set("slot_Centroid", names[0] + "_Centroid")
        schema.addField(unresolvedName, type=float)
        schema.addField("deblend_nChild", type=np.int32)
        for flag in [
            "base_PixelFlags_flag_edge",
            "base_PixelFlags_flag_nodata",
            "base_PixelFlags_flag_interpolatedCenter",
            "base_PixelFlags_flag_saturatedCenter",
            "base_PixelFlags_flag_crCenter",
            "base_PixelFlags_flag_bad",
            "base_PixelFlags_flag_interpolated",
            "base_PixelFlags_flag_saturated",
        ]:
            schema.addField(flag, type="Flag")
        config.refFluxName = names[0]
        config.sourceSelector["science"].signalToNoise.fluxField = names[0] + "_instFlux"
        config.sourceSelector["science"].signalToNoise.errField = names[0] + "_instFluxErr"
        self.meas_apCorr_task = measureApCorr.MeasureApCorrTask(schema=schema, config=config)
        self.names = names
        self.apNameStr = apNameStr
        self.schema = schema
        self.exposure = lsst.afw.image.ExposureF(10, 10)
        self.unresolvedName = unresolvedName

    def tearDown(self):
        del self.schema
        del self.meas_apCorr_task
        del self.exposure

    def testAddFields(self):
        """Instantiating the task should add one field to the schema."""
        for name in self.names:
            self.assertIn("apcorr_" + name + self.apNameStr + "_used", self.schema.getNames())
        sortedNames = sorted(self.names)
        key0 = self.schema.find("apcorr_" + sortedNames[0] + self.apNameStr + "_used").key
        key1 = self.schema.find("apcorr_" + sortedNames[1] + self.apNameStr + "_used").key
        # Check that the apCorr fields were added in a sorted order (not
        # foolproof as this could have happened by fluke, but it's the best
        # we can do to test this here (having added the two fields in an anti-
        # sorted order).
        self.assertLess(key0.getOffset() + key0.getBit(), key1.getOffset() + key1.getBit())

    def testReturnApCorrMap(self):
        """The measureApCorr task should return a structure with a single key 'apCorrMap'."""
        struct = self.meas_apCorr_task.run(catalog=self.makeCatalog(), exposure=self.exposure)
        self.assertEqual(list(struct.getDict().keys()), ['apCorrMap'])

    def testApCorrMapKeys(self):
        """An apCorrMap structure should have two keys per name supplied to addApCorrName()."""
        key_names = []
        for name in self.names:
            apFluxName = name + self.apNameStr + "_instFlux"
            apFluxErrName = name + self.apNameStr + "_instFluxErr"
            struct = self.meas_apCorr_task.run(catalog=self.makeCatalog(), exposure=self.exposure)
            key_names.append(apFluxName)
            key_names.append(apFluxErrName)
        self.assertEqual(set(struct.apCorrMap.keys()), set(key_names))

    def testTooFewSources(self):
        """ If there are too few sources, check that an exception is raised."""
        # Create an empty catalog with no sources to process.
        catalog = afwTable.SourceCatalog(self.schema)
        with self.assertRaisesRegex(measureApCorr.MeasureApCorrError,
                                    "Unable to measure aperture correction for 'test1Ap'"):
            self.meas_apCorr_task.run(catalog=catalog, exposure=self.exposure)

        # We now try again after declaring that the aperture correction is
        # allowed to fail. This should run without raising an exception, but
        # will log a warning.
        for name in self.names:
            self.meas_apCorr_task.config.allowFailure.append(name + self.apNameStr)
        with self.assertLogs(level=logging.WARNING) as cm:
            self.meas_apCorr_task.run(catalog=catalog, exposure=self.exposure)
        self.assertIn("Unable to measure aperture correction for 'test1Ap'", cm.output[0])

    def testSourceNotUsed(self):
        """ Check that a source outside the bounding box is flagged as not used (False)."""
        sourceCat = self.makeCatalog()
        source = sourceCat.addNew()
        nameAp = self.names[0] + "Ap"
        source[nameAp + "_instFlux"] = 5.1
        source[nameAp + "_instFluxErr"] = 1e-3
        source[self.meas_apCorr_task.config.refFluxName + "_instFlux"] = 5.1
        source[self.meas_apCorr_task.config.refFluxName + "_instFluxErr"] = 1e-3
        centroidKey = afwTable.Point2DKey(self.schema["slot_Centroid"])
        source.set(centroidKey, lsst.geom.Point2D(15, 7.1))
        apCorrFlagName = "apcorr_" + nameAp + "_used"

        self.meas_apCorr_task.run(catalog=sourceCat, exposure=self.exposure)
        # Check that all but the final source are used.
        self.assertTrue(sourceCat[apCorrFlagName][0: -1].all())
        # Check that the final source is not used.
        self.assertFalse(sourceCat[apCorrFlagName][-1])

    def testSourceUsed(self):
        """Check that valid sources inside the bounding box that are used have their flags set to True."""
        sourceCat = self.makeCatalog()
        self.meas_apCorr_task.run(catalog=sourceCat, exposure=self.exposure)
        for name in self.names:
            self.assertTrue(sourceCat["apcorr_" + name + self.apNameStr + "_used"].all())

    def testApertureMeasOnes(self):
        """ Check that sources with aperture fluxes exactly the same as their catalog fluxes
            returns an aperture correction map of 1s"""
        apFluxName = self.names[0] + self.apNameStr + "_instFlux"
        sourceCat = self.makeCatalog()
        struct = self.meas_apCorr_task.run(catalog=sourceCat, exposure=self.exposure)
        default_fill = apCorrDefaultMap(value=1.0, bbox=self.exposure.getBBox())
        test_fill = afwImage.ImageF(self.exposure.getBBox())
        struct.apCorrMap[apFluxName].fillImage(test_fill)
        np.testing.assert_allclose(test_fill.getArray(), default_fill.getArray())

    def testApertureMeasTens(self):
        """Check that aperture correction scales source fluxes in the correct direction."""
        apCorr_factor = 10.
        sourceCat = self.makeCatalog(apCorrScale=apCorr_factor)
        apFluxName = self.names[0] + self.apNameStr + "_instFlux"
        struct = self.meas_apCorr_task.run(catalog=sourceCat, exposure=self.exposure)
        default_fill = apCorrDefaultMap(value=apCorr_factor, bbox=self.exposure.getBBox())
        test_fill = afwImage.ImageF(self.exposure.getBBox())
        struct.apCorrMap[apFluxName].fillImage(test_fill)
        np.testing.assert_allclose(test_fill.getArray(), default_fill.getArray())

    def testFilterBadValue(self):
        """Check that the aperture correction filters a bad value."""
        sourceCat = self.makeCatalog()
        source = sourceCat.addNew()
        nameAp = self.names[0] + self.apNameStr
        source[nameAp + "_instFlux"] = 100.0
        source[nameAp + "_instFluxErr"] = 1e-3
        source[self.meas_apCorr_task.config.refFluxName + "_instFlux"] = 5.1
        source[self.meas_apCorr_task.config.refFluxName + "_instFluxErr"] = 1e-3
        source[self.unresolvedName] = 0.0
        centroidKey = afwTable.Point2DKey(self.schema["slot_Centroid"])
        x = self.exposure.getX0() + 1
        y = self.exposure.getY0() + 1
        source.set(centroidKey, lsst.geom.Point2D(x, y))

        self.meas_apCorr_task.run(catalog=sourceCat, exposure=self.exposure)

        # Check that both Ap fluxes are removed as outliers; one is due
        # to being unfilled (nan), the other is a large outlier.
        for name in self.names:
            apCorrFlagName = "apcorr_" + name + self.apNameStr + "_used"
            # Check that all but the final source are used.
            self.assertTrue(sourceCat[apCorrFlagName][0: -1].all())
            # Check that the final source is not used.
            self.assertFalse(sourceCat[apCorrFlagName][-1])

    def testTooFewSourcesAfterFiltering(self):
        """Check that the aperture correction fails when too many are filtered."""
        sourceCat = self.makeCatalog()
        self.meas_apCorr_task.config.minDegreesOfFreedom = 4

        for name in self.names:
            nameAp = name + self.apNameStr
            sourceCat[nameAp + "_instFlux"][0] = 100.0

        with self.assertRaisesRegex(measureApCorr.MeasureApCorrError,
                                    f"Unable to measure aperture correction for '{nameAp}'"):
            self.meas_apCorr_task.run(catalog=sourceCat, exposure=self.exposure)

        # We now try again after declaring that the aperture correction is
        # allowed to fail. This should run cleanly without raising an exception.
        for name in self.names:
            self.meas_apCorr_task.config.allowFailure.append(name + self.apNameStr)
        self.meas_apCorr_task.run(catalog=sourceCat, exposure=self.exposure)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
