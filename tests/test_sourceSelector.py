#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
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
import astropy.units as u
import warnings

import lsst.afw.image
import lsst.afw.table
import lsst.meas.algorithms
import lsst.meas.base.tests
import lsst.pipe.base
import lsst.utils.tests

from lsst.meas.algorithms import ColorLimit


class SourceSelectorTester:
    """Mixin for testing

    This provides a base class for doing tests common to the
    ScienceSourceSelectorTask and ReferenceSourceSelectorTask.
    """
    Task = None

    def setUp(self):
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        schema.addField("flux", float, "Flux value")
        schema.addField("flux_flag", "Flag", "Bad flux?")
        schema.addField("other_flux", float, "Flux value 2")
        schema.addField("other_flux_flag", "Flag", "Bad flux 2?")
        schema.addField("other_fluxErr", float, "Flux error 2")
        schema.addField("goodFlag", "Flag", "Flagged if good")
        schema.addField("badFlag", "Flag", "Flagged if bad")
        schema.addField("starGalaxy", float, "0=star, 1=galaxy")
        schema.addField("nChild", np.int32, "Number of children")
        schema.addField("detect_isPrimary", "Flag", "Is primary detection?")
        schema.addField("sky_source", "Flag", "Empty sky region.")

        self.xCol = "centroid_x"
        self.yCol = "centroid_y"
        schema.addField(self.xCol, float, "Centroid x value.")
        schema.addField(self.yCol, float, "Centroid y value.")

        self.catalog = lsst.afw.table.SourceCatalog(schema)
        self.catalog.reserve(10)
        self.config = self.Task.ConfigClass()
        self.exposure = None

    def tearDown(self):
        del self.catalog

    def check(self, expected):
        task = self.Task(config=self.config)
        results = task.run(self.catalog, exposure=self.exposure)
        self.assertListEqual(results.selected.tolist(), expected)
        self.assertListEqual([src.getId() for src in results.sourceCat],
                             [src.getId() for src, ok in zip(self.catalog, expected) if ok])

        # Check with pandas.DataFrame version of catalog
        results = task.run(self.catalog.asAstropy().to_pandas(), exposure=self.exposure)
        self.assertListEqual(results.selected.tolist(), expected)
        self.assertListEqual(list(results.sourceCat['id']),
                             [src.getId() for src, ok in zip(self.catalog, expected) if ok])

        # Check with astropy.table.Table version of catalog
        results = task.run(self.catalog.asAstropy(), exposure=self.exposure)
        self.assertListEqual(results.selected.tolist(), expected)
        self.assertListEqual(list(results.sourceCat['id']),
                             [src.getId() for src, ok in zip(self.catalog, expected) if ok])

    def testFlags(self):
        bad1 = self.catalog.addNew()
        bad1.set("goodFlag", False)
        bad1.set("badFlag", False)
        bad2 = self.catalog.addNew()
        bad2.set("goodFlag", True)
        bad2.set("badFlag", True)
        bad3 = self.catalog.addNew()
        bad3.set("goodFlag", False)
        bad3.set("badFlag", True)
        good = self.catalog.addNew()
        good.set("goodFlag", True)
        good.set("badFlag", False)
        self.catalog["flux"] = 1.0
        self.catalog["other_flux"] = 1.0
        self.config.flags.good = ["goodFlag"]
        self.config.flags.bad = ["badFlag"]
        self.check([False, False, False, True])

    def testSignalToNoise(self):
        low = self.catalog.addNew()
        low.set("other_flux", 1.0)
        low.set("other_fluxErr", 1.0)
        good = self.catalog.addNew()
        good.set("other_flux", 1.0)
        good.set("other_fluxErr", 0.1)
        high = self.catalog.addNew()
        high.set("other_flux", 1.0)
        high.set("other_fluxErr", 0.001)
        self.config.doSignalToNoise = True
        self.config.signalToNoise.fluxField = "other_flux"
        self.config.signalToNoise.errField = "other_fluxErr"
        self.config.signalToNoise.minimum = 5.0
        self.config.signalToNoise.maximum = 100.0
        self.check([False, True, False])

    def testSignalToNoiseNoWarn(self):
        low = self.catalog.addNew()
        low.set("other_flux", np.nan)
        low.set("other_fluxErr", np.nan)
        self.config.doSignalToNoise = True
        self.config.signalToNoise.fluxField = "other_flux"
        self.config.signalToNoise.errField = "other_fluxErr"
        # Ensure no warnings are raised.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.check([False])


class ScienceSourceSelectorTaskTest(SourceSelectorTester, lsst.utils.tests.TestCase):
    Task = lsst.meas.algorithms.ScienceSourceSelectorTask

    def setUp(self):
        SourceSelectorTester.setUp(self)
        self.config.fluxLimit.fluxField = "flux"
        self.config.flags.bad = []
        self.config.doFluxLimit = True
        self.config.doFlags = True
        self.config.doUnresolved = False
        self.config.doIsolated = False

    def testFluxLimit(self):
        tooBright = self.catalog.addNew()
        tooBright.set("flux", 1.0e10)
        tooBright.set("flux_flag", False)
        good = self.catalog.addNew()
        good.set("flux", 1000.0)
        good.set("flux_flag", False)
        bad = self.catalog.addNew()
        bad.set("flux", good.get("flux"))
        bad.set("flux_flag", True)
        tooFaint = self.catalog.addNew()
        tooFaint.set("flux", 1.0)
        tooFaint.set("flux_flag", False)
        self.config.fluxLimit.minimum = 10.0
        self.config.fluxLimit.maximum = 1.0e6
        self.config.fluxLimit.fluxField = "flux"
        self.check([False, True, False, False])

        # Works with no maximum set?
        self.config.fluxLimit.maximum = None
        self.check([True, True, False, False])

        # Works with no minimum set?
        self.config.fluxLimit.minimum = None
        self.check([True, True, False, True])

    def testUnresolved(self):
        num = 5
        for _ in range(num):
            self.catalog.addNew()
        self.catalog["flux"] = 1.0
        starGalaxy = np.linspace(0.0, 1.0, num, False)
        self.catalog["starGalaxy"] = starGalaxy
        self.config.doUnresolved = True
        self.config.unresolved.name = "starGalaxy"
        minimum, maximum = 0.3, 0.7
        self.config.unresolved.minimum = minimum
        self.config.unresolved.maximum = maximum
        self.check(((starGalaxy > minimum) & (starGalaxy < maximum)).tolist())

        # Works with no minimum set?
        self.config.unresolved.minimum = None
        self.check((starGalaxy < maximum).tolist())

        # Works with no maximum set?
        self.config.unresolved.minimum = minimum
        self.config.unresolved.maximum = None
        self.check((starGalaxy > minimum).tolist())

    def testIsolated(self):
        num = 5
        for _ in range(num):
            self.catalog.addNew()
        self.catalog["flux"] = 1.0
        parent = np.array([0, 0, 10, 0, 0], dtype=int)
        nChild = np.array([2, 0, 0, 0, 0], dtype=int)
        self.catalog["parent"] = parent
        self.catalog["nChild"] = nChild
        self.config.doIsolated = True
        self.config.isolated.parentName = "parent"
        self.config.isolated.nChildName = "nChild"
        self.check(((parent == 0) & (nChild == 0)).tolist())

    def testRequireFiniteRaDec(self):
        num = 5
        for _ in range(num):
            self.catalog.addNew()
        ra = np.array([np.nan, np.nan, 0, 0, 0], dtype=float)
        dec = np.array([2, np.nan, 0, 0, np.nan], dtype=float)
        self.catalog["coord_ra"] = ra
        self.catalog["coord_dec"] = dec
        self.config.doRequireFiniteRaDec = True
        self.config.requireFiniteRaDec.raColName = "coord_ra"
        self.config.requireFiniteRaDec.decColName = "coord_dec"
        self.check((np.isfinite(ra) & np.isfinite(dec)).tolist())

    def testRequirePrimary(self):
        num = 5
        for _ in range(num):
            self.catalog.addNew()
        primary = np.array([True, True, False, True, False], dtype=bool)
        self.catalog["detect_isPrimary"] = primary
        self.config.doRequirePrimary = True
        self.config.requirePrimary.primaryColName = "detect_isPrimary"
        self.check(primary.tolist())

    def testSkySource(self):
        num = 5
        for _ in range(num):
            self.catalog.addNew()
        sky = np.array([True, True, False, True, False], dtype=bool)
        self.catalog["sky_source"] = sky
        # This is a union, not an intersection, so include another selection
        # that would otherwise reject everything.
        self.config.doRequirePrimary = True
        self.config.doSkySources = True
        self.check(sky.tolist())


class ReferenceSourceSelectorTaskTest(SourceSelectorTester, lsst.utils.tests.TestCase):
    Task = lsst.meas.algorithms.ReferenceSourceSelectorTask

    def setUp(self):
        SourceSelectorTester.setUp(self)
        self.config.magLimit.fluxField = "flux"
        self.config.doMagLimit = True
        self.config.doFlags = True
        self.config.doUnresolved = False
        self.config.doRequireFiniteRaDec = False

    def testMagnitudeLimit(self):
        tooBright = self.catalog.addNew()
        tooBright.set("flux", 1.0e10)
        tooBright.set("flux_flag", False)
        good = self.catalog.addNew()
        good.set("flux", 1000.0)
        good.set("flux_flag", False)
        bad = self.catalog.addNew()
        bad.set("flux", good.get("flux"))
        bad.set("flux_flag", True)
        tooFaint = self.catalog.addNew()
        tooFaint.set("flux", 1.0)
        tooFaint.set("flux_flag", False)
        # Note: magnitudes are backwards, so the minimum flux is the maximum magnitude
        self.config.magLimit.minimum = (1.0e6*u.nJy).to_value(u.ABmag)
        self.config.magLimit.maximum = (10.0*u.nJy).to_value(u.ABmag)
        self.config.magLimit.fluxField = "flux"
        self.check([False, True, False, False])

        # Works with no minimum set?
        self.config.magLimit.minimum = None
        self.check([True, True, False, False])

        # Works with no maximum set?
        self.config.magLimit.maximum = None
        self.check([True, True, False, True])

    def testMagErrorLimit(self):
        # Using an arbitrary field as if it was a magnitude error to save adding a new field
        field = "other_fluxErr"
        tooFaint = self.catalog.addNew()
        tooFaint.set(field, 0.5)
        tooBright = self.catalog.addNew()
        tooBright.set(field, 0.00001)
        good = self.catalog.addNew()
        good.set(field, 0.2)

        self.config.doMagError = True
        self.config.magError.minimum = 0.01
        self.config.magError.maximum = 0.3
        self.config.magError.magErrField = field
        self.check([False, False, True])

    def testColorLimits(self):
        num = 10
        for _ in range(num):
            self.catalog.addNew()
        color = np.linspace(-0.5, 0.5, num, True)
        flux = 1000.0*u.nJy
        # Definition: color = mag(flux) - mag(otherFlux)
        otherFlux = (flux.to(u.ABmag) - color*u.mag).to_value(u.nJy)
        self.catalog["flux"] = flux.value
        self.catalog["other_flux"] = otherFlux
        minimum, maximum = -0.1, 0.2
        self.config.colorLimits = {"test": ColorLimit(primary="flux", secondary="other_flux",
                                                      minimum=minimum, maximum=maximum)}
        self.check(((color > minimum) & (color < maximum)).tolist())

        # Works with no minimum set?
        self.config.colorLimits["test"].minimum = None
        self.check((color < maximum).tolist())

        # Works with no maximum set?
        self.config.colorLimits["test"].maximum = None
        self.config.colorLimits["test"].minimum = minimum
        self.check((color > minimum).tolist())

        # Multiple limits
        self.config.colorLimits = {"test": ColorLimit(primary="flux", secondary="other_flux",
                                                      minimum=minimum),
                                   "other": ColorLimit(primary="flux", secondary="other_flux",
                                                       maximum=maximum)}
        assert maximum > minimum  # To be non-mutually-exclusive
        self.check(((color > minimum) & (color < maximum)).tolist())

        # Multiple mutually-exclusive limits
        self.config.colorLimits["test"] = ColorLimit(primary="flux", secondary="other_flux", maximum=-0.1)
        self.config.colorLimits["other"] = ColorLimit(primary="flux", secondary="other_flux", minimum=0.1)
        self.check([False]*num)

    def testUnresolved(self):
        num = 5
        for _ in range(num):
            self.catalog.addNew()
        self.catalog["flux"] = 1.0
        starGalaxy = np.linspace(0.0, 1.0, num, False)
        self.catalog["starGalaxy"] = starGalaxy
        self.config.doUnresolved = True
        self.config.unresolved.name = "starGalaxy"
        minimum, maximum = 0.3, 0.7
        self.config.unresolved.minimum = minimum
        self.config.unresolved.maximum = maximum
        self.check(((starGalaxy > minimum) & (starGalaxy < maximum)).tolist())

        # Works with no minimum set?
        self.config.unresolved.minimum = None
        self.check((starGalaxy < maximum).tolist())

        # Works with no maximum set?
        self.config.unresolved.minimum = minimum
        self.config.unresolved.maximum = None
        self.check((starGalaxy > minimum).tolist())

    def testFiniteRaDec(self):
        "Test that non-finite RA and Dec values are caught."
        num = 5
        for _ in range(num):
            self.catalog.addNew()
        self.catalog["coord_ra"][:] = 1.0
        self.catalog["coord_dec"][:] = 1.0
        self.catalog["coord_ra"][0] = np.nan
        self.catalog["coord_dec"][1] = np.inf
        self.config.doRequireFiniteRaDec = True

        self.check([False, False, True, True, True])

    def testCullFromMaskedRegion(self):
        # Test that objects whose centroids land on specified mask(s) are
        # culled.
        maskNames = ["NO_DATA", "BLAH"]
        noDataPoints = [[0, 0], [3, 2]]
        self.exposure = lsst.afw.image.ExposureF(5, 5)
        mask = self.exposure.mask
        for maskName in maskNames:
            if maskName not in mask.getMaskPlaneDict():
                mask.addMaskPlane(maskName)
        num = 5
        for _ in range(num):
            self.catalog.addNew()
        self.catalog[self.xCol][:] = 5.0
        self.catalog[self.yCol][:] = 5.0

        # Set first two entries in catalog to land maskNames region.
        for i, noDataPoint in enumerate(noDataPoints):
            # Flip x & y for numpy array convention.
            mask.array[noDataPoint[1]][noDataPoint[0]] = mask.getPlaneBitMask(
                maskNames[min(i, len(maskNames) - 1)]
            )
            self.catalog[self.xCol][i] = noDataPoint[0]
            self.catalog[self.yCol][i] = noDataPoint[1]

        self.config.doCullFromMaskedRegion = True
        self.config.cullFromMaskedRegion.xColName = self.xCol
        self.config.cullFromMaskedRegion.yColName = self.yCol
        self.config.cullFromMaskedRegion.badMaskNames = maskNames
        self.check([False, False, True, True, True])
        # Reset config back to False and None for other tests.
        self.config.doCullFromMaskedRegion = False
        self.exposure = None


class TestBaseSourceSelector(lsst.utils.tests.TestCase):
    """Test the API of the Abstract Base Class with a trivial example."""
    def setUp(self):
        schema = lsst.meas.base.tests.TestDataset.makeMinimalSchema()
        self.selectedKeyName = "is_selected"
        schema.addField(self.selectedKeyName, type="Flag")
        self.catalog = lsst.afw.table.SourceCatalog(schema)
        for i in range(4):
            self.catalog.addNew()

        self.sourceSelector = lsst.meas.algorithms.NullSourceSelectorTask()

    def testRun(self):
        """Test that run() returns a catalog and boolean selected array."""
        result = self.sourceSelector.run(self.catalog)
        for i, x in enumerate(self.catalog['id']):
            self.assertIn(x, result.sourceCat['id'])
            self.assertTrue(result.selected[i])

    def testRunSourceSelectedField(self):
        """Test that the selected flag is set in the original catalog."""
        self.sourceSelector.run(self.catalog, sourceSelectedField=self.selectedKeyName)
        np.testing.assert_array_equal(self.catalog[self.selectedKeyName], True)

    def testRunNonContiguousRaises(self):
        """Cannot do source selection on non-contiguous catalogs."""
        del self.catalog[1]  # take one out of the middle to make it non-contiguous.
        self.assertFalse(self.catalog.isContiguous(), "Catalog is contiguous: the test won't work.")

        with self.assertRaises(RuntimeError):
            self.sourceSelector.run(self.catalog)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    import sys
    setup_module(sys.modules[__name__])
    unittest.main()
