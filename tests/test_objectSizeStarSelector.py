# This file is part of meas_algorithms.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
import numpy as np

import lsst.afw.table as afwTable
from lsst.meas.algorithms import sourceSelector, NoPsfLikeStarsError
import lsst.meas.base.tests
import lsst.utils.tests

fluxField = "base_GaussianFlux_instFlux"


def addGoodSource(sourceCat, num=0):
    """Insert a likely-good source into the catalog.

    Parameters
    ----------
    sourceCat : `lsst.afw.table.SourceCatalog`
       The source catalog for which a "good" source is to be added for testing.
    num : `float` or `int`
       A number that is added to various values to distinguish them in catalogs with multiple objects.
    """
    sourceCat.addNew()
    sourceCat["coord_ra"][-1] = 1.0 + num
    sourceCat["coord_dec"][-1] = 2.0 + num
    # Add some variability to the fluxes to form a cluster with non-finite "width" in the mag-size plane
    fluxFactor = np.random.uniform(low=0.98, high=1.02)
    sourceCat[fluxField][-1] = 100.0*fluxFactor  # We set fluxMin = 50 in setUp
    sourceCat[fluxField + "Err"][-1] = 1.0
    # Add some variability to the shapes to form a cluster with non-finite "width" in the mag-size plane
    widthFactor = np.random.uniform(low=0.98, high=1.02)
    sourceCat["truth_xx"][-1] = 3.0*widthFactor
    sourceCat["truth_yy"][-1] = 3.0*widthFactor
    sourceCat["truth_xy"][-1] = 1.0


class TestObjectSizeSourceSelector(lsst.utils.tests.TestCase):

    def setUp(self):
        np.random.seed(100)

        self.sourceSelector = sourceSelector.sourceSelectorRegistry["objectSize"]()
        self.badFlags = self.sourceSelector.config.badFlags
        schema = lsst.meas.base.tests.TestDataset.makeMinimalSchema()
        self.sourceSelector.config.sourceFluxField = fluxField
        self.sourceSelector.config.fluxMin = 50.0
        schema.addField(fluxField, type=np.float64)
        schema.addField(fluxField + "Err", type=np.float64)
        for flag in self.badFlags:
            schema.addField(flag, type="Flag")
        self.sourceCat = afwTable.SourceCatalog(schema)

    def tearDown(self):
        del self.sourceCat
        del self.badFlags
        del self.sourceSelector

    def testNoInputSources(self):
        with self.assertRaises(NoPsfLikeStarsError):
            self.sourceSelector.selectSources(self.sourceCat)

    def testSelectSourcesGood(self):
        for i in range(5):
            addGoodSource(self.sourceCat, i)
        result = self.sourceSelector.selectSources(self.sourceCat)
        for src in self.sourceCat["id"]:
            self.assertIn(src, self.sourceCat[result.selected]["id"])

    def testSelectSourcesIndividualBadFlags(self):
        for i in range(len(self.badFlags)):
            addGoodSource(self.sourceCat, i)
        for i in range(len(self.badFlags)):
            for j, flag in enumerate(self.badFlags):
                self.sourceCat[j].set(flag, True) if i == j else self.sourceCat[j].set(flag, False)
            result = self.sourceSelector.selectSources(self.sourceCat)
            self.assertNotIn(self.sourceCat[i]["id"], self.sourceCat[result.selected]["id"],
                             "should not have found %s" % flag)

    def testSelectSourcesAllBadFlags(self):
        for i, flag in enumerate(self.badFlags):
            addGoodSource(self.sourceCat, i)
            self.sourceCat[i].set(flag, True)
        with self.assertRaises(NoPsfLikeStarsError):
            self.sourceSelector.selectSources(self.sourceCat)

    def testSelectSourcesSignalToNoiseCuts(self):
        for i in range(10):
            addGoodSource(self.sourceCat, i)

        self.sourceCat[fluxField + "Err"][0] = self.sourceCat[fluxField][0]/20.0
        self.sourceCat[fluxField + "Err"][1] = self.sourceCat[fluxField][1]/500.0
        self.sourceCat[fluxField + "Err"][2] = self.sourceCat[fluxField][2]/2000.0

        self.sourceSelector.config.signalToNoiseMax = 1000
        result = self.sourceSelector.run(self.sourceCat)
        self.assertNotIn(self.sourceCat[0]["id"], result.sourceCat["id"])
        self.assertIn(self.sourceCat[1]["id"], result.sourceCat["id"])
        self.assertNotIn(self.sourceCat[2]["id"], result.sourceCat["id"])

    def testSelectSourcesNoSignalToNoiseCut(self):
        for i in range(5):
            addGoodSource(self.sourceCat, i)
        self.sourceCat[fluxField + "Err"][0] = self.sourceCat[fluxField][0]*1e+9  # S/N ~1e-9
        self.sourceSelector.config.signalToNoiseMin = 0
        result = self.sourceSelector.run(self.sourceCat)
        self.assertIn(self.sourceCat[0]["id"], result.sourceCat["id"])

    def testSelectSourcesFluxCuts(self):
        for i in range(10):
            addGoodSource(self.sourceCat, i)

        self.sourceSelector.config.doSignalToNoiseLimit = False
        self.sourceSelector.config.doFluxLimit = True
        self.sourceSelector.config.fluxMin = 98.0  # Just outside of range allowed in addGoodSource()
        self.sourceCat[fluxField][0] = 97.9
        self.sourceSelector.config.fluxMax = 101.3  # Just outside of range allowed in addGoodSource()
        self.sourceCat[fluxField][2] = 101.4
        result = self.sourceSelector.run(self.sourceCat)
        self.assertNotIn(self.sourceCat[0]["id"], result.sourceCat["id"])
        self.assertIn(self.sourceCat[1]["id"], result.sourceCat["id"])
        self.assertNotIn(self.sourceCat[2]["id"], result.sourceCat["id"])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
