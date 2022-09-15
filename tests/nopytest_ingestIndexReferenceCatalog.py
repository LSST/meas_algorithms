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

# This file is excluded from running through pytest due to concerns about the
# interaction between multiprocessing as invoked by this code, and the process
# pool used by pytest.
#
# Note that it is invoked independently by SCons, so the tests are still run
# as part of the build.

import os.path
import tempfile
import unittest
import unittest.mock

import numpy as np

import lsst.daf.butler
from lsst.daf.butler import DatasetType, DeferredDatasetHandle
from lsst.daf.butler.script import ingest_files
from lsst.meas.algorithms import (ConvertReferenceCatalogTask, ReferenceObjectLoader)
from lsst.meas.algorithms.htmIndexer import HtmIndexer
from lsst.meas.algorithms.ingestIndexReferenceTask import addRefCatMetadata
from lsst.meas.algorithms.convertRefcatManager import ConvertRefcatManager
from lsst.meas.algorithms.readTextCatalogTask import ReadTextCatalogTask
import lsst.utils

import ingestIndexTestBase


class TestConvertReferenceCatalogParallel(ingestIndexTestBase.ConvertReferenceCatalogTestBase,
                                          lsst.utils.tests.TestCase):
    """Test converting a refcat with multiprocessing turned on."""
    def testIngestTwoFilesTwoCores(self):
        def runTest(withRaDecErr):
            # Generate a second catalog, with different ids
            inTempDir1 = tempfile.TemporaryDirectory()
            inPath1 = inTempDir1.name
            skyCatalogFile1, _, skyCatalog1 = self.makeSkyCatalog(inPath1, idStart=25, seed=123)
            inTempDir2 = tempfile.TemporaryDirectory()
            inPath2 = inTempDir2.name
            skyCatalogFile2, _, skyCatalog2 = self.makeSkyCatalog(inPath2, idStart=5432, seed=11)
            # override some field names, and use multiple cores
            config = ingestIndexTestBase.makeConvertConfig(withRaDecErr=withRaDecErr, withMagErr=True,
                                                           withPm=True, withPmErr=True)
            # use a very small HTM pixelization depth to ensure there will be collisions when
            # ingesting the files in parallel
            depth = 2
            config.dataset_config.indexer.active.depth = depth
            # np.savetxt prepends '# ' to the header lines, so use a reader that understands that
            config.file_reader.format = 'ascii.commented_header'
            config.n_processes = 2  # use multiple cores for this test only
            config.id_name = 'id'  # Use the ids from the generated catalogs
            repoPath = os.path.join(self.outPath, "output_multifile_parallel",
                                    "_withRaDecErr" if withRaDecErr else "_noRaDecErr")

            # Convert the input data files to our HTM indexed format.
            dataTempDir = tempfile.TemporaryDirectory()
            dataPath = dataTempDir.name
            converter = ConvertReferenceCatalogTask(output_dir=dataPath, config=config)
            converter.run([skyCatalogFile1, skyCatalogFile2])

            # Make a temporary butler to ingest them into.
            butler = self.makeTemporaryRepo(repoPath, config.dataset_config.indexer.active.depth)
            dimensions = [f"htm{depth}"]
            datasetType = DatasetType(config.dataset_config.ref_dataset_name,
                                      dimensions,
                                      "SimpleCatalog",
                                      universe=butler.registry.dimensions,
                                      isCalibration=False)
            butler.registry.registerDatasetType(datasetType)

            # Ingest the files into the new butler.
            run = "testingRun"
            htmTableFile = os.path.join(dataPath, "filename_to_htm.ecsv")
            ingest_files(repoPath,
                         config.dataset_config.ref_dataset_name,
                         run,
                         htmTableFile,
                         transfer="auto")

            # Test if we can get back the catalogs, with a new butler.
            butler = lsst.daf.butler.Butler(repoPath)
            datasetRefs = list(butler.registry.queryDatasets(config.dataset_config.ref_dataset_name,
                                                             collections=[run]).expanded())
            handlers = []
            for dataRef in datasetRefs:
                handlers.append(DeferredDatasetHandle(butler=butler, ref=dataRef, parameters=None))
            loaderConfig = ReferenceObjectLoader.ConfigClass()
            loader = ReferenceObjectLoader([dataRef.dataId for dataRef in datasetRefs],
                                           handlers,
                                           name="testRefCat",
                                           config=loaderConfig,
                                           log=self.logger)
            self.checkAllRowsInRefcat(loader, skyCatalog1, config)
            self.checkAllRowsInRefcat(loader, skyCatalog2, config)

            inTempDir1.cleanup()
            inTempDir2.cleanup()
            dataTempDir.cleanup()

        runTest(withRaDecErr=True)
        runTest(withRaDecErr=False)


class TestConvertRefcatManager(ingestIndexTestBase.ConvertReferenceCatalogTestBase,
                               lsst.utils.tests.TestCase):
    """Unittests of various methods of ConvertRefcatManager.

    Uses mocks to force particular behavior regarding e.g. catalogs.
    """
    def setUp(self):
        np.random.seed(10)

        self.tempDir = tempfile.TemporaryDirectory()
        tempPath = self.tempDir.name
        self.log = lsst.log.Log.getLogger("lsst.TestIngestIndexManager")
        self.config = ingestIndexTestBase.makeConvertConfig(withRaDecErr=True)
        self.config.id_name = 'id'
        self.depth = 2  # very small depth, for as few pixels as possible.
        self.indexer = HtmIndexer(self.depth)
        self.htm = lsst.sphgeom.HtmPixelization(self.depth)
        ingester = ConvertReferenceCatalogTask(output_dir=tempPath, config=self.config)
        dtype = [('id', '<f8'), ('ra', '<f8'), ('dec', '<f8'), ('ra_err', '<f8'), ('dec_err', '<f8'),
                 ('a', '<f8'), ('a_err', '<f8')]
        self.schema, self.key_map = ingester.makeSchema(dtype)
        self.fileReader = ReadTextCatalogTask()

        self.fakeInput = self.makeSkyCatalog(outPath=None, size=5, idStart=6543)
        self.matchedPixels = np.array([1, 1, 2, 2, 3])
        self.tempDir2 = tempfile.TemporaryDirectory()
        tempPath = self.tempDir2.name
        self.filenames = {x: os.path.join(tempPath, "%d.fits" % x) for x in set(self.matchedPixels)}

        self.worker = ConvertRefcatManager(self.filenames,
                                           self.config,
                                           self.fileReader,
                                           self.indexer,
                                           self.schema,
                                           self.key_map,
                                           self.htm.universe()[0],
                                           addRefCatMetadata,
                                           self.log)

    def _createFakeCatalog(self, nOld=5, nNew=0, idStart=42):
        """Create a fake output SimpleCatalog, populated with nOld+nNew elements.

        Parameters
        ----------
        nOld : `int`, optional
            The number of filled in sources to put in the catalog.
        nNew : `int`, optional
            The number of empty sources to put in the catalog.
        idStart : `int`, optional
            The start id of the ``nOld`` sources.

        Returns
        -------
        catalog : `lsst.afw.table.SimpleCatalog`
            A catalog populated with random data and contiguous ids.
        """
        catalog = lsst.afw.table.SimpleCatalog(self.schema)
        catalog.resize(nOld)
        for x in self.schema:
            catalog[x.key] = np.random.random(nOld)
        # do the ids separately, so there are no duplicates
        catalog['id'] = np.arange(idStart, idStart + nOld)
        catalog.resize(nOld + nNew)  # make space for the elements we will add
        return catalog.copy(deep=True)

    def test_doOnePixelNewData(self):
        """Test that we can add new data to an existing catalog."""
        pixelId = 1  # the pixel we are going to test

        nOld = 5
        nNew = sum(self.matchedPixels == pixelId)
        catalog = self._createFakeCatalog(nOld=nOld, nNew=nNew)
        self.worker.getCatalog = unittest.mock.Mock(self.worker.getCatalog, return_value=catalog)

        self.worker._doOnePixel(self.fakeInput, self.matchedPixels, pixelId, {}, {})
        newcat = lsst.afw.table.SimpleCatalog.readFits(self.filenames[pixelId])

        # check that the "pre" catalog is unchanged, exactly
        np.testing.assert_equal(newcat[:nOld]['id'], catalog[:nOld]['id'])
        self.assertFloatsEqual(newcat[:nOld]['coord_ra'], catalog[:nOld]['coord_ra'])
        self.assertFloatsEqual(newcat[:nOld]['coord_dec'], catalog[:nOld]['coord_dec'])

        # check that the new catalog elements are set correctly
        newElements = self.fakeInput[self.matchedPixels == pixelId]
        np.testing.assert_equal(newcat[nOld:]['id'], newElements['id'])
        self.assertFloatsAlmostEqual(newcat[nOld:]['coord_ra'], newElements['ra_icrs']*np.pi/180)
        self.assertFloatsAlmostEqual(newcat[nOld:]['coord_dec'], newElements['dec_icrs']*np.pi/180)

    def test_doOnePixelNoData(self):
        """Test that we can put new data into an empty catalog."""
        pixelId = 2

        nOld = 0
        nNew = sum(self.matchedPixels == pixelId)
        catalog = self._createFakeCatalog(nOld=nOld, nNew=nNew)
        self.worker.getCatalog = unittest.mock.Mock(self.worker.getCatalog, return_value=catalog)

        self.worker._doOnePixel(self.fakeInput, self.matchedPixels, pixelId, {}, {})
        newcat = lsst.afw.table.SimpleCatalog.readFits(self.filenames[pixelId])

        # check that the new catalog elements are set correctly
        newElements = self.fakeInput[self.matchedPixels == pixelId]
        np.testing.assert_equal(newcat['id'], newElements['id'])
        self.assertFloatsAlmostEqual(newcat['coord_ra'], newElements['ra_icrs']*np.pi/180)
        self.assertFloatsAlmostEqual(newcat['coord_dec'], newElements['dec_icrs']*np.pi/180)

    def test_getCatalog(self):
        """Test that getCatalog returns a properly expanded new catalog."""
        pixelId = 3
        nOld = 10
        nNewElements = 5
        # save a catalog to disk that we can check against the getCatalog()'s return
        catalog = self._createFakeCatalog(nOld=nOld, nNew=0)
        catalog.writeFits(self.filenames[pixelId])
        newcat = self.worker.getCatalog(pixelId, self.schema, nNewElements)

        self.assertEqual(len(newcat), nOld + nNewElements)

        np.testing.assert_equal(newcat[:len(catalog)]['id'], catalog['id'])
        self.assertFloatsEqual(newcat[:len(catalog)]['coord_ra'], catalog['coord_ra'])
        self.assertFloatsEqual(newcat[:len(catalog)]['coord_dec'], catalog['coord_dec'])

    def tearDown(self):
        self.tempDir.cleanup()
        self.tempDir2.cleanup()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
