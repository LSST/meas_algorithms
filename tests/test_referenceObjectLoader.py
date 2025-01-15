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

import os.path
import tempfile
import unittest
import glob

import numpy as np
from smatch.matcher import sphdist
import astropy.time

import lsst.daf.butler
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.daf.butler import DatasetType, DeferredDatasetHandle
from lsst.daf.butler.script import ingest_files
from lsst.meas.algorithms import (ConvertReferenceCatalogTask, ReferenceObjectLoader,
                                  getRefFluxField, getRefFluxKeys)
from lsst.meas.algorithms.testUtils import MockReferenceObjectLoaderFromFiles
from lsst.meas.algorithms.convertReferenceCatalog import _makeSchema
import lsst.utils
import lsst.geom

import convertReferenceCatalogTestBase


class ReferenceObjectLoaderGenericTests(lsst.utils.tests.TestCase):
    """Test parts of the reference loader that don't depend on loading a
    catalog, for example schema creation, filter maps, units, and metadata.
    """
    def testFilterMapVsAnyFilterMapsToThis(self):
        config = ReferenceObjectLoader.ConfigClass()
        # check that a filterMap-only config passes validation
        config.filterMap = {"b": "a"}
        try:
            config.validate()
        except lsst.pex.config.FieldValidationError:
            self.fail("`filterMap`-only LoadReferenceObjectsConfig should not fail validation.")

        # anyFilterMapsToThis and filterMap are mutually exclusive
        config.anyFilterMapsToThis = "c"
        with self.assertRaises(lsst.pex.config.FieldValidationError):
            config.validate()

        # check that a anyFilterMapsToThis-only config passes validation
        config.filterMap = {}
        try:
            config.validate()
        except lsst.pex.config.FieldValidationError:
            self.fail("`anyFilterMapsToThis`-only LoadReferenceObjectsConfig should not fail validation.")

    def testFilterAliasMap(self):
        """Make a schema with filter aliases."""
        for filterMap in ({}, {"camr": "r"}):
            config = ReferenceObjectLoader.ConfigClass()
            config.filterMap = filterMap
            loader = ReferenceObjectLoader(None, [], name=None, config=config)
            refSchema = _makeSchema(filterNameList="r")
            loader._addFluxAliases(refSchema,
                                   anyFilterMapsToThis=config.anyFilterMapsToThis,
                                   filterMap=config.filterMap)

            self.assertIn("r_flux", refSchema)
            self.assertIn("r_fluxErr", refSchema)

            # camera filters aliases are named <filter>_camFlux
            if "camr" in filterMap:
                self.assertEqual(getRefFluxField(refSchema, "camr"), "camr_camFlux")
            else:
                with self.assertRaisesRegex(RuntimeError,
                                            r"Could not find flux field\(s\) camr_camFlux, camr_flux"):
                    getRefFluxField(refSchema, "camr")

            refCat = afwTable.SimpleCatalog(refSchema)
            refObj = refCat.addNew()
            refObj["r_flux"] = 1.23
            self.assertAlmostEqual(refCat[0].get(getRefFluxField(refSchema, "r")), 1.23)
            if "camr" in filterMap:
                self.assertAlmostEqual(refCat[0].get(getRefFluxField(refSchema, "camr")), 1.23)
            refObj["r_fluxErr"] = 0.111
            if "camr" in filterMap:
                self.assertEqual(refCat[0].get("camr_camFluxErr"), 0.111)
            fluxKey, fluxErrKey = getRefFluxKeys(refSchema, "r")
            self.assertEqual(refCat[0].get(fluxKey), 1.23)
            self.assertEqual(refCat[0].get(fluxErrKey), 0.111)
            if "camr" in filterMap:
                fluxKey, fluxErrKey = getRefFluxKeys(refSchema, "camr")
                self.assertEqual(refCat[0].get(fluxErrKey), 0.111)
            else:
                with self.assertRaises(RuntimeError):
                    getRefFluxKeys(refSchema, "camr")

    def testAnyFilterMapsToThisAlias(self):
        # test anyFilterMapsToThis
        config = ReferenceObjectLoader.ConfigClass()
        config.anyFilterMapsToThis = "gg"
        loader = ReferenceObjectLoader(None, [], name=None, config=config)
        refSchema = _makeSchema(filterNameList=["gg"])
        loader._addFluxAliases(refSchema,
                               anyFilterMapsToThis=config.anyFilterMapsToThis,
                               filterMap=config.filterMap)
        self.assertEqual(getRefFluxField(refSchema, "r"), "gg_flux")
        # raise if "gg" is not in the refcat filter list
        with self.assertRaises(RuntimeError):
            refSchema = _makeSchema(filterNameList=["rr"])
            refSchema = loader._addFluxAliases(refSchema,
                                               anyFilterMapsToThis=config.anyFilterMapsToThis,
                                               filterMap=config.filterMap)

    def testGetMetadataCircle(self):
        center = lsst.geom.SpherePoint(100*lsst.geom.degrees, 45*lsst.geom.degrees)
        radius = lsst.geom.Angle(1*lsst.geom.degrees)
        loader = ReferenceObjectLoader(None, [], name=None)
        metadata = loader.getMetadataCircle(center, radius, "fakeR")
        self.assertEqual(metadata['RA'], center.getLongitude().asDegrees())
        self.assertEqual(metadata['DEC'], center.getLatitude().asDegrees())
        self.assertEqual(metadata['RADIUS'], radius.asDegrees())
        self.assertEqual(metadata['SMATCHV'], 2)
        self.assertEqual(metadata['FILTER'], 'fakeR')
        self.assertEqual(metadata['JEPOCH'], None)
        self.assertEqual(metadata['TIMESYS'], 'TAI')

        epoch = astropy.time.Time(2023.0, format="jyear", scale="tai")
        metadata = loader.getMetadataCircle(center, radius, "fakeR", epoch=epoch)
        self.assertEqual(metadata['JEPOCH'], epoch.jyear)


class ReferenceObjectLoaderLoadTests(convertReferenceCatalogTestBase.ConvertReferenceCatalogTestBase,
                                     lsst.utils.tests.TestCase):
    """Tests of loading reference catalogs, using an in-memory generated fake
    sky catalog that is converted to an LSST refcat.

    This effectively is a partial integration test of the refcat conversion,
    ingestion, and loading sequence, focusing mostly on testing the different
    ways to load a refcat. It significantly overlaps in coverage with
    ``nopytest_convertReferenceCatalog.py``, but uses a very trivial test
    refcat and only one core during the conversion.
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Generate a catalog, with arbitrary ids
        inTempDir = tempfile.TemporaryDirectory()
        inPath = inTempDir.name
        skyCatalogFile, _, skyCatalog = cls.makeSkyCatalog(inPath, idStart=25, seed=123)

        cls.skyCatalog = skyCatalog

        # override some field names.
        config = convertReferenceCatalogTestBase.makeConvertConfig(withRaDecErr=True, withMagErr=True,
                                                                   withPm=True, withParallax=True,
                                                                   withFullPositionInformation=True)
        # use a very small HTM pixelization depth
        depth = 2
        config.dataset_config.indexer.active.depth = depth
        # np.savetxt prepends '# ' to the header lines, so use a reader that understands that
        config.file_reader.format = 'ascii.commented_header'
        config.n_processes = 1
        config.id_name = 'id'  # Use the ids from the generated catalogs
        cls.repoTempDir = tempfile.TemporaryDirectory()
        repoPath = cls.repoTempDir.name

        # Convert the input data files to our HTM indexed format.
        dataTempDir = tempfile.TemporaryDirectory()
        dataPath = dataTempDir.name
        converter = ConvertReferenceCatalogTask(output_dir=dataPath, config=config)
        converter.run([skyCatalogFile])

        # Make a temporary butler to ingest them into.
        butler = cls.makeTemporaryRepo(repoPath, config.dataset_config.indexer.active.depth)
        dimensions = [f"htm{depth}"]
        datasetType = DatasetType(config.dataset_config.ref_dataset_name,
                                  dimensions,
                                  "SimpleCatalog",
                                  universe=butler.dimensions,
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
        handles = []
        for dataRef in datasetRefs:
            handles.append(DeferredDatasetHandle(butler=butler, ref=dataRef, parameters=None))

        cls.datasetRefs = datasetRefs
        cls.handles = handles

        inTempDir.cleanup()
        dataTempDir.cleanup()

    def test_loadSkyCircle(self):
        """Test the loadSkyCircle routine."""
        loader = ReferenceObjectLoader([dataRef.dataId for dataRef in self.datasetRefs],
                                       self.handles,
                                       name="testrefcat")
        center = lsst.geom.SpherePoint(180.0*lsst.geom.degrees, 0.0*lsst.geom.degrees)
        cat = loader.loadSkyCircle(
            center,
            30.0*lsst.geom.degrees,
            filterName='a',
        ).refCat
        # Check that the max distance is less than the radius
        dist = sphdist(180.0, 0.0, np.rad2deg(cat['coord_ra']), np.rad2deg(cat['coord_dec']))
        self.assertLess(np.max(dist), 30.0)

        # Check that all the objects from the two catalogs are here.
        dist = sphdist(180.0, 0.0, self.skyCatalog['ra'], self.skyCatalog['dec'])
        inside, = (dist < 30.0).nonzero()
        self.assertEqual(len(cat), len(inside))

        self.assertTrue(cat.isContiguous())
        self.assertEqual(len(np.unique(cat['id'])), len(cat))
        # A default-loaded sky circle should not have centroids
        self.assertNotIn('centroid_x', cat.schema)
        self.assertNotIn('centroid_y', cat.schema)
        self.assertNotIn('hasCentroid', cat.schema)

    def test_loadPixelBox(self):
        """Test the loadPixelBox routine."""
        # This will create a box 50 degrees on a side.
        loaderConfig = ReferenceObjectLoader.ConfigClass()
        loaderConfig.pixelMargin = 0
        loader = ReferenceObjectLoader([dataRef.dataId for dataRef in self.datasetRefs],
                                       self.handles,
                                       name="testrefcat",
                                       config=loaderConfig)
        bbox = lsst.geom.Box2I(corner=lsst.geom.Point2I(0, 0), dimensions=lsst.geom.Extent2I(1000, 1000))
        crpix = lsst.geom.Point2D(500, 500)
        crval = lsst.geom.SpherePoint(180.0*lsst.geom.degrees, 0.0*lsst.geom.degrees)
        cdMatrix = afwGeom.makeCdMatrix(scale=0.05*lsst.geom.degrees)
        wcs = afwGeom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)

        cat = loader.loadPixelBox(bbox, wcs, 'a', bboxToSpherePadding=0).refCat

        # This is a sanity check on the ranges; the exact selection depends
        # on cos(dec) and the tangent-plane projection.
        self.assertLess(np.max(np.rad2deg(cat['coord_ra'])), 180.0 + 25.0)
        self.assertGreater(np.max(np.rad2deg(cat['coord_ra'])), 180.0 - 25.0)
        self.assertLess(np.max(np.rad2deg(cat['coord_dec'])), 25.0)
        self.assertGreater(np.min(np.rad2deg(cat['coord_dec'])), -25.0)

        # The following is to ensure the reference catalog coords are
        # getting corrected for proper motion when an epoch is provided.
        # Use an extreme epoch so that differences in corrected coords
        # will be significant.  Note that this simply tests that the coords
        # do indeed change when the epoch is passed.  It makes no attempt
        # at assessing the correctness of the change.  This is left to the
        # explicit testProperMotion() test below.
        catWithEpoch = loader.loadPixelBox(
            bbox,
            wcs,
            'a',
            bboxToSpherePadding=0,
            epoch=astropy.time.Time(30000, format='mjd', scale='tai')).refCat

        self.assertFloatsNotEqual(cat['coord_ra'], catWithEpoch['coord_ra'], rtol=1.0e-4)
        self.assertFloatsNotEqual(cat['coord_dec'], catWithEpoch['coord_dec'], rtol=1.0e-4)

    def test_filterMap(self):
        """Test filterMap parameters."""
        loaderConfig = ReferenceObjectLoader.ConfigClass()
        loaderConfig.filterMap = {'aprime': 'a'}
        loader = ReferenceObjectLoader([dataRef.dataId for dataRef in self.datasetRefs],
                                       self.handles,
                                       name="testrefcat",
                                       config=loaderConfig)
        center = lsst.geom.SpherePoint(180.0*lsst.geom.degrees, 0.0*lsst.geom.degrees)
        result = loader.loadSkyCircle(
            center,
            30.0*lsst.geom.degrees,
            filterName='aprime',
        )
        self.assertEqual(result.fluxField, 'aprime_camFlux')
        self.assertFloatsEqual(result.refCat['aprime_camFlux'], result.refCat['a_flux'])

    def test_properMotion(self):
        """Test proper motion correction."""
        loaderConfig = ReferenceObjectLoader.ConfigClass()
        loaderConfig.filterMap = {'aprime': 'a'}
        loader = ReferenceObjectLoader([dataRef.dataId for dataRef in self.datasetRefs],
                                       self.handles,
                                       name="testrefcat",
                                       config=loaderConfig)
        center = lsst.geom.SpherePoint(180.0*lsst.geom.degrees, 0.0*lsst.geom.degrees)
        cat = loader.loadSkyCircle(
            center,
            30.0*lsst.geom.degrees,
            filterName='a'
        ).refCat

        # Zero epoch change --> no proper motion correction (except minor numerical effects)
        cat_pm = loader.loadSkyCircle(
            center,
            30.0*lsst.geom.degrees,
            filterName='a',
            epoch=self.epoch
        ).refCat

        self.assertFloatsAlmostEqual(cat_pm['coord_ra'], cat['coord_ra'], rtol=1.0e-14)
        self.assertFloatsAlmostEqual(cat_pm['coord_dec'], cat['coord_dec'], rtol=1.0e-14)
        self.assertFloatsEqual(cat_pm['coord_raErr'], cat['coord_raErr'])
        self.assertFloatsEqual(cat_pm['coord_decErr'], cat['coord_decErr'])

        # One year difference
        cat_pm = loader.loadSkyCircle(
            center,
            30.0*lsst.geom.degrees,
            filterName='a',
            epoch=self.epoch + 1.0*astropy.units.yr
        ).refCat

        self.assertFloatsEqual(cat_pm['pm_raErr'], cat['pm_raErr'])
        self.assertFloatsEqual(cat_pm['pm_decErr'], cat['pm_decErr'])

        separations = np.array([cat[i].getCoord().separation(cat_pm[i].getCoord()).asArcseconds()
                                for i in range(len(cat))])
        bearings = np.array([cat[i].getCoord().bearingTo(cat_pm[i].getCoord()).asArcseconds()
                             for i in range(len(cat))])
        self.assertFloatsAlmostEqual(separations, self.properMotionAmt.asArcseconds(), rtol=1.0e-10)
        self.assertFloatsAlmostEqual(bearings, self.properMotionDir.asArcseconds(), rtol=1.0e-10)

        predictedRaErr = np.hypot(cat["coord_raErr"], cat["pm_raErr"])
        predictedDecErr = np.hypot(cat["coord_decErr"], cat["pm_decErr"])
        self.assertFloatsAlmostEqual(cat_pm["coord_raErr"], predictedRaErr)
        self.assertFloatsAlmostEqual(cat_pm["coord_decErr"], predictedDecErr)

        # One year negative difference. This demonstrates a fix for DM-38808,
        # when the refcat epoch is later in time than the data.
        cat_pm = loader.loadSkyCircle(
            center,
            30.0*lsst.geom.degrees,
            filterName='a',
            epoch=self.epoch - 1.0*astropy.units.yr
        ).refCat

        self.assertFloatsEqual(cat_pm['pm_raErr'], cat['pm_raErr'])
        self.assertFloatsEqual(cat_pm['pm_decErr'], cat['pm_decErr'])

        separations = np.array([cat[i].getCoord().separation(cat_pm[i].getCoord()).asArcseconds()
                                for i in range(len(cat))])
        bearings = np.array([cat[i].getCoord().bearingTo(cat_pm[i].getCoord()).asArcseconds()
                             for i in range(len(cat))])
        reverse_proper_motion_dir = self.properMotionDir + 180 * lsst.geom.degrees
        self.assertFloatsAlmostEqual(separations, self.properMotionAmt.asArcseconds(), rtol=1.0e-10)
        self.assertFloatsAlmostEqual(bearings, reverse_proper_motion_dir.asArcseconds(), rtol=1.0e-10)

        predictedRaErr = np.hypot(cat["coord_raErr"], cat["pm_raErr"])
        predictedDecErr = np.hypot(cat["coord_decErr"], cat["pm_decErr"])
        self.assertFloatsAlmostEqual(cat_pm["coord_raErr"], predictedRaErr)
        self.assertFloatsAlmostEqual(cat_pm["coord_decErr"], predictedDecErr)

    def test_requireProperMotion(self):
        """Tests of the requireProperMotion config field."""
        loaderConfig = ReferenceObjectLoader.ConfigClass()
        loaderConfig.requireProperMotion = True
        loader = ReferenceObjectLoader([dataRef.dataId for dataRef in self.datasetRefs],
                                       self.handles,
                                       name="testrefcat",
                                       config=loaderConfig)
        center = lsst.geom.SpherePoint(180.0*lsst.geom.degrees, 0.0*lsst.geom.degrees)

        # Test that we require an epoch set.
        msg = 'requireProperMotion=True but epoch not provided to loader'
        with self.assertRaisesRegex(RuntimeError, msg):
            loader.loadSkyCircle(
                center,
                30.0*lsst.geom.degrees,
                filterName='a'
            )


class Version0Version1ReferenceObjectLoaderTestCase(lsst.utils.tests.TestCase):
    """Test cases for reading version 0 and version 1 catalogs."""
    def testLoadVersion0(self):
        """Attempting to read version 0 refcats should raise.
        """
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'version0',
            'ref_cats',
            'cal_ref_cat'
        )

        filenames = sorted(glob.glob(os.path.join(path, '????.fits')))

        loader = MockReferenceObjectLoaderFromFiles(filenames, name='cal_ref_cat', htmLevel=4)
        with self.assertRaisesRegex(ValueError, "Version 0 refcats are no longer supported"):
            loader.loadSkyCircle(convertReferenceCatalogTestBase.make_coord(10, 20),
                                 5*lsst.geom.degrees,
                                 'a')

    def testLoadVersion1(self):
        """Test reading a format_version=1 catalog (fluxes unchanged)."""
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'version1',
            'ref_cats',
            'cal_ref_cat'
        )

        filenames = sorted(glob.glob(os.path.join(path, '????.fits')))

        loader = MockReferenceObjectLoaderFromFiles(filenames, name='cal_ref_cat', htmLevel=4)
        result = loader.loadSkyCircle(convertReferenceCatalogTestBase.make_coord(10, 20),
                                      5*lsst.geom.degrees,
                                      'a')

        # version>=1 should not change units on read (they're already nJy).
        catalog = afwTable.SimpleCatalog.readFits(filenames[0])
        self.assertFloatsEqual(catalog['a_flux'], result.refCat['a_flux'])
        self.assertFloatsEqual(catalog['a_fluxErr'], result.refCat['a_fluxErr'])
        self.assertFloatsEqual(catalog['b_flux'], result.refCat['b_flux'])
        self.assertFloatsEqual(catalog['b_fluxErr'], result.refCat['b_fluxErr'])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
