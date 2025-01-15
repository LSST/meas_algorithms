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

__all__ = ["getRefFluxField", "getRefFluxKeys", "LoadReferenceObjectsConfig",
           "ReferenceObjectLoader"]

import logging

import astropy.time
import astropy.units
import numpy

import lsst.geom as geom
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst import sphgeom
from lsst.daf.base import PropertyList

from .convertReferenceCatalog import LATEST_FORMAT_VERSION


def getFormatVersionFromRefCat(refCat):
    """"Return the format version stored in a reference catalog header.

    Parameters
    ----------
    refCat : `lsst.afw.table.SimpleCatalog`
        Reference catalog to inspect.

    Returns
    -------
    version : `int`
        Format version integer.

    Raises
    ------
    ValueError
        Raised if the catalog is version 0, has no metadata, or does not
        include a "REFCAT_FORMAT_VERSION" key.
    """
    errMsg = "Version 0 refcats are no longer supported: refcat fluxes must have nJy units."
    md = refCat.getMetadata()
    if md is None:
        raise ValueError(f"No metadata found in refcat header. {errMsg}")

    try:
        version = md.getScalar("REFCAT_FORMAT_VERSION")
        if version == 0:
            raise ValueError(errMsg)
        else:
            return version
    except KeyError:
        raise ValueError(f"No version number found in refcat header metadata. {errMsg}")


class _FilterCatalog:
    """This is a private helper class which filters catalogs by
    row based on the row being inside the region used to initialize
    the class.

    Parameters
    ----------
    region : `lsst.sphgeom.Region`
        The spatial region which all objects should lie within
    """
    def __init__(self, region):
        self.region = region

    def __call__(self, refCat, catRegion):
        """This call method on an instance of this class takes in a reference
        catalog, and the region from which the catalog was generated.

        If the catalog region is entirely contained within the region used to
        initialize this class, then all the entries in the catalog must be
        within the region and so the whole catalog is returned.

        If the catalog region is not entirely contained, then the location for
        each record is tested against the region used to initialize the class.
        Records which fall inside this region are added to a new catalog, and
        this catalog is then returned.

        Parameters
        ---------
        refCat : `lsst.afw.table.SourceCatalog`
            SourceCatalog to be filtered.
        catRegion : `lsst.sphgeom.Region`
            Region in which the catalog was created
        """
        if catRegion.isWithin(self.region):
            # no filtering needed, region completely contains refcat
            return refCat

        filteredRefCat = type(refCat)(refCat.table)
        for record in refCat:
            if self.region.contains(record.getCoord().getVector()):
                filteredRefCat.append(record)
        return filteredRefCat


class LoadReferenceObjectsConfig(pexConfig.Config):
    pixelMargin = pexConfig.RangeField(
        doc="Padding to add to 4 all edges of the bounding box (pixels)",
        dtype=int,
        default=250,
        min=0,
    )
    anyFilterMapsToThis = pexConfig.Field(
        doc=("Always use this reference catalog filter, no matter whether or what filter name is "
             "supplied to the loader. Effectively a trivial filterMap: map all filter names to this filter."
             " This can be set for purely-astrometric catalogs (e.g. Gaia DR2) where there is only one "
             "reasonable choice for every camera filter->refcat mapping, but not for refcats used for "
             "photometry, which need a filterMap and/or colorterms/transmission corrections."),
        dtype=str,
        default=None,
        optional=True
    )
    filterMap = pexConfig.DictField(
        doc=("Mapping of camera filter name: reference catalog filter name; "
             "each reference filter must exist in the refcat."
             " Note that this does not perform any bandpass corrections: it is just a lookup."),
        keytype=str,
        itemtype=str,
        default={},
    )
    requireProperMotion = pexConfig.Field(
        doc="Require that the fields needed to correct proper motion "
            "(epoch, pm_ra and pm_dec) are present?",
        dtype=bool,
        default=False,
    )

    def validate(self):
        super().validate()
        if self.filterMap != {} and self.anyFilterMapsToThis is not None:
            msg = "`filterMap` and `anyFilterMapsToThis` are mutually exclusive"
            raise pexConfig.FieldValidationError(LoadReferenceObjectsConfig.anyFilterMapsToThis,
                                                 self, msg)


class ReferenceObjectLoader:
    """This class facilitates loading reference catalogs.

    The QuantumGraph generation will create a list of datasets that may
    possibly overlap a given region. These datasets are then used to construct
    an instance of this class. The class instance should then be passed into
    a task which needs reference catalogs. These tasks should then determine
    the exact region of the sky reference catalogs will be loaded for, and
    call a corresponding method to load the reference objects.

    Parameters
    ----------
    dataIds : iterable of `lsst.daf.butler.DataCoordinate`
        An iterable object of data IDs that point to reference catalogs.
    refCats : iterable of `lsst.daf.butler.DeferredDatasetHandle`
        Handles to load refCats on demand.
    name : `str`, optional
        The name of the refcat that this object will load. This name is used
        for applying colorterms, for example.
    config : `LoadReferenceObjectsConfig`
        Configuration of this reference loader.
    log : `lsst.log.Log`, `logging.Logger` or `None`, optional
        Logger object used to write out messages. If `None` a default
        logger will be used.
    """
    ConfigClass = LoadReferenceObjectsConfig

    def __init__(self, dataIds, refCats, name=None, log=None, config=None):
        if config is None:
            config = self.ConfigClass()
        self.config = config
        self.dataIds = dataIds
        self.refCats = list(refCats)
        self.name = name
        self.log = log or logging.getLogger(__name__).getChild("ReferenceObjectLoader")

    def applyProperMotions(self, catalog, epoch):
        """Apply proper motion correction to a reference catalog.

        Adjust position and position error in the ``catalog``
        for proper motion to the specified ``epoch``,
        modifying the catalog in place.

        Parameters
        ----------
        catalog : `lsst.afw.table.SimpleCatalog`
            Catalog of positions, containing at least these fields:

            - Coordinates, retrieved by the table's coordinate key.
            - ``coord_raErr`` : Error in Right Ascension (rad).
            - ``coord_decErr`` : Error in Declination (rad).
            - ``pm_ra`` : Proper motion in Right Ascension (rad/yr,
                East positive)
            - ``pm_raErr`` : Error in ``pm_ra`` (rad/yr), optional.
            - ``pm_dec`` : Proper motion in Declination (rad/yr,
                North positive)
            - ``pm_decErr`` : Error in ``pm_dec`` (rad/yr), optional.
            - ``epoch`` : Mean epoch of object (an astropy.time.Time)
        epoch : `astropy.time.Time`
            Epoch to which to correct proper motion.
            If None, do not apply PM corrections or raise if
            ``config.requireProperMotion`` is True.

        Raises
        ------
        RuntimeError
            Raised if ``config.requireProperMotion`` is set but we cannot
            apply the proper motion correction for some reason.
        """
        if epoch is None:
            if self.config.requireProperMotion:
                raise RuntimeError("requireProperMotion=True but epoch not provided to loader.")
            else:
                self.log.debug("No epoch provided: not applying proper motion corrections to refcat.")
                return

        # Warn/raise for a catalog in an incorrect format, if epoch was specified.
        if "pm_ra" in catalog.schema:
            pm_ra_radians = False
            field = catalog.schema["pm_ra"].asField()
            if field.getTypeString() == "Angle" or field.getUnits() == "rad":
                pm_ra_radians = True

            if self.config.requireProperMotion and not pm_ra_radians:
                raise RuntimeError(
                    "requireProperMotion=True but refcat pm_ra field is not an Angle or radians.",
                )
            elif not pm_ra_radians:
                self.log.warning(
                    "Reference catalog pm_ra field is not an Angle or radians; cannot apply proper motion.",
                )
                return

        if ("epoch" not in catalog.schema or "pm_ra" not in catalog.schema):
            if self.config.requireProperMotion:
                raise RuntimeError("requireProperMotion=True but PM data not available from catalog.")
            else:
                self.log.warning("Proper motion correction not available for this reference catalog.")
            return

        applyProperMotionsImpl(self.log, catalog, epoch)

    @staticmethod
    def _remapReferenceCatalogSchema(refCat, *, anyFilterMapsToThis=None,
                                     filterMap=None, centroids=False):
        """This function takes in a reference catalog and returns a new catalog
        with additional columns defined from the remaining function arguments.

        Parameters
        ----------
        refCat : `lsst.afw.table.SimpleCatalog`
            Reference catalog to map to new catalog
        anyFilterMapsToThis : `str`, optional
            Always use this reference catalog filter.
            Mutually exclusive with `filterMap`
        filterMap : `dict` [`str`,`str`], optional
            Mapping of camera filter name: reference catalog filter name.
        centroids : `bool`, optional
            Add centroid fields to the loaded Schema. ``loadPixelBox`` expects
            these fields to exist.

        Returns
        -------
        expandedCat : `lsst.afw.table.SimpleCatalog`
            Deep copy of input reference catalog with additional columns added
        """
        if anyFilterMapsToThis or filterMap:
            ReferenceObjectLoader._addFluxAliases(refCat.schema, anyFilterMapsToThis, filterMap)

        mapper = afwTable.SchemaMapper(refCat.schema, True)
        mapper.addMinimalSchema(refCat.schema, True)
        mapper.editOutputSchema().disconnectAliases()

        if centroids:
            # Add and initialize centroid and hasCentroid fields (these are
            # added after loading to avoid wasting space in the saved catalogs).
            # The new fields are automatically initialized to (nan, nan) and
            # False so no need to set them explicitly.
            mapper.editOutputSchema().addField("centroid_x", type=float, doReplace=True)
            mapper.editOutputSchema().addField("centroid_y", type=float, doReplace=True)
            mapper.editOutputSchema().addField("hasCentroid", type="Flag", doReplace=True)
            mapper.editOutputSchema().getAliasMap().set("slot_Centroid", "centroid")

        expandedCat = afwTable.SimpleCatalog(mapper.getOutputSchema())
        expandedCat.setMetadata(refCat.getMetadata())
        expandedCat.extend(refCat, mapper=mapper)

        return expandedCat

    @staticmethod
    def _addFluxAliases(schema, anyFilterMapsToThis=None, filterMap=None):
        """Add aliases for camera filter fluxes to the schema.

        For each camFilter: refFilter in filterMap, adds these aliases:
            <camFilter>_camFlux:      <refFilter>_flux
            <camFilter>_camFluxErr: <refFilter>_fluxErr, if the latter exists
        or sets `anyFilterMapsToThis` in the schema.

        Parameters
        ----------
        schema : `lsst.afw.table.Schema`
            Schema for reference catalog.
        anyFilterMapsToThis : `str`, optional
            Always use this reference catalog filter.
            Mutually exclusive with `filterMap`.
        filterMap : `dict` [`str`,`str`], optional
            Mapping of camera filter name: reference catalog filter name.
            Mutually exclusive with `anyFilterMapsToThis`.

        Raises
        ------
        RuntimeError
            Raised if any required reference flux field is missing from the
            schema.
        """
        # Fail on any truthy value for either of these.
        if anyFilterMapsToThis and filterMap:
            raise ValueError("anyFilterMapsToThis and filterMap are mutually exclusive!")

        aliasMap = schema.getAliasMap()

        if anyFilterMapsToThis is not None:
            refFluxName = anyFilterMapsToThis + "_flux"
            if refFluxName not in schema:
                msg = f"Unknown reference filter for anyFilterMapsToThis='{refFluxName}'"
                raise RuntimeError(msg)
            aliasMap.set("anyFilterMapsToThis", refFluxName)
            return  # this is mutually exclusive with filterMap

        def addAliasesForOneFilter(filterName, refFilterName):
            """Add aliases for a single filter

            Parameters
            ----------
            filterName : `str` (optional)
                Camera filter name. The resulting alias name is
                <filterName>_camFlux
            refFilterName : `str`
                Reference catalog filter name; the field
                <refFilterName>_flux must exist.
            """
            camFluxName = filterName + "_camFlux"
            refFluxName = refFilterName + "_flux"
            if refFluxName not in schema:
                raise RuntimeError("Unknown reference filter %s" % (refFluxName,))
            aliasMap.set(camFluxName, refFluxName)
            refFluxErrName = refFluxName + "Err"
            if refFluxErrName in schema:
                camFluxErrName = camFluxName + "Err"
                aliasMap.set(camFluxErrName, refFluxErrName)

        if filterMap is not None:
            for filterName, refFilterName in filterMap.items():
                addAliasesForOneFilter(filterName, refFilterName)

    @staticmethod
    def _makeBoxRegion(BBox, wcs, BBoxPadding):
        outerLocalBBox = geom.Box2D(BBox)
        innerLocalBBox = geom.Box2D(BBox)

        # Grow the bounding box to allow for effects not fully captured by the
        # wcs provided (which represents the current best-guess wcs solution
        # associated with the dataset for which the calibration is to be
        # computed using the loaded and trimmed reference catalog being defined
        # here).  These effects could include pointing errors and/or an
        # insufficient optical distorition model for the instrument.  The idea
        # is to ensure the spherical geometric region created contains the
        # entire region covered by the bbox.
        # Also create an inner region that is sure to be inside the bbox.
        outerLocalBBox.grow(BBoxPadding)
        innerLocalBBox.grow(-1*BBoxPadding)

        # Handle the case where the inner bounding box shrank to a zero sized
        # region (which will be the case if the shrunken size of either
        # dimension is less than or equal to zero).  In this case, the inner
        # bounding box is set to the original input bounding box.  This is
        # probably not the best way to handle an empty inner bounding box, but
        # it is what the calling code currently expects.
        if innerLocalBBox.getDimensions() == geom.Extent2D(0, 0):
            innerLocalBBox = geom.Box2D(BBox)

        # Convert the corners of the bounding boxes to sky coordinates.
        innerBoxCorners = innerLocalBBox.getCorners()
        innerSphCorners = [wcs.pixelToSky(corner).getVector() for corner in innerBoxCorners]
        innerSkyRegion = sphgeom.ConvexPolygon(innerSphCorners)

        outerBoxCorners = outerLocalBBox.getCorners()
        outerSphCorners = [wcs.pixelToSky(corner).getVector() for corner in outerBoxCorners]
        outerSkyRegion = sphgeom.ConvexPolygon(outerSphCorners)

        return innerSkyRegion, outerSkyRegion, innerSphCorners, outerSphCorners

    @staticmethod
    def _calculateCircle(bbox, wcs, pixelMargin):
        """Compute on-sky center and radius of search region.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            Pixel bounding box.
        wcs : `lsst.afw.geom.SkyWcs`
            WCS; used to convert pixel positions to sky coordinates.
        pixelMargin : `int`
            Padding to add to 4 all edges of the bounding box (pixels).

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            A Struct containing:

            - coord : `lsst.geom.SpherePoint`
                ICRS center of the search region.
            - radius : `lsst.geom.Angle`
                Radius of the search region.
            - bbox : `lsst.geom.Box2D`
                Bounding box used to compute the circle.
        """
        bbox = geom.Box2D(bbox)  # we modify the box, so use a copy
        bbox.grow(pixelMargin)
        coord = wcs.pixelToSky(bbox.getCenter())
        radius = max(coord.separation(wcs.pixelToSky(pp)) for pp in bbox.getCorners())
        return pipeBase.Struct(coord=coord, radius=radius, bbox=bbox)

    @staticmethod
    def getMetadataCircle(coord, radius, filterName, epoch=None):
        """Return metadata about the loaded reference catalog, in an on-sky
        circle.

        This metadata is used for reloading the catalog (e.g. for
        reconstituting a normalized match list).

        Parameters
        ----------
        coord : `lsst.geom.SpherePoint`
            ICRS center of the search region.
        radius : `lsst.geom.Angle`
            Radius of the search region.
        filterName : `str`
            Name of the camera filter.
        epoch : `astropy.time.Time` or `None`, optional
            Epoch that proper motion and parallax were corrected to, or `None`
            if no such corrections were applied.

        Returns
        -------
        md : `lsst.daf.base.PropertyList`
            Metadata about the catalog.
        """
        md = PropertyList()
        md.add('RA', coord.getRa().asDegrees(), 'field center in degrees')
        md.add('DEC', coord.getDec().asDegrees(), 'field center in degrees')
        md.add('RADIUS', radius.asDegrees(), 'field radius in degrees, minimum')
        # Version 1: Initial version
        # Version 2: JEPOCH for TAI Julian Epoch year of PM/parallax correction
        md.add('SMATCHV', 2, 'SourceMatchVector version number')
        md.add('FILTER', filterName, 'camera filter name for photometric data')
        md.add('TIMESYS', "TAI", "time scale of time keywords")
        md.add('JEPOCH', None if epoch is None else epoch.tai.jyear,
               'Julian epoch (TAI Julian Epoch year) for catalog')
        return md

    def getMetadataBox(self, bbox, wcs, filterName, epoch=None,
                       bboxToSpherePadding=100):
        """Return metadata about the loaded reference catalog, in an
        on-detector box.

        This metadata is used for reloading the catalog (e.g., for
        reconstituting a normalised match list).

        Parameters
        ----------
        bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            Bounding box for the pixels.
        wcs : `lsst.afw.geom.SkyWcs`
            The WCS object associated with ``bbox``.
        filterName : `str`
            Name of the camera filter.
        epoch : `astropy.time.Time` or `None`,  optional
            Epoch that proper motion and parallax were corrected to, or `None`
            if no such corrections were applied.
        bboxToSpherePadding : `int`, optional
            Padding in pixels to account for translating a set of corners into
            a spherical (convex) boundary that is certain to encompass the
            enitre area covered by the box.

        Returns
        -------
        md : `lsst.daf.base.PropertyList`
            The metadata detailing the search parameters used for this
            dataset.
        """
        circle = self._calculateCircle(bbox, wcs, self.config.pixelMargin)
        md = self.getMetadataCircle(circle.coord, circle.radius, filterName, epoch=epoch)

        paddedBbox = circle.bbox
        _, _, innerCorners, outerCorners = self._makeBoxRegion(paddedBbox, wcs, bboxToSpherePadding)
        for box, corners in zip(("INNER", "OUTER"), (innerCorners, outerCorners)):
            for (name, corner) in zip(("UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT"),
                                      corners):
                md.add(f"{box}_{name}_RA", geom.SpherePoint(corner).getRa().asDegrees(), f"{box}_corner")
                md.add(f"{box}_{name}_DEC", geom.SpherePoint(corner).getDec().asDegrees(), f"{box}_corner")
        return md

    def loadPixelBox(self, bbox, wcs, filterName, epoch=None,
                     bboxToSpherePadding=100):
        """Load reference objects that are within a pixel-based rectangular
        region.

        This algorithm works by creating a spherical box whose corners
        correspond to the WCS converted corners of the input bounding box
        (possibly padded).  It then defines a filtering function which looks at
        the pixel position of the reference objects and accepts only those that
        lie within the specified bounding box.

        The spherical box region and filtering function are passed to the
        generic loadRegion method which loads and filters the reference objects
        from the datastore and returns a single catalog containing the filtered
        set of reference objects.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            Box which bounds a region in pixel space.
        wcs : `lsst.afw.geom.SkyWcs`
            Wcs object defining the pixel to sky (and inverse) transform for
            the supplied ``bbox``.
        filterName : `str`
            Name of camera filter.
        epoch : `astropy.time.Time` or `None`, optional
            Epoch to which to correct proper motion and parallax, or `None`
            to not apply such corrections.
        bboxToSpherePadding : `int`, optional
            Padding to account for translating a set of corners into a
            spherical (convex) boundary that is certain to encompase the
            enitre area covered by the box.

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``refCat``
                Catalog containing reference objects inside the specified
                bounding box (padded by self.config.pixelMargin).
            ``fluxField``
                Name of the field containing the flux associated with
                ``filterName``.

        Raises
        ------
        RuntimeError
            Raised if no reference catalogs could be found for the specified
            region.
        TypeError
            Raised if the loaded reference catalogs do not have matching
            schemas.
        """
        paddedBbox = geom.Box2D(bbox)
        paddedBbox.grow(self.config.pixelMargin)
        innerSkyRegion, outerSkyRegion, _, _ = self._makeBoxRegion(paddedBbox, wcs, bboxToSpherePadding)

        def _filterFunction(refCat, region):
            # Perform an initial "pre filter" step based on the refCat coords
            # and the outerSkyRegion created from the self.config.pixelMargin-
            # paddedBbox plus an "extra" padding of bboxToSpherePadding and the
            # raw wcs.  This should ensure a large enough projected area on the
            # sky that accounts for any projection/distortion issues, but small
            # enough to filter out loaded reference objects that lie well
            # beyond the projected detector of interest.  This step is required
            # due to the very local nature of the wcs available for the
            # sky <--> pixel conversions.
            preFiltFunc = _FilterCatalog(outerSkyRegion)
            refCat = preFiltFunc(refCat, region)

            # Add columns to the pre-filtered reference catalog relating their
            # coordinates to equivalent pixel positions for the wcs provided
            # and use to populate those columns.
            refCat = self._remapReferenceCatalogSchema(refCat, centroids=True)
            afwTable.updateRefCentroids(wcs, refCat)
            # No need to filter the catalog if it is entirely contained in the
            # region defined by the inner sky region.
            if innerSkyRegion.contains(region):
                return refCat
            # Create a new reference catalog, and populate it only with records
            # that fall inside the padded bbox.
            filteredRefCat = type(refCat)(refCat.table)
            centroidKey = afwTable.Point2DKey(refCat.schema['centroid'])
            for record in refCat:
                pixCoords = record[centroidKey]
                if paddedBbox.contains(geom.Point2D(pixCoords)):
                    filteredRefCat.append(record)
            return filteredRefCat
        return self.loadRegion(outerSkyRegion, filterName, filtFunc=_filterFunction, epoch=epoch)

    def loadRegion(self, region, filterName, filtFunc=None, epoch=None):
        """Load reference objects within a specified region.

        This function loads the DataIds used to construct an instance of this
        class which intersect or are contained within the specified region. The
        reference catalogs which intersect but are not fully contained within
        the input region are further filtered by the specified filter function.
        This function returns a single source catalog containing all reference
        objects inside the specified region.

        Parameters
        ----------
        region : `lsst.sphgeom.Region`
            This can be any type that is derived from `lsst.sphgeom.Region` and
            should define the spatial region for which reference objects are to
            be loaded.
        filtFunc : callable or `None`, optional
            This optional parameter should be a callable object that takes a
            reference catalog and its corresponding region as parameters,
            filters the catalog by some criteria and returns the filtered
            reference catalog. If `None`, an internal filter function is used
            which filters according to if a reference object falls within the
            input region.
        filterName : `str`
            Name of camera filter.
        epoch : `astropy.time.Time` or `None`, optional
            Epoch to which to correct proper motion and parallax, or `None` to
            not apply such corrections.

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``refCat``
                Catalog containing reference objects which intersect the
                input region, filtered by the specified filter function.
            ``fluxField``
                Name of the field containing the flux associated with
                ``filterName``.

        Raises
        ------
        RuntimeError
            Raised if no reference catalogs could be found for the specified
            region.
        TypeError
            Raised if the loaded reference catalogs do not have matching
            schemas.
        """
        regionLat = region.getBoundingBox().getLat()
        regionLon = region.getBoundingBox().getLon()
        self.log.info("Loading reference objects from %s in region bounded by "
                      "[%.8f, %.8f], [%.8f, %.8f] RA Dec",
                      self.name,
                      regionLon.getA().asDegrees(), regionLon.getB().asDegrees(),
                      regionLat.getA().asDegrees(), regionLat.getB().asDegrees())
        if filtFunc is None:
            filtFunc = _FilterCatalog(region)
        # filter out all the regions supplied by the constructor that do not overlap
        overlapList = []
        for dataId, refCat in zip(self.dataIds, self.refCats):
            # SphGeom supports some objects intersecting others, but is not symmetric,
            # try the intersect operation in both directions
            try:
                intersects = dataId.region.intersects(region)
            except TypeError:
                intersects = region.intersects(dataId.region)

            if intersects:
                overlapList.append((dataId, refCat))

        if len(overlapList) == 0:
            raise RuntimeError("No reference tables could be found for input region")

        firstCat = overlapList[0][1].get()
        refCat = filtFunc(firstCat, overlapList[0][0].region)
        trimmedAmount = len(firstCat) - len(refCat)

        # Load in the remaining catalogs
        for dataId, inputRefCat in overlapList[1:]:
            tmpCat = inputRefCat.get()

            if tmpCat.schema != firstCat.schema:
                raise TypeError("Reference catalogs have mismatching schemas")

            filteredCat = filtFunc(tmpCat, dataId.region)
            refCat.extend(filteredCat)
            trimmedAmount += len(tmpCat) - len(filteredCat)

        self.log.debug("Trimmed %d refCat objects lying outside padded region, leaving %d",
                       trimmedAmount, len(refCat))
        self.log.info("Loaded %d reference objects", len(refCat))

        # Ensure that the loaded reference catalog is continuous in memory
        if not refCat.isContiguous():
            refCat = refCat.copy(deep=True)

        self.applyProperMotions(refCat, epoch)

        expandedCat = self._remapReferenceCatalogSchema(refCat,
                                                        anyFilterMapsToThis=self.config.anyFilterMapsToThis,
                                                        filterMap=self.config.filterMap)

        # Ensure that the returned reference catalog is continuous in memory
        if not expandedCat.isContiguous():
            expandedCat = expandedCat.copy(deep=True)

        fluxField = getRefFluxField(expandedCat.schema, filterName)

        if expandedCat.schema[fluxField].asField().getUnits() != "nJy":
            # if the flux field is not in nJy, check the refcat format version
            version = getFormatVersionFromRefCat(refCat)
            if version > LATEST_FORMAT_VERSION:
                raise ValueError(f"Unsupported refcat format version: {version} > {LATEST_FORMAT_VERSION}.")

        return pipeBase.Struct(refCat=expandedCat, fluxField=fluxField)

    def loadSkyCircle(self, ctrCoord, radius, filterName, epoch=None):
        """Load reference objects that lie within a circular region on the sky.

        This method constructs a circular region from an input center and
        angular radius, loads reference catalogs which are contained in or
        intersect the circle, and filters reference catalogs which intersect
        down to objects which lie within the defined circle.

        Parameters
        ----------
        ctrCoord : `lsst.geom.SpherePoint`
            Point defining the center of the circular region.
        radius : `lsst.geom.Angle`
            Defines the angular radius of the circular region.
        filterName : `str`
            Name of camera filter.
        epoch : `astropy.time.Time` or `None`, optional
            Epoch to which to correct proper motion and parallax, or `None` to
            not apply such corrections.

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``refCat``
                Catalog containing reference objects inside the specified
                search circle.
            ``fluxField``
                Name of the field containing the flux associated with
                ``filterName``.
        """
        centerVector = ctrCoord.getVector()
        sphRadius = sphgeom.Angle(radius.asRadians())
        circularRegion = sphgeom.Circle(centerVector, sphRadius)
        return self.loadRegion(circularRegion, filterName, epoch=epoch)

    def loadSchema(self, filterName):
        """Load the schema for the reference catalog.

        Parameters
        ----------
        filterName : `str`
            Name of camera filter.

         Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``schema``
                Schema of the reference catalogs returned by other 'load'
                methods.
            ``fluxField``
                Name of the field containing the flux associated with
                ``filterName``.
        """
        if not self.refCats:
            raise RuntimeError("No reference tables could be found.")
        cat = self.refCats[0].get()
        # Replace the original handle with an in-memory one that caches what
        # we've already read, since there's a good chance we'll want to read it
        # later.
        self.refCats[0] = pipeBase.InMemoryDatasetHandle(cat, dataId=self.refCats[0].dataId, copy=False)
        emptyCat = type(cat)(cat.table.clone())
        expandedEmptyCat = self._remapReferenceCatalogSchema(
            emptyCat,
            anyFilterMapsToThis=self.config.anyFilterMapsToThis,
            filterMap=self.config.filterMap,
        )
        fluxField = getRefFluxField(expandedEmptyCat.schema, filterName)
        if expandedEmptyCat.schema[fluxField].asField().getUnits() != "nJy":
            # if the flux field is not in nJy, check the refcat format version
            version = getFormatVersionFromRefCat(emptyCat)
            if version > LATEST_FORMAT_VERSION:
                raise ValueError(f"Unsupported refcat format version: {version} > {LATEST_FORMAT_VERSION}.")
        return pipeBase.Struct(schema=expandedEmptyCat.schema, fluxField=fluxField)


def getRefFluxField(schema, filterName):
    """Get the name of a flux field from a schema.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`
        Reference catalog schema.
    filterName : `str`
        Name of camera filter.

    Returns
    -------
    fluxFieldName : `str`
        Name of flux field.

    Notes
    -----
    Return the alias of ``anyFilterMapsToThis``, if present
    else, return ``*filterName*_camFlux`` if present,
    else, return ``*filterName*_flux`` if present (camera filter name
    matches reference filter name), else raise an exception.

    Raises
    ------
    RuntimeError
        Raised if an appropriate field is not found.
    """
    if not isinstance(schema, afwTable.Schema):
        raise RuntimeError("schema=%s is not a schema" % (schema,))
    try:
        return schema.getAliasMap().get("anyFilterMapsToThis")
    except LookupError:
        pass  # try the filterMap next

    fluxFieldList = [filterName + "_camFlux", filterName + "_flux"]
    for fluxField in fluxFieldList:
        if fluxField in schema:
            return fluxField

    raise RuntimeError("Could not find flux field(s) %s" % (", ".join(fluxFieldList)))


def getRefFluxKeys(schema, filterName):
    """Return keys for flux and flux error.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`
        Reference catalog schema.
    filterName : `str`
        Name of camera filter.

    Returns
    -------
    keys : `tuple` of (`lsst.afw.table.Key`, `lsst.afw.table.Key`)
        Two keys:

        - flux key
        - flux error key, if present, else None

    Raises
    ------
    RuntimeError
        If flux field not found.
    """
    fluxField = getRefFluxField(schema, filterName)
    fluxErrField = fluxField + "Err"
    fluxKey = schema[fluxField].asKey()
    try:
        fluxErrKey = schema[fluxErrField].asKey()
    except Exception:
        fluxErrKey = None
    return (fluxKey, fluxErrKey)


def applyProperMotionsImpl(log, catalog, epoch):
    """Apply proper motion correction to a reference catalog.

    Adjust position and position error in the ``catalog``
    for proper motion to the specified ``epoch``,
    modifying the catalog in place.

    Parameters
    ----------
    log : `lsst.log.Log` or `logging.Logger`
        Log object to write to.
    catalog : `lsst.afw.table.SimpleCatalog`
        Catalog of positions, containing:

        - Coordinates, retrieved by the table's coordinate key.
        - ``coord_raErr`` : Error in Right Ascension (rad).
        - ``coord_decErr`` : Error in Declination (rad).
        - ``pm_ra`` : Proper motion in Right Ascension (rad/yr,
            East positive)
        - ``pm_raErr`` : Error in ``pm_ra`` (rad/yr), optional.
        - ``pm_dec`` : Proper motion in Declination (rad/yr,
            North positive)
        - ``pm_decErr`` : Error in ``pm_dec`` (rad/yr), optional.
        - ``epoch`` : Mean epoch of object (an astropy.time.Time)
    epoch : `astropy.time.Time`
        Epoch to which to correct proper motion.
    """
    if "epoch" not in catalog.schema or "pm_ra" not in catalog.schema or "pm_dec" not in catalog.schema:
        log.warning("Proper motion correction not available from catalog")
        return
    if not catalog.isContiguous():
        raise RuntimeError("Catalog must be contiguous")
    catEpoch = astropy.time.Time(catalog["epoch"], scale="tai", format="mjd")
    log.info("Correcting reference catalog for proper motion to %r", epoch)
    # Use `epoch.tai` to make sure the time difference is in TAI
    timeDiffsYears = (epoch.tai - catEpoch).to(astropy.units.yr).value
    coordKey = catalog.table.getCoordKey()
    # Compute the offset of each object due to proper motion
    # as components of the arc of a great circle along RA and Dec
    pmRaRad = catalog["pm_ra"]
    pmDecRad = catalog["pm_dec"]
    offsetsRaRad = pmRaRad*timeDiffsYears
    offsetsDecRad = pmDecRad*timeDiffsYears
    # Compute the corresponding bearing and arc length of each offset
    # due to proper motion, and apply the offset.
    # The factor of 1e6 for computing bearing is intended as
    # a reasonable scale for typical values of proper motion
    # in order to avoid large errors for small values of proper motion;
    # using the offsets is another option, but it can give
    # needlessly large errors for short duration.
    offsetBearingsRad = numpy.arctan2(offsetsDecRad*1e6, offsetsRaRad*1e6)
    offsetAmountsRad = numpy.hypot(offsetsRaRad, offsetsDecRad)
    for record, bearingRad, amountRad in zip(catalog, offsetBearingsRad, offsetAmountsRad):
        record.set(coordKey,
                   record.get(coordKey).offset(bearing=bearingRad*geom.radians,
                                               amount=amountRad*geom.radians))
    # TODO DM-36979: this needs to incorporate the full covariance!
    # Increase error in RA and Dec based on error in proper motion
    if "coord_raErr" in catalog.schema:
        catalog["coord_raErr"] = numpy.hypot(catalog["coord_raErr"],
                                             catalog["pm_raErr"]*timeDiffsYears)
    if "coord_decErr" in catalog.schema:
        catalog["coord_decErr"] = numpy.hypot(catalog["coord_decErr"],
                                              catalog["pm_decErr"]*timeDiffsYears)
