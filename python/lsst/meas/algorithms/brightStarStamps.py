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

"""Collection of small images (stamps), each centered on a bright star."""

__all__ = ["BrightStarStamp", "BrightStarStamps"]

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import reduce
from operator import ior

import numpy as np
from lsst.afw.geom import SpanSet, Stencil
from lsst.afw.image import MaskedImageF
from lsst.afw.math import Property, StatisticsControl, makeStatistics, stringToStatisticsProperty
from lsst.afw.table.io import Persistable
from lsst.daf.base import PropertyList
from lsst.geom import Point2I

from .stamps import StampBase, Stamps, readFitsWithOptions

logger = logging.getLogger(__name__)


@dataclass
class BrightStarStamp(StampBase):
    """Single stamp centered on a bright star, normalized by its annularFlux.

    Parameters
    ----------
    stamp_im : `~lsst.afw.image.MaskedImage`
        Pixel data for this postage stamp
    gaiaGMag : `float`
        Gaia G magnitude for the object in this stamp
    gaiaId : `int`
        Gaia object identifier
    position : `~lsst.geom.Point2I`
        Origin of the stamps in its origin exposure (pixels)
    archive_elements : `~collections.abc.Mapping`[ `str` , \
                `~lsst.afw.table.io.Persistable`], optional
            Archive elements (e.g. Transform / WCS) associated with this stamp.
    annularFlux : `float` or None, optional
        Flux in an annulus around the object
    """

    stamp_im: MaskedImageF
    gaiaGMag: float
    gaiaId: int
    position: Point2I
    archive_elements: Mapping[str, Persistable] | None = None
    annularFlux: float | None = None
    minValidAnnulusFraction: float = 0.0
    validAnnulusFraction: float | None = None
    optimalInnerRadius: int | None = None
    optimalOuterRadius: int | None = None

    @classmethod
    def factory(cls, stamp_im, metadata, idx, archive_elements=None, minValidAnnulusFraction=0.0):
        """This method is needed to service the FITS reader. We need a standard
        interface to construct objects like this. Parameters needed to
        construct this object are passed in via a metadata dictionary and then
        passed to the constructor of this class.  This particular factory
        method requires keys: G_MAGS, GAIA_IDS, and ANNULAR_FLUXES. They should
        each point to lists of values.

        Parameters
        ----------
        stamp_im : `~lsst.afw.image.MaskedImage`
            Pixel data to pass to the constructor
        metadata : `dict`
            Dictionary containing the information
            needed by the constructor.
        idx : `int`
            Index into the lists in ``metadata``
        archive_elements : `~collections.abc.Mapping`[ `str` , \
                `~lsst.afw.table.io.Persistable`], optional
            Archive elements (e.g. Transform / WCS) associated with this stamp.
        minValidAnnulusFraction : `float`, optional
            The fraction of valid pixels within the normalization annulus of a
            star.

        Returns
        -------
        brightstarstamp : `BrightStarStamp`
            An instance of this class
        """
        if "X0" in metadata and "Y0" in metadata:
            x0 = metadata["X0"]
            y0 = metadata["X0"]
            position = Point2I(x0, y0)
        else:
            position = None
        return cls(
            stamp_im=stamp_im,
            gaiaGMag=metadata["G_MAGS"],
            gaiaId=metadata["GAIA_IDS"],
            position=position,
            archive_elements=archive_elements,
            annularFlux=metadata["ANNULAR_FLUXES"],
            minValidAnnulusFraction=minValidAnnulusFraction,
            validAnnulusFraction=metadata["VALID_PIXELS_FRACTION"],
        )

    def _getMaskedImage(self):
        return self.stamp_im

    def _getArchiveElements(self):
        return self.archive_elements

    def _getMetadata(self) -> PropertyList | None:
        md = PropertyList()
        md["G_MAG"] = self.gaiaGMag
        md["GAIA_ID"] = self.gaiaId
        md["X0"] = self.position.x
        md["Y0"] = self.position.y
        md["ANNULAR_FLUX"] = self.annularFlux
        md["VALID_PIXELS_FRACTION"] = self.validAnnulusFraction
        return md

    def measureAndNormalize(
        self,
        annulus: SpanSet,
        statsControl: StatisticsControl = StatisticsControl(),
        statsFlag: Property = stringToStatisticsProperty("MEAN"),
        badMaskPlanes: Collection[str] = ("BAD", "SAT", "NO_DATA"),
    ):
        """Compute "annularFlux", the integrated flux within an annulus
        around an object's center, and normalize it.

        Since the center of bright stars are saturated and/or heavily affected
        by ghosts, we measure their flux in an annulus with a large enough
        inner radius to avoid the most severe ghosts and contain enough
        non-saturated pixels.

        Parameters
        ----------
        annulus : `~lsst.afw.geom.spanSet.SpanSet`
            SpanSet containing the annulus to use for normalization.
        statsControl : `~lsst.afw.math.statistics.StatisticsControl`, optional
            StatisticsControl to be used when computing flux over all pixels
            within the annulus.
        statsFlag : `~lsst.afw.math.statistics.Property`, optional
            statsFlag to be passed on to ``afwMath.makeStatistics`` to compute
            annularFlux. Defaults to a simple MEAN.
        badMaskPlanes : `collections.abc.Collection` [`str`]
            Collection of mask planes to ignore when computing annularFlux.
        """
        stampSize = self.stamp_im.getDimensions()
        # Create image: science pixel values within annulus, NO_DATA elsewhere
        maskPlaneDict = self.stamp_im.mask.getMaskPlaneDict()
        annulusImage = MaskedImageF(stampSize, planeDict=maskPlaneDict)
        annulusMask = annulusImage.mask
        annulusMask.array[:] = 2 ** maskPlaneDict["NO_DATA"]
        annulus.copyMaskedImage(self.stamp_im, annulusImage)
        # Set mask planes to be ignored.
        andMask = reduce(ior, (annulusMask.getPlaneBitMask(bm) for bm in badMaskPlanes))
        statsControl.setAndMask(andMask)

        annulusStat = makeStatistics(annulusImage, statsFlag, statsControl)
        # Determine the number of valid (unmasked) pixels within the annulus.
        unMasked = annulusMask.array.size - np.count_nonzero(annulusMask.array)
        self.validAnnulusFraction = unMasked / annulus.getArea()
        logger.info(
            "The Star's annulus contains %s valid pixels and the annulus itself contains %s pixels.",
            unMasked,
            annulus.getArea(),
        )
        if unMasked > (annulus.getArea() * self.minValidAnnulusFraction):
            # Compute annularFlux.
            self.annularFlux = annulusStat.getValue()
            logger.info("Annular flux is: %s", self.annularFlux)
        else:
            raise RuntimeError(
                f"Less than {self.minValidAnnulusFraction * 100}% of pixels within the annulus are valid."
            )
        if np.isnan(self.annularFlux):
            raise RuntimeError("Annular flux computation failed, likely because there are no valid pixels.")
        if self.annularFlux < 0:
            raise RuntimeError("The annular flux is negative. The stamp can not be normalized!")
        # Normalize stamps.
        self.stamp_im.image.array /= self.annularFlux
        return None


class BrightStarStamps(Stamps):
    """Collection of bright star stamps and associated metadata.

    Parameters
    ----------
    starStamps : `collections.abc.Sequence` [`BrightStarStamp`]
        Sequence of star stamps. Cannot contain both normalized and
        unnormalized stamps.
    innerRadius : `int`, optional
        Inner radius value, in pixels. This and ``outerRadius`` define the
        annulus used to compute the ``"annularFlux"`` values within each
        ``starStamp``. Must be provided if ``normalize`` is True.
    outerRadius : `int`, optional
        Outer radius value, in pixels. This and ``innerRadius`` define the
        annulus used to compute the ``"annularFlux"`` values within each
        ``starStamp``. Must be provided if ``normalize`` is True.
    nb90Rots : `int`, optional
        Number of 90 degree rotations required to compensate for detector
        orientation.
    metadata : `~lsst.daf.base.PropertyList`, optional
        Metadata associated with the bright stars.
    use_mask : `bool`
        If `True` read and write mask data. Default `True`.
    use_variance : `bool`
        If ``True`` read and write variance data. Default ``False``.
    use_archive : `bool`
        If ``True`` read and write an Archive that contains a Persistable
        associated with each stamp. In the case of bright stars, this is
        usually a ``TransformPoint2ToPoint2``, used to warp each stamp
        to the same pixel grid before stacking.

    Raises
    ------
    ValueError
        Raised if one of the star stamps provided does not contain the
        required keys.
    AttributeError
        Raised if there is a mix-and-match of normalized and unnormalized
        stamps, stamps normalized with different annulus definitions, or if
        stamps are to be normalized but annular radii were not provided.

    Notes
    -----
    A butler can be used to read only a part of the stamps, specified by a
    bbox:

    >>> starSubregions = butler.get(
            "brightStarStamps",
            dataId,
            parameters={"bbox": bbox}
        )
    """

    def __init__(
        self,
        starStamps,
        innerRadius=None,
        outerRadius=None,
        nb90Rots=None,
        metadata=None,
        use_mask=True,
        use_variance=False,
        use_archive=False,
    ):
        super().__init__(starStamps, metadata, use_mask, use_variance, use_archive)
        # From v2 onwards, stamps are now always assumed to be unnormalized
        self.normalized = False
        self.nb90Rots = nb90Rots

    @classmethod
    def initAndNormalize(
        cls,
        starStamps,
        innerRadius,
        outerRadius,
        nb90Rots=None,
        metadata=None,
        use_mask=True,
        use_variance=False,
        use_archive=False,
        imCenter=None,
        discardNanFluxObjects=True,
        forceFindFlux=False,
        statsControl=StatisticsControl(),
        statsFlag=stringToStatisticsProperty("MEAN"),
        badMaskPlanes=("BAD", "SAT", "NO_DATA"),
    ):
        """Normalize a set of bright star stamps and initialize a
        BrightStarStamps instance.

        Since the center of bright stars are saturated and/or heavily affected
        by ghosts, we measure their flux in an annulus with a large enough
        inner radius to avoid the most severe ghosts and contain enough
        non-saturated pixels.

        Parameters
        ----------
        starStamps : `collections.abc.Sequence` [`BrightStarStamp`]
            Sequence of star stamps. Cannot contain both normalized and
            unnormalized stamps.
        innerRadius : `int`
            Inner radius value, in pixels. This and ``outerRadius`` define the
            annulus used to compute the ``"annularFlux"`` values within each
            ``starStamp``.
        outerRadius : `int`
            Outer radius value, in pixels. This and ``innerRadius`` define the
            annulus used to compute the ``"annularFlux"`` values within each
            ``starStamp``.
        nb90Rots : `int`, optional
            Number of 90 degree rotations required to compensate for detector
            orientation.
        metadata : `~lsst.daf.base.PropertyList`, optional
            Metadata associated with the bright stars.
        use_mask : `bool`
            If `True` read and write mask data. Default `True`.
        use_variance : `bool`
            If ``True`` read and write variance data. Default ``False``.
        use_archive : `bool`
            If ``True`` read and write an Archive that contains a Persistable
            associated with each stamp. In the case of bright stars, this is
            usually a ``TransformPoint2ToPoint2``, used to warp each stamp
            to the same pixel grid before stacking.
        imCenter : `collections.abc.Sequence`, optional
            Center of the object, in pixels. If not provided, the center of the
            first stamp's pixel grid will be used.
        discardNanFluxObjects : `bool`
            Whether objects with NaN annular flux should be discarded.
            If False, these objects will not be normalized.
        forceFindFlux : `bool`
            Whether to try to find the flux of objects with NaN annular flux
            at a different annulus.
        statsControl : `~lsst.afw.math.statistics.StatisticsControl`, optional
            StatisticsControl to be used when computing flux over all pixels
            within the annulus.
        statsFlag : `~lsst.afw.math.statistics.Property`, optional
            statsFlag to be passed on to ``~lsst.afw.math.makeStatistics`` to
            compute annularFlux. Defaults to a simple MEAN.
        badMaskPlanes : `collections.abc.Collection` [`str`]
            Collection of mask planes to ignore when computing annularFlux.

        Raises
        ------
        ValueError
            Raised if one of the star stamps provided does not contain the
            required keys.
        AttributeError
            Raised if there is a mix-and-match of normalized and unnormalized
            stamps, stamps normalized with different annulus definitions, or if
            stamps are to be normalized but annular radii were not provided.
        """
        stampSize = starStamps[0].stamp_im.getDimensions()
        if imCenter is None:
            imCenter = stampSize[0] // 2, stampSize[1] // 2

        # Create SpanSet of annulus.
        outerCircle = SpanSet.fromShape(outerRadius, Stencil.CIRCLE, offset=imCenter)
        innerCircle = SpanSet.fromShape(innerRadius, Stencil.CIRCLE, offset=imCenter)
        annulusWidth = outerRadius - innerRadius
        if annulusWidth < 1:
            raise ValueError("The annulus width must be greater than 1 pixel.")
        annulus = outerCircle.intersectNot(innerCircle)

        # Initialize (unnormalized) brightStarStamps instance.
        bss = cls(
            starStamps,
            innerRadius=None,
            outerRadius=None,
            nb90Rots=nb90Rots,
            metadata=metadata,
            use_mask=use_mask,
            use_variance=use_variance,
            use_archive=use_archive,
        )

        # Ensure that no stamps have already been normalized.
        bss._checkNormalization(True, innerRadius, outerRadius)
        bss._innerRadius, bss._outerRadius = innerRadius, outerRadius

        # Apply normalization.
        rejects = []
        badStamps = []
        for stamp in bss._stamps:
            try:
                stamp.measureAndNormalize(
                    annulus, statsControl=statsControl, statsFlag=statsFlag, badMaskPlanes=badMaskPlanes
                )
                # Stars that are missing from input bright star stamps may
                # still have a flux within the normalization annulus. The
                # following two lines make sure that these stars are included
                # in the subtraction process. Failing to assign the optimal
                # radii values may result in an error in the `createAnnulus`
                # method of the `SubtractBrightStarsTask` class. An alternative
                # to handle this is to create two types of stamps that are
                # missing from the input brightStarStamps object. One for those
                # that have flux within the normalization annulus and another
                # for those that do not have a flux within the normalization
                # annulus.
                stamp.optimalOuterRadius = outerRadius
                stamp.optimalInnerRadius = innerRadius
            except RuntimeError as err:
                logger.error(err)
                # Optionally keep NaN flux objects, for bookkeeping purposes,
                # and to avoid having to re-find and redo the preprocessing
                # steps needed before bright stars can be subtracted.
                if discardNanFluxObjects:
                    rejects.append(stamp)
                elif forceFindFlux:
                    newInnerRadius = innerRadius
                    newOuterRadius = outerRadius
                    while True:
                        newOuterRadius += annulusWidth
                        newInnerRadius += annulusWidth
                        if newOuterRadius > min(imCenter):
                            logger.info("No flux found for the star with Gaia ID of %s", stamp.gaiaId)
                            stamp.annularFlux = None
                            badStamps.append(stamp)
                            break
                        newOuterCircle = SpanSet.fromShape(newOuterRadius, Stencil.CIRCLE, offset=imCenter)
                        newInnerCircle = SpanSet.fromShape(newInnerRadius, Stencil.CIRCLE, offset=imCenter)
                        newAnnulus = newOuterCircle.intersectNot(newInnerCircle)
                        try:
                            stamp.measureAndNormalize(
                                newAnnulus,
                                statsControl=statsControl,
                                statsFlag=statsFlag,
                                badMaskPlanes=badMaskPlanes,
                            )

                        except RuntimeError:
                            stamp.annularFlux = np.nan
                            logger.error(
                                "The annular flux was not found for radii %d and %d",
                                newInnerRadius,
                                newOuterRadius,
                            )
                        if stamp.annularFlux and stamp.annularFlux > 0:
                            logger.info("The flux is found within an optimized annulus.")
                            logger.info(
                                "The optimized annulus radii are %d and %d and the flux is %f",
                                newInnerRadius,
                                newOuterRadius,
                                stamp.annularFlux,
                            )
                            stamp.optimalOuterRadius = newOuterRadius
                            stamp.optimalInnerRadius = newInnerRadius
                            break
                else:
                    stamp.annularFlux = np.nan

        # Remove rejected stamps.
        bss.normalized = True
        if discardNanFluxObjects:
            for reject in rejects:
                bss._stamps.remove(reject)
        elif forceFindFlux:
            for badStamp in badStamps:
                bss._stamps.remove(badStamp)
            bss._innerRadius, bss._outerRadius = None, None
            return bss, badStamps
        return bss

    @classmethod
    def readFits(cls, filename):
        """Build an instance of this class from a file.

        Parameters
        ----------
        filename : `str`
            Name of the file to read.
        """
        return cls.readFitsWithOptions(filename, None)

    @classmethod
    def readFitsWithOptions(cls, filename, options):
        """Build an instance of this class with options.

        Parameters
        ----------
        filename : `str`
            Name of the file to read.
        options : `PropertyList`
            Collection of metadata parameters.
        """
        stamps, metadata = readFitsWithOptions(filename, BrightStarStamp.factory, options)
        nb90Rots = metadata["NB_90_ROTS"] if "NB_90_ROTS" in metadata else None
        # For backwards compatibility, always assume stamps are unnormalized.
        # This allows older stamps to be read in successfully.
        return cls(
            stamps,
            nb90Rots=nb90Rots,
            metadata=metadata,
            use_mask=metadata["HAS_MASK"],
            use_variance=metadata["HAS_VARIANCE"],
            use_archive=metadata["HAS_ARCHIVE"],
        )

    def append(self, item, innerRadius=None, outerRadius=None):
        """Add an additional bright star stamp.

        Parameters
        ----------
        item : `BrightStarStamp`
            Bright star stamp to append.
        innerRadius : `int`, optional
            Inner radius value, in pixels. This and ``outerRadius`` define the
            annulus used to compute the ``"annularFlux"`` values within each
            ``BrightStarStamp``.
        outerRadius : `int`, optional
            Outer radius value, in pixels. This and ``innerRadius`` define the
            annulus used to compute the ``"annularFlux"`` values within each
            ``BrightStarStamp``.
        """
        if not isinstance(item, BrightStarStamp):
            raise ValueError(f"Can only add instances of BrightStarStamp, got {type(item)}.")
        if (item.annularFlux is None) == self.normalized:
            raise AttributeError(
                "Trying to append an unnormalized stamp to a normalized BrightStarStamps "
                "instance, or vice-versa."
            )
        else:
            self._checkRadius(innerRadius, outerRadius)
        self._stamps.append(item)
        return None

    def extend(self, bss):
        """Extend BrightStarStamps instance by appending elements from another
        instance.

        Parameters
        ----------
        bss : `BrightStarStamps`
            Other instance to concatenate.
        """
        if not isinstance(bss, BrightStarStamps):
            raise ValueError(f"Can only extend with a BrightStarStamps object. Got {type(bss)}.")
        self._checkRadius(bss._innerRadius, bss._outerRadius)
        self._stamps += bss._stamps

    def getMagnitudes(self):
        """Retrieve Gaia G-band magnitudes for each star.

        Returns
        -------
        gaiaGMags : `list` [`float`]
            Gaia G-band magnitudes for each star.
        """
        return [stamp.gaiaGMag for stamp in self._stamps]

    def getGaiaIds(self):
        """Retrieve Gaia IDs for each star.

        Returns
        -------
        gaiaIds : `list` [`int`]
            Gaia IDs for each star.
        """
        return [stamp.gaiaId for stamp in self._stamps]

    def getAnnularFluxes(self):
        """Retrieve normalization factor for each star.

        These are computed by integrating the flux in annulus centered on the
        bright star, far enough from center to be beyond most severe ghosts and
        saturation.
        The inner and outer radii that define the annulus can be recovered from
        the metadata.

        Returns
        -------
        annularFluxes : `list` [`float`]
            Annular fluxes which give the normalization factor for each star.
        """
        return [stamp.annularFlux for stamp in self._stamps]

    def getValidPixelsFraction(self):
        """Retrieve the fraction of valid pixels within the normalization
        annulus for each star.

        Returns
        -------
        validPixelsFractions : `list` [`float`]
            Fractions of valid pixels within the normalization annulus for each
            star.
        """
        return [stamp.validAnnulusFraction for stamp in self._stamps]

    def selectByMag(self, magMin=None, magMax=None):
        """Return the subset of bright star stamps for objects with specified
        magnitude cuts (in Gaia G).

        Parameters
        ----------
        magMin : `float`, optional
            Keep only stars fainter than this value.
        magMax : `float`, optional
            Keep only stars brighter than this value.
        """
        subset = [
            stamp
            for stamp in self._stamps
            if (magMin is None or stamp.gaiaGMag > magMin) and (magMax is None or stamp.gaiaGMag < magMax)
        ]
        # This saves looping over init when guaranteed to be the correct type.
        instance = BrightStarStamps(
            (), innerRadius=self._innerRadius, outerRadius=self._outerRadius, metadata=self._metadata
        )
        instance._stamps = subset
        return instance

    def _checkRadius(self, innerRadius, outerRadius):
        """Ensure provided annulus radius is consistent with that already
        present in the instance, or with arguments passed on at initialization.
        """
        if innerRadius != self._innerRadius or outerRadius != self._outerRadius:
            raise AttributeError(
                f"Trying to mix stamps normalized with annulus radii {innerRadius, outerRadius} with those "
                "of BrightStarStamp instance\n"
                f"(computed with annular radii {self._innerRadius, self._outerRadius})."
            )

    def _checkNormalization(self, normalize, innerRadius, outerRadius):
        """Ensure there is no mixing of normalized and unnormalized stars, and
        that, if requested, normalization can be performed.
        """
        noneFluxCount = self.getAnnularFluxes().count(None)
        nStamps = len(self)
        nFluxVals = nStamps - noneFluxCount
        if noneFluxCount and noneFluxCount < nStamps:
            # At least one stamp contains an annularFlux value (i.e. has been
            # normalized), but not all of them do.
            raise AttributeError(
                f"Only {nFluxVals} stamps contain an annularFlux value.\nAll stamps in a BrightStarStamps "
                "instance must either be normalized with the same annulus definition, or none of them can "
                "contain an annularFlux value."
            )
        elif normalize:
            # Stamps are to be normalized; ensure annular radii are specified
            # and they have no annularFlux.
            if innerRadius is None or outerRadius is None:
                raise AttributeError(
                    "For stamps to be normalized (normalize=True), please provide a valid value (in pixels) "
                    "for both innerRadius and outerRadius."
                )
            elif noneFluxCount < nStamps:
                raise AttributeError(
                    f"{nFluxVals} stamps already contain an annularFlux value. For stamps to be normalized, "
                    "all their annularFlux must be None."
                )
        elif innerRadius is not None and outerRadius is not None:
            # Radii provided, but normalize=False; check that stamps already
            # contain annularFluxes.
            if noneFluxCount:
                raise AttributeError(
                    f"{noneFluxCount} stamps contain no annularFlux, but annular radius values were provided "
                    "and normalize=False.\nTo normalize stamps, set normalize to True."
                )
        else:
            # At least one radius value is missing; ensure no stamps have
            # already been normalized.
            if nFluxVals:
                raise AttributeError(
                    f"{nFluxVals} stamps contain an annularFlux value. If stamps have been normalized, the "
                    "innerRadius and outerRadius values used must be provided."
                )
        return None
