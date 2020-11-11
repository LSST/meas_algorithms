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
#
"""Collection of small images (stamps), each centered on a bright star.
"""

__all__ = ["BrightStarStamp", "BrightStarStamps"]

import collections.abc
from typing import NamedTuple
from enum import Enum, auto
from dataclasses import dataclass

import lsst.afw.image as afwImage
import lsst.afw.fits as afwFits
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.daf.base import PropertySet
from .stamps import Stamp, Stamps


class RadiiEnum(Enum):
    INNER_RADIUS = auto()
    OUTER_RADIUS = auto()

    def __str__(self):
        return self.name


class BrightStarStamp(Stamp):
    """Single stamp centered on a bright star, normalized by its
    annularFlux.
    """
    gaiaGMag: float
    gaiaId: int
    annularFlux: float

    @classmethod
    def factory(cls, stamp, metadata, idx):
        gaiaGMags = metadata.getArray("G_MAGS")
        gaiaIds = metadata.getArray("GAIA_IDS")
        annularFluxes = metadata.getArray("ANNULAR_FLUXES")
        return cls(stamp=stamp, ra=Angle(numpy.nan), dec=Angle(numpy.nan), size=-1,
                gaiaGMag=gaiaGMags[idx], gaiaId=gaiaIds[idx], annularFlux=annularFluxes[idx])
 

class BrightStarStamps(Stamps):
    """Collection of bright star stamps and associated metadata.

    Parameters
    ----------
    starStamps : `collections.abc.Sequence` [`BrightStarStamp`]
        Sequence of star stamps.
    innerRadius : `int`, optional
        Inner radius value, in pixels. This and ``outerRadius`` define the
        annulus used to compute the ``"annularFlux"`` values within each
        ``starStamp``. Must be provided if ``"INNER_RADIUS"`` and
        ``"OUTER_RADIUS"`` are not present in ``metadata``.
    outerRadius : `int`, optional
        Outer radius value, in pixels. This and ``innerRadius`` define the
        annulus used to compute the ``"annularFlux"`` values within each
        ``starStamp``. Must be provided if ``"INNER_RADIUS"`` and
        ``"OUTER_RADIUS"`` are not present in ``metadata``.
    metadata : `lsst.daf.base.PropertyList`, optional
        Metadata associated with the bright stars.

    Raises
    ------
    ValueError
        Raised if one of the star stamps provided does not contain the
        required keys.
    AttributeError
        Raised if the definition of the annulus used to compute each star's
        normalization factor are not provided, that is, if ``"INNER_RADIUS"``
        and ``"OUTER_RADIUS"`` are not present in ``metadata`` _and_
        ``innerRadius`` and ``outerRadius`` are not provided.

    Notes
    -----
    A (gen2) butler can be used to read only a part of the stamps,
    specified by a bbox:

    >>> starSubregions = butler.get("brightStarStamps_sub", dataId, bbox=bbox)
    """

    def __init__(self, starStamps, innerRadius=None, outerRadius=None,
                 metadata=None, has_mask=True, has_variance=False):
        super.__init__(starStamps, metadata, has_mask, has_variance)
        for item in starStamps:
            if not isinstance(item, BrightStarStamp):
                raise ValueError(f"Can only add instances of BrightStarStamp, got {type(item)}")
        # Add inner and outer radii to metadata
        self._checkRadius(innerRadius, RadiiEnum.INNER_RADIUS)
        self._innerRadius = innerRadius
        self._checkRadius(outerRadius, RadiiEnum.OUTER_RADIUS)
        self._outerRadius = outerRadius

    def _refresh_metadata(self):
        # ensure metadata contains current number of objects
        self._metadata["N_STARS"] = len(self)

        # add full list of Gaia magnitudes, IDs and annularFlxes to shared
        # metadata
        self._metadata["G_MAGS"] = self.getMagnitudes()
        self._metadata["GAIA_IDS"] = self.getGaiaIds()
        self._metadata["ANNULAR_FLUXES"] = self.getAnnularFluxes()
        return None

    def append(self, item, innerRadius, outerRadius):
        """Add an additional bright star stamp.

        Parameters
        ----------
        item : `BrightStarStamp`
            Bright star stamp to append.
        innerRadius : `int`
            Inner radius value, in4 pixels. This and ``outerRadius`` define the
            annulus used to compute the ``"annularFlux"`` values within each
            ``starStamp``.
        outerRadius : `int`, optional
            Outer radius value, in pixels. This and ``innerRadius`` define the
            annulus used to compute the ``"annularFlux"`` values within each
            ``starStamp``.
        """
        if not isinstance(item, BrightStarStamp):
            raise ValueError(f"Can only add instances of BrightStarStamp, got {type(item)}.")
        self._checkRadius(innerRadius, RadiiEnum.INNER_RADIUS)
        self._checkRadius(outerRadius, RadiiEnum.OUTER_RADIUS)
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
        self._checkRadius(bss._innerRadius, RadiiEnum.INNER_RADIUS)
        self._checkRadius(bss._outerRadius, RadiiEnum.OUTER_RADIUS)
        self._stamps += bss._stamps

    def getMagnitudes(self):
        """Retrieve Gaia G magnitudes for each star.

        Returns
        -------
        gaiaGMags : `list` [`float`]
        """
        return [stamp.gaiaGMag for stamp in self._stamps]

    def getGaiaIds(self):
        """Retrieve Gaia IDs for each star.

        Returns
        -------
        gaiaIds : `list` [`int`]
        """
        return [stamp.gaiaId for stamp in self._stamps]

    def getAnnularFluxes(self):
        """Retrieve normalization factors for each star.

        These are computed by integrating the flux in annulus centered on the
        bright star, far enough from center to be beyond most severe ghosts and
        saturation. The inner and outer radii that define the annulus can be
        recovered from the metadata.

        Returns
        -------
        annularFluxes : list[`float`]
        """
        return [stamp.annularFlux for stamp in self._stamps]

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
        subset = [stamp for stamp in self._stamps
                  if (magMin is None or stamp.gaiaGMag > magMin)
                  and (magMax is None or stamp.gaiaGMag < magMax)]
        # This is an optimization to save looping over the init argument when
        # it is already guaranteed to be the correct type
        instance = BrightStarStamps((), metadata=self._metadata)
        instance._stamps = subset
        return instance

    def _checkRadius(self, radiusValue, metadataEnum):
        """Ensure provided annulus radius is consistent with that present
        in metadata. If metadata does not contain annulus radius, add it.
        """
        # if a radius value is already present in metadata, ensure it matches
        # the one given
        metadataName = str(metadataEnum)
        if self._metadata.exists(metadataName):
            if radiusValue is not None:
                if self._metadata[metadataName] != radiusValue:
                    raise AttributeError("BrightStarStamps instance already contains different annulus radii "
                                         + f"values ({metadataName}).")
        # if not already in metadata, a value must be provided
        elif radiusValue is None:
            raise AttributeError("No radius value provided for the AnnularFlux measurement "
                                 + f"({metadataName}), and none present in metadata.")
        else:
            self._metadata[metadataName] = radiusValue
        return None
