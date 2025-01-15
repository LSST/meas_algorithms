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

from __future__ import annotations

"""Collection of small images (stamps), each centered on a bright star."""

__all__ = ["BrightStarStamp", "BrightStarStamps"]

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from lsst.afw.detection import Psf
from lsst.afw.geom import SkyWcs
from lsst.afw.image import MaskedImageF
from lsst.afw.table.io import Persistable
from lsst.daf.base import PropertyList
from lsst.geom import Point2D

from .stamps import StampBase, StampsBase, readFitsWithOptions

logger = logging.getLogger(__name__)


@dataclass
class BrightStarStamp(StampBase):
    """Single stamp centered on a bright star."""

    maskedImage: MaskedImageF
    psf: Psf
    wcs: SkyWcs
    visit: int
    detector: int
    refId: int
    refMag: float
    position: Point2D
    scale: float | None
    scaleErr: float | None
    pedestal: float | None
    pedestalErr: float | None
    pedestalScaleCov: float | None
    xGradient: float | None
    yGradient: float | None
    globalReducedChiSquared: float | None
    globalDegreesOfFreedom: int | None
    psfReducedChiSquared: float | None
    psfDegreesOfFreedom: int | None
    psfMaskedFluxFrac: float | None

    @classmethod
    def _getMaskedImageClass(cls) -> type[MaskedImageF]:
        return MaskedImageF

    @classmethod
    def _getArchiveElementNames(cls) -> list[str]:
        return ["PSF", "WCS"]

    @classmethod
    def factory(
        cls,
        maskedImage: MaskedImageF,
        metadata: PropertyList,
        idx: int,
        archive_elements: Mapping[str, Persistable] | None = None,
    ):
        assert archive_elements is not None
        return cls(
            maskedImage=maskedImage,
            psf=archive_elements["PSF"],
            wcs=archive_elements["WCS"],
            visit=metadata["VISIT"],
            detector=metadata["DETECTOR"],
            refId=metadata["REFID"],
            refMag=metadata["REFMAG"],
            position=Point2D(metadata["X_FA"], metadata["Y_FA"]),
            scale=metadata["SCALE"],
            scaleErr=metadata["SCALE_ERR"],
            pedestal=metadata["PEDESTAL"],
            pedestalErr=metadata["PEDESTAL_ERR"],
            pedestalScaleCov=metadata["PEDESTAL_SCALE_COV"],
            xGradient=metadata["X_GRADIENT"],
            yGradient=metadata["Y_GRADIENT"],
            globalReducedChiSquared=metadata["GLOBAL_REDUCED_CHI_SQUARED"],
            globalDegreesOfFreedom=metadata["GLOBAL_DEGREES_OF_FREEDOM"],
            psfReducedChiSquared=metadata["PSF_REDUCED_CHI_SQUARED"],
            psfDegreesOfFreedom=metadata["PSF_DEGREES_OF_FREEDOM"],
            psfMaskedFluxFrac=metadata["PSF_MASKED_FLUX_FRAC"],
        )

    def _getMaskedImage(self):
        return self.maskedImage

    def _getArchiveElements(self):
        return {"PSF": self.psf, "WCS": self.wcs}

    def _getMetadata(self) -> PropertyList | None:
        md = PropertyList()
        md["VISIT"] = self.visit
        md["DETECTOR"] = self.detector
        md["REFID"] = self.refId
        md["REFMAG"] = self.refMag
        md["X_FA"] = self.position.x
        md["Y_FA"] = self.position.y
        md["SCALE"] = self.scale
        md["SCALE_ERR"] = self.scaleErr
        md["PEDESTAL"] = self.pedestal
        md["PEDESTAL_ERR"] = self.pedestalErr
        md["PEDESTAL_SCALE_COV"] = self.pedestalScaleCov
        md["X_GRADIENT"] = self.xGradient
        md["Y_GRADIENT"] = self.yGradient
        md["GLOBAL_REDUCED_CHI_SQUARED"] = self.globalReducedChiSquared
        md["GLOBAL_DEGREES_OF_FREEDOM"] = self.globalDegreesOfFreedom
        md["PSF_REDUCED_CHI_SQUARED"] = self.psfReducedChiSquared
        md["PSF_DEGREES_OF_FREEDOM"] = self.psfDegreesOfFreedom
        md["PSF_MASKED_FLUX_FRAC"] = self.psfMaskedFluxFrac
        return md


class BrightStarStamps(StampsBase):

    def __init__(
        self,
        starStamps,
        metadata=None,
    ):
        super().__init__(starStamps, metadata, useMask=True, useVariance=True, useArchive=True)
        self.byRefId = {stamp.refId: stamp for stamp in self}

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
        stamps, metadata = readFitsWithOptions(filename, BrightStarStamp, options)
        return cls(stamps, metadata=metadata)
