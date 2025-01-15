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

"""Collection of small images (postage stamps)."""

__all__ = ["StampBase", "StampsBase", "writeFits", "readFitsWithOptions"]

import abc
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from lsst.afw.fits import Fits, readMetadata
from lsst.afw.image import ImageFitsReader, MaskedImage, MaskedImageF, MaskFitsReader
from lsst.afw.table.io import InputArchive, OutputArchive, Persistable
from lsst.daf.base import PropertyList
from lsst.geom import Angle, SpherePoint
from lsst.utils import doImport
from lsst.utils.introspection import get_full_type_name

DEFAULT_ARCHIVE_ELEMENT_NAME = "ELEMENT"


def writeFits(
    filename: str,
    stamps: Sequence[StampBase],
    metadata: PropertyList,
    typeName: str,
    writeMask: bool,
    writeVariance: bool,
    writeArchive: bool = False,
):
    """Write a single FITS file containing all stamps.

    Parameters
    ----------
    filename : `str`
        A string indicating the output filename.
    stamps : iterable of `StampBase`
        An iterable of Stamp objects.
    metadata : `PropertyList`
        A collection of key:value metadata pairs written to the primary header.
    typeName : `str`
        Python type name of the StampsBase subclass to use.
    writeMask : `bool`
        Write the mask data to the output file?
    writeVariance : `bool`
        Write the variance data to the output file?
    writeArchive : `bool`, optional
        Write an archive which stores Persistables along with each stamp?
    """
    # Stored metadata in the primary HDU
    metadata["HAS_MASK"] = writeMask
    metadata["HAS_VARIANCE"] = writeVariance
    metadata["HAS_ARCHIVE"] = writeArchive
    metadata["N_STAMPS"] = len(stamps)
    metadata["STAMPCLS"] = typeName
    metadata["VERSION"] = 2  # Record version number in case of future code changes

    # Create the primary HDU with global metadata
    fitsFile = Fits(filename, "w")
    fitsFile.createEmpty()

    # Store Persistables in an OutputArchive and write it to the primary HDU
    if writeArchive:
        stamps_archiveElementNames = set()
        stamps_archiveElementIds = []
        oa = OutputArchive()
        for stamp in stamps:
            stamp_archiveElements = stamp._getArchiveElements()
            stamps_archiveElementNames.update(stamp_archiveElements.keys())
            stamps_archiveElementIds.append(
                {name: oa.put(persistable) for name, persistable in stamp_archiveElements.items()}
            )
        fitsFile.writeMetadata(metadata)
        oa.writeFits(fitsFile)
    else:
        stamps_archiveElementIds = [None] * len(stamps)
        fitsFile.writeMetadata(metadata)
    fitsFile.closeFile()

    # Add all pixel data to extension HDUs; optionally write mask/variance info
    for i, (stamp, stamp_archiveElementIds) in enumerate(zip(stamps, stamps_archiveElementIds)):
        metadata = PropertyList()
        extVer = i + 1  # EXTVER should be 1-based; the index from enumerate is 0-based
        metadata.update({"EXTVER": extVer, "EXTNAME": "IMAGE"})
        if stampMetadata := stamp._getMetadata():
            metadata.update(stampMetadata)
        if stamp_archiveElementIds:
            metadata.update(stamp_archiveElementIds)
            for stamps_archiveElementName in sorted(stamps_archiveElementNames):
                metadata.add("ARCHIVE_ELEMENT", stamps_archiveElementName)
        stamp.maskedImage.getImage().writeFits(filename, metadata=metadata, mode="a")
        if writeMask:
            metadata = PropertyList()
            metadata.update({"EXTVER": extVer, "EXTNAME": "MASK"})
            stamp.maskedImage.getMask().writeFits(filename, metadata=metadata, mode="a")
        if writeVariance:
            metadata = PropertyList()
            metadata.update({"EXTVER": extVer, "EXTNAME": "VARIANCE"})
            stamp.maskedImage.getVariance().writeFits(filename, metadata=metadata, mode="a")
    return None


def readFitsWithOptions(
    filename: str,
    stamp_cls: type[StampBase],
    options: PropertyList,
):
    """Read stamps from FITS file, allowing for only a subregion of the stamps
    to be read.

    Parameters
    ----------
    filename : `str`
        A string indicating the file to read
    stampFactory : classmethod
        A factory function defined on a dataclass for constructing
        stamp objects a la `~lsst.meas.algorithm.Stamp`
    options : `PropertyList` or `dict`
        A collection of parameters.

    Returns
    -------
    stamps : `list` of dataclass objects like `Stamp`, PropertyList
        A tuple of a list of `Stamp`-like objects
    metadata : `PropertyList`
        The metadata

    Notes
    -----
    The data are read using the data type expected by the
    `~lsst.afw.image.MaskedImage` class attached to the `StampBase`
    dataclass associated with the factory method.
    """
    # Extract necessary info from metadata
    metadata = readMetadata(filename, hdu=0)
    nStamps = metadata["N_STAMPS"]
    hasArchive = metadata["HAS_ARCHIVE"]
    stamps_archiveElementNames = None
    stamps_archiveElementIds_v1 = None
    if hasArchive:
        if metadata["VERSION"] < 2:
            stamps_archiveElementIds_v1 = metadata.getArray("ARCHIVE_IDS")
        else:
            stamps_archiveElementNames = stamp_cls._getArchiveElementNames()
    with Fits(filename, "r") as fitsFile:
        nExtensions = fitsFile.countHdus()
        stampParts = {}

        # Determine the dtype from the factory.
        # This allows a Stamp class to be defined in terms of MaskedImageD or
        # MaskedImageI without forcing everything to floats.
        maskedImageCls = stamp_cls._getMaskedImageClass()
        default_dtype = np.dtype(maskedImageCls.dtype)
        variance_dtype = np.dtype(np.float32)  # Variance is always the same type

        # We need to be careful because nExtensions includes the primary HDU
        stampMetadata = {}
        stamps_archiveElementIds = {}
        for idx in range(nExtensions - 1):
            dtype = None
            hduNum = idx + 1
            md = readMetadata(filename, hdu=hduNum)
            # Skip binary tables that aren't images or archives.
            if md["XTENSION"] == "BINTABLE" and not ("ZIMAGE" in md and md["ZIMAGE"]):
                if md["EXTNAME"] != "ARCHIVE_INDEX":
                    continue
            if md["EXTNAME"] in ("IMAGE", "VARIANCE"):
                stampId = md["EXTVER"]
                reader = ImageFitsReader(filename, hdu=hduNum)
                if md["EXTNAME"] == "VARIANCE":
                    dtype = variance_dtype
                else:
                    dtype = default_dtype
                    if stamps_archiveElementNames is not None:
                        stamps_archiveElementIds[stampId] = {
                            name: archiveId
                            for name in stamps_archiveElementNames
                            if (archiveId := md.pop(name, None))
                        }
                    # md.remove("EXTNAME")
                    # md.remove("EXTVER")
                    stampMetadata[stampId] = md
            elif md["EXTNAME"] == "MASK":
                stampId = md["EXTVER"]
                reader = MaskFitsReader(filename, hdu=hduNum)
            elif md["EXTNAME"] == "ARCHIVE_INDEX":
                fitsFile.setHdu(hduNum)
                archive = InputArchive.readFits(fitsFile)
                continue
            elif md["EXTTYPE"] == "ARCHIVE_DATA":
                continue
            else:
                raise ValueError(f"Unknown extension type: {md['EXTNAME']}")
            stampParts.setdefault(stampId, {})[md["EXTNAME"].lower()] = reader.read(dtype=dtype)

    if len(stampParts) != nStamps:
        raise ValueError(
            f"Number of stamps read ({len(stampParts)}) does not agree with the "
            f"number of stamps recorded in the metadata ({nStamps})."
        )
    # Construct the stamps themselves
    stamps = []
    for k in range(nStamps):
        # Need to increment by one since EXTVER starts at 1
        maskedImage = maskedImageCls(**stampParts[k + 1])
        if stamps_archiveElementIds_v1 is not None:
            stamp_archiveElements = {
                DEFAULT_ARCHIVE_ELEMENT_NAME: archive.get(stamps_archiveElementIds_v1[k])
            }
        elif stamps_archiveElementNames is not None:
            stamp_archiveElementIds = stamps_archiveElementIds.get(k + 1, {})
            stamp_archiveElements = {name: archive.get(id) for name, id in stamp_archiveElementIds.items()}
        else:
            stamp_archiveElements = None
        if metadata["VERSION"] < 2:
            stamps.append(stamp_cls.factory(maskedImage, metadata, k, stamp_archiveElements))
        else:
            stamps.append(stamp_cls.factory(maskedImage, stampMetadata[k + 1], k, stamp_archiveElements))

    return stamps, metadata


def _defaultPosition():
    # SpherePoint is nominally mutable in C++ so we must use a factory
    # and return an entirely new SpherePoint each time a Stamps is created.
    return SpherePoint(Angle(np.nan), Angle(np.nan))


@dataclass
class StampBase(abc.ABC):
    """Single abstract postage stamp.

    Notes
    -----
    Inherit from this class to add metadata to the postage stamp.
    """

    @classmethod
    @abc.abstractmethod
    def _getMaskedImageClass(cls) -> type[MaskedImage]:
        """Return the class of the MaskedImage object to be used."""
        raise NotImplementedError()

    @classmethod
    def _getArchiveElementNames(cls) -> list[str]:
        return []

    @classmethod
    @abc.abstractmethod
    def factory(
        cls,
        maskedImage: MaskedImageF,
        metadata: PropertyList,
        index: int,
        archiveElements: Mapping[str, Persistable] | None = None,
    ) -> typing.Self:
        """This method is needed to service the FITS reader.
        We need a standard interface to construct objects like this.
        Parameters needed to construct this object are passed in via a metadata
        dictionary and then passed to the constructor of this class.

        Parameters
        ----------
        maskedImage : `~lsst.afw.image.MaskedImageF`
            Pixel data to pass to the constructor
        metadata : `PropertyList`
            Dictionary containing the information needed by the constructor.
        index : `int`
            Index into the lists in ``metadata``
        archiveElements : `~collections.abc.Mapping`[ `str` , \
                `~lsst.afw.table.io.Persistable`], optional
            Archive elements (e.g. Transform / WCS) associated with this stamp.

        Returns
        -------
        stamp : `StampBase`
            An instance of this class
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _getMaskedImage(self):
        """Return the image data."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _getArchiveElements(self):
        """Return the archive elements.

        Keys should be upper case names that will be used directly as FITS
        header keys.
        """
        raise NotImplementedError()

    def _getMetadata(self) -> PropertyList | None:
        """Return the metadata."""
        return None


class StampsBase(abc.ABC, Sequence):
    """Collection of stamps and associated metadata.

    Parameters
    ----------
    stamps : iterable
        This should be an iterable of dataclass objects
        a la ``~lsst.meas.algorithms.Stamp``.
    metadata : `~lsst.daf.base.PropertyList`, optional
        Metadata associated with the objects within the stamps.
    useMask : `bool`, optional
        If ``True`` read and write the mask data. Default ``True``.
    useVariance : `bool`, optional
        If ``True`` read and write the variance data. Default ``True``.
    useArchive : `bool`, optional
        If ``True``, read and write an Archive that contains a Persistable
        associated with each stamp, for example a Transform or a WCS.
        Default ``False``.
    """

    def __init__(
        self,
        stamps: Sequence[StampBase],
        metadata: PropertyList | None = None,
        useMask: bool = True,
        useVariance: bool = True,
        useArchive: bool = False,
    ):
        for stamp in stamps:
            if not isinstance(stamp, StampBase):
                raise ValueError(f"The entries in stamps must inherit from StampBase. Got {type(stamp)}.")
        self._stamps = list(stamps)
        self._metadata = PropertyList() if metadata is None else metadata.deepCopy()
        self.useMask = useMask
        self.useVariance = useVariance
        self.useArchive = useArchive

    @classmethod
    def readFits(cls, filename: str):
        """Build an instance of this class from a file.

        Parameters
        ----------
        filename : `str`
            Name of the file to read.
        """
        return cls.readFitsWithOptions(filename=filename, options=None)

    @classmethod
    def readFitsWithOptions(cls, filename: str, options: PropertyList):
        """Build an instance of this class from a file, with options.

        Parameters
        ----------
        filename : `str`
            Name of the file to read.
        options : `PropertyList`
            Collection of metadata parameters.
        """
        # To avoid problems since this is no longer an abstract base method.
        # TO-DO: Consider refactoring this method. This class check was added
        # to allow the butler formatter to use a generic type but still end up
        # giving the correct type back, ensuring that the abstract base class
        # is not used by mistake. Perhaps this logic can be optimised.
        if cls is not StampsBase:
            raise NotImplementedError(f"Please implement specific FITS reader for class {cls}")

        # Load metadata to get the class
        metadata = readMetadata(filename, hdu=0)
        typeName = metadata.get("STAMPCLS")
        if typeName is None:
            raise RuntimeError(
                f"No class name in file {filename}. Unable to instantiate correct stamps subclass. "
                "Is this an old version format Stamps file?"
            )

        # Import class and override `cls`
        stampType = doImport(typeName)
        cls = stampType

        return cls.readFitsWithOptions(filename, options)

    def writeFits(self, filename: str):
        """Write this object to a FITS file.

        Parameters
        ----------
        filename : `str`
            Name of the FITS file to write.
        """
        typeName = get_full_type_name(self)
        writeFits(
            filename=filename,
            stamps=self._stamps,
            metadata=self._metadata,
            typeName=typeName,
            writeMask=self.useMask,
            writeVariance=self.useVariance,
            writeArchive=self.useArchive,
        )

    def __len__(self):
        return len(self._stamps)

    def __getitem__(self, index):
        return self._stamps[index]

    def __iter__(self):
        return iter(self._stamps)

    @property
    def metadata(self):
        return self._metadata
