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
from __future__ import absolute_import, division, print_function

__all__ = ["BaseSourceSelectorConfig", "BaseSourceSelectorTask", "sourceSelectorRegistry",
           "ColorLimit", "MagnitudeLimit", "SignalToNoiseLimit", "MagnitudeErrorLimit",
           "RequireFlags", "RequireUnresolved",
           "ScienceSourceSelectorConfig", "ScienceSourceSelectorTask",
           "ReferenceSourceSelectorConfig", "ReferenceSourceSelectorTask",
           ]

import abc
import numpy as np

import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.image
from future.utils import with_metaclass


class BaseSourceSelectorConfig(pexConfig.Config):
    bad_flags = pexConfig.ListField(
        doc="List of flags which cause a source to be rejected as bad",
        dtype=str,
        default=[
            "base_PixelFlags_flag_edge",
            "base_PixelFlags_flag_interpolatedCenter",
            "base_PixelFlags_flag_saturatedCenter",
            "base_PixelFlags_flag_crCenter",
            "base_PixelFlags_flag_bad",
        ],
    )


@with_metaclass(abc.ABCMeta)
class BaseSourceSelectorTask(pipeBase.Task):
    """Base class for source selectors

    Write paragraph about what a source selector is/does.

    Register all source selectors with the sourceSelectorRegistry using:
        sourceSelectorRegistry.register(name, class)

    Attributes
    ----------
    usesMatches : bool
        A boolean variable specifiy if the inherited source selector uses
        matches.
    """

    ConfigClass = BaseSourceSelectorConfig
    _DefaultName = "source_selector"
    uses_matches = False

    def __init__(self, **kwargs):
        """Create a source selector.

        Source selectors may have their own special arguments plus
        any kwargs for lsst.pipe.base.Task
        """
        pipeBase.Task.__init__(self, **kwargs)

    def run(self, source_cat, source_selected_field=None, masked_image=None,
            matches=None):
        """Select sources and return them.

        Parameters:
        -----------
        source_cat : lsst.afw.table.SourceCatalog
            Catalog of sources that may be sources
        source_selected_field : {None} lsst.afw.table.Key
            Key specifying a location field to write to. If None then no data
            is written to the output source cat.
        masked_image : {None} lsst.afw.image.MaskedImage
            Masked image containing the sources. A few source selectors may use
            this in their selection but most will just use it for plotting.
        matches : {None} list of lsst.afw.table.ReferenceMatch
            A list of lsst.afw.table.ReferenceMatch objects. If use_matches set
            in source selector, this field is required otherwise ignored.

        Return
        ------
        lsst.pipe.base.Struct
            The struct contains the following data:

            source_cat : lsst.afw.table.SourceCatalog
                The catalog of sources that were selected.
        """
        return self.select_sources(source_cat=source_cat,
                                   source_selected_field=None,
                                   masked_image=masked_image,
                                   matches=matches)

    @abc.abstractmethod
    def select_sources(self, source_cat, source_selected_field=None,
                       masked_image=None, matches=None):
        """Return a catalog of sources selected by specified criteria.

        We would prefer the input catalog to be contiguous in memory
        for speed and simplicity of the code. However, the code does
        allows for the input of non contiguous catalogs. Through
        a different code path.

        Parameters
        ----------
        source_cat : lsst.afw.table.SourceCatalog
            catalog of sources that may be sources
        source_selected_field : {None} lsst.afw.table.Key
            Key specifying a location field to write to
            if the key was set.
        masked_image : {None} lsst.afw.image
            An image containing the sources tests or for plotting.
        matches : {None} list of lsst.afw.table.ReferenceMatch
            A list of lsst.afw.table.ReferenceMatch objects

        Return
        ------
        lsst.pipe.base.Struct
            The struct contains the following data:

            result : lsst.afw.table.SourceCatalog
                The catalog of sources that were selected.
        """

        # NOTE: example implementation, returning all sources that have no bad
        # flags set.
        result = afwTable.SourceCatalog(source_cat.table)

        is_bad_array = self._isBad(source_cat)
        if source_selected_field is not None:
            result.get(source_selected_field) = np.logical_not(
                is_bad_array)
            return pipeBase.Struct(source_cat=result)
        return pipeBase.Struct(source_cat=result[np.logical_not(is_bad_array)])

    def _is_bad(self, source):
        """Return True if any of config.badFlags are set for this source.

        Parameters
        ----------
        source : lsst.afw.tabel.SourceRecord
            Source record object to test again the selection.

        Return
        ------
        bool array
            True if source fails any of the selections.
        """
        return any(source.get(flag) for flag in self.config.bad_flags)



sourceSelectorRegistry = pexConfig.makeRegistry(
    doc="A registry of source selectors (subclasses of BaseSourceSelectorTask)",
)


class BaseLimit(pexConfig.Config):
    """Base class for selecting sources by applying a limit

    This object can be used as a `lsst.pex.config.Config` for configuring
    the limit, and then the `apply` method can be used to identify sources
    in the catalog that match the configured limit.

    This provides the `maximum` and `minimum` fields in the Config, and
    a method to apply the limits to an array of values calculated by the
    subclass.
    """
    minimum = pexConfig.Field(dtype=float, optional=True, doc="Select objects with value greater than this")
    maximum = pexConfig.Field(dtype=float, optional=True, doc="Select objects with value less than this")

    def apply(self, values):
        """Apply the limits to an array of values

        Subclasses should calculate the array of values and then
        return the result of calling this method.

        Parameters
        ----------
        values : `numpy.ndarray`
            Array of values to which to apply limits.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        selected = np.ones(len(values), dtype=bool)
        with np.errstate(invalid="ignore"):  # suppress NAN warnings
            if self.minimum is not None:
                selected &= values > self.minimum
            if self.maximum is not None:
                selected &= values < self.maximum
        return selected


class ColorLimit(BaseLimit):
    """Select sources using a color limit

    This object can be used as a `lsst.pex.config.Config` for configuring
    the limit, and then the `apply` method can be used to identify sources
    in the catalog that match the configured limit.

    We refer to 'primary' and 'secondary' flux measurements; these are the
    two components of the color, which is:

        instFluxToMag(cat[primary]) - instFluxToMag(cat[secondary])
    """
    primary = pexConfig.Field(dtype=str, doc="Name of column with primary flux measurement")
    secondary = pexConfig.Field(dtype=str, doc="Name of column with secondary flux measurement")

    def apply(self, catalog):
        """Apply the color limit to a catalog

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to which the limit will be applied.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        primary = lsst.afw.image.abMagFromFlux(catalog[self.primary])
        secondary = lsst.afw.image.abMagFromFlux(catalog[self.secondary])
        color = primary - secondary
        return BaseLimit.apply(self, color)


class FluxLimit(BaseLimit):
    """Select sources using a flux limit

    This object can be used as a `lsst.pex.config.Config` for configuring
    the limit, and then the `apply` method can be used to identify sources
    in the catalog that match the configured limit.
    """
    fluxField = pexConfig.Field(dtype=str, default="slot_CalibFlux_flux",
                                doc="Name of the source flux field to use.")

    def apply(self, catalog):
        """Apply the flux limits to a catalog

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to which the limit will be applied.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        flagField = self.fluxField + "_flag"
        if flagField in catalog.schema:
            selected = np.logical_not(catalog[flagField])
        else:
            selected = np.ones(len(catalog), dtype=bool)

        flux = catalog[self.fluxField]
        selected &= BaseLimit.apply(self, flux)
        return selected


class MagnitudeLimit(BaseLimit):
    """Select sources using a magnitude limit

    Note that this assumes that a zero-point has already been applied and
    the fluxes are in AB fluxes in Jansky. It is therefore principally
    intended for reference catalogs rather than catalogs extracted from
    science images.

    This object can be used as a `lsst.pex.config.Config` for configuring
    the limit, and then the `apply` method can be used to identify sources
    in the catalog that match the configured limit.
    """
    fluxField = pexConfig.Field(dtype=str, default="flux",
                                doc="Name of the source flux field to use.")

    def apply(self, catalog):
        """Apply the magnitude limits to a catalog

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to which the limit will be applied.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        flagField = self.fluxField + "_flag"
        if flagField in catalog.schema:
            selected = np.logical_not(catalog[flagField])
        else:
            selected = np.ones(len(catalog), dtype=bool)

        magnitude = lsst.afw.image.abMagFromFlux(catalog[self.fluxField])
        selected &= BaseLimit.apply(self, magnitude)
        return selected


class SignalToNoiseLimit(BaseLimit):
    """Select sources using a flux signal-to-noise limit

    This object can be used as a `lsst.pex.config.Config` for configuring
    the limit, and then the `apply` method can be used to identify sources
    in the catalog that match the configured limit.
    """
    fluxField = pexConfig.Field(dtype=str, default="flux",
                                doc="Name of the source flux field to use.")
    errField = pexConfig.Field(dtype=str, default="flux_err",
                               doc="Name of the source flux error field to use.")

    def apply(self, catalog):
        """Apply the signal-to-noise limits to a catalog

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to which the limit will be applied.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        flagField = self.fluxField + "_flag"
        if flagField in catalog.schema:
            selected = np.logical_not(catalog[flagField])
        else:
            selected = np.ones(len(catalog), dtype=bool)

        signalToNoise = catalog[self.fluxField]/catalog[self.errField]
        selected &= BaseLimit.apply(self, signalToNoise)
        return selected


class MagnitudeErrorLimit(BaseLimit):
    """Select sources using a magnitude error limit

    Because the magnitude error is the inverse of the signal-to-noise
    ratio, this also works to select sources by signal-to-noise when
    you only have a magnitude.

    This object can be used as a `lsst.pex.config.Config` for configuring
    the limit, and then the `apply` method can be used to identify sources
    in the catalog that match the configured limit.
    """
    magErrField = pexConfig.Field(dtype=str, default="mag_err",
                                  doc="Name of the source flux error field to use.")

    def apply(self, catalog):
        """Apply the magnitude error limits to a catalog

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to which the limit will be applied.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        return BaseLimit.apply(self, catalog[self.magErrField])


class RequireFlags(pexConfig.Config):
    """Select sources using flags

    This object can be used as a `lsst.pex.config.Config` for configuring
    the limit, and then the `apply` method can be used to identify sources
    in the catalog that match the configured limit.
    """
    good = pexConfig.ListField(dtype=str, default=[],
                               doc="List of source flag fields that must be set for a source to be used.")
    bad = pexConfig.ListField(dtype=str, default=[],
                              doc="List of source flag fields that must NOT be set for a source to be used.")

    def apply(self, catalog):
        """Apply the flag requirements to a catalog

        Returns whether the source is selected.

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to which the requirements will be applied.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        selected = np.ones(len(catalog), dtype=bool)
        for flag in self.good:
            selected &= catalog[flag]
        for flag in self.bad:
            selected &= ~catalog[flag]
        return selected


class RequireUnresolved(BaseLimit):
    """Select sources using star/galaxy separation

    This object can be used as a `lsst.pex.config.Config` for configuring
    the limit, and then the `apply` method can be used to identify sources
    in the catalog that match the configured limit.
    """
    name = pexConfig.Field(dtype=str, default="base_ClassificationExtendedness_value",
                           doc="Name of column for star/galaxy separation")

    def apply(self, catalog):
        """Apply the flag requirements to a catalog

        Returns whether the source is selected.

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to which the requirements will be applied.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        selected = np.ones(len(catalog), dtype=bool)
        value = catalog[self.name]
        return BaseLimit.apply(self, value)

class RequireIsolated(pexConfig.Config):
    """Select sources based on whether they are isolated

    This object can be used as a `lsst.pex.config.Config` for configuring
    the column names to check for "parent" and "nChild" keys.

    Note that this should only be run on a catalog that has had the
    deblender already run (or else deblend_nChild does not exist).
    """
    parentName = pexConfig.Field(dtype=str, default="parent",
                                 doc="Name of column for parent")
    nChildName = pexConfig.Field(dtype=str, default="deblend_nChild",
                                 doc="Name of column for nChild")

    def apply(self, catalog):
        """Apply the isolation requirements to a catalog

        Returns whether the source is selected.

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to which the requirements will be applied.

        Returns
        -------
        selected : `numpy.ndarray`
            Boolean array indicating for each source whether it is selected
            (True means selected).
        """
        selected = ((catalog[self.parentName] == 0) &
                    (catalog[self.nChildName] == 0))
        return selected

class ScienceSourceSelectorConfig(pexConfig.Config):
    """Configuration for selecting science sources"""
    doFluxLimit = pexConfig.Field(dtype=bool, default=False, doc="Apply flux limit?")
    doFlags = pexConfig.Field(dtype=bool, default=False, doc="Apply flag limitation?")
    doUnresolved = pexConfig.Field(dtype=bool, default=False, doc="Apply unresolved limitation?")
    doSignalToNoise = pexConfig.Field(dtype=bool, default=False, doc="Apply signal-to-noise limit?")
    doIsolated = pexConfig.Field(dtype=bool, default=False, doc="Apply isolated limitation?")
    fluxLimit = pexConfig.ConfigField(dtype=FluxLimit, doc="Flux limit to apply")
    flags = pexConfig.ConfigField(dtype=RequireFlags, doc="Flags to require")
    unresolved = pexConfig.ConfigField(dtype=RequireUnresolved, doc="Star/galaxy separation to apply")
    signalToNoise = pexConfig.ConfigField(dtype=SignalToNoiseLimit, doc="Signal-to-noise limit to apply")
    isolated = pexConfig.ConfigField(dtype=RequireIsolated, doc="Isolated criteria to apply")

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.flags.bad = ["base_PixelFlags_flag_edge", "base_PixelFlags_flag_saturated", "base_PsfFlux_flags"]
        self.signalToNoise.fluxField = "base_PsfFlux_flux"
        self.signalToNoise.errField = "base_PsfFlux_fluxSigma"


class ScienceSourceSelectorTask(BaseSourceSelectorTask):
    """Science source selector

    By "science" sources, we mean sources that are on images that we
    are processing, as opposed to sources from reference catalogs.

    This selects (science) sources by (optionally) applying each of a
    magnitude limit, flag requirements and star/galaxy separation.
    """
    ConfigClass = ScienceSourceSelectorConfig

    def selectSources(self, catalog, matches=None):
        """Return a catalog of selected sources

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to select.
        matches : `lsst.afw.table.ReferenceMatchVector`, optional
            List of matches; ignored.

        Return struct
        -------------
        sourceCat : `lsst.afw.table.SourceCatalog`
            Catalog of selected sources, non-contiguous.
        """
        selected = np.ones(len(catalog), dtype=bool)
        if self.config.doFluxLimit:
            selected &= self.config.fluxLimit.apply(catalog)
        if self.config.doFlags:
            selected &= self.config.flags.apply(catalog)
        if self.config.doUnresolved:
            selected &= self.config.unresolved.apply(catalog)
        if self.config.doSignalToNoise:
            selected &= self.config.signalToNoise.apply(catalog)
        if self.config.doIsolated:
            selected &= self.config.isolated.apply(catalog)

        self.log.info("Selected %d/%d sources", selected.sum(), len(catalog))

        return pipeBase.Struct(sourceCat=catalog[selected],
                               selection=selected)

class ReferenceSourceSelectorConfig(pexConfig.Config):
    doMagLimit = pexConfig.Field(dtype=bool, default=False, doc="Apply magnitude limit?")
    doFlags = pexConfig.Field(dtype=bool, default=False, doc="Apply flag limitation?")
    doSignalToNoise = pexConfig.Field(dtype=bool, default=False, doc="Apply signal-to-noise limit?")
    doMagError = pexConfig.Field(dtype=bool, default=False, doc="Apply magnitude error limit?")
    magLimit = pexConfig.ConfigField(dtype=MagnitudeLimit, doc="Magnitude limit to apply")
    flags = pexConfig.ConfigField(dtype=RequireFlags, doc="Flags to require")
    signalToNoise = pexConfig.ConfigField(dtype=SignalToNoiseLimit, doc="Signal-to-noise limit to apply")
    magError = pexConfig.ConfigField(dtype=MagnitudeErrorLimit, doc="Magnitude error limit to apply")
    colorLimits = pexConfig.ConfigDictField(keytype=str, itemtype=ColorLimit, default={},
                                            doc="Color limits to apply; key is used as a label only")


class ReferenceSourceSelectorTask(BaseSourceSelectorTask):
    """Reference source selector

    This selects reference sources by (optionally) applying each of a
    magnitude limit, flag requirements and color limits.
    """
    ConfigClass = ReferenceSourceSelectorConfig

    def selectSources(self, catalog, matches=None):
        """Return a catalog of selected reference sources

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of sources to select.
        matches : `lsst.afw.table.ReferenceMatchVector`, optional
            List of matches; ignored.

        Return struct
        -------------
        sourceCat : `lsst.afw.table.SourceCatalog`
            Catalog of selected sources, non-contiguous.
        """
        selected = np.ones(len(catalog), dtype=bool)
        if self.config.doMagLimit:
            selected &= self.config.magLimit.apply(catalog)
        if self.config.doFlags:
            selected &= self.config.flags.apply(catalog)
        if self.config.doSignalToNoise:
            selected &= self.config.signalToNoise.apply(catalog)
        if self.config.doMagError:
            selected &= self.config.magError.apply(catalog)
        for limit in self.config.colorLimits.values():
            selected &= limit.apply(catalog)

        self.log.info("Selected %d/%d references", selected.sum(), len(catalog))

        result = type(catalog)(catalog.table)  # Empty catalog based on the original
        for source in catalog[selected]:
            result.append(source)
        return pipeBase.Struct(sourceCat=result, selection=selected)


sourceSelectorRegistry.register("science", ScienceSourceSelectorTask)
sourceSelectorRegistry.register("references", ReferenceSourceSelectorTask)
