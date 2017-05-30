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

__all__ = ["BaseSourceSelectorConfig", "BaseSourceSelectorTask",
           "sourceSelectorRegistry"]

import abc

import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from future.utils import with_metaclass


class BaseSourceSelectorConfig(pexConfig.Config):
    pass


@with_metaclass(abc.ABCMeta)
class BaseSourceSelectorTask(pipeBase.Task):
    """Base class for source selectors

    Write paragraph about what a source selector is/does.

    Register all source selectors with the sourceSelectorRegistry using:
        sourceSelectorRegistry.register(name, class)

    Attributes
    ----------
    usesMatches : bool
        A boolean variable specify if the inherited source selector uses
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

        The input catalog must be contiguous in memory.

        Parameters:
        -----------
        source_cat : lsst.afw.table.SourceCatalog
            Catalog of sources that may be sources
        source_selected_field : {None} lsst.afw.table.Key
            Key of flag field in source_cat to set for selected sources.
            If set, will modify source_cat in-place.
        masked_image : {None} lsst.afw.image.MaskedImage
            Masked image containing the sources. A few source selectors may
            use this in their selection but most will just use it for
            plotting.
        matches : {None} list of lsst.afw.table.ReferenceMatch
            A list of lsst.afw.table.ReferenceMatch objects. If use_matches
            set in source selector, this field is required otherwise ignored.

        Return
        ------
        lsst.pipe.base.Struct
            The struct contains the following data:

            source_cat : lsst.afw.table.SourceCatalog
                The catalog of sources that were selected.
                (may not be memory-contiguous)
        """
        result = self.select_sources(source_cat=source_cat,
                                     masked_image=masked_image,
                                     matches=matches)

        if source_selected_field is not None:
            # TODO: Remove for loop when DM-6981 is completed.
            for source, flag in zip(source_cat, result.selected):
                source.set(source_selected_field) = flag
        return pipeBase.Struct(source_cat=source_cat[result.selected])

    @abc.abstractmethod
    def select_sources(self, source_cat, masked_image=None, matches=None):
        """Return a catalog of sources selected by specified criteria.

        The input catalog must be contiguous in memory.

        Parameters
        ----------
        source_cat : lsst.afw.table.SourceCatalog
            catalog of sources that may be sources
        masked_image : {None} lsst.afw.image
            An image containing the sources tests or for plotting.
        matches : {None} list of lsst.afw.table.ReferenceMatch
            A list of lsst.afw.table.ReferenceMatch objects

        Return
        ------
        lsst.pipe.base.Struct
            The struct contains the following data:

            selected : bool array
                Boolean array of sources that were selected, same length as
                source_cat.
        """
        raise NotImplementedError("BaseSourceSelectorTask is abstract")

sourceSelectorRegistry = pexConfig.makeRegistry(
    doc="A registry of source selectors (subclasses of "
        "BaseSourceSelectorTask)",
)
