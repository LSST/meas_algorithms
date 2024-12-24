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

__all__ = ["SetPrimaryFlagsConfig", "SetPrimaryFlagsTask"]

import numpy as np

from lsst.pex.config import Config, ListField
from lsst.pipe.base import Task
from lsst.geom import Box2D


class SetPrimaryFlagsConfig(Config):
    pseudoFilterList = ListField(dtype=str, default=['sky'],
                                 doc="Names of filters which should never be primary")


class SetPrimaryFlagsTask(Task):
    """Set the ``isPrimary`` flag and either various blendedness, or
    patch/tract flags to a catalog (for single frame or coadd catalogs,
    respectively), based on other properties of the sources.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`
        Source catalog schema to add fields to.
    isSingleFrame : `bool`
        Flag specifying if task is operating with single frame imaging.
    includeDeblend : `bool`
        Include deblend information in isPrimary and add blendedness fields?

    Notes
    -----
    The tests for this task still live in
    ``pipe_tasks/tests/test_isPrimaryFlag.py``; see discussion on DM-42720.
    """

    _DefaultName = "setPrimaryFlags"
    ConfigClass = SetPrimaryFlagsConfig

    def __init__(self, *, schema, isSingleFrame=False, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema
        self.isSingleFrame = isSingleFrame
        self.includeDeblend = False
        if not self.isSingleFrame:
            primaryDoc = ("True if source has no children and is in the inner region of a coadd patch "
                          "and is in the inner region of a coadd tract "
                          "and is not \"detected\" in a pseudo-filter (see config.pseudoFilterList)")
            self.isPatchInnerKey = self.schema.addField(
                "detect_isPatchInner", type="Flag",
                doc="True if source is in the inner region of a coadd patch",
            )
            self.isTractInnerKey = self.schema.addField(
                "detect_isTractInner", type="Flag",
                doc="True if source is in the inner region of a coadd tract",
            )
        else:
            primaryDoc = "True if source has no children and is not a sky source."
        self.isPrimaryKey = self.schema.addField(
            "detect_isPrimary", type="Flag",
            doc=primaryDoc,
        )

        if "deblend_nChild" in schema.getNames():
            self.includeDeblend = True
            self.isDeblendedSourceKey = self.schema.addField(
                "detect_isDeblendedSource", type="Flag",
                doc=primaryDoc + " and is either an unblended isolated source or a "
                                 "deblended child from a parent with 'deblend_nChild' > 1")
            self.fromBlendKey = self.schema.addField(
                "detect_fromBlend", type="Flag",
                doc="This source is deblended from a parent with more than one child."
            )
            self.isIsolatedKey = self.schema.addField(
                "detect_isIsolated", type="Flag",
                doc="This source is not a part of a blend."
            )
            if "deblend_scarletFlux" in schema.getNames():
                self.isDeblendedModelKey = self.schema.addField(
                    "detect_isDeblendedModelSource", type="Flag",
                    doc=primaryDoc + " and is a deblended child")
            else:
                self.isDeblendedModelKey = None

    def run(self, sources, skyMap=None, tractInfo=None, patchInfo=None):
        """Set isPrimary and related flags on sources.

        For coadded imaging, the `isPrimary` flag returns True when an object
        has no children, is in the inner region of a coadd patch, is in the
        inner region of a coadd trach, and is not detected in a pseudo-filter
        (e.g., a sky_object).
        For single frame imaging, the isPrimary flag returns True when a
        source has no children and is not a sky source.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            A sourceTable. Reads in centroid fields and an nChild field.
            Writes is-patch-inner, is-tract-inner, and is-primary flags.
        skyMap : `lsst.skymap.BaseSkyMap`
            Sky tessellation object
        tractInfo : `lsst.skymap.TractInfo`, optional
            Tract object; required if ``self.isSingleFrame`` is False.
        patchInfo : `lsst.skymap.PatchInfo`
            Patch object; required if ``self.isSingleFrame`` is False.
        """
        # Mark whether sources are contained within the inner regions of the
        # given tract/patch and are not "pseudo" (e.g. sky) sources.
        if not self.isSingleFrame:
            isPatchInner = getPatchInner(sources, patchInfo)
            isTractInner = getTractInner(sources, tractInfo, skyMap)
            isPseudo = self._getPseudoSources(sources)
            isPrimary = isTractInner & isPatchInner & ~isPseudo

            sources[self.isPatchInnerKey] = isPatchInner
            sources[self.isTractInnerKey] = isTractInner
        else:
            # Mark all of the sky sources in SingleFrame images
            # (if they were added)
            if "sky_source" in sources.schema:
                isSky = sources["sky_source"]
            else:
                isSky = np.zeros(len(sources), dtype=bool)
            isPrimary = ~isSky

        if self.includeDeblend:
            result = getDeblendPrimaryFlags(sources)
            fromBlend, isIsolated, isDeblendedSource, isDeblendedModelSource = result
            sources[self.fromBlendKey] = fromBlend
            sources[self.isIsolatedKey] = isIsolated
            sources[self.isDeblendedSourceKey] = isDeblendedSource
            if self.isDeblendedModelKey is not None:
                sources[self.isDeblendedModelKey] = isDeblendedModelSource
            isPrimary = isPrimary & isDeblendedSource

        sources[self.isPrimaryKey] = isPrimary

    def _getPseudoSources(self, sources):
        """Get a flag that marks pseudo sources.

        Some categories of sources, for example sky objects, are not really
        detected sources and should not be considered primary sources.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            The catalog of sources for which to identify "pseudo"
            (e.g. sky) objects.

        Returns
        -------
        isPseudo : array-like of `bool`
            True for each source that is a pseudo source.
            Note: to remove pseudo sources use `~isPseudo`.
        """
        # Filter out sources that should never be primary.
        isPseudo = np.zeros(len(sources), dtype=bool)
        for filt in self.config.pseudoFilterList:
            try:
                pseudoFilterKey = self.schema.find("merge_peak_%s" % filt).getKey()
                isPseudo |= sources[pseudoFilterKey]
            except KeyError:
                self.log.warning("merge_peak is not set for pseudo-filter %s", filt)
        return isPseudo


def getPatchInner(sources, patchInfo):
    """Set a flag for each source if it is in the innerBBox of a patch.

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog with pre-calculated centroids.
    patchInfo : `lsst.skymap.PatchInfo`
        Information about a `SkyMap` `Patch`.

    Returns
    -------
    isPatchInner : array-like of `bool`
        `True` for each source that has a centroid
        in the inner region of a patch.
    """
    # Extract the centroid position for all the sources
    x = sources["slot_Centroid_x"]
    y = sources["slot_Centroid_y"]
    centroidFlag = sources["slot_Centroid_flag"]

    # set inner flags for each source and set primary flags for
    # sources with no children (or all sources if deblend info not available)
    innerFloatBBox = Box2D(patchInfo.getInnerBBox())
    inInner = innerFloatBBox.contains(x, y)

    # When the centroider fails, we can still fall back to the peak,
    # but we don't trust that quite as much -
    # so we use a slightly smaller box for the patch comparison.
    shrunkInnerFloatBBox = Box2D(innerFloatBBox)
    shrunkInnerFloatBBox.grow(-1)
    inShrunkInner = shrunkInnerFloatBBox.contains(x, y)

    # Flag sources contained in the inner region of a patch
    isPatchInner = (centroidFlag & inShrunkInner) | (~centroidFlag & inInner)
    return isPatchInner


def getTractInner(sources, tractInfo, skyMap):
    """Set a flag for each source that the skyMap includes in tractInfo.

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog with pre-calculated centroids.
    tractInfo : `lsst.skymap.TractInfo`
        Tract object
    skyMap : `lsst.skymap.BaseSkyMap`
        Sky tessellation object

    Returns
    -------
    isTractInner : array-like of `bool`
        True if the skyMap.findTract method returns
        the same tract as tractInfo.
    """
    tractId = tractInfo.getId()
    isTractInner = np.array([skyMap.findTract(s.getCoord()).getId() == tractId for s in sources])
    return isTractInner


def getDeblendPrimaryFlags(sources):
    """Get flags generated by the deblender.

    scarlet is different than meas_deblender in that it is not
    (necessarily) flux conserving. For consistency in scarlet,
    all of the parents with only a single child (isolated sources)
    need to be deblended. This creates a question: which type
    of isolated source should we make measurements on, the
    undeblended "parent" or the deblended child?
    For that reason we distinguish between a DeblendedSource,
    which is a source that has no children and uses the
    isolated parents, and a DeblendedModelSource, which uses
    the scarlet models for both isolated and blended sources.
    In the case of meas_deblender, DeblendedModelSource is
    `None` because it is not contained in the output catalog.

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog that has already been deblended using
        either meas_extensions_scarlet or meas_deblender.

    Returns
    -------
    fromBlend : array-like of `bool`
        True for each source modeled by the deblender from a `Peak`
        in a parent footprint that contained at least one other `Peak`.
        While these models can be approximated as isolated,
        and measurements are made on them as if that's the case,
        we know deblending to introduce biases in the shape and centroid
        of objects and it is important to know that the sources that these
        models are based on are all bleneded in the true image.
    isIsolated : array-like of `bool`
        True for isolated sources, regardless of whether or not they
        were modeled by the deblender.
    isDeblendedSource : array-like of `bool`
        True for each source that is a "DeblendedSource" as defined above.
    isDeblendedModelSource : array-like of `bool`
        True for each source that is a "DeblendedSourceModel"
        as defined above.
    """
    nChildKey = "deblend_nChild"
    nChild = sources[nChildKey]
    parent = sources["parent"]

    if "deblend_scarletFlux" in sources.schema:
        # The number of peaks in the sources footprint.
        # This (may be) different than nChild,
        # the number of deblended sources in the catalog,
        # because some peaks might have been culled during deblending.
        nPeaks = sources["deblend_nPeaks"]
        nChild = sources["deblend_nChild"]
        parentNPeaks = sources["deblend_parentNPeaks"]
        depth = sources["deblend_depth"]
        # It is possible for a catalog to contain a hierarchy of sources,
        # so we mark the leaves (end nodes of the hierarchy tree with no
        # children).
        isLeaf = nChild == 0
        isChild = depth > 0
        isSibling = parentNPeaks > 1
        isDeblendedModelSource = isLeaf & isChild
        fromBlend = isDeblendedModelSource & (isSibling | depth > 1)
        isIsolatedParent = (depth == 0) & (nPeaks == 1)
        isIsolated = isIsolatedParent | ((depth == 1) & ~isSibling)
        isDeblendedSource = fromBlend | isIsolatedParent
    else:
        # Set the flags for meas_deblender
        fromBlend = parent != 0
        isIsolated = (nChild == 0) & (parent == 0)
        isDeblendedSource = nChild == 0
        isDeblendedModelSource = None
    return fromBlend, isIsolated, isDeblendedSource, isDeblendedModelSource
