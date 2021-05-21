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
import lsst.sphgeom


class HtmIndexer:
    """Manage a spatial index of hierarchical triangular mesh (HTM)
    shards.

    Parameters
    ----------
    depth : `int`
        Depth of the HTM hierarchy to construct.
    """
    def __init__(self, depth=8):
        self.pixelization = lsst.sphgeom.HtmPixelization(depth)

    def getShardIds(self, ctrCoord, radius):
        """Get the IDs of all shards that touch a circular aperture.

        Parameters
        ----------
        ctrCoord : `lsst.geom.SpherePoint`
            ICRS center of search region.
        radius : `lsst.geom.Angle`
            Radius of search region.

        Returns
        -------
        results : `tuple`
            A tuple containing:

            - shardIdList : `list` of `int`
                List of shard IDs
            - isOnBoundary : `list` of `bool`
                For each shard in ``shardIdList`` is the shard on the
                boundary (not fully enclosed by the search region)?
        """
        circle = lsst.sphgeom.Circle(ctrCoord.getVector(), lsst.sphgeom.Angle.fromRadians(radius.asRadians()))
        interior = self.pixelization.interior(circle)
        shardIdList = []
        isOnBoundary = []
        for begin, end in self.pixelization.envelope(circle):
            for shardId in range(begin, end):
                shardIdList.append(shardId)
                isOnBoundary.append(not interior.contains(shardId))
        return shardIdList, isOnBoundary

    def indexPoints(self, raList, decList):
        """Generate shard IDs for sky positions.

        Parameters
        ----------
        raList : `list` of `float`
            List of right ascensions, in degrees.
        decList : `list` of `float`
            List of declinations, in degrees.

        Returns
        -------
        shardIds : `list` of `int`
            List of shard IDs
        """
        return [
            self.pixelization.index(lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees).getVector())
            for ra, dec in zip(raList, decList)
        ]

    @staticmethod
    def makeDataId(shardId, datasetName):
        """Make a data id from a shard ID.

        Parameters
        ----------
        shardId : `int`
            ID of shard in question.
        datasetName : `str`
            Name of dataset to use.

        Returns
        -------
        dataId : `dict`
            Data ID for shard.
        """
        if shardId is None:
            # NoneType doesn't format, so make dummy pixel
            shardId = 0
        return {'pixel_id': shardId, 'name': datasetName}
