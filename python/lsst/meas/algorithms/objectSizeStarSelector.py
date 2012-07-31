# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import collections
import math

import numpy
try:
    import matplotlib.pyplot as pyplot
except ImportError:
    pyplot = None

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDetection
import lsst.afw.display.ds9 as ds9
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as geomEllip
import lsst.afw.cameraGeom as cameraGeom
from . import algorithmsLib
from . import measurement
from lsst.meas.algorithms.starSelectorRegistry import starSelectorRegistry

class ObjectSizeStarSelectorConfig(pexConfig.Config):
    fluxMin = pexConfig.Field(
        doc = "specify the minimum psfFlux for good Psf Candidates",
        dtype = float,
        default = 12500.0,
#        minValue = 0.0,
        check = lambda x: x >= 0.0,
    )
    fluxMax = pexConfig.Field(
        doc = "specify the maximum psfFlux for good Psf Candidates (ignored if == 0)",
        dtype = float,
        default = 0.0,
        check = lambda x: x >= 0.0,
    )
    kernelSize = pexConfig.Field(
        doc = "size of the kernel to create",
        dtype = int,
        default = 21,
    )
    borderWidth = pexConfig.Field(
        doc = "number of pixels to ignore around the edge of PSF candidate postage stamps",
        dtype = int,
        default = 0,
    )
    badFlags = pexConfig.ListField(
        doc = "List of flags which cause a source to be rejected as bad",
        dtype = str,
        default = ["flags.pixel.edge", "flags.pixel.interpolated.center", "flags.pixel.saturated.center"]
        )
    histSize = pexConfig.Field(
        doc = "Number of bins in size histogram",
        dtype = int,
        default = 64,
        )
    widthMin = pexConfig.Field(
        doc = "minimum width to include in histogram",
        dtype = float,
        default = 0.0,
        check = lambda x: x >= 0.0,
        )
    widthMax = pexConfig.Field(
        doc = "maximum width to include in histogram",
        dtype = float,
        default = 10.0,
        check = lambda x: x >= 0.0,
        )

class ObjectSizeStarSelector(object):
    ConfigClass = ObjectSizeStarSelectorConfig

    def __init__(self, config, schema=None, key=None):
        """Construct a star selector that uses second moments
        
        This is a naive algorithm and should be used with caution.
        
        @param[in] config: An instance of ObjectSizeStarSelectorConfig
        @param[in,out] schema: An afw.table.Schema to register the selector's flag field.
                               If None, the sources will not be modified.
        @param[in] key: An existing Flag Key to use instead of registering a new field.
        """
        self._kernelSize  = config.kernelSize
        self._borderWidth = config.borderWidth
        self._widthMin = config.widthMin
        self._widthMax = config.widthMax
        self._fluxMin  = config.fluxMin
        self._fluxMax  = config.fluxMax
        self._badFlags = config.badFlags
        self._histSize = config.histSize
        if key is not None:
            self._key = key
            if schema is not None and key not in schema:
                raise LookupError("The key passed to the star selector is not present in the schema")
        elif schema is not None:
            self._key = schema.addField("classification.objectSize.star", type="Flag",
                                        doc="selected as a star by ObjectSizeStarSelector")
        else:
            self._key = None
            
    def selectStars(self, exposure, catalog, matches=None):
        """Return a list of PSF candidates that represent likely stars
        
        A list of PSF candidates may be used by a PSF fitter to construct a PSF.
        
        @param[in] exposure: the exposure containing the sources
        @param[in] catalog: a SourceCatalog containing sources that may be stars
        @param[in] matches: astrometric matches; ignored by this star selector
        
        @return psfCandidateList: a list of PSF candidates.
        """
        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displayExposure = lsstDebug.Info(__name__).displayExposure     # display the Exposure + spatialCells
        plotMagSize = lsstDebug.Info(__name__).plotMagSize             # display the magnitude-size relation

	detector = exposure.getDetector()
	distorter = None
	xy0 = afwGeom.Point2D(0,0)
	if not detector is None:
	    cPix = detector.getCenterPixel()
	    detSize = detector.getSize()
	    xy0.setX(cPix.getX() - int(0.5*detSize.getMm()[0]))
	    xy0.setY(cPix.getY() - int(0.5*detSize.getMm()[1]))
	    distorter = detector.getDistortion()
        #
        # Look at the distribution of stars in the magnitude-size plane
        #
        flux = catalog.get("initial.flux.gaussian")
        mag = -2.5*numpy.log10(flux)

        xx = numpy.empty(len(catalog))
        xy = numpy.empty_like(xx)
        yy = numpy.empty_like(xx)
        for i, source in enumerate(catalog):
            Ixx, Ixy, Iyy = source.getIxx(), source.getIxy(), source.getIyy()
            if distorter:
                xpix, ypix = source.getX() + xy0.getX(), source.getY() + xy0.getY()
                p = afwGeom.Point2D(xpix, ypix)
                m = distorter.undistort(p, geomEllip.Quadrupole(Ixx, Iyy, Ixy), detector)
                Ixx, Ixy, Iyy = m.getIxx(), m.getIxy(), m.getIyy()

            xx[i], xy[i], yy[i] = Ixx, Ixy, Iyy
            
        width = numpy.sqrt(xx + yy)

        bad = reduce(lambda x, y: numpy.logical_or(x, catalog.get(y)), self._badFlags, False)
        bad = numpy.logical_or(bad, flux < self._fluxMin)
        if self._fluxMax > 0:
            bad = numpy.logical_or(bad, flux > self._fluxMax)
        good = numpy.logical_not(bad)

        #
        # Look for the maximum in the size histogram, then search upwards for the minimum that separates
        # the initial peak (of, we presume, stars) from the galaxies
        #
        widthHist, bins = numpy.histogram(width, bins=self._histSize, range=(self._widthMin, self._widthMax))
        imax = numpy.where(widthHist == max(widthHist))[0][0]
        for i in range(imax + 1, len(widthHist)):
            if widthHist[i] == 0 or widthHist[i] > widthHist[i-1]:
                break
        widthCrit = 0.5*(bins[i - 1] + bins[i])
        
        stellar = width < widthCrit
        #
        # We know enough to plot, if so requested
        #
        if display and plotMagSize and pyplot:
            fig = pyplot.figure()
            axes = fig.add_axes((0.1, 0.1, 0.85, 0.80));

            l = numpy.logical_and(good, stellar)
            axes.plot(mag[l], width[l], "o", markersize=3, markeredgewidth=0, color="green")
            l = numpy.logical_and(good, numpy.logical_not(stellar))
            axes.plot(mag[l], width[l], "o", markersize=3, markeredgewidth=0, color="red")
            axes.set_ylim(0, 10)

            axes.set_xlabel("model")
            axes.set_ylabel(r"$\sqrt{I_{xx} + I_{yy}}$")
            fig.show()

            #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            fig = pyplot.figure()
            color = [("g" if bins[i] < widthCrit else "r") for i in range(len(widthHist))]
            axes = fig.add_axes((0.1, 0.1, 0.85, 0.80));
            axes.bar(bins[:-1], widthHist, width=bins[1] - bins[0], color=color)
            axes.set_xlabel(r"$\sqrt{I_{xx} + I_{yy}}$")
            axes.set_ylabel("N")
            fig.show()

            try:
                reply = raw_input("continue? [y q(uit) p(db)] ").strip()
            except EOFError:
                reply = "y"

            if reply:
                if reply[0] == "p":
                    import pdb; pdb.set_trace()
                elif reply[0] == 'q':
                    sys.exit(1)
        
        if display and displayExposure:
            frame = 0
            mi = exposure.getMaskedImage()
            ds9.mtv(mi, frame=frame, title="PSF candidates")
    
            with ds9.Buffering():
                for i, source in enumerate(catalog):
                    if good[i]:
                        ctype = ds9.GREEN # star candidate
                    else:
                        ctype = ds9.RED # not star
			
                    ds9.dot("+", source.getX() - mi.getX0(),
                            source.getY() - mi.getY0(), frame=frame, ctype=ctype)
        #
        # Time to use that stellar classification to generate psfCandidateList
        #
        with ds9.Buffering():
            psfCandidateList = []
            for i, source in enumerate(catalog):
                if not (stellar[i] and good[i]):
                    continue
                
                try:
                    psfCandidate = algorithmsLib.makePsfCandidate(source, exposure)
                    # The setXXX methods are class static, but it's convenient to call them on
                    # an instance as we don't know Exposure's pixel type
                    # (and hence psfCandidate's exact type)
                    if psfCandidate.getWidth() == 0:
                        psfCandidate.setBorderWidth(self._borderWidth)
                        psfCandidate.setWidth(self._kernelSize + 2*self._borderWidth)
                        psfCandidate.setHeight(self._kernelSize + 2*self._borderWidth)

                    im = psfCandidate.getMaskedImage().getImage()
                    vmax = afwMath.makeStatistics(im, afwMath.MAX).getValue()
                    if not numpy.isfinite(vmax):
                        continue
                    if self._key is not None:
                        source.set(self._key, True)
                    psfCandidateList.append(psfCandidate)

                    if display and displayExposure:
                        ds9.dot("o", source.getX() - mi.getX0(), source.getY() - mi.getY0(),
                                size=4, frame=frame, ctype=ds9.CYAN)
                except Exception as err:
                    pass # FIXME: should log this!

        return psfCandidateList

starSelectorRegistry.register("objectSize", ObjectSizeStarSelector)
