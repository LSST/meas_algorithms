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

"""
Convert an external reference catalog into the hierarchical triangular mesh
(HTM) sharded LSST-style format, to be ingested into the butler.
"""

__all__ = ["ConvertReferenceCatalogTask", "ConvertReferenceCatalogConfig", "DatasetConfig"]

import argparse
import glob
import os
import pathlib
import logging

import astropy

import lsst.afw.table
import lsst.pipe.base
import lsst.pex.config as pexConfig
from lsst.daf.base import PropertyList

from .indexerRegistry import IndexerRegistry
from .readTextCatalogTask import ReadTextCatalogTask
from . import convertRefcatManager
from . import ReferenceObjectLoader

# The most recent Indexed Reference Catalog on-disk format version.
LATEST_FORMAT_VERSION = 1


def addRefCatMetadata(catalog):
    """Add metadata to a new (not yet populated) reference catalog.

    Parameters
    ----------
    catalog : `lsst.afw.table.SimpleCatalog`
        Catalog to which metadata should be attached.  Will be modified
        in-place.
    """
    md = catalog.getMetadata()
    if md is None:
        md = PropertyList()
    md.set("REFCAT_FORMAT_VERSION", LATEST_FORMAT_VERSION)
    catalog.setMetadata(md)


class DatasetConfig(pexConfig.Config):
    """Description of the on-disk storage format for the converted reference
    catalog.
    """
    format_version = pexConfig.Field(
        dtype=int,
        doc="Version number of the persisted on-disk storage format."
        "\nVersion 0 had Jy as flux units (default 0 for unversioned catalogs)."
        "\nVersion 1 had nJy as flux units.",
        default=0  # This needs to always be 0, so that unversioned catalogs are interpreted as version 0.
    )
    ref_dataset_name = pexConfig.Field(
        dtype=str,
        doc="Name of this reference catalog; this should match the name used during butler ingest.",
    )
    indexer = IndexerRegistry.makeField(
        default='HTM',
        doc='Name of indexer algoritm to use.  Default is HTM',
    )


class ConvertReferenceCatalogConfig(pexConfig.Config):
    dataset_config = pexConfig.ConfigField(
        dtype=DatasetConfig,
        doc="Configuration for reading the ingested data",
    )
    n_processes = pexConfig.Field(
        dtype=int,
        doc=("Number of python processes to use when ingesting."),
        default=1
    )
    manager = pexConfig.ConfigurableField(
        target=convertRefcatManager.ConvertRefcatManager,
        doc="Multiprocessing manager to perform the actual conversion of values, file-by-file."
    )
    file_reader = pexConfig.ConfigurableField(
        target=ReadTextCatalogTask,
        doc='Task to use to read the files.  Default is to expect text files.'
    )
    ra_name = pexConfig.Field(
        dtype=str,
        doc="Name of RA column (values in decimal degrees)",
    )
    dec_name = pexConfig.Field(
        dtype=str,
        doc="Name of Dec column (values in decimal degrees)",
    )
    ra_err_name = pexConfig.Field(
        dtype=str,
        doc="Name of RA error column",
        optional=True,
    )
    dec_err_name = pexConfig.Field(
        dtype=str,
        doc="Name of Dec error column",
        optional=True,
    )
    coord_err_unit = pexConfig.Field(
        dtype=str,
        doc="Unit of RA/Dec error fields (astropy.unit.Unit compatible)",
        optional=True
    )
    mag_column_list = pexConfig.ListField(
        dtype=str,
        doc="The values in the reference catalog are assumed to be in AB magnitudes. "
            "List of column names to use for photometric information.  At least one entry is required."
    )
    mag_err_column_map = pexConfig.DictField(
        keytype=str,
        itemtype=str,
        default={},
        doc="A map of magnitude column name (key) to magnitude error column (value)."
    )
    is_photometric_name = pexConfig.Field(
        dtype=str,
        optional=True,
        doc='Name of column stating if satisfactory for photometric calibration (optional).'
    )
    is_resolved_name = pexConfig.Field(
        dtype=str,
        optional=True,
        doc='Name of column stating if the object is resolved (optional).'
    )
    is_variable_name = pexConfig.Field(
        dtype=str,
        optional=True,
        doc='Name of column stating if the object is measured to be variable (optional).'
    )
    id_name = pexConfig.Field(
        dtype=str,
        optional=True,
        doc='Name of column to use as an identifier (optional).'
    )
    pm_ra_name = pexConfig.Field(
        dtype=str,
        doc="Name of proper motion RA column",
        optional=True,
    )
    pm_dec_name = pexConfig.Field(
        dtype=str,
        doc="Name of proper motion Dec column",
        optional=True,
    )
    pm_ra_err_name = pexConfig.Field(
        dtype=str,
        doc="Name of proper motion RA error column",
        optional=True,
    )
    pm_dec_err_name = pexConfig.Field(
        dtype=str,
        doc="Name of proper motion Dec error column",
        optional=True,
    )
    pm_scale = pexConfig.Field(
        dtype=float,
        doc="Scale factor by which to multiply proper motion values to obtain units of milliarcsec/year",
        default=1.0,
    )
    parallax_name = pexConfig.Field(
        dtype=str,
        doc="Name of parallax column",
        optional=True,
    )
    parallax_err_name = pexConfig.Field(
        dtype=str,
        doc="Name of parallax error column",
        optional=True,
    )
    parallax_scale = pexConfig.Field(
        dtype=float,
        doc="Scale factor by which to multiply parallax values to obtain units of milliarcsec",
        default=1.0,
    )
    epoch_name = pexConfig.Field(
        dtype=str,
        doc="Name of epoch column",
        optional=True,
    )
    epoch_format = pexConfig.Field(
        dtype=str,
        doc="Format of epoch column: any value accepted by astropy.time.Time, e.g. 'iso' or 'unix'",
        optional=True,
    )
    epoch_scale = pexConfig.Field(
        dtype=str,
        doc="Scale of epoch column: any value accepted by astropy.time.Time, e.g. 'utc'",
        optional=True,
    )
    extra_col_names = pexConfig.ListField(
        dtype=str,
        default=[],
        doc='Extra columns to add to the reference catalog.'
    )

    def setDefaults(self):
        # Newly ingested reference catalogs always have the latest format_version.
        self.dataset_config.format_version = LATEST_FORMAT_VERSION
        # gen3 refcats are all depth=7
        self.dataset_config.indexer['HTM'].depth = 7

    def validate(self):
        pexConfig.Config.validate(self)

        def assertAllOrNone(*names):
            """Raise ValueError unless all the named fields are set or are
            all none (or blank).
            """
            setNames = [name for name in names if bool(getattr(self, name))]
            if len(setNames) in (len(names), 0):
                return
            prefix = "Both or neither" if len(names) == 2 else "All or none"
            raise ValueError("{} of {} must be set, but only {} are set".format(
                prefix, ", ".join(names), ", ".join(setNames)))

        if not (self.ra_name and self.dec_name and self.mag_column_list):
            raise ValueError(
                "ra_name and dec_name and at least one entry in mag_column_list must be supplied.")
        if self.mag_err_column_map and set(self.mag_column_list) != set(self.mag_err_column_map.keys()):
            raise ValueError(
                "mag_err_column_map specified, but keys do not match mag_column_list: {} != {}".format(
                    sorted(self.mag_err_column_map.keys()), sorted(self.mag_column_list)))
        assertAllOrNone("ra_err_name", "dec_err_name", "coord_err_unit")
        if self.coord_err_unit is not None:
            result = astropy.units.Unit(self.coord_err_unit, parse_strict='silent')
            if isinstance(result, astropy.units.UnrecognizedUnit):
                msg = f"{self.coord_err_unit} is not a valid astropy unit string."
                raise pexConfig.FieldValidationError(ConvertReferenceCatalogConfig.coord_err_unit, self, msg)

        assertAllOrNone("epoch_name", "epoch_format", "epoch_scale")
        assertAllOrNone("pm_ra_name", "pm_dec_name")
        assertAllOrNone("pm_ra_err_name", "pm_dec_err_name")
        assertAllOrNone("parallax_name", "parallax_err_name")
        if self.pm_ra_err_name and not self.pm_ra_name:
            raise ValueError('"pm_ra/dec_name" must be specified if "pm_ra/dec_err_name" are specified')
        if (self.pm_ra_name or self.parallax_name) and not self.epoch_name:
            raise ValueError(
                '"epoch_name" must be specified if "pm_ra/dec_name" or "parallax_name" are specified')


class ConvertReferenceCatalogTask(lsst.pipe.base.Task):
    """Class for producing HTM-indexed reference catalogs from external
    catalog data.

    This implements an indexing scheme based on hierarchical triangular
    mesh (HTM). The term index really means breaking the catalog into
    localized chunks called shards.  In this case each shard contains
    the entries from the catalog in a single HTM trixel

    For producing catalogs this task makes the following assumptions
    about the input catalogs:

    - RA, Dec are in decimal degrees.
    - Epoch is available in a column, in a format supported by astropy.time.Time.
    - There are no off-diagonal covariance terms, such as covariance
      between RA and Dec, or between PM RA and PM Dec. Support for such
      covariance would have to be added to to the config, including consideration
      of the units in the input catalog.

    Parameters
    ----------
    output_dir : `str`
        The path to write the output files to, in a subdirectory defined by
        ``DatasetConfig.ref_dataset_name``.
    """
    canMultiprocess = False
    ConfigClass = ConvertReferenceCatalogConfig
    _DefaultName = 'ConvertReferenceCatalogTask'

    def __init__(self, *, output_dir=None, **kwargs):
        super().__init__(**kwargs)
        if output_dir is None:
            raise RuntimeError("Must specify output_dir.")
        self.base_dir = output_dir
        self.output_dir = os.path.join(output_dir, self.config.dataset_config.ref_dataset_name)
        self.ingest_table_file = os.path.join(self.base_dir, "filename_to_htm.ecsv")
        self.indexer = IndexerRegistry[self.config.dataset_config.indexer.name](
            self.config.dataset_config.indexer.active)
        self.makeSubtask('file_reader')

    def run(self, inputFiles):
        """Index a set of files comprising a reference catalog.

        Outputs are persisted in the butler repository.

        Parameters
        ----------
        inputFiles : `list`
            A list of file paths to read.
        """
        # Create the output path, if it doesn't exist; fail if the path exists:
        # we don't want to accidentally append to existing files.
        pathlib.Path(self.output_dir).mkdir(exist_ok=False)

        schema, key_map = self._writeMasterSchema(inputFiles[0])
        # create an HTM we can interrogate about pixel ids
        htm = lsst.sphgeom.HtmPixelization(self.indexer.htm.get_depth())
        filenames = self._getOutputFilenames(htm)
        worker = self.config.manager.target(filenames,
                                            self.config,
                                            self.file_reader,
                                            self.indexer,
                                            schema,
                                            key_map,
                                            htm.universe()[0],
                                            addRefCatMetadata,
                                            self.log)
        result = worker.run(inputFiles)

        self._writeConfig()
        self._writeIngestHelperFile(result)

    def _writeIngestHelperFile(self, result):
        """Write the astropy table containing the htm->filename relationship,
        used for the ``butler ingest-files`` command after this task completes.
        """
        dimension = f"htm{self.config.dataset_config.indexer.active.depth}"
        table = astropy.table.Table(names=("filename", dimension), dtype=('str', 'int'))
        for key in result:
            table.add_row((result[key], key))
        table.write(self.ingest_table_file)

    def _writeConfig(self):
        """Write the config that was used to generate the refcat."""
        filename = os.path.join(self.output_dir, "config.py")
        with open(filename, 'w') as file:
            self.config.dataset_config.saveToStream(file)

    def _getOutputFilenames(self, htm):
        """Get filenames from the butler for each output htm pixel.

        Parameters
        ----------
        htm : `lsst.sphgeom.HtmPixelization`
            The HTM pixelization scheme to be used to build filenames.

        Returns
        -------
        filenames : `list [str]`
            List of filenames to write each HTM pixel to.
        """
        filenames = {}
        start, end = htm.universe()[0]
        path = os.path.join(self.output_dir, f"{self.indexer.htm}.fits")
        base = os.path.join(os.path.dirname(path), "%d"+os.path.splitext(path)[1])
        for pixelId in range(start, end):
            filenames[pixelId] = base % pixelId

        return filenames

    def makeSchema(self, dtype):
        """Make the schema to use in constructing the persisted catalogs.

        Parameters
        ----------
        dtype : `numpy.dtype`
            Data type describing each entry in ``config.extra_col_names``
            for the catalogs being ingested.

        Returns
        -------
        schemaAndKeyMap : `tuple` of (`lsst.afw.table.Schema`, `dict`)
            A tuple containing two items:
            - The schema for the output source catalog.
            - A map of catalog keys to use in filling the record
        """
        # make a schema with the standard fields
        schema = ReferenceObjectLoader.makeMinimalSchema(
            filterNameList=self.config.mag_column_list,
            addCentroid=False,
            addIsPhotometric=bool(self.config.is_photometric_name),
            addIsResolved=bool(self.config.is_resolved_name),
            addIsVariable=bool(self.config.is_variable_name),
            coordErrDim=2 if bool(self.config.ra_err_name) else 0,
            addProperMotion=2 if bool(self.config.pm_ra_name) else 0,
            properMotionErrDim=2 if bool(self.config.pm_ra_err_name) else 0,
            addParallax=bool(self.config.parallax_name),
        )
        keysToSkip = set(("id", "centroid_x", "centroid_y", "hasCentroid"))
        key_map = {fieldName: schema[fieldName].asKey() for fieldName in schema.getOrderedNames()
                   if fieldName not in keysToSkip}

        def addField(name):
            if dtype[name].kind == 'U':
                # dealing with a string like thing.  Need to get type and size.
                at_size = dtype[name].itemsize
                return schema.addField(name, type=str, size=at_size)
            else:
                at_type = dtype[name].type
                return schema.addField(name, at_type)

        for col in self.config.extra_col_names:
            key_map[col] = addField(col)
        return schema, key_map

    def _writeMasterSchema(self, inputfile):
        """Generate and save the master catalog schema.

        Parameters
        ----------
        inputfile : `str`
            An input file to read to get the input dtype.
        """
        arr = self.file_reader.run(inputfile)
        schema, key_map = self.makeSchema(arr.dtype)

        catalog = lsst.afw.table.SimpleCatalog(schema)
        addRefCatMetadata(catalog)
        outputfile = os.path.join(self.output_dir, "master_schema.fits")
        catalog.writeFits(outputfile)
        return schema, key_map

    def _reduce_kwargs(self):
        # Need to be able to pickle this class to use the multiprocess manager.
        kwargs = super()._reduce_kwargs()
        kwargs['output_dir'] = self.base_dir
        return kwargs


def build_argparser():
    """Construct an argument parser for the ``convertReferenceCatalog`` script.

    Returns
    -------
    argparser : `argparse.ArgumentParser`
        The argument parser that defines the ``convertReferenceCatalog``
        command-line interface.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='More information is available at https://pipelines.lsst.io.'
    )
    parser.add_argument("outputDir",
                        help="Path to write the output shard files, configs, and `ingest-files` table to.")
    parser.add_argument("configFile",
                        help="File containing the ConvertReferenceCatalogConfig fields.")
    # Use a "+"-list here, so we can produce a more useful error if the user
    # uses an unquoted glob that gets shell expanded.
    parser.add_argument("fileglob", nargs="+",
                        help="Quoted glob for the files to be read in and converted."
                             " Example (note required quotes to prevent shell expansion):"
                             ' "gaia_source/csv/GaiaSource*"')
    return parser


def run_convert(outputDir, configFile, fileglob):
    """Run `ConvertReferenceCatalogTask` on the input arguments.

    Parameters
    ----------
    outputDir : `str`
        Path to write the output files to.
    configFile : `str`
        File specifying the ``ConvertReferenceCatalogConfig`` fields.
    fileglob : `str`
        Quoted glob for the files to be read in and converted.
    """
    # We have to initialize the logger manually when running from the commandline.
    logging.basicConfig(level=logging.INFO, format="{name} {levelname}: {message}", style="{")

    config = ConvertReferenceCatalogTask.ConfigClass()
    config.load(configFile)
    config.validate()
    converter = ConvertReferenceCatalogTask(output_dir=outputDir, config=config)
    files = glob.glob(fileglob)
    converter.run(files)
    with open(os.path.join(outputDir, "convertReferenceCatalogConfig.py"), "w") as outfile:
        converter.config.saveToStream(outfile)
    msg = ("Completed refcat conversion.\n\n"
           "Ingest the resulting files with the following commands, substituting the path\n"
           "to your butler repo for `REPO`, and the ticket number you are tracking this\n"
           "ingest on for `DM-NNNNN`:\n"
           f"\n    butler register-dataset-type REPO {config.dataset_config.ref_dataset_name} "
           "SimpleCatalog htm7"
           "\n    butler ingest-files -t direct REPO gaia_dr2 refcats/DM-NNNNN "
           f"{converter.ingest_table_file}"
           "\n    butler collection-chain REPO --mode extend refcats refcats/DM-NNNNN")
    print(msg)


def main():
    args = build_argparser().parse_args()
    if len(args.fileglob) > 1:
        raise RuntimeError("Final argument must be a quoted file glob, not a shell-expanded list of files.")
    # Fileglob comes out as a length=1 list, so we can test it above.
    run_convert(args.outputDir, args.configFile, args.fileglob[0])
