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

__all__ = ["IngestIndexedReferenceConfig", "IngestIndexedReferenceTask", "DatasetConfig"]

import math

import astropy.time
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.geom
import lsst.afw.table as afwTable
from lsst.afw.image import fluxFromABMag, fluxErrFromABMagErr
from .indexerRegistry import IndexerRegistry
from .readTextCatalogTask import ReadTextCatalogTask
from .loadReferenceObjects import LoadReferenceObjectsTask

_RAD_PER_DEG = math.pi / 180
_RAD_PER_MILLIARCSEC = _RAD_PER_DEG/(3600*1000)


class IngestReferenceRunner(pipeBase.TaskRunner):
    """!Task runner for the reference catalog ingester

    Data IDs are ignored so the runner should just run the task on the parsed command.
    """

    def run(self, parsedCmd):
        """!Run the task.
        Several arguments need to be collected to send on to the task methods.

        @param[in] parsedCmd  Parsed command including command line arguments.
        @returns Struct containing the result of the indexing.
        """
        files = parsedCmd.files
        butler = parsedCmd.butler
        task = self.TaskClass(config=self.config, log=self.log, butler=butler)
        task.writeConfig(parsedCmd.butler, clobber=self.clobberConfig, doBackup=self.doBackup)

        result = task.create_indexed_catalog(files)
        if self.doReturnResults:
            return pipeBase.Struct(
                result=result,
            )


class DatasetConfig(pexConfig.Config):
    ref_dataset_name = pexConfig.Field(
        dtype=str,
        default='cal_ref_cat',
        doc='String to pass to the butler to retrieve persisted files.',
    )
    indexer = IndexerRegistry.makeField(
        default='HTM',
        doc='Name of indexer algoritm to use.  Default is HTM',
    )


class IngestIndexedReferenceConfig(pexConfig.Config):
    dataset_config = pexConfig.ConfigField(
        dtype=DatasetConfig,
        doc="Configuration for reading the ingested data",
    )
    file_reader = pexConfig.ConfigurableField(
        target=ReadTextCatalogTask,
        doc='Task to use to read the files.  Default is to expect text files.'
    )
    ra_name = pexConfig.Field(
        dtype=str,
        doc="Name of RA column",
    )
    dec_name = pexConfig.Field(
        dtype=str,
        doc="Name of Dec column",
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

    def validate(self):
        pexConfig.Config.validate(self)

        def assertAllOrNone(*names):
            """Raise ValueError unless all the named fields are set or are
            all none (or blank)
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
        assertAllOrNone("ra_err_name", "dec_err_name")
        assertAllOrNone("epoch_name", "epoch_format", "epoch_scale")
        assertAllOrNone("pm_ra_name", "pm_dec_name")
        assertAllOrNone("pm_ra_err_name", "pm_dec_err_name")
        if self.pm_ra_err_name and not self.pm_ra_name:
            raise ValueError('"pm_ra/dec_name" must be specified if "pm_ra/dec_err_name" are specified')
        if (self.pm_ra_name or self.parallax_name) and not self.epoch_name:
            raise ValueError(
                '"epoch_name" must be specified if "pm_ra/dec_name" or "parallax_name" are specified')


class IngestIndexedReferenceTask(pipeBase.CmdLineTask):
    """!Class for both producing indexed reference catalogs and for loading them.

    This implements an indexing scheme based on hierarchical triangular mesh (HTM).
    The term index really means breaking the catalog into localized chunks called
    shards.  In this case each shard contains the entries from the catalog in a single
    HTM trixel

    For producing catalogs this task makes the following assumptions
    about the input catalogs:
    - RA, Dec, RA error and Dec error are all in decimal degrees.
    - Epoch is available in a column, in a format supported by astropy.time.Time.
    - There are no off-diagonal covariance terms, such as covariance
        between RA and Dec, or between PM RA and PM Dec. Gaia is a well
        known example of a catalog that has such terms, and thus should not
        be ingested with this task.
    """
    canMultiprocess = False
    ConfigClass = IngestIndexedReferenceConfig
    RunnerClass = IngestReferenceRunner
    _DefaultName = 'IngestIndexedReferenceTask'

    _flags = ['photometric', 'resolved', 'variable']

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser

        This overrides the original because we need the file arguments
        """
        parser = pipeBase.InputOnlyArgumentParser(name=cls._DefaultName)
        parser.add_argument("files", nargs="+", help="Names of files to index")
        return parser

    def __init__(self, *args, **kwargs):
        """!Constructor for the HTM indexing engine

        @param[in] butler  dafPersistence.Butler object for reading and writing catalogs
        """
        self.butler = kwargs.pop('butler')
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.indexer = IndexerRegistry[self.config.dataset_config.indexer.name](
            self.config.dataset_config.indexer.active)
        self.makeSubtask('file_reader')

    def create_indexed_catalog(self, files):
        """!Index a set of files comprising a reference catalog.  Outputs are persisted in the
        data repository.

        @param[in] files  A list of file names to read.
        """
        rec_num = 0
        first = True
        for filename in files:
            arr = self.file_reader.run(filename)
            index_list = self.indexer.index_points(arr[self.config.ra_name], arr[self.config.dec_name])
            if first:
                schema, key_map = self.make_schema(arr.dtype)
                # persist empty catalog to hold the master schema
                dataId = self.indexer.make_data_id('master_schema',
                                                   self.config.dataset_config.ref_dataset_name)
                self.butler.put(self.get_catalog(dataId, schema), 'ref_cat',
                                dataId=dataId)
                first = False
            pixel_ids = set(index_list)
            for pixel_id in pixel_ids:
                dataId = self.indexer.make_data_id(pixel_id, self.config.dataset_config.ref_dataset_name)
                catalog = self.get_catalog(dataId, schema)
                els = np.where(index_list == pixel_id)
                for row in arr[els]:
                    record = catalog.addNew()
                    rec_num = self._fill_record(record, row, rec_num, key_map)
                self.butler.put(catalog, 'ref_cat', dataId=dataId)
        dataId = self.indexer.make_data_id(None, self.config.dataset_config.ref_dataset_name)
        self.butler.put(self.config.dataset_config, 'ref_cat_config', dataId=dataId)

    @staticmethod
    def compute_coord(row, ra_name, dec_name):
        """!Create an ICRS SpherePoint from a np.array row
        @param[in] row  dict like object with ra/dec info in degrees
        @param[in] ra_name  name of RA key
        @param[in] dec_name  name of Dec key
        @returns ICRS SpherePoint constructed from the RA/Dec values
        """
        return lsst.geom.SpherePoint(row[ra_name], row[dec_name], lsst.geom.degrees)

    def _set_coord_err(self, record, row, key_map):
        """Set coordinate error from the input

        The errors are read from the specified columns, and installed
        in the appropriate columns of the output.

        Parameters
        ----------
        record : `lsst.afw.table.SimpleRecord`
            Record to modify.
        row : `dict`-like
            Row from numpy table.
        key_map : `dict` mapping `str` to `lsst.afw.table.Key`
            Map of catalog keys.
        """
        if self.config.ra_err_name:  # IngestIndexedReferenceConfig.validate ensures all or none
            record.set(key_map["coord_raErr"], row[self.config.ra_err_name]*_RAD_PER_DEG)
            record.set(key_map["coord_decErr"], row[self.config.dec_err_name]*_RAD_PER_DEG)

    def _set_flags(self, record, row, key_map):
        """!Set the flags for a record.  Relies on the _flags class attribute
        @param[in,out] record  SimpleCatalog record to modify
        @param[in] row  dict like object containing flag info
        @param[in] key_map  Map of catalog keys to use in filling the record
        """
        names = record.schema.getNames()
        for flag in self._flags:
            if flag in names:
                attr_name = 'is_{}_name'.format(flag)
                record.set(key_map[flag], bool(row[getattr(self.config, attr_name)]))

    def _set_mags(self, record, row, key_map):
        """!Set the flux records from the input magnitudes
        @param[in,out] record  SimpleCatalog record to modify
        @param[in] row  dict like object containing magnitude values
        @param[in] key_map  Map of catalog keys to use in filling the record
        """
        for item in self.config.mag_column_list:
            record.set(key_map[item+'_flux'], fluxFromABMag(row[item]))
        if len(self.config.mag_err_column_map) > 0:
            for err_key in self.config.mag_err_column_map.keys():
                error_col_name = self.config.mag_err_column_map[err_key]
                record.set(key_map[err_key+'_fluxErr'],
                           fluxErrFromABMagErr(row[error_col_name], row[err_key]))

    def _set_proper_motion(self, record, row, key_map):
        """Set the proper motions from the input

        The proper motions are read from the specified columns,
        scaled appropriately, and installed in the appropriate
        columns of the output.

        Parameters
        ----------
        record : `lsst.afw.table.SimpleRecord`
            Record to modify.
        row : `dict`-like
            Row from numpy table.
        key_map : `dict` mapping `str` to `lsst.afw.table.Key`
            Map of catalog keys.
        """
        if self.config.pm_ra_name is None:  # IngestIndexedReferenceConfig.validate ensures all or none
            return
        radPerOriginal = _RAD_PER_MILLIARCSEC*self.config.pm_scale
        record.set(key_map["pm_ra"], row[self.config.pm_ra_name]*radPerOriginal*lsst.geom.radians)
        record.set(key_map["pm_dec"], row[self.config.pm_dec_name]*radPerOriginal*lsst.geom.radians)
        record.set(key_map["epoch"], self._epoch_to_mjd_tai(row[self.config.epoch_name]))
        if self.config.pm_ra_err_name is not None:  # pm_dec_err_name also, by validation
            record.set(key_map["pm_raErr"], row[self.config.pm_ra_err_name]*radPerOriginal)
            record.set(key_map["pm_decErr"], row[self.config.pm_dec_err_name]*radPerOriginal)

    def _epoch_to_mjd_tai(self, nativeEpoch):
        """Convert an epoch in native format to TAI MJD (a float)
        """
        return astropy.time.Time(nativeEpoch, format=self.config.epoch_format,
                                 scale=self.config.epoch_scale).tai.mjd

    def _set_extra(self, record, row, key_map):
        """!Copy the extra column information to the record
        @param[in,out] record  SimpleCatalog record to modify
        @param[in] row  dict like object containing the column values
        @param[in] key_map  Map of catalog keys to use in filling the record
        """
        for extra_col in self.config.extra_col_names:
            value = row[extra_col]
            # If data read from a text file contains string like entires,
            # numpy stores this as its own internal type, a numpy.str_
            # object. This seems to be a consequence of how numpy stores
            # string like objects in fixed column arrays. This checks
            # if any of the values to be added to the catalog are numpy
            # string types, and if they are, casts them to a python string
            # which is what the python c++ records expect
            if isinstance(value, np.str_):
                value = str(value)
            record.set(key_map[extra_col], value)

    def _fill_record(self, record, row, rec_num, key_map):
        """!Fill a record to put in the persisted indexed catalogs

        @param[in,out] record  afwTable.SimpleRecord in a reference catalog to fill.
        @param[in] row  A row from a numpy array constructed from the input catalogs.
        @param[in] rec_num  Starting integer to increment for the unique id
        @param[in] key_map  Map of catalog keys to use in filling the record
        """
        record.setCoord(self.compute_coord(row, self.config.ra_name, self.config.dec_name))
        if self.config.id_name:
            record.setId(row[self.config.id_name])
        else:
            rec_num += 1
            record.setId(rec_num)

        self._set_coord_err(record, row, key_map)
        self._set_flags(record, row, key_map)
        self._set_mags(record, row, key_map)
        self._set_proper_motion(record, row, key_map)
        self._set_extra(record, row, key_map)
        return rec_num

    def get_catalog(self, dataId, schema):
        """!Get a catalog from the butler or create it if it doesn't exist

        @param[in] dataId  Identifier for catalog to retrieve
        @param[in] schema  Schema to use in catalog creation if the butler can't get it
        @returns table (an lsst.afw.table.SimpleCatalog) for the specified identifier
        """
        if self.butler.datasetExists('ref_cat', dataId=dataId):
            return self.butler.get('ref_cat', dataId=dataId)
        return afwTable.SimpleCatalog(schema)

    def make_schema(self, dtype):
        """!Make the schema to use in constructing the persisted catalogs.

        @param[in] dtype  A np.dtype describing the type of each entry in
            config.extra_col_names.

        @returns a pair of items:
        - The schema for the output source catalog.
        - A map of catalog keys to use in filling the record
        """
        self.config.validate()  # just to be sure

        # make a schema with the standard fields
        schema = LoadReferenceObjectsTask.makeMinimalSchema(
            filterNameList=self.config.mag_column_list,
            addFluxErr=bool(self.config.mag_err_column_map),
            addCentroid=False,
            addIsPhotometric=bool(self.config.is_photometric_name),
            addIsResolved=bool(self.config.is_resolved_name),
            addIsVariable=bool(self.config.is_variable_name),
            coordErrDim=2 if bool(self.config.ra_err_name) else 0,
            addProperMotion=2 if bool(self.config.pm_ra_name) else 0,
            properMotionErrDim=2 if bool(self.config.pm_ra_err_name) else 0,
            addParallax=bool(self.config.parallax_name),
            addParallaxErr=bool(self.config.parallax_err_name),
        )
        keysToSkip = set(("id", "centroid_x", "centroid_y", "hasCentroid"))
        key_map = {fieldName: schema[fieldName].asKey() for fieldName in schema.getOrderedNames()
                   if fieldName not in keysToSkip}

        def add_field(name):
            if dtype[name].kind == 'U':
                # dealing with a string like thing.  Need to get type and size.
                at_size = dtype[name].itemsize
                return schema.addField(name, type=str, size=at_size)
            else:
                at_type = dtype[name].type
                return schema.addField(name, at_type)

        for col in self.config.extra_col_names:
            key_map[col] = add_field(col)
        return schema, key_map
