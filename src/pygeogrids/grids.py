# Copyright (c) 2018, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN, DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
The grids module defines the grid classes.
"""

import numpy as np
import numpy.testing as nptest

try:
    from osgeo import ogr

    ogr_installed = True
except ImportError:
    ogr_installed = False

try:
    from itertools import izip as zip
except ImportError:
    pass  # python3

import pygeogrids.nearest_neighbor as NN
from pygeogrids.geodetic_datum import GeodeticDatum
from pygeogrids.subset import SubsetCollection, Subset
from netCDF4 import Dataset
from pygeogrids.netcdf import load_grid_definition, filled_no_mask, save_lonlat
import warnings


class GridDefinitionError(Exception):
    pass


class GridIterationError(Exception):
    pass


class BasicGrid(object):
    """
    Grid that just has lat,lon coordinates and can find the
    nearest neighbour. It can also yield the gpi, lat, lon
    information in order.

    Parameters
    ----------
    lon : numpy.array
        longitudes of the points in the grid
    lat : numpy.array
        latitudes of the points in the grid
    geodatum : basestring
        Name of the geodatic datum associated with the grid
    gpis : numpy.array, optional
        if the gpi numbers are in a different order than the
        lon and lat arrays an array containing the gpi numbers
        can be given
        if no array is given here the lon lat arrays are given
        gpi numbers starting at 0
    subset : numpy.array, optional
        if the active_subset part of the array is only a subset of
        all the points then the subset array which is a index
        into lon and lat can be given here.
    setup_kdTree : boolean, optional
        if set (default) then the kdTree for nearest neighbour
        search will be built on initialization
    shape : tuple, optional
        The shape of the grid array in 2-d space [lat, lon].
        e.g. for a 1x1 degree global regular grid the shape would be (180,360).
        if given the grid can be reshaped into the given shape
        this indicates that it is a regular grid and fills the
        attributes self.lon2d and self.lat2d which
        define the grid only be the meridian coordinates(self.lon2d) and
        the coordinates of the circles of latitude(self.lat2d).
        The shape has to be given as (lat2d, lon2d)
        It it is not given the shape is set to the length of the input
        lon and lat arrays.

    Attributes
    ----------
    arrlon : numpy.array
        1D array of all longitudes of the grid
    arrlat : numpy.array
        1D array of all latitudes of the grid
    n_gpi : int
        number of gpis in the grid
    gpidirect : boolean
        if true the gpi number is equal to the index
        of arrlon and arrlat
    gpis : numpy.array
        gpi number for elements in arrlon and arrlat
        gpi[i] is located at arrlon[i],arrlat[i]
    subset : numpy.array
        if given then this contains the indices of a subset of
        the grid. This can be used if only a part of a grid is
        interesting for a application. e.g. land points, or only
        a specific country
    allpoints : boolean
        if False only a subset of the grid is active_subset
    activearrlon : numpy.array
        array of longitudes that are active_subset, is defined by
        arrlon[subset] if a subset is given otherwise equal to
        arrlon
    activearrlat : numpy.array
        array of latitudes that are active_subset, is defined by
        arrlat[subset] if a subset is given otherwise equal to
        arrlat
    activegpis : numpy.array
        array of gpis that are active_subset, is defined by
        gpis[subset] if a subset is given otherwise equal to
        gpis
    geodatum : object
        pygeogrids.geodatic_datum object (reference ellipsoid) associated
        with the grid
    issplit : boolean
        if True then the array was split in n parts with
        the self.split function
    kdTree : object
        grid.nearest_neighbor.findGeoNN object for
        nearest neighbor search
    shape : tuple, optional
        if given during initialization then this is
        the shape the grid can be reshaped to
    lat2d : numpy.array, optional
        if shape is given this attribute contains all latitudes according to
        the provided 2d-shape that make up the grid
    lon2d : numpy.array, optional
        if shape is given this attribute contains all longitudes according to
        the provided 2d-shape that make up the grid
    """

    def __init__(self, lon, lat, gpis=None, geodatum='WGS84', subset=None,
                 setup_kdTree=True, shape=None):
        """
        init method, prepares lon and lat arrays for _transform_lonlats if
        necessary

        """
        lon = np.asanyarray(lon)
        lat = np.asanyarray(lat)
        if gpis is not None:
            gpis = np.asanyarray(gpis)
        if subset is not None:
            subset = np.asanyarray(subset)

        if lat.shape != lon.shape:
            raise GridDefinitionError(
                "lat and lon np.arrays have to have equal shapes")

        self.n_gpi = len(lon)

        self.arrlon = lon
        self.arrlat = lat

        self.shape = None

        if shape is not None and len(shape) == 2:
            if len(self.arrlat) % shape[0] != 0:
                msg = ("Given shape does not have the correct first dimension."
                       " Length of lat array is not divisible by shape[0] "
                       " without rest")
                raise GridDefinitionError(msg)

            if len(self.arrlon) % shape[1] != 0:
                msg = ("Given shape does not have the correct second "
                       "dimension. Length of lon array is not divisible by "
                       "shape[1] without rest")
                raise GridDefinitionError(msg)

            self.shape = shape
            self.lat2d = np.reshape(self.arrlat, self.shape)
            self.lon2d = np.reshape(self.arrlon, self.shape)

        else:
            self.shape = tuple([len(self.arrlon)])

        self.geodatum = GeodeticDatum(geodatum)

        if gpis is None:
            self.gpis = np.arange(self.n_gpi, dtype=int)
            self.gpidirect = True
        else:
            if lat.shape != gpis.shape:
                raise GridDefinitionError("lat, lon gpi np.arrays have to "
                                          "have equal shapes")
            self.gpis = gpis
            self.gpidirect = False

        self.subset = subset

        if subset is not None:
            self._create_subset(subset)
        else:
            self._empty_subset()

        self.issplit = False

        self.kdTree = None
        if setup_kdTree:
            self._setup_kdtree()

    def _setup_kdtree(self):
        """
        Setup kdTree
        """
        if self.kdTree is None:
            self.kdTree = NN.findGeoNN(self.activearrlon, self.activearrlat,
                                       self.geodatum)
            self.kdTree._build_kdtree()

    def _create_subset(self, subset):
        # Create
        self.activearrlon = self.arrlon[subset]
        self.activearrlat = self.arrlat[subset]
        self.activegpis = self.gpis[subset]
        self.allpoints = False

    def _empty_subset(self):
        self.activearrlon = self.arrlon
        self.activearrlat = self.arrlat
        self.activegpis = self.gpis
        self.allpoints = True

    @classmethod
    def from_file(cls, filename, subset_flag='subset_flag', subset_value=1,
                  location_var_name='gpi'):
        """
        load a grid from netCDF file, only for basic or cell grids

        Parameters
        ----------
        filename : str
            Path to netcdf file to load.
        subset_flag : str, optional (default: 'subset_flag')
            name of the subset to load.
        subset_value : int or list, optional (default: 1)
            Value(s) of the subset variable that points are loaded for.
        location_var_name: str, optional (default: 'gpi')
            variable name under which the grid point locations
            are stored

        Returns
        -------
        grid : BasicGrid
            grid instance initialized with the loaded data
        """
        subsets = {subset_flag: subset_value}

        lons, lats, gpis, _, subsets, geodatumName, shape = \
            load_grid_definition(filename, location_var_name, subsets)

        return cls(lons,
                   lats,
                   gpis=gpis,
                   geodatum=geodatumName,
                   subset=subsets[subset_flag]['gpis'],
                   shape=shape)

    def split(self, n):
        """
        Function splits the grid into n parts this changes not function but
        grid_points() which takes the argument n and will only iterate through
        this part of the grid.

        Parameters
        ----------
        n : int
            Number of parts the grid should be split into
        """
        self.issplit = True
        self.subarrlats = np.array_split(self.activearrlat, n)
        self.subarrlons = np.array_split(self.activearrlon, n)
        self.subgpis = np.array_split(self.activegpis, n)

    def unite(self):
        """
        Unites a split array, so that it can be iterated over as a whole
        again.
        """
        self.issplit = False

    def grid_points(self, *args):
        """
        Yields all grid points in order

        Parameters
        ----------
        n : int, optional
            if the grid is split in n parts using the split function
            then this iterator will only iterate of the nth part of the
            grid

        Returns
        -------
        gpi : long
            grid point index
        lon : float
            longitude of gpi
        lat : float
            longitude of gpi
        """

        if not self.issplit and len(args) == 0:
            return self._normal_grid_points()
        elif self.issplit and len(args) == 1:
            return self._split_grid_points(args[0])

        raise GridIterationError("this function only takes an argument if "
                                 "the grid is split, and takes no argument "
                                 "if the grid is not split")

    def get_grid_points(self, *args):
        """
        Returns all active_subset grid points.

        Parameters
        ----------
        n : int, optional
            if the grid is split in n parts using the split function
            then this function will only return the nth part of the
            grid

        Returns
        -------
        gpis : numpy.ndarray
            Grid point indices.
        arrlon : numpy.ndarray
            Longitudes.
        arrlat : numpy.ndarray
            Latitudes.
        """

        if not self.issplit and len(args) == 0:
            return (self.activegpis,
                    self.activearrlon,
                    self.activearrlat)

        elif self.issplit and len(args) == 1:
            n = args[0]
            return (self.subgpis[n],
                    self.subarrlons[n],
                    self.subarrlats[n])

    def _normal_grid_points(self):
        """
        Yields all grid points in order.

        Returns
        -------
        gpi : long
            grid point index
        lon : float
            longitude of gpi
        lat : float
            longitude of gpi
        """
        for i, (lon, lat) in enumerate(zip(self.activearrlon,
                                           self.activearrlat)):
            yield self.activegpis[i], lon, lat

    def _split_grid_points(self, n):
        """
        Yields all grid points or split grid in order.

        Parameters
        ----------
        n : int
            Number of subgrid to yield

        Returns
        -------
        gpi : long
            Grid point index.
        lon : float
            Longitude of gpi.
        lat : float
            Longitude of gpi.
        """

        for i, (lon, lat) in enumerate(zip(self.subarrlons[n],
                                           self.subarrlats[n])):
            yield self.subgpis[n][i], lon, lat

    def find_nearest_gpi(self, lon, lat, max_dist=np.Inf):
        """
        Finds nearest gpi, builds kdTree if it does not yet exist.

        Parameters
        ----------
        lon : float or iterable
            Longitude of point.
        lat : float or iterable
            Latitude of point.
        max_dist : float, optional
            Maximum distance to consider for search (default: np.Inf).

        Returns
        -------
        gpi : long
            Grid point index.
        distance : float
            Distance of gpi to given lon, lat.
            At the moment not on a great circle but in spherical
            cartesian coordinates.
        """
        gpi, distance = self.find_k_nearest_gpi(lon, lat, max_dist=max_dist, k=1)

        if not _element_iterable(lon):
            gpi = gpi[0]
            distance = distance[0]

        return gpi, distance

    def find_k_nearest_gpi(self, lon, lat, max_dist=np.Inf, k=1):
        """
        Find k nearest gpi, builds kdTree if it does not yet exist.

        Parameters
        ----------
        lon : float or iterable
            Longitude of point.
        lat : float or iterable
            Latitude of point.
        max_dist : float, optional
            Maximum distance to consider for search (default: np.Inf).
        k : int, optional
            The number of nearest neighbors to return (default: 1).

        Returns
        -------
        gpi : long
            Grid point index.
        distance : float
            Distance of gpi to given lon, lat.
            At the moment not on a great circle but in spherical
            cartesian coordinates.
        """
        if self.kdTree is None:
            self._setup_kdtree()

        distance, ind = self.kdTree.find_nearest_index(
            lon, lat, max_dist=max_dist, k=k)

        if self.gpidirect and self.allpoints:
            gpi = ind
        else:
            gpi = self.activegpis[ind]

        return gpi, distance

    def gpi2lonlat(self, gpi):
        """
        Longitude and latitude for given gpi.

        Parameters
        ----------
        gpi : int32 or iterable
            Grid point index.

        Returns
        -------
        lon : float
            Longitude of gpi.
        lat : float
            Latitude of gpi
        """
        # check if iterable
        iterable = _element_iterable(gpi)

        gpi = np.atleast_1d(gpi)
        if self.gpidirect:
            lons, lats = self.arrlon[gpi], self.arrlat[gpi]
        else:
            # get the indices that would sort the gpis
            gpisorted = np.argsort(self.gpis)
            # find the position where the gpis fit in the sorted array
            pos = np.searchsorted(self.gpis[gpisorted], gpi)
            index = gpisorted[pos]
            lons, lats = self.arrlon[index], self.arrlat[index]

        if not iterable:
            lons = lons[0]
            lats = lats[0]

        return lons, lats

    def gpi2rowcol(self, gpi):
        """
        If the grid can be reshaped into a sensible 2D shape then this
        function gives the row(latitude dimension) and
        column(longitude dimension) indices of the gpi in the 2D grid.

        Parameters
        ----------
        gpi : int32
            Grid point index.

        Returns
        -------
        row : int
            Row in 2D array.
        col : int
            Column in 2D array.
        """
        # check if iterable
        iterable = _element_iterable(gpi)

        gpi = np.atleast_1d(gpi)
        if len(self.shape) == 2:
            if self.gpidirect:
                index = gpi
            else:
                # get the indices that would sort the gpis
                gpisorted = np.argsort(self.gpis)
                # find the position where the gpis fit in the sorted array
                pos = np.searchsorted(self.gpis[gpisorted], gpi)
                index = gpisorted[pos]

            index_lat = (index / self.shape[1]).astype(np.int)
            index_lon = index % self.shape[1]
            if not iterable:
                index_lat = index_lat[0]
                index_lon = index_lon[0]
            return index_lat, index_lon

        else:
            raise (GridDefinitionError("Grid has no 2D shape"))

    def calc_lut(self, other, max_dist=np.Inf, into_subset=False):
        """
        Takes other BasicGrid or CellGrid objects and computes a lookup table
        between them. The lut will have the size of self.n_gpis and will
        for every grid point have the nearest index into other.arrlon etc.

        Parameters
        ----------
        other : grid object
            to which to calculate the lut to
        max_dist : float, optional
            maximum allowed distance in meters
        into_subset : boolean, optional
            if set the returned lut will have the index into the subset
            if the other grid is a subset of a grid.
            Example:
            if e.g. ind_l is used for the warp_grid some datasets will
            be given as arrays with len(ind_l) elements. These
            datasets can not be indexed with gpi numbers but have to
            be indexed with indices into the subset
        """

        if self.kdTree is None:
            self._setup_kdtree()

        if other.kdTree is None:
            other._setup_kdtree()

        if self.kdTree.kdtree is not None and other.kdTree.kdtree is not None:
            dist, index = other.kdTree.find_nearest_index(
                self.activearrlon, self.activearrlat, max_dist=max_dist)

            valid_index = np.where(dist != np.inf)[0]
            dist = dist[valid_index]
            index = index[valid_index]
            if not other.gpidirect or not other.allpoints:
                if not into_subset:
                    index = other.activegpis[index]

            active_lut = np.empty_like(self.activearrlat, dtype=np.int64)
            active_lut.fill(-1)
            active_lut[valid_index] = index

            if not self.allpoints:
                gpi_lut = np.empty_like(self.gpis)
                gpi_lut.fill(-1)
                gpi_lut[self.gpis[self.subset]] = active_lut
            elif not self.gpidirect:
                gpi_lut = np.empty(np.max(self.activegpis) + 1, dtype=np.int64)
                gpi_lut.fill(-1)
                gpi_lut[self.activegpis] = active_lut
            else:
                gpi_lut = active_lut

            return gpi_lut

    def get_shp_grid_points(self, ply):
        """
        Returns all grid points located in a submitted shapefile,
        optinal as coordinates. Currently only works in WGS84.

        Parameters
        ----------
        ply: object, OGRGeometryShadow
            the Geometry of the Feature as returned from ogr.GetGeometryRef

        Returns
        -------
        grid : BasicGrid
            Subgrid.
        """

        if ogr_installed:
            lonmin, lonmax, latmin, latmax = ply.GetEnvelope()
            gpis, lats, lons = self.get_bbox_grid_points(latmin, latmax,
                                                         lonmin, lonmax,
                                                         both=True)

            lon_ip = []
            lat_ip = []
            gpi_ip = []

            if len(gpis) > 0:
                for gpi, lon, lat in zip(gpis, lons, lats):
                    pt = ogr.Geometry(ogr.wkbPoint)
                    pt.SetPoint_2D(0, float(lon), float(lat))
                    if ply.Contains(pt):
                        lon_ip.append(lon)
                        lat_ip.append(lat)
                        gpi_ip.append(gpi)
            if len(gpi_ip) > 0:
                return self.subgrid_from_gpis(gpi_ip)
            else:
                return None

        else:
            raise Exception("No supported implementation installed.\
                            Please install gdal and osgeo.")

    def get_bbox_grid_points(self, latmin=-90, latmax=90, lonmin=-180,
                             lonmax=180, coords=False, both=False):
        """
        Returns all grid points located in a submitted geographic box,
        optinal as coordinates

        Parameters
        ----------
        latmin : float, optional
            minimum latitude
        latmax : float, optional
            maximum latitude
        lonmin : float, optional
            minimum latitude
        lonmax : float, optional
            maximum latitude
        coords : boolean, optional
            set to True if coordinates should be returned
        both: boolean, optional
            set to True if gpis and coordinates should be returned

        Returns
        -------
        gpi : numpy.ndarray
            grid point indices, if coords=False
        lat : numpy.ndarray
            longitudes of gpis, if coords=True
        lon : numpy.ndarray
            longitudes of gpis, if coords=True
        """

        gp_info = np.array(list(self.grid_points()))
        gpis = gp_info[:, 0].astype(int)
        lons = gp_info[:, 1]
        lats = gp_info[:, 2]
        index = np.where((lats <= latmax) &
                         (lats >= latmin) &
                         (lons <= lonmax) &
                         (lons >= lonmin))

        if coords is True:
            return lats[index], lons[index]
        elif both:
            return gpis[index], lats[index], lons[index]
        else:
            return gpis[index]

    def to_cell_grid(self, cellsize=5.0, cellsize_lat=None, cellsize_lon=None):
        """
        Convert grid to cellgrid with a cell partition of cellsize.

        Parameters
        ----------
        cellsize : float, optional
            Cell size in degrees
        cellsize_lon : float, optional
            Cell size in degrees on the longitude axis
        cellsize_lat : float, optional
            Cell size in degrees on the latitude axis

        Returns
        -------
        cell_grid : CellGrid object
            Cell grid object.
        """
        cells = lonlat2cell(self.arrlon, self.arrlat, cellsize=cellsize,
                            cellsize_lat=cellsize_lat,
                            cellsize_lon=cellsize_lon)

        if self.gpidirect:
            gpis = None
        else:
            gpis = self.gpis

        return CellGrid(self.arrlon, self.arrlat, cells, gpis=gpis,
                        subset=self.subset, shape=self.shape)

    def subgrid_from_gpis(self, gpis):
        """
        Generate a subgrid for given gpis.

        Parameters
        ----------
        gpis : int, numpy.ndarray
            Grid point indices.

        Returns
        -------
        grid : BasicGrid
            Subgrid.
        """
        sublons, sublats = self.gpi2lonlat(gpis)

        return BasicGrid(sublons, sublats, gpis, geodatum=self.geodatum.name)

    def __eq__(self, other):
        """
        Compare arrlon, arrlat, gpis, subsets and shape.

        Returns
        -------
        result : boolean
            Returns True if grids are equal.
        """
        # only test to certain significance for float variables
        # grids are assumed to be the same if the gpi, lon, lat tuples are the
        # same
        idx_gpi = np.argsort(self.gpis)
        idx_gpi_other = np.argsort(other.gpis)
        gpisame = np.all(self.gpis[idx_gpi] == other.gpis[idx_gpi_other])
        try:
            nptest.assert_allclose(self.arrlon[idx_gpi],
                                   other.arrlon[idx_gpi_other])
            lonsame = True
        except AssertionError:
            lonsame = False
        try:
            nptest.assert_allclose(self.arrlat[idx_gpi],
                                   other.arrlat[idx_gpi_other])
            latsame = True
        except AssertionError:
            latsame = False
        if self.subset is not None and other.subset is not None:
            subsetsame = np.all(
                sorted(self.gpis[self.subset]) == sorted(other.gpis[other.subset]))
        elif self.subset is None and other.subset is None:
            subsetsame = True
        else:
            subsetsame = False

        if self.shape is None and other.shape is None:
            shapesame = True
        elif self.shape is not None and other.shape is not None:
            shapesame = self.shape == other.shape
        else:
            shapesame = False

        if self.geodatum.name == other.geodatum.name:
            geosame = True
        else:
            geosame = False

        return np.all([lonsame, latsame, gpisame, subsetsame, shapesame,
                       geosame])


class CellGrid(BasicGrid):
    """
    Grid that has lat,lon coordinates as well as cell informatin. It can find
    nearest neighbour. It can also yield the gpi, lat, lon, cell information
    in cell order. This is important if the data on the grid is saved in cell
    files on disk as we can go through all grid points with optimized
    IO performance.

    Parameters
    ----------
    lon : numpy.ndarray
        Longitudes of the points in the grid.
    lat : numpy.ndarray
        Latitudes of the points in the grid.
    cells : numpy.ndarray
        Of same shape as lon and lat, containing the cell number of each gpi.
    gpis : numpy.ndarray, optional
        If the gpi numbers are in a different order than the lon and lat
        arrays an array containing the gpi numbers can be given.
    subset : numpy.array, optional
        If the active_subset part of the array is only a subset of all the points
        then the subset array which is a index into lon, lat and cells can
        be given here.

    Attributes
    ----------
    arrcell : numpy.ndarray
        Array of cell number with same shape as arrlon, arrlat.
    activearrcell : numpy.ndarray
        Array of longitudes that are active_subset, is defined by arrlon[subset]
        if a subset is given otherwise equal to arrlon.
    """

    def __init__(self, lon, lat, cells, gpis=None, geodatum='WGS84',
                 subset=None, setup_kdTree=False, **kwargs):

        super(CellGrid, self).__init__(lon, lat, gpis=gpis,
                                       geodatum=geodatum, subset=subset,
                                       setup_kdTree=setup_kdTree, **kwargs)

        self.gpi_lut = None
        cells = np.asanyarray(cells)

        if self.arrlon.shape != cells.shape:
            raise GridDefinitionError(
                "lat, lon and cells np.arrays have to have equal shapes")
        self.arrcell = cells

        if subset is not None:
            self.activearrcell = self.arrcell[subset]
        else:
            self.activearrcell = self.arrcell

    @classmethod
    def from_file(cls, filename, subset_flag='subset_flag', subset_value=1,
                  location_var_name='gpi'):
        """
        load a cell grid from netCDF file

        Parameters
        ----------
        filename : string
            filename
        subset_flag : string, optional (default: 'subset_flag')
            name of the subset to load.
        subset_value : int or list, optional (default: 1)
            Value(s) of the subset variable that points are loaded for.
        location_var_name: string, optional (default: 'gpi')
            variable name under which the grid point locations
            are stored

        Returns
        -------
        grid : CellGrid
            grid instance initialized with the loaded data
        """
        subsets = {subset_flag: subset_value}

        lons, lats, gpis, arrcell, subsets, geodatumName, shape = \
            load_grid_definition(filename, location_var_name, subsets)

        return cls(lons,
                   lats,
                   arrcell,
                   gpis=gpis,
                   geodatum=geodatumName,
                   subset=subsets[subset_flag]['gpis'],
                   shape=shape)

    def gpi2cell(self, gpi):
        """
        Cell for given gpi.

        Parameters
        ----------
        gpi : int32 or iterable
            Grid point index.

        Returns
        -------
        cell : int or iterable
            Cell number of gpi.
        """
        # check if iterable
        iterable = _element_iterable(gpi)

        gpi = np.atleast_1d(gpi)
        if self.gpidirect:
            cell = self.arrcell[gpi]
        else:
            if self.gpi_lut is None:
                self.gpi_lut = np.zeros(self.gpis.max() + 1, dtype=np.int32)
                self.gpi_lut[self.gpis] = np.arange(self.gpis.size)

            cell = self.arrcell[self.gpi_lut[gpi]]

        if not iterable:
            cell = cell[0]

        return cell

    def get_cells(self):
        """
        Function to get all cell numbers of the grid.

        Returns
        -------
        cells : numpy.ndarray
            Unique cell numbers.
        """
        return np.unique(self.activearrcell)

    def get_grid_points(self, *args):
        """
        Returns all active_subset grid points.

        Parameters
        ----------
        n : int, optional
            If the grid is split in n parts using the split function then this
            function will only return the nth part of the grid.

        Returns
        -------
        gpis : numpy.ndarray
            Grid point indices.
        arrlon : numpy.ndarray
            Longitudes.
        arrlat :numpy.ndarray
            Latitudes.
        cells : numpy.ndarray
            Cell numbers.
        """
        if not self.issplit and len(args) == 0:
            return (self.activegpis,
                    self.activearrlon,
                    self.activearrlat,
                    self.activearrcell)

        elif self.issplit and len(args) == 1:
            n = args[0]
            return (self.subgpis[n],
                    self.subarrlons[n],
                    self.subarrlats[n],
                    self.subcells[n])

    def grid_points_for_cell(self, cells):
        """
        Get all grid points for a given cell number.

        Parameters
        ----------
        cell : int, numpy.ndarray
            Cell numbers.

        Returns
        -------
        gpis : numpy.ndarray
            Gpis belonging to cell.
        lons : numpy.array
            Longitudes belonging to the gpis.
        lats : numpy.array
            Latitudes belonging to the gpis.
        """
        cells = np.atleast_1d(cells)

        gpis = []
        lons = []
        lats = []
        for cell in cells:
            cell_index = np.where(cell == self.activearrcell)
            gpis.append(self.activegpis[cell_index])
            lons.append(self.activearrlon[cell_index])
            lats.append(self.activearrlat[cell_index])

        gpis = np.hstack(gpis)
        lons = np.hstack(lons)
        lats = np.hstack(lats)

        return gpis, lons, lats

    def split(self, n):
        """
        Function splits the grid into n parts this changes not function but
        grid_points() which takes the argument n and will only iterate through
        this part of the grid.

        Parameters
        ----------
        n : int
            Number of parts the grid should be split into.
        """
        self.issplit = True
        # sort by cell number to split correctly
        sorted_index = np.argsort(self.activearrcell)
        self.subarrlats = np.array_split(self.activearrlat[sorted_index], n)
        self.subarrlons = np.array_split(self.activearrlon[sorted_index], n)
        self.subgpis = np.array_split(self.activegpis[sorted_index], n)
        self.subcells = np.array_split(self.activearrcell[sorted_index], n)

    def _normal_grid_points(self):
        """
        Yields all grid points in cell order.

        Returns
        -------
        gpi : long
            Grid point index.
        lon : float
            Longitude of gpi.
        lat : float
            Longitude of gpi.
        cell : int
            Cell number.
        """
        uniq_cells = np.unique(self.activearrcell)

        for cell in uniq_cells:
            cell_gpis = np.where(cell == self.activearrcell)[0]
            for gpi in cell_gpis:
                yield self.activegpis[gpi], self.activearrlon[gpi], \
                      self.activearrlat[gpi], cell

    def _split_grid_points(self, n):
        """
        Yields all grid points in cell order.

        Parameters
        ----------
        n : int
            Number of subgrid to yield.

        Returns
        -------
        gpi : long
            Grid point index.
        lon : float
            Longitude of gpi.
        lat : float
            Latitude of gpi.
        cell : int
            Cell number.
        """
        uniq_cells = np.unique(self.subcells[n])

        for cell in uniq_cells:
            cell_gpis = np.where(cell == self.subcells[n])[0]
            for gpi in cell_gpis:
                yield self.subgpis[n][gpi], self.subarrlons[n][gpi], \
                      self.subarrlats[n][gpi], cell

    def subgrid_from_gpis(self, gpis):
        """
        Generate a subgrid for given gpis.

        Parameters
        ----------
        gpis : int, numpy.ndarray
            Grid point indices.

        Returns
        -------
        grid : BasicGrid
            Subgrid.
        """
        sublons, sublats = self.gpi2lonlat(gpis)
        subcells = self.gpi2cell(gpis)

        return CellGrid(sublons, sublats, subcells, gpis,
                        geodatum=self.geodatum.name)

    def subgrid_from_cells(self, cells):
        """
        Generate a subgrid for given cells.

        Parameters
        ----------
        cells : int, numpy.ndarray
            Cell numbers.

        Returns
        -------
        grid : CellGrid
            Subgrid.
        """
        subgpis, sublons, sublats = self.grid_points_for_cell(cells)
        subcells = self.gpi2cell(subgpis)

        return CellGrid(sublons, sublats, subcells, subgpis,
                        geodatum=self.geodatum.name)

    def __eq__(self, other):
        """
        Compare cells.

        Returns
        -------
        result : boolean
            Returns true if equal.
        """
        basicsame = super(CellGrid, self).__eq__(other)
        idx_gpi = np.argsort(self.gpis)
        idx_gpi_other = np.argsort(other.gpis)
        cellsame = np.all(self.arrcell[idx_gpi]
                          == other.arrcell[idx_gpi_other])
        return np.all([basicsame, cellsame])


class MetaGrid():
    """
    MetaGrid is a version of a Basic or CellGrid that contains a subset collection
    to quickly create, activate, combine and store multiple subsets.
    """

    # todo: shape is not intuitive should be (lat,lon(, but is (lon,lat)... this is a BasicGrid Problme
    #

    def __init__(self, lon, lat, cells=None, gpis=None, geodatum='WGS84',
                 setup_kdTree=True, shape=None, subsets=None):
        """

        Parameters
        ----------
        lon : numpy.ndarray
            Longitudes of the points in the grid.
        lat : numpy.ndarray
            Latitudes of the points in the grid.
        cells : numpy.ndarray, optional (default: None)
            Of same shape as lon and lat, containing the cell number of each gpi.
            If None is given, a 5x5 deg cell sampling is chosen.
        gpis : numpy.array, optional
            if the gpi numbers are in a different order than the
            lon and lat arrays an array containing the gpi numbers
            can be given
            if no array is given here the lon lat arrays are given
            gpi numbers starting at 0
        geodatum : basestring
            Name of the geodatic datum associated with the grid
        setup_kdTree : boolean, optional
            if set (default) then the kdTree for nearest neighbour
            search will be built on initialization
        shape : tuple, optional
            The shape of the grid array in 2-d space.
            e.g. for a 1x1 degree global regular grid the shape would be (180,360).
            if given the grid can be reshaped into the given shape
            this indicates that it is a regular grid and fills the
            attributes self.lon2d and self.lat2d which
            define the grid only be the meridian coordinates(self.lon2d) and
            the coordinates of the circles of latitude(self.lat2d).
            The shape has to be given as (lat2d, lon2d) # todo: this is wrong?
            It it is not given the shape is set to the length of the input
            lon and lat arrays.
            # todo: shape is not intuitive....

        shape : tuple, optional (default: None)
            Number of elements in the grid in a 2d array. Order: (lon, lat)
        subsets : list, optional (default: None)
            A list of pygeogrids.subset.Subsets that are assigned to the grid
            upon initialisation or a pygeogrids.subset.SubsetCollection.
        """

        if cells is None:
            self.base = BasicGrid(lon=lon, lat=lat, gpis=gpis,
                                  geodatum=geodatum, setup_kdTree=setup_kdTree,
                                  subset=None, shape=shape)
        else:
            self.base = CellGrid(lon=lon, lat=lat, gpis=gpis, cells=cells,
                                 geodatum=geodatum, setup_kdTree=setup_kdTree,
                                 subset=None, shape=shape)

        self.active_subset = None  # active_subset subset, set by activate()

        if isinstance(subsets, SubsetCollection):
            self.subsets = subsets
        else:
            self.subsets = SubsetCollection(subsets=subsets)

        # fill value has to match with the value from save_lonlat(), i.e. 0!
        # Therefore this can't be changed for now.
        self.subsets_fill_value = 0

    def __eq__(self, other):
        """ Compare grids and subset collections """
        basicsame = self.base == other
        subsetsame = self.subsets == other.subset_coll
        return all([basicsame, subsetsame])

    @property
    def subset_names(self):
        return self.subsets.names

    @classmethod
    def load_grid(cls, filename, location_var_name='gpi', subsets='all'):
        """
        Load meta grid with the selected subsets from file.

        Parameters
        ----------
        filename : str
            Path to the grid nc file.
        location_var_name : str, optional (default: 'gpi')
            Name of the index variable
        subsets : dict or list, optional (default: 'all')
            as dict: {name1: [values1], ...}
            as list: [name1, name2,...]
            Subset names or subset names and a list of values to consider.
            If 'all' is passed, all variables (except the reserved_names) are
            interpreted as subsets.

        Returns
        -------
        grid : MetaGrid
            A MetaGrid with a subset collection loaded from file
        """

        # Variables and variable values that are always ignored
        reserved_names = ['lon', 'lat', location_var_name, 'crs', 'cells']
        fill_value = 0

        def subset_defs(filter_subsets=()) -> dict:
            definitions = {}
            with Dataset(filename, 'r') as nc_data:
                for var in nc_data.variables:
                    if var in reserved_names: continue
                    if var in filter_subsets: continue
                    subset = filled_no_mask(np.unique(nc_data.variables[var][:]))
                    definitions[var] = np.delete(subset, fill_value)
            return definitions

        if subsets is None:
            definitions = None
        elif subsets.lower() == 'all':
            definitions = subset_defs()
        else:
            filter_subsets = subsets
            definitions = subset_defs(filter_subsets=filter_subsets)

        lons, lats, gpis, arrcell, subsets_kwargs, geodatumName, shape = \
            load_grid_definition(filename, location_var_name, subsets=definitions)

        subsets = []
        for name, kwargs in subsets_kwargs.items():
            subsets.append(Subset(name, **kwargs))

        return cls(lons,
                   lats,
                   gpis=gpis,
                   cells=arrcell,
                   geodatum=geodatumName,
                   setup_kdTree=True,
                   shape=shape,
                   subsets=subsets)

    @classmethod
    def from_grid(cls, grid, input_subset_name='input_subset', subsets=None):
        """
        Create a MetaGrid object from a passed BasicGrid or CellGrid

        Parameters
        ----------
        grid : BasicGrid or CellGrid
            The grid to use as the basis for coords and gpis in the MetaGrid.
        input_subset_name : str, optional (default: 'input_subset')
            If there is already a subset active in the passed grid, then the
            subset will have this name in the MetaGrid SubetCollection. If there
            is no subset, this is ignored.
        subsets : dict or list, optional (default: 'all')
            as dict: {name1: [values1], ...}
            as list: [name1, name2,...]
            Subset names or subset names and a list of values to consider.
            If 'all' is passed, all variables (except the reserved_names) are
            interpreted as subsets.

        Returns
        -------
        grid : MetaGrid
            A MetaGrid with a subset collection loaded from file
        """

        lons = grid.arrlon
        lats = grid.arrlat
        gpis = grid.gpis

        try:
            cells = grid.arrcell
        except AttributeError:
            cells = None

        if not grid.allpoints:
            meaning = 'Subset from grid used in creation of Metagrid'
            oss = Subset(name=input_subset_name, gpis=grid.gpis, meaning=meaning)
        else:
            oss = None

        if subsets is None:
            subsets = oss
        elif isinstance(subsets, SubsetCollection):
            if oss is not None: subsets.add(oss)
        else:
            subsets = subsets + oss if oss is not None else subsets

        subsets = None if len(subsets) == 0 else subsets

        return cls(lons,
                   lats,
                   gpis=gpis,
                   cells=cells,
                   geodatum=grid.geodatum.name,
                   setup_kdTree=True,
                   shape=grid.shape,
                   subsets=subsets)

    def subsets_as_dicts(self, format='all'):
        """
        Return subset points as a dictionary.

        Parameters
        ----------
        format : str
            See the description of Subset.to_dict()
            one of: all, save_lonlat, gpis

        Returns
        -------
        subset_dict : dict
            Dict of subset points in the selected format.
        """

        subset_dicts = {}
        for subset in self.subsets:
            subset_dicts.update(subset.as_dict(format))

        return subset_dicts

    def save_grid(self, filename, global_attrs=None):
        """
        Save MetaGrid and all subsets to netcdf file.

        Parameters
        ----------
        filename : str
            Path where the .nc file is created.
        global_attrs : dict, optional (default: None)
            Additional global attributes that are passed when writing the nc
            file.
        """

        try:
            arrcell = self.base.arrcell
        except AttributeError:
            arrcell = None

        gpis = self.base.gpis

        if self.base.shape is not None:
            if global_attrs is None:
                global_attrs = {}
            global_attrs['shape'] = self.base.shape

        global_attrs['grid_type'] = self.__class__.__name__  # MetaGrid

        subset_dicts = None
        if not self.subsets.empty:
            subset_dicts = self.subsets_as_dicts(format='save_lonlat')

        var_attrs = {subset.name: subset.attrs for subset in self.subsets}

        save_lonlat(filename, self.base.arrlon, self.base.arrlat, self.base.geodatum,
                    arrcell=arrcell, gpis=gpis, subsets=subset_dicts, zlib=True,
                    global_attrs=global_attrs, var_attrs=var_attrs)

    def add_subset(self, subset: {Subset, dict}):
        """
        Add a new subset to the subset collection either from a Subset object
        or from a dict with name and gpis as key and value.

        Parameters
        ----------
        subset : Subset or dict
            If a Subset is passed directy it is simply added. If a dict of the
            for {name : gpis} is passed, a new basic subset is created.
        """

        if isinstance(subset, Subset):
            pass
        elif isinstance(subset, dict):
            for name, gpis in subset.items():
                subset = Subset(name, gpis)
        else:
            raise ValueError("Either pass a Subset or a dict like {name: gpis}")

        self.subsets.add(subset)

    def subset_from_bbox(self, latmin=-90, latmax=90, lonmin=-180, lonmax=180,
                         name=None, **subset_kwargs):
        """
        Create a new subset from a bounding box and add it to the collection.
        This applies to the currently active subset.

        Parameters
        ----------
        latmin : float
            Lat of the lower left corner of the bbox
        latmax : float
            Lat of the upper right corner of the bbox
        lonmin : float
            Lon of the lower left corner of the bbox
        lonmax : float
            Lon of the upper right corner of the bbox
        name : str
            Name of the subset, if None is passed a name is generated from the
            bbox
        subset_kwargs :
            Additional keywords are passed to the subset generation
        """
        # if self.active_subset is None:
        #    raise ValueError("No subset is currently active")

        gpis = self.base.get_bbox_grid_points(latmin, latmax, lonmin, lonmax)

        if name is None:
            bbox_str = '_'.join([str(f) for f in [latmin, latmax, lonmin, lonmax]])
            name = f"bbox_{bbox_str}"

        self.add_subset(Subset(name, gpis, **subset_kwargs))

    def filter_active_subset(self, vals: {int: list}, **subset_kwargs):
        """
        Create a new subset from the currently active one by filtering with
        the passed values.
        """
        # todo: rename to filter active
        # todo create a similar fct subset_from_values(name, vals, new_name)
        # todo: add name to select a ss instead of using the current one
        if self.active_subset is None:
            raise ValueError('No subset active to be filtered. '
                             'Activate one with activate_subset(<name>)')
        else:
            subset = self.active_subset

        self.add_subset(subset.select_by_val(vals, **subset_kwargs))

    def deactivate_subset(self):
        """
        Deactivate the current subset. I.e. go back to the initialisation state.
        Also revert splitting if any splitting was performed.
        """
        self.base._empty_subset()
        self.base.unite()

    def activate_subset(self, name, vals=None):
        """
        Activate a subset from the collection. Only one subset can be active_subset at
        a time.

        Parameters
        ----------
        name : str
            Name of the subset to activate, must be in the collection.
        vals : {int,float,list}
            Subset values are filtered for these values directly. without creating
            a new subset
        """

        if self.active_subset is not None:
            self.deactivate_subset()

        subset = self.subsets[name]
        if vals is not None:
            subset = subset.select_by_val(vals)

        self.active_subset = subset

        subset_gpis = self.active_subset.gpis

        self.base.activearrlon = self.base.arrlon[subset_gpis]
        self.base.activearrlat = self.base.arrlat[subset_gpis]
        self.base.activegpis = self.base.gpis[subset_gpis]
        self.base.allpoints = False

        self.subset = subset_gpis

    def combine_subsets(self, names, new_name, method='intersect', **subset_kwargs):
        """
        Combine two or more subsets and create a new one which is added to the
        current subset collction.

        Parameters
        ----------
        names : Iterable
            List of names of subsets to combine
        new_name : str
            Name of the subset that is created
        method : str, optional (default: 'intersect')
            Name of a method to use to combine the subsets.
        kwargs:
        Additional kwargs are used when creating the new subset.
        """

        self.subsets.combine(subset_names=names,
                             new_name=new_name,
                             method=method, **subset_kwargs)

    def merge_subsets(self, names: list, new_name, layer_vals=None, keep_merged=True):
        """
        Merge multiple subsets into a single, new one. Merge down layers in
        the given order. Optionally set a new value for each subset.
        Points that are present in multiple merged subsets, will have
        the value from the last merged subset.

        Parameters
        ----------
        names : list
            Names of subsets to merge. If a GPI is in multiple subsets,
            the value of the later subset will be used.
        new_name : str
            Name of the new subset that is created. Must be different from
            subsets that are already in the collection.
        keep_merged : bool, optional (default: True)
            Keep the original input subsets as well as the newly created one.
        """

        self.subsets.merge(subset_names=names,
                           new_name=new_name,
                           new_vals=layer_vals,
                           keep=keep_merged)

    def plot(self, only_subset=False, visualise_vals=False, ax=None, **kwargs):
        """ Draw a basic map of grid points and current active subset """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('Plotting needs matplotlib installed, which is not found')
            return
        from pygeogrids.plotting import points_on_map
        imax = ax

        if not only_subset:
            not_ss_kwargs = {k: v for k, v in kwargs.items() if k not in ['cmap', 'color']}
            imax = points_on_map(self.base.arrlon, self.base.arrlat,
                                 c=None, color='grey', imax=imax, **not_ss_kwargs)

        if visualise_vals:
            c = self.active_subset.values
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.get_cmap('jet')
        else:
            c = None
            if 'color' not in kwargs:
                kwargs['color'] = 'green'

        if 'set_auto_extent' not in kwargs:
            set_auto_extent = True if only_subset else False
        else:
            set_auto_extent = kwargs.pop('set_auto_extent')

        imax = points_on_map(self.base.activearrlon, self.base.activearrlat,
                             c=c, imax=imax,
                             set_auto_extent=set_auto_extent,
                             **kwargs)

        return imax


def lonlat2cell(lon, lat, cellsize=5., cellsize_lon=None, cellsize_lat=None):
    """
    Partition lon, lat points into cells.

    Parameters
    ----------
    lat: float64, or numpy.ndarray
        Latitude.
    lon: float64, or numpy.ndarray
        Longitude.
    cellsize: float
        Cell size in degrees.
    cellsize_lon : float, optional
        Cell size in degrees on the longitude axis.
    cellsize_lat : float, optional
        Cell size in degrees on the latitude axis.

    Returns
    -------
    cell: int32, or numpy.ndarray
        Cell numbers.
    """
    if cellsize_lon is None:
        cellsize_lon = cellsize
    if cellsize_lat is None:
        cellsize_lat = cellsize
    y = np.clip(np.floor((np.double(lat) +
                          (np.double(90.0) + 1e-9)) / cellsize_lat), 0, 180)
    x = np.clip(np.floor((np.double(lon) + (np.double(180.0) + 1e-9))
                         / cellsize_lon), 0, 360)
    cells = np.int32(x * (np.double(180.0) / cellsize_lat) + y)

    max_cells = ((np.double(180.0) / cellsize_lat) *
                 (np.double(360.0)) / cellsize_lon)
    cells = np.where(cells > max_cells - 1, cells - max_cells, cells)
    return np.int32(cells)


def gridfromdims(londim, latdim, origin='top', **kwargs):
    """
    Defines new grid object from latitude and longitude dimensions. Latitude
    and longitude dimensions are 1D arrays that give the latitude and
    longitude values of a 2D latitude-longitude array.

    Parameters
    ----------
    londim : numpy.ndarray
        longitude dimension
    latdim : numpy.ndarray
        latitude dimension
    origin : {'bottom', 'top'}, optional (default: 'bottom')
        If bottom is selected, the GPI origin is at (min_lon, min_lat),
        i.e. in the bottom left corner.
        If 'top' is selected, the origin is at (min_lon, max_lat),
        i.e. in the top left corner

    Returns
    -------
    grid : BasicGrid
        New grid object.
    """
    lons, lats = np.meshgrid(londim, latdim)
    if origin.lower() == 'bottom':
        lats = np.flipud(lats)
    elif origin.lower() == 'top':
        pass
    else:
        raise ValueError(f"Unexpected origin passed, expected 'top' or 'bottom' "
                         f"got {origin.lower()}")

    return BasicGrid(lons.flatten(), lats.flatten(),
                     shape=(len(latdim), len(londim)), **kwargs)


def genreg_grid(grd_spc_lat=1, grd_spc_lon=1, minlat=-90.0, maxlat=90.0,
                minlon=-180.0, maxlon=180.0, **kwargs):
    """
    Define a global regular lon lat grid which starts in the North Western
    Corner of minlon, maxlat. The grid points are defined to be in the middle
    of a grid cell. e.g. the first point on a 1x1 degree grid with
    minlon -180.0 and maxlat 90.0 will be at -179.5 longitude, 89.5 latitude.

    Parameters
    ----------
    grd_spc_lat: float, optional
        Grid spacing in latitude direction.
    grd_spc_lon: float, optional
        Grid spacing in longitude direction.
    minlat : float, optional
        Minimum latitude of the grid.
    maxlat : float, optional
        Maximum latitude of the grid.
    minlon : float, optional
        Minimum longitude of the grid.
    maxlon : float, optional
        Maximum longitude of the grid.
    """
    lon_dim = np.arange(minlon + grd_spc_lon / 2.0, maxlon, grd_spc_lon)
    lat_dim = np.arange(maxlat - grd_spc_lat / 2.0, minlat, -grd_spc_lat)

    return gridfromdims(lon_dim, lat_dim, **kwargs)


def _element_iterable(el):
    """
    Test if a element is iterable

    Parameters
    ----------
    el: object


    Returns
    -------
    iterable: boolean
       if True then then el is iterable
       if Fales then not
    """
    try:
        el[0]
        iterable = True
    except (TypeError, IndexError):
        iterable = False

    return iterable


def reorder_to_cellsize(grid, cellsize_lat, cellsize_lon):
    """
    Reorder grid points in one grid to follow the
    ordering of differently sized cells. This is useful if
    e.g. a 10x10 degree CellGrid should be traversed
    in an order compatible with a 5x5 degree CellGrid.

    Parameters
    ----------
    grid: :py:class:`pygeogrids.grids.CellGrid`
        input grid
    cellsize_lat: float
        cellsize in latitude direction
    cellsize_lon: float
        cellsize in longitude direction

    Returns
    -------
    new_grid: :py:class:`pygeogrids.grids.CellGrid`
        output grid with original cell sizes but
        different ordering.
    """

    cell_grid = grid.to_cell_grid(cellsize_lat=cellsize_lat,
                                  cellsize_lon=cellsize_lon)
    cell_sort = np.argsort(cell_grid.arrcell)
    new_arrlon = grid.arrlon[cell_sort]
    new_arrlat = grid.arrlat[cell_sort]
    new_arrcell = grid.arrcell[cell_sort]
    new_gpis = grid.gpis[cell_sort]
    new_subset = None

    if grid.subset is not None:
        full_subset = np.zeros(new_arrlon.size)
        full_subset[grid.subset] = 1
        new_full_subset = full_subset[cell_sort]
        new_subset = np.where(new_full_subset == 1)[0]

    return CellGrid(new_arrlon, new_arrlat, new_arrcell,
                    gpis=new_gpis,
                    subset=new_subset)


if __name__ == '__main__':
    import numpy as np
    from pygeogrids.subset import SubsetCollection
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    base_grid = genreg_grid(grd_spc_lat=0.25, grd_spc_lon=0.25, origin='bottom')
    subsets = SubsetCollection.from_file(r'H:\code\pygeogrids\docs\examples\metagrid\europe.nc')

    ax = plt.axes(projection=ccrs.Mercator())
    metagrid = MetaGrid.from_grid(base_grid, subsets=subsets)
    metagrid.activate_subset('land')
    metagrid.plot(False, visualise_vals=False, markersize=10, title='test',
                  ax=ax)

    metagrid.activate_subset('country')
    metagrid.plot(True, visualise_vals=True, markers_scalef=0.5)

    metagrid.activate_subset('landcover_class')
    metagrid.plot(True, visualise_vals=True, markers_scalef=1.)

    # collection = SubsetCollection.from_file(r'C:\Temp\coll\europe.nc')
    #
    # grid = MetaGrid(lon=subgrid.arrlon, lat=subgrid.arrlat,
    #                 gpis=subgrid.gpis, subsets=collection)
    #
    #
    # for subset in ['land', 'landcover_class', 'country']:
    #     subgrid = load_grid(r"U:\users_temp\subsets.nc")
