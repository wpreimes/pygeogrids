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

import warnings
try:
    import matplotlib.pyplot as plt
    import matplotlib as mp
except ImportError:
    warnings.warn("Matplotlib is necessary for plotting grids.")
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    warnings.warn("Basemap is necessary for plotting grids.")
import numpy as np
import pygeogrids.grids as grids

def points_on_map(lons, lats, figsize=(12, 6), color='blue', imax=None,
                  llc_lonlat=None, urc_lonlat=None, set_auto_extent=True):
    # create a simple scatter plot of points on a map
    try:
        import cartopy
        import cartopy.crs as ccrs
    except ImportError:
        print("Plotting maps needs cartopy, install with 'conda install cartopy'")
        return
    data_crs = ccrs.PlateCarree()
    if not imax:
        plt.figure(num=None, figsize=figsize, facecolor='w', edgecolor='k')
        imax = plt.axes(projection=ccrs.Mercator())

    imax.coastlines(resolution='110m', color='black', linewidth=0.25)
    imax.add_feature(cartopy.feature.BORDERS, linewidth=0.1, zorder=2)

    #imax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    if llc_lonlat is None:
        llc_lonlat = (min(lons), min(lats))
    if urc_lonlat is None:
        urc_lonlat = (max(lons), max(lats))

    if set_auto_extent:
        imax.set_extent([llc_lonlat[0], urc_lonlat[0],
                         llc_lonlat[1], urc_lonlat[1]], crs=data_crs)
    lon_interval = max([llc_lonlat[0], urc_lonlat[0]]) - \
                   min([llc_lonlat[0], urc_lonlat[0]])
    markersize = 1.5 * (360 / lon_interval)

    plt.scatter(lons, lats, s=markersize, zorder=3, transform=data_crs, color=color)

    return imax



def plot_cell_grid_partitioning(output,
                                cellsize_lon=5.,
                                cellsize_lat=5.0,
                                figsize=(12, 6)):
    """
    Plot an overview of a global cell partitioning.

    Parameters
    ----------
    output: string
        output file name
    """
    mp.rcParams['font.size'] = 10
    mp.rcParams['text.usetex'] = True
    plt.figure(figsize=figsize, dpi=300)
    ax = plt.axes([0, 0, 1, 1])

    map = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90,
                  llcrnrlon=-180, urcrnrlon=180, ax=ax)
    map.drawparallels(np.arange(-90, 90, cellsize_lat), labels=[1, 0, 0, 0],
                      linewidth=0.5)
    map.drawmeridians(np.arange(-180, 180, cellsize_lon),
                      labels=[0, 0, 0, 1], rotation='vertical', linewidth=0.5)
    # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='0.6', lake_color='aqua')
    label_lats = np.arange(-90 + cellsize_lat / 2., 90, cellsize_lat)
    label_lons = np.arange(-180 + cellsize_lon / 2., 180, cellsize_lon)
    lons, lats = np.meshgrid(label_lons, label_lats)
    x, y = map(lons.flatten(), lats.flatten())
    cells = grids.lonlat2cell(lons.flatten(), lats.flatten(),
                              cellsize_lon=cellsize_lon, cellsize_lat=cellsize_lat)
    for xt, yt, cell in zip(x, y, cells):
        plt.text(xt, yt, "{:}".format(cell), fontsize=4,
                 va="center", ha="center", weight="bold")
    plt.savefig(output, format='png', dpi=300)
    plt.close()
