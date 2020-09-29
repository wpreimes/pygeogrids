# -*- coding: utf-8 -*-

"""
These tests run the (exact code used in the examples).
"""

def test_subets_example():
    import numpy as np
    from pygeogrids.subset import Subset

    # for the first subset pass gpis as 1d arrays, all points are assigned the default value "1".
    subset1 = Subset(name='Subset1', gpis=np.array([10, 11, 12, 13]))

    # for the second subset pass gpis as a 2d array, not all points are assigned the same value.
    subset2 = Subset(name='Subset2', meaning='Advanced case', gpis=np.array([[11, 12], [19, 20]]), values=[1, 1, 2, 2],
                     attrs={'AnyAttributeName': 'AttributeValue'})
    print(f'{subset1.name} has values {subset1.values} and shape {subset1.shape}')
    print(f'{subset2.name} has values {subset2.values} and shape {subset2.shape}')

    from pprint import pprint
    print("When returning `all` properties: \n")
    pprint(subset2.as_dict('all'))
    print("When returning `gpis` only: \n")
    print(subset2.as_dict('gpis'))

    filtered_subset2 = subset2.select_by_val(vals=[1], meaning='Filtered version of subset2')

    print("The filtered subset looks like this: \n")
    pprint(filtered_subset2.as_dict())

    inter = subset2.intersect(subset1, meaning='intersection of subset1 and subset2')
    pprint(inter.as_dict())

    union = subset2.union(subset1, values=3)
    pprint(union.as_dict())

    diff = subset2.diff(subset1, name='subset_diff', attrs={'method': 'subset2 minus subset1'})
    pprint(diff.as_dict())

    from pygeogrids.subset import SubsetCollection

    # create a subset collection from a list of subsets:
    collection = SubsetCollection(subsets=[subset1, subset2])

    subset_collection_dict = collection.as_dict()
    pprint(subset_collection_dict)

    collection_from_dict = SubsetCollection.from_dict(subset_collection_dict)

    collection.add(Subset('added_subset', gpis=np.array([5, 12, 19, 26]), values=5.))
    pprint(collection.as_dict())

    collection.combine(['Subset1', 'added_subset'], method='union', new_name='combined_union')
    pprint(collection.as_dict())

    collection.merge(['Subset1', 'added_subset'], new_name='merged_subsets')
    pprint(collection.as_dict())

    import tempfile
    import os

    filename = os.path.join(tempfile.mkdtemp(), 'ssc.nc')

    collection.to_file(filename)
    pprint(collection.as_dict())

    loaded_collection = SubsetCollection.from_file(filename)
    pprint(loaded_collection.as_dict())
    assert collection == loaded_collection

def test_metagrid_example():
    from pygeogrids import MetaGrid
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        makemaps = True
    except ImportError:
        makemaps = False
        print('Parts of this tutorial need matplotlib and cartopy installed.')
        print("Run 'conda install cartopy'")

    lon, lat = np.meshgrid(np.arange(-180., 180., 9.), np.arange(-90., 90., 9.))
    metagrid = MetaGrid(lon.flatten(), np.flipud(lat).flatten())
    metagrid.plot(False, markersize=5.)

    metagrid.subset_from_bbox(-60, 60, -50, 50, name='center_box')
    metagrid.subset_from_bbox(-20, 20, -160, 160, name='lon_box')
    metagrid.combine_subsets(['lon_box', 'center_box'], method='intersect', new_name='inter')

    if makemaps:
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), subplot_kw={'projection': ccrs.Orthographic(0, 0)})
        metagrid.activate_subset('center_box')
        metagrid.plot(only_subset=False, markersize=15., title='center_box', color='blue', ax=axs[0],
                      set_auto_extent=False)
        metagrid.activate_subset('lon_box')
        metagrid.plot(only_subset=False, markersize=15., title='lon_box', color='red', ax=axs[1], set_auto_extent=False)
        metagrid.activate_subset('inter')
        metagrid.plot(only_subset=False, markersize=15., title='inter', color='orange', ax=axs[2],
                      set_auto_extent=False)
    else:
        print('This part of the tutorial needs cartopy/matplotlib installed')

    try:
        import matplotlib.pyplot as plt
        import cartopy
        makemaps = True
    except ImportError:
        makemaps = False
        print('Parts of this tutorial need matplotlib and cartopy installed.')
        print("Run 'conda install cartopy'")
    import numpy as np
    import os
    from pprint import pprint
    from pygeogrids import src_path
    from pygeogrids.subset import SubsetCollection
    from pygeogrids.grids import genreg_grid

    base_grid = genreg_grid(grd_spc_lat=0.25, grd_spc_lon=0.25, origin='bottom')
    subsets = SubsetCollection.from_file(os.path.join(src_path, '..', 'docs', 'examples', 'metagrid', 'europe.nc'))
    subsets.names
    subsets['land'].as_dict()
    print(subsets['country'].as_dict())

    from pygeogrids.grids import MetaGrid
    metagrid = MetaGrid.from_grid(base_grid, subsets=subsets)

    metagrid.activate_subset('land')
    if makemaps:
        fig, axs = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.Mercator()})
        metagrid.plot(only_subset=False, title='Europe Land Mask Subset on global grid', markersize=0.5, ax=axs[0])
        metagrid.plot(only_subset=True, title='Europe Land Mask Subset only', markersize=0.5, ax=axs[1])
    else:
        print('This part of the tutorial needs cartopy/matplotlib installed')

    if makemaps:
        fig, axs = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.Mercator()})
        metagrid.activate_subset('country')
        metagrid.plot(only_subset=True, visualise_vals=True, title='Europe Country Name Subset', markersize=1.0,
                      ax=axs[0])
        metagrid.activate_subset('landcover_class')
        metagrid.plot(only_subset=True, visualise_vals=True, title='Europe CCI Landcover Subset', markersize=1.0,
                      ax=axs[1])
    else:
        print('Install matplotlib and cartopy to run this part of the tutorial')

    # Filter subset for France and add it to the metagrid
    idx = subsets['country'].attrs['flag_meanings'].index('France')
    val_france = subsets['country'].attrs['flag_values'][idx]
    metagrid.activate_subset('country')
    metagrid.filter_active_subset(vals=[val_france], name='france')
    print(metagrid.subsets.names)

    for lc_class in [120, 121, 122, 130]:
        i = subsets['landcover_class'].attrs['flag_values'].tolist().index(lc_class)
        meaning = subsets['landcover_class'].attrs['flag_meanings'][i]
        print(f"{lc_class} = {meaning}")

    metagrid.activate_subset('landcover_class')
    metagrid.filter_active_subset(vals=[120, 121, 122, 130], name='grassland')
    print(metagrid.subsets.names)

    metagrid.combine_subsets(['grassland', 'france'], method='intersect', new_name='france_grassland')

    # %%

    if makemaps:
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), subplot_kw={'projection': ccrs.Mercator()})
        metagrid.activate_subset('france')
        metagrid.plot(only_subset=True, visualise_vals=True, title='France Points', markersize=1.0, ax=axs[0])
        metagrid.activate_subset('grassland')
        metagrid.plot(only_subset=True, visualise_vals=True, title='Grassland Points', markersize=1.0, ax=axs[1])
        metagrid.activate_subset('france_grassland')
        metagrid.plot(only_subset=True, visualise_vals=True, title='Grassland in France Points', markersize=1.0,
                      ax=axs[2])
    else:
        print('This part of the tutorial needs cartopy/matplotlib installed')

    gpis = metagrid.base.activegpis
    lons, lats = metagrid.base.activearrlon, metagrid.base.activearrlat
    print('Grassland grid points are: \n')
    print('GPIs:', gpis.tolist())
    print('LONs:', lons.tolist())
    print('LATs:', lats.tolist())

    plt.close('all')

if __name__ == '__main__':
    test_metagrid_example()
    test_subets_example()
