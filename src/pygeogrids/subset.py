# -*- coding: utf-8 -*-

import numpy as np
import warnings
from netCDF4 import Dataset
from collections import Iterable

class Subset():
    """
    A subset is an arbitrary group of GPIs on a grid (e.g. land points etc.).
    """
    def __init__(self, name, gpis, meaning='', values=1, attrs=None):
        # todo: sort GPIs when creating subset?
        """
        Parameters
        ----------
        name : str
            Name of the subset
        gpis : np.array
            Array of GPIs that identify the subset in a grid
            Can be a 1d ar a 2d array, this affects the shape attribute but are
            always stored as a sorted 1d array.
        meaning : str, optional (default: '')
            Short description of the points in the subset.
        values : int or np.array, optional (default: 1)
            Integer values that represent the subset in the variable.
        attrs : dict, optional (default: None)
            Attributes that are stored together with the subset.
            Attributes are not compared in __eq__
        """
        gpis = np.asanyarray(gpis)

        self.name = name

        if gpis.ndim == 1:
            self.shape = (len(gpis),)
        elif gpis.ndim == 2:
            self.shape = gpis.shape
            gpis = gpis.flatten()
        else:
            raise ValueError("GPIs must be passed as 1d or 2d array")

        idx = np.argsort(gpis)
        self.gpis = gpis[idx]

        self.meaning = '' if meaning is None else meaning

        if isinstance(values, int):
            self.values = np.repeat(values, len(self.gpis))
        else:
            values = np.asanyarray(values)
            if len(values) != len(gpis):
                raise ValueError(f"Shape of values array does not match "
                                 f"to gpis with {len(gpis)} elements")
            self.values = values[idx]

        self.attrs = attrs if attrs is not None else {}

    def __eq__(self, other):
        try:
            assert self.name == other.name
            assert self.meaning == other.meaning
            assert self.shape == other.shape
            np.testing.assert_equal(self.gpis, other.gpis)
            np.testing.assert_equal(self.values, other.values)
            return True
        except AssertionError:
            return False

    def as_dict(self, format='all') -> dict:
        """
        Return subset attributes as a dictionary

        Parameters
        ----------
        format : {'all','save_lonlat','gpis'}
            Format definition string on what values are returned and how.
            * 'gpis': Return in form: {name: [gpis], ... }
            * 'save_lonlat': Return in form as used in save_lonlat():
                    {name: {'points': gpis,
                            'meaning': meaning,
                            'value': values }}
            * 'all' : Return all data and metadata
                    {name: {'points': gpis,
                            'meaning': meaning,
                            'value': values,
                            'shape': shape,
                            'attrs' : {attributes}}}

        Returns
        -------
        subset_dict : dict
            Subset as dict
        """

        if format.lower() == 'gpis':
            return {self.name: self.gpis}
        elif format.lower() == 'save_lonlat':
            return {self.name: {'points': self.gpis, 'value': self.values,
                                'meaning': self.meaning}}
        elif format.lower() == 'all':
            return {self.name: {'points': self.gpis, 'value': self.values,
                                'meaning': self.meaning, 'shape': self.shape,
                                'attrs': self.attrs}}
        else:
            raise ValueError(f"{format} is not a known format definiton")

    def select_by_val(self, vals, **subset_kwargs):
        """
        Filter subset points to points with certain values.

        Parameters
        ----------
        vals : list or int
            Whitelist of values
        subset_kwargs :
            Additional kwargs are used to create the new subset

        Returns
        -------
        filtered : Subset
            New, filtered Subset
        """

        idx = np.isin(self.values, vals)

        if not any(idx):
            raise ValueError('No points in subset {self.name} found with passed value(s)')

        if 'name' not in subset_kwargs:
            name = f"filtered_{self.name}"
        else:
            name = subset_kwargs.pop('name')

        return Subset(name=name, gpis=self.gpis[idx], values=self.values[idx],
                      **subset_kwargs)

    def merge(self, other, new_name=None, new_meaning=None, new_val_self=None,
              new_val_other=None, prioritize_other=True):
        """
        Merge this subset with another subset. Merging means that GPIs and values
        from both subsets are included in the merged subset. If there are points
        that are in both subsets, the preference can be set to self/other.
        Values can be manually overridden, so that after merging there are not more
        then 2 different values in the merged subset.

        Parameters
        ----------
        other : Subset
            Another Subset.
        new_name : str, optional (default: None)
            Name of the merged subset.
        new_meaning: str, optional (default: None)
            New meaning of the merged subsets.
        new_val_self : int, optional (default: None)
            If a value is given, then values in self.values are overridden
            with this value in the merged subset.
        new_val_other : int, optional (default: None)
            If a value is given, then values in other.values are overridden
            with this value in the merged subset.
        prioritize_other : bool, optional (default: True)
            If true, points that are in self AND other will have values
            from other. Otherwise the values of self are preferred.

        Returns
        -------
        merged_subset : Subset
            The new subset
        """

        # this defines for points in both subsets, which value is used.
        if prioritize_other:
            n = 1 # concat(other_gpis, self_gpis)
        else:
            n = -1 # concat(self_gpis, other_gpis)

        gpis = np.concatenate((other.gpis, self.gpis)[::n])
        other_values, self_values = other.values.copy(), self.values.copy()

        if new_val_other:
            other_values = np.repeat(new_val_other, len(other_values))

        if new_val_self:
            self_values = np.repeat(new_val_self, len(self_values))

        values = np.concatenate((other_values, self_values)[::n])

        gpis, indices = np.unique(gpis, return_index=True)

        if new_name is None:
            new_name = f"{self.name}_merge_{other.name}"

        return Subset(name=new_name, gpis=gpis, values=values[indices],
                      meaning=new_meaning)

    def intersect(self, other, new_name=None, **subset_kwargs):
        """
        Intersect 2 subset, to include points that are in A AND B,
        create a new subset from these points with a new value.

        Parameters
        ----------
        other: Subset
            Another subset that is intersected with this subset.
        new_name : str, optional (default: None)
            Name of the Subset that is created from the intersection.
        subset_kwargs :
            Additional kwargs are used to create the new subset
        """

        gpis = np.intersect1d(self.gpis, other.gpis, return_indices=False)

        if new_name is None:
            new_name = f"{self.name}_inter_{other.name}"

        return Subset(name=new_name, gpis=gpis, **subset_kwargs)

    def union(self, other, new_name=None, **subset_kwargs):
        """
        Unite 2 subsets, to include points from A and B,
        create a new subset from these points with a new value.

        Parameters
        ----------
        other: Subset
            Another subset that is united with this subset.
        new_name : str, optional (default: None)
            Name of the Subset that is created from the union of gpis.
        subset_kwargs :
            Additional kwargs are used to create the new subset
        """

        gpis = np.union1d(self.gpis, other.gpis)

        if new_name is None:
            new_name = f"{self.name}_union_{other.name}"

        return Subset(name=new_name, gpis=gpis, **subset_kwargs)

    def diff(self, other, new_name=None, **subset_kwargs):
        """
        Difference of 2 subsets, to include points from A without points
        from B, create a new subset from these points with a new value.

        Parameters
        ----------
        other: Subset
            Another subset that is subtracted from this subset.
        new_name : str, optional (default: None)
            Name of the Subset that is created from the difference gpis.
        subset_kwargs :
            Additional kwargs are used to create the new subset
        """

        gpis = np.setdiff1d(self.gpis, other.gpis)

        if new_name is None:
            new_name = f"{self.name}_diff_{other.name}"

        return Subset(name=new_name, gpis=gpis, **subset_kwargs)

    def _apply(self, other, method, *args, **kwargs):
        return self.__getattribute__(method)(other, *args, **kwargs)

class SubsetCollection():
    """
    A SubsetCollection holds multiple subsets and provides functions to add,
    drop and combine/merge multiple of them.
    Can be written to / read from netcdf.
    """

    # todo: is functionality to read/write to nc files needed?
    # todo: Allow setting all subset params when creating coll from dict?

    def __init__(self, subsets=None):
        """
        Parameters
        ----------
        subsets : list, optional (default: None)
            List of initial Subsets
        """

        self.subsets = [] if subsets is None else subsets

    @property
    def names(self) -> list:
        return sorted([s.name for s in self.subsets])

    @property
    def empty(self) -> bool:
        return True if len(self) is 0 else False

    def __len__(self) -> int:
        return len(self.subsets)

    def __getitem__(self, item:{str, int}) -> Subset:
        """ Get subset by name or by index """
        if isinstance(item, int): # by index, for __iter__
            return self.subsets[item]
        else: # by name
            for s in self.subsets:
                if s.name == item: return s
            raise KeyError(f"No subset with name or index {item} found")

    def __eq__(self, other) -> bool:
        """ Compare 2 collections for equal gpis in all subsets"""
        try:
            assert np.all(self.names == other.names)

            for name in self.names:
                assert name in other.names
                assert self[name] == other[name] # compare subsets

            return True

        except AssertionError:
            return False

    @classmethod
    def from_dict(cls, subsets_dict: dict):
        """
        Create a subset collection from gpis passed as a dict.
        This does NOT allow to set values, meaning and shape... # todo: add?

        Parameters
        ----------
        subsets_dict : dict
            Subset dict with subset names as keys and gpis as values
            {'subset1' : [1,2,3], 'subset2': [1,3,5]}

        Returns
        -------
        cls : SubsetCollection
            The collection initiated from dict values
        """

        subsets = []
        for name, gpis in subsets_dict.items():
            subsets.append(Subset(name, gpis))

        return cls(subsets=subsets)

    @classmethod
    def from_file(cls, filename:str):
        # todo: keep this, not needed to load grid
        """
        Load subset collection from a stored netcdf file.

        Parameters
        ----------
        filename : str
            Path to file that was created using the to_file() function

        Returns
        -------
        cls : SubsetCollection
            The collection loaded from file
        """

        subsets = []
        with Dataset(filename, 'r') as ncfile:
            for varname in ncfile.variables:
                var = ncfile.variables[varname]
                if var.ndim == 1: continue
                subset_kwargs = {}
                try:
                    subset_kwargs['meaning'] = var.getncattr('meaning')
                except KeyError:
                    pass
                try:
                    shape = var.getncattr('shape')
                    shape = (shape,) if not isinstance(shape, Iterable) else shape
                    subset_kwargs['shape'] = shape
                except KeyError:
                    pass

                subset = Subset(varname, gpis=var[:][0], values=var[:][1],
                                **subset_kwargs)

                subsets.append(subset)

        return cls(subsets=subsets)

    def to_file(self, filepath:str):
        # todo: keep this, not needed to save grid
        """
        Store subsets as variables in netcdf format.

        Parameters
        ----------
        filepath : str
            Path to netcdf file to create
        """
        if self.empty:
            raise IOError("Cannot write empty subset to file.")

        with Dataset(filepath, "w", format="NETCDF4") as ncfile:
            ncfile.createDimension("points", None)
            ncfile.createDimension("props", 2) # gpis & values

            props = ncfile.createVariable('_props', 'str', zlib=True,
                                          dimensions=('props',))

            props[:] = np.array(['gpis', 'vals'])

            for subset in self.subsets:
                ss = ncfile.createVariable(subset.name, 'int', zlib=True,
                                           dimensions=('props', 'points'))
                ss[:] = np.array([subset.gpis, subset.values])

                ss.setncatts({'meaning': subset.meaning,
                              'shape': subset.shape})

    def as_dict(self, format='all') -> dict:
        """
        Return subset points as a dictionary.

        Parameters
        ----------
        format : {'all','save_lonlat','gpis'}
            See the description of Subset.to_dict()

        Returns
        -------
        subsets_dict : dict
            Subsets as dictionary
        """

        subset_dicts = {}
        for subset in self.subsets:
            subset_dicts.update(subset.as_dict(format))

        return subset_dicts

    def add(self, subset:Subset):
        """
        Append another subset to the collection.

        Parameters
        ----------
        subset : Subset
            The subset to add.
        """

        if subset.name in self.names:
            raise KeyError(f"A subset {subset.name} already exists in the collection")

        self.subsets.append(subset)

    def drop(self, name:str):
        """
        Drop a subset from the collection

        Parameters
        ----------
        name : str
            Name of the subset to drop.
        """

        for i, s in enumerate(self.subsets):
            if s.name == name:
                self.subsets.pop(i)

    def combine(self, subset_names:list, new_name:str, method='intersect',
                **subset_kwargs) -> Subset:
        """
        Combine 2 or more subsets, to get the common gpis. This is not the
        same as merging them!

        Parameters
        ----------
        subset_names : list
            Names of subsets to concatenate. If a GPI is in multiple subsets,
            the value of the later subset will be used.
        new_name : str, optional (default: None)
            Name of the new subset that is created. If None is passed, a name
            is created.
        method : {'intersect', 'union', 'diff'}
            An implemented method to combine subset.
            * intersect: Points that are in both subsets
            * union: Points that are in subset A or B
            * diff: Points from A without points from B
        subset_kwargs:
            Kwargs are passed to create the new subset.

        Returns
        -------
        combined : Subset
            A new subset created by combining the selected Subsets.
        """

        if len(subset_names) < 2:
            raise IOError(f"At least 2 subsets are expected for intersection, "
                          f"got {len(subset_names)}")

        subset = self[subset_names[0]]

        for i, other_name in enumerate(subset_names[1:]):
            subset = subset._apply(self[other_name], method, new_name=new_name,
                                   **subset_kwargs)

        return subset

    def merge(self, subset_names:list, new_name=None, new_vals=None, keep=False):
        """
        Merge down multiple layers. This means that gpis and values for subsets
        in the provided order are combined and gpis that are in multiple subsets
        will have the value of the subset that was merged down last.

        Parameters
        ----------
        subset_names : list
            Names of subsets to merge. If a GPI is in multiple subsets,
            the value of the later subset will be used.
        new_name : str, optional (default: None)
            Name of the new subset that is created. If None is passed, a name
            is created.
        keep : bool, optional (default: False)
            Keep the original input subsets as well as the newly created one.
        new_vals : dict, optional (default: None)
            New values that are assigned to the respective, merged subsets.
            Structure: {subset_name: subset_value, ...}
            Any subset named that is selected here, must be in subset_names as
            well.
        Additional kwargs are passed to create the new subset.

        Returns
        -------
        merged : Subset
            A new subset created by merging the selected Subsets.
        """

        if len(subset_names) < 2:
            raise IOError(f"At least 2 subsets are expected for merging, "
                          f"got {len(subset_names)}")

        if new_vals is None:
            new_vals = {}

        if any([e in subset_names for e in new_vals.keys()]):
            raise ValueError("Names in new_vals must match with subset_names passed.")

        subset = self[subset_names[0]]

        for i, other_name in enumerate(subset_names[1:]):

            new_val_self = new_vals[subset.name] \
                if subset.name in new_vals.keys() else None
            new_val_other = new_vals[other_name] \
                if other_name in new_vals.keys() else None

            subset = subset.merge(self[other_name], new_val_self=new_val_self,
                                  new_val_other=new_val_other)
            if not keep:
                self.drop(other_name)

        if not keep:
            self.drop(subset_names[0])

        if new_name is None:
            new_name = f"merge_{'_'.join(subset_names)}"

        subset.name = new_name
        subset.meaning = f"Merged subsets {', '.join(subset_names)}"

        return subset


if __name__ == '__main__':

    ss = Subset('test', np.array([1,2,3,4,5]), values=1)
    sc = SubsetCollection([ss])
    t2 = Subset('test2', np.array([1,2,3,4,5])*2, values=2)
    sc.add(t2)

    inter = sc.combine(['test', 'test2'], new_name='inter', values=3,
                       how='intersect', shape=(10,10))
    sc.add(inter)

    merge = sc.merge(['test', 'test2'], new_name='merged', keep=False)
    sc.add(merge)

    sc.to_file(r'C:\Temp\ssc\ssc.nc')

    sc = SubsetCollection.from_file(r'C:\Temp\ssc\ssc.nc')

    pass




