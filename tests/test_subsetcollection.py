# -*- coding: utf-8 -*-
"""
Created on Aug 30 14:20 2020

@author: wolfgang
"""

from pygeogrids.subset import SubsetCollection, Subset
import numpy as np
import tempfile
import os
import unittest

class TestSubsetCollection(unittest.TestCase):
    def setUp(self):
        self.test = Subset('test', np.array([1, 2, 3, 4, 5]), values=1,
                            meaning='testset1', attrs={'attr': 'value'})
        sc = SubsetCollection([self.test])
        self.test2 = Subset('test2', np.array([1, 2, 3, 4, 5]) * 2, values=2,
                            meaning='testset2', attrs={'attr': 'value'})
        sc.add(self.test2)

        self.sc = sc

    def test_copy(self):
        assert self.sc.copy() == self.sc

    def test_props(self):
        assert self.sc.names == ['test', 'test2']
        assert self.sc.empty is False
        assert len(self.sc) == 2
        assert self.sc['test'] == self.test

    def test_from_dict(self):
        d = {'test': {'gpis': [1,2,3,4,5], 'values': 1,
                      'meaning': 'testset1',
                      'attrs': {'attr': 'value'}},
             'test2': {'gpis': [2,4,6,8,10], 'values': 2,
                      'meaning': 'testset2',
                      'attrs': {'attr': 'value'}}}
        assert self.sc == SubsetCollection.from_dict(d)

        assert SubsetCollection([Subset('a', [1,2,3])]) == \
               SubsetCollection.from_dict({'a': [1,2,3]})

    @staticmethod
    def test_empty():
        s = SubsetCollection()
        assert s.empty == True
        assert len(s.names) == 0
        try:
            a = s[0]
            assert False # no Error was thrown
        except (IndexError, KeyError):
            assert True # Error is ok
        assert s == SubsetCollection()
        assert not s.as_dict()

    def test_as_dict(self):
        d_all = self.sc.as_dict('all')
        d_lonlat = self.sc.as_dict('save_lonlat')
        d_gpis = self.sc.as_dict('gpis')

        assert sorted(list(d_gpis.keys())) == ['test', 'test2']

        assert sorted(list(d_lonlat.keys())) == ['test', 'test2']
        assert sorted(d_lonlat['test'].keys()) == sorted(['points', 'meaning', 'value'])
        d_gpis['test'] = self.sc['test'].gpis


        assert sorted(d_all.keys()) == ['test', 'test2']
        assert sorted(d_all['test'].keys()) == sorted(['gpis', 'meaning',
                                                       'values', 'attrs'])

        for n, d in zip(['gpis', 'points'], [d_all, d_lonlat]):
            np.testing.assert_array_equal(d['test'][n], self.sc['test'].gpis)

        np.testing.assert_array_equal(d_gpis['test'], self.sc['test'].gpis)


    def test_read_write(self):
        with tempfile.TemporaryDirectory() as outpath:
            self.sc.to_file(os.path.join(outpath, 'test.nc'))
            other = SubsetCollection.from_file(os.path.join(outpath, 'test.nc'))
            assert self.sc == other

    def test_combine_intersect(self):
        method = 'intersect'
        self.sc.combine(subset_names=['test', 'test2'], new_name=f'from_{method}',
                        method=method, values=99)
        assert f"from_{method}" in self.sc.names
        np.testing.assert_array_equal(self.sc['from_intersect'].gpis,
                                      np.array([2, 4]))
        np.testing.assert_array_equal(self.sc['from_intersect'].values,
                                      np.array([99, 99]))

    def test_combine_union(self):
        method = 'union'
        self.sc.combine(subset_names=['test', 'test2'], new_name=f'from_{method}',
                        method=method, values=99)
        assert f"from_{method}" in self.sc.names
        np.testing.assert_array_equal(self.sc['from_union'].gpis,
                                      np.array(list(range(1,6)) + [6,8,10]))
        np.testing.assert_array_equal(self.sc['from_union'].values,
                                      np.array([99]*8))

    def test_combine_diff(self):
        method = 'diff'
        self.sc.combine(subset_names=['test', 'test2'], new_name=f'from_{method}',
                        method=method, values=99)
        assert f"from_{method}" in self.sc.names
        np.testing.assert_array_equal(self.sc['from_diff'].gpis,
                                      np.array([1,3,5]))
        np.testing.assert_array_equal(self.sc['from_diff'].values,
                                      np.array([99]*3))

    def test_merge(self):
        self.sc.merge(['test', 'test2'], new_name='merged_new_single_val', new_vals=99,
                      keep=True)

        # case when overwriting all values
        gpis_should = np.array(list(range(1,6)) + [6,8,10]) # like union

        np.testing.assert_array_equal(self.sc['merged_new_single_val'].gpis,
                                      gpis_should)
        np.testing.assert_array_equal(self.sc['merged_new_single_val'].values,
                                      np.array([99]*8))
        assert self.sc['merged_new_single_val'].meaning == "Merged subsets test, test2"

        # case when overwriting values for each layer
        self.sc.merge(['test', 'test2'], new_name='merged_new_other_vals',
                      new_vals={'test': 5, 'test2': 6}, keep=True)
        np.testing.assert_array_equal(self.sc['merged_new_other_vals'].gpis,
                                      gpis_should)
        np.testing.assert_array_equal(self.sc['merged_new_other_vals'].values,
                                      np.array([5,6,5,6,5,6,6,6]))
        assert self.sc['merged_new_other_vals'].meaning == "Merged subsets test, test2"

        # case when using the original values for each layer
        self.sc.merge(['test', 'test2'], keep=True)
        np.testing.assert_array_equal(self.sc['merge_test_test2'].gpis,
                                      gpis_should)
        np.testing.assert_array_equal(self.sc['merge_test_test2'].values,
                                      np.array([1,2,1,2,1,2,2,2]))
        assert self.sc['merge_test_test2'].meaning == "Merged subsets test, test2"


        assert ('test' in self.sc.names) and ('test2' in self.sc.names)

        # reverse order, test1 has higher priority now
        self.sc.merge(['test2', 'test'], keep=False)
        assert self.sc.names == ['merge_test2_test', 'merge_test_test2',
                                 'merged_new_other_vals', 'merged_new_single_val']
        np.testing.assert_array_equal(self.sc['merge_test2_test'].gpis,
                                      gpis_should)
        np.testing.assert_array_equal(self.sc['merge_test2_test'].values,
                                      np.array([1,1,1,1,1,2,2,2]))

        assert not self.sc['merge_test_test2'] == self.sc['merge_test2_test']