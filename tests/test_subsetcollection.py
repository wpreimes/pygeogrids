# -*- coding: utf-8 -*-
"""
Created on Aug 30 14:20 2020

@author: wolfgang
"""

from pygeogrids.subset import SubsetCollection, Subset
import unittest
import numpy as np
import pytest

def get_data():
    ss = Subset('test', np.array([1, 2, 3, 4, 5]), values=1, meaning='testset1',
                attrs={'attr': 'value'})
    sc = SubsetCollection([ss])
    t2 = Subset('test2', np.array([1, 2, 3, 4, 5]) * 2, values=2, meaning='testset2',
                attrs={'attr': 'value'})
    sc.add(t2)
    return  sc

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




@pytest.mark.parametrize("format,fields",
                         [("all", ['points', 'meaning', 'value', 'shape', 'attrs']),
                          ("save_lonlat", ['points', 'meaning', 'value']),
                          ("gpis", ['points'])])
def test_as_dict(format, fields):
    sc = get_data()
    d = sc.as_dict(format)
    for field in fields:
        assert d[field]



if __name__ == '__main__':
    test_empty()