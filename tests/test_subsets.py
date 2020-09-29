# -*- coding: utf-8 -*-

from pygeogrids.subset import Subset
import numpy as np

def test_subset():
    subset1 = Subset('test1', gpis=np.arange(100), values=1, meaning='subset1')
    subset2 = Subset('test2', gpis = np.arange(50,150), values=2, meaning='subset 2')

    np.testing.assert_array_equal(subset1.as_dict('gpis')['test1'], np.arange(100))
    np.testing.assert_array_equal(subset1.as_dict('all')['test1']['values'],
                                  np.repeat(1, 100))
    assert subset2.as_dict('save_lonlat')['test2']['meaning'] == 'subset 2'

    assert not subset1 == subset2
    assert subset2 == Subset('test2', gpis=np.arange(50,150), values=2, meaning='subset 2')

    return subset1, subset2


def test_select():
    subset = Subset('test', gpis=np.array([5,4,3,2,1]), values=np.array([1,2,1,2,1]))
    selection = subset.select_by_val([1,2])
    np.testing.assert_array_equal(subset.gpis, selection.gpis)
    np.testing.assert_array_equal(subset.values, selection.values)

    np.testing.assert_array_equal(subset.select_by_val(1).gpis, np.array([1, 3, 5]))
    assert subset.select_by_val(2).name == 'filtered_test'
    assert subset.select_by_val(2).meaning == ''


def test_inter():
    subset1, subset2 = test_subset()
    inter = subset1.intersect(subset2, name='inter', values=3)
    assert all(inter.values == 3)
    np.testing.assert_array_equal(inter.gpis, np.arange(50,100))
    assert inter.name == 'inter'

def test_union():
    subset1, subset2 = test_subset()
    union = subset1.union(subset2)
    assert all(union.values == 1)
    np.testing.assert_array_equal(union.gpis, np.arange(150))
    assert union.name == 'test1_union_test2'

def test_diff():
    subset1, subset2 = test_subset()
    diff = subset1.diff(subset2, values=np.arange(1000,1050))
    assert all(diff.values == np.arange(1000,1050))
    np.testing.assert_array_equal(diff.gpis, np.arange(50))
    assert diff.name == 'test1_diff_test2'

def test_merge():
    subset1, subset2 = test_subset()
    merged = subset1.merge(subset2)
    np.testing.assert_equal(merged.values,
                            np.concatenate((np.repeat(1, 50), np.repeat(2, 100))))
    assert merged.name == 'test1_merge_test2'
    assert merged.meaning == ''
    np.testing.assert_array_equal(merged.gpis,
        np.unique(np.concatenate((subset1.gpis, subset2.gpis))))

    # all setts
    merged = subset1.merge(subset2, new_name='test', new_meaning='merged',
                           new_val_self=3, new_val_other=4, prioritize_other=False)
    np.testing.assert_equal(merged.values,
          np.concatenate((np.repeat(3, 100), np.repeat(4, 50))))
    assert merged.name == 'test'
    assert merged.meaning == 'merged'
    np.testing.assert_array_equal(merged.gpis,
        np.unique(np.concatenate((subset1.gpis, subset2.gpis))))
