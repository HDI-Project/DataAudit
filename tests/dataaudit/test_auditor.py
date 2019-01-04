#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from numpy import nan

from dataaudit import auditor


@pytest.fixture()
def create_df():
    return pd.DataFrame({'identifier': [0, 1, 2, 3],
                         'gender': ['null', 'Nan', 'male', 'female'],
                         'birthdate': ['10/21/2000', 'unknown', '1/10/2000', '9/16/2000'],
                         'address': ['undefined', '4060246413000', '406026413022', '40677564000']})


@pytest.fixture()
def total_nans_df():
    return pd.DataFrame({'identifier': [0, 1, 2, 3]})


@pytest.fixture()
def get_check_list():
    return [{'gender': [{'percenatge': {'female': 0, 'male': 0, 'other': 0, 'unknown': 0}},
                        {'nan': {'number': 0, 'percenatge': '0.00%'}}]},
            {'birthdate': [{'nan': {'number': 0, 'percenatge': '0.00%'}}]}]


def test_check_total_nans(total_nans_df):
    expected_output = [{'Patient': [{'identifier': {'nan': {'number': 0, 'percenatge': '0.00%'}}},
                                    {'gender': {'nan': {'number': 2, 'percenatge': '50.00%'}}},
                                    {'birthdate': {'nan': {'number': 1, 'percenatge': '25.00%'}}},
                                    {'address': {'nan': {'number': 1, 'percenatge': '25.00%'}}}]}]
    expected_output = expected_output[0]['Patient'][0]
    output = auditor.Auditor().check_total_nans('Patient', total_nans_df)
    output = output[0]['Patient'][0]

    assert output['identifier'] == expected_output['identifier']


def test_check_nan(create_df):
    expected_output = {'nan': {'number': 2, 'percenatge': '50.00%'}}
    attr_value = create_df['gender'].values
    assert auditor.Auditor().check_nan(attr_value) == expected_output


def test_find_type(create_df, get_check_list):
    expected_output = [{'gender': [{'percenatge': {'male': '25.00%', 'female': '25.00%',
                                                   'Nan': '25.00%', 'null': '25.00%'}},
                                   {'nan': {'number': 2, 'percenatge': '50.00%'}}]},
                       {'birthdate': [{'nan': {'number': 1, 'percenatge': '25.00%'}}]}]
    assert auditor.Auditor().find_type(create_df, get_check_list) == expected_output


def test_find_percentage():
    expected_output = {'percenatge': {'male': '25.00%', 'female': '25.00%',
                                      'Nan': '25.00%', 'null': '25.00%'}}
    assert auditor.Auditor().find_percentage(
        'gender', ['null', 'Nan', 'male', 'female']) == expected_output


def test_find_frequency():
    expected_output = {'female': 1, 'male': 2}
    assert auditor.Auditor().find_frequency(
        ['female', 'male', 'male']) == expected_output


def test_find_minimum_error():
    with pytest.raises(TypeError):
        auditor.Auditor().find_minimum([2, "f", 3])


def test_find_minimum():
    assert auditor.Auditor().find_minimum([2, nan, 3]) == {'min': 2}


def test_find_maximum_error():
    with pytest.raises(TypeError):
        auditor.Auditor().find_maximum([2, "f", 3])


def test_find_maximum():
    assert auditor.Auditor().find_maximum([2, nan, 3]) == {'max': 3}


def test_find_mean_error():
    with pytest.raises(TypeError):
        auditor.Auditor().find_mean([2, "f", 3])


def test_find_mean():
    assert auditor.Auditor().find_mean([2, 4, 3]) == {'mean': 3}


def test_compare_distributions():
    expected_output = {'ks_test': {'statistic': 1.0, 'pvalue': 0.011065637015803861}}
    assert auditor.Auditor().compare_distributions([1, 2, 3, 4], [5, 6, 9, 8]) == expected_output


def test_apply_predefined_distribution_check():
    expected_output = {
        'normal': {
            'ks_test': {
                'statistic': 0.8396,
                'pvalue': 0.002544122126156039}}}
    assert auditor.Auditor().apply_predefined_distribution_check(
        [1, 2, 3, 4], dist_type="normal",
        dist_charecteristics={"mean": 1999, "std": 2000}) == expected_output
