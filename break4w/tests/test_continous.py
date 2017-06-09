from unittest import TestCase, main

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from break4w.continous import Continous, _check_limits


class ContinousTest(TestCase):

    def setUp(self):
        self.map_ = pd.DataFrame([['1', '2', '2', '4'],
                                  ['TBD', 'True', 'True', 'False'],
                                  ['Striker', 'D-man', 'D-man', 'Goalie']],
                                 index=['years_on_team',
                                        'team_captain', 'position'],
                                 columns=['Bitty', 'Ransom', 'Holster',
                                          'Johnson'],
                                 ).T
        self.name = 'years_on_team'
        self.description = ("How many years the player has been on SMH during "
                            "Bitty's frog year")
        self.dtype = int
        self.units = 'years'

        self.c = Continous(self.name,
                           self.description,
                           units=self.units,
                           dtype=self.dtype,
                           limits=[1, None],
                           outliers=[None, 5])

    def test_init_default(self):
        test = Continous(
            name=self.name,
            description=self.description,
            units=self.units
            )
        self.assertEqual(test.units, self.units)
        self.assertEqual('Continous', test.type)
        self.assertEqual(test.dtype, float)
        self.assertEqual(test.bound_lower, None)
        self.assertEqual(test.bound_upper, None)
        self.assertEqual(test.outlier_lower, None)
        self.assertEqual(test.outlier_upper, None)
        self.assertEqual(test.sig_figs, None)

    def test_init(self):
        self.assertEqual(self.c.dtype, self.dtype)
        self.assertEqual(self.c.bound_lower, 1)
        self.assertEqual(self.c.bound_upper, None)
        self.assertEqual(self.c.outlier_lower, None)
        self.assertEqual(self.c.outlier_upper, 5)

    def test_update_outliers(self):
        self.assertEqual(self.c.outlier_lower, None)
        self.assertEqual(self.c.outlier_upper, 5)
        self.c.dictionary_update_outliers([2, 7])
        self.assertEqual(self.c.outlier_lower, 2)
        self.assertEqual(self.c.outlier_upper, 7)

    def test_analysis_drop_outliers_none(self):
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series(['1', '2', '2', '4'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.c.dictionary_update_outliers([None, None])
        self.c.analysis_drop_outliers(self.map_)
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series([1, 2, 2, 4],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.assertEqual(self.c.log[2]['transformation'], 'No values dropped')

    def test_analysis_drop_outliers_left(self):
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series(['1', '2', '2', '4'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.c.dictionary_update_outliers([2, None])
        self.c.analysis_drop_outliers(self.map_)
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series([np.nan, 2, 2, 4],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.assertEqual(self.c.log[2]['transformation'], 'values less than 2')

    def test_analysis_drop_outliers_right(self):
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series(['1', '2', '2', '4'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.c.dictionary_update_outliers([None, 3])
        self.c.analysis_drop_outliers(self.map_)
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series([1, 2, 2, np.nan],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.assertEqual(self.c.log[2]['transformation'],
                         'values greater than 3')

    def test_analysis_drop_outliers_both(self):
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series(['1', '2', '2', '4'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.c.dictionary_update_outliers([2, 3])
        self.c.analysis_drop_outliers(self.map_)
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series([np.nan, 2, 2, np.nan],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.assertEqual(self.c.log[2]['transformation'],
                         'values outside [2, 3]')

    def test_analysis_set_sig_figs_error(self):
        with self.assertRaises(ValueError):
            self.c.analysis_set_sig_figs(self.map_)

    def test_analysis_set_sig_figs(self):
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series(['1', '2', '2', '4'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.c.sig_figs = 0.1
        self.c.analysis_set_sig_figs(self.map_)
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series([1, 2, 2, 4],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.assertEqual(self.c.bound_lower, 1.0)
        self.assertEqual(self.c.bound_upper, None)
        self.assertEqual(self.c.outlier_lower, None)
        self.assertEqual(self.c.outlier_upper, 5.0)

    def test_validate_none_pass(self):
        self.c.bound_upper = None
        self.c.bound_lower = None
        self.c.validate(self.map_)
        self.assertEqual(self.c.log[1]['transformation'],
                         'There were no limits specified.')

    def test_validate_lower_pass(self):
        self.c.validate(self.map_)
        self.assertEqual(self.c.log[1]['transformation'],
                         'The values were greater than or equal to 1 years.')

    def test_validate_lower_error(self):
        self.c.bound_lower = 2
        with self.assertRaises(ValueError):
            self.c.validate(self.map_)
        self.assertEqual(self.c.log[1]['transformation'],
                         'There are values less than 2 years.')

    def test_validate_upper_pass(self):
        self.c.bound_upper = 5
        self.c.bound_lower = None
        self.c.validate(self.map_)
        self.assertEqual(self.c.log[1]['transformation'],
                         'The values were less than or equal to 5 years.')

    def test_validate_upper_error(self):
        self.c.bound_upper = 1
        with self.assertRaises(ValueError):
            self.c.validate(self.map_)
        self.assertEqual(self.c.log[1]['transformation'],
                         'There are values greater than 1 years.')

    def test_validate_both_pass(self):
        self.c.bound_upper = 5
        self.c.validate(self.map_)
        self.assertEqual(self.c.log[1]['transformation'],
                         'The values were between 1 and 5 years.')

    def test_validate_both_error(self):
        self.c.bound_lower = 2
        self.c.bound_upper = 3
        with self.assertRaises(ValueError):
            self.c.validate(self.map_)
        self.assertEqual(self.c.log[1]['transformation'],
                         'There are values less than 2 and greater than'
                         ' 3 years.')

    def test_check_limits_none(self):
        limits = None
        lower, upper = _check_limits(limits, 'limits')
        self.assertEqual(lower, None)
        self.assertEqual(upper, None)

    def test_check_limits_left_none(self):
        limits = [None, 5]
        lower, upper = _check_limits(limits, 'limits')
        self.assertEqual(lower, None)
        self.assertEqual(upper, 5)

    def test_check_limits_right_none(self):
        limits = [-3, None]
        lower, upper = _check_limits(limits, 'limits')
        self.assertEqual(lower, -3)
        self.assertEqual(upper, None)

    def test_check_limits_all_pass(self):
        limits = [-3, 5]
        lower, upper = _check_limits(limits, 'limits')
        self.assertEqual(lower, -3)
        self.assertEqual(upper, 5)

    def test_check_limits_all_error(self):
        limits = [8, 5]
        with self.assertRaises(ValueError):
            _check_limits(limits, 'limits')

if __name__ == '__main__':
    main()
