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
                           outliers=[None, 5],
                           )

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

    def test_dictionary_update_outliers(self):
        self.assertEqual(self.c.outlier_lower, None)
        self.assertEqual(self.c.outlier_upper, 5)
        self.c.dictionary_update_outliers([2, 7])
        self.assertEqual(self.c.outlier_lower, 2)
        self.assertEqual(self.c.outlier_upper, 7)

        log_ = self.c.log[0]
        self.assertEqual(log_['command'], 'update outlier values')
        self.assertEqual(log_['transform_type'], 'update dictionary')
        self.assertEqual(log_['transformation'], 'Outlier values have been '
                         'updated: None > 2 and 5 > 7.')

    def test_validate_fail_dtype(self):
        self.c.name = 'position'
        with self.assertRaises(TypeError):
            self.c.validate(self.map_)
        self.assertEqual(len(self.c.log), 1)
        log_ = self.c.log[0]
        self.assertEqual(log_['command'], 'validate')
        self.assertEqual(log_['transform_type'], 'error')
        self.assertEqual(log_['transformation'],
                         'the data cannot be cast to int')

    def test_validate_no_lim_no_blank_no_ambigious_pass(self):
        self.c.bound_upper = None
        self.c.bound_lower = None
        self.c.blanks = None
        self.c.validate(self.map_)
        self.assertEqual(len(self.c.log), 2)

        log0 = self.c.log[0]
        self.assertEqual(log0['command'], 'validate')
        self.assertEqual(log0['transform_type'], 'pass')
        self.assertEqual(log0['transformation'],
                         'the data can be cast to int')

        log1 = self.c.log[1]
        self.assertEqual(log1['command'], 'validate')
        self.assertEqual(log1['transform_type'], 'pass')
        self.assertEqual(log1['transformation'],
                         'there were no limits specified')

    def test_validate_lower_pass_blank_list_ambigious_list(self):
        self.c.blanks = ['missing']
        self.c.ambiguous = ['test']
        self.map_.loc['Ransom', 'years_on_team'] = 'missing'
        self.c.validate(self.map_)
        self.assertEqual(len(self.c.log), 2)

        log0 = self.c.log[0]
        self.assertEqual(log0['command'], 'validate')
        self.assertEqual(log0['transform_type'], 'pass')
        self.assertEqual(log0['transformation'],
                         'the data can be cast to int')

        log1 = self.c.log[1]
        self.assertEqual(log1['command'], 'validate')
        self.assertEqual(log1['transform_type'], 'pass')
        self.assertEqual(log1['transformation'],
                         'The values were greater than or equal to 1 years.')

    def test_validate_greater_pass_blank_str_ambigious_str(self):
        self.c.blanks = 'missing'
        self.c.ambiguous = 'missing'
        self.c.test = 'test'
        self.c.bound_upper = 5
        self.c.bound_lower = None
        self.map_.loc['Ransom', 'years_on_team'] = 'missing'
        self.c.validate(self.map_)
        self.assertEqual(len(self.c.log), 2)

        log0 = self.c.log[0]
        self.assertEqual(log0['command'], 'validate')
        self.assertEqual(log0['transform_type'], 'pass')
        self.assertEqual(log0['transformation'],
                         'the data can be cast to int')

        log1 = self.c.log[1]
        self.assertEqual(log1['command'], 'validate')
        self.assertEqual(log1['transform_type'], 'pass')
        self.assertEqual(log1['transformation'],
                         'The values were less than or equal to 5 years.')

    def test_validate_both_lim(self):
        self.c.bound_upper = 5
        self.c.validate(self.map_)
        self.assertEqual(len(self.c.log), 2)

        log1 = self.c.log[1]
        self.assertEqual(log1['command'], 'validate')
        self.assertEqual(log1['transform_type'], 'pass')
        self.assertEqual(log1['transformation'],
                         'The values were between 1 and 5 years.')

    def test_validate_lower_error(self):
        self.c.bound_lower = 2
        with self.assertRaises(ValueError):
            self.c.validate(self.map_)
        self.assertEqual(len(self.c.log), 2)

        log1 = self.c.log[1]
        self.assertEqual(log1['command'], 'validate')
        self.assertEqual(log1['transform_type'], 'error')
        self.assertEqual(log1['transformation'],
                         'There are values less than 2 years.')

    def test_validate_upper_error(self):
        self.c.bound_upper = 3
        with self.assertRaises(ValueError):
            self.c.validate(self.map_)
        self.assertEqual(len(self.c.log), 2)

        log1 = self.c.log[1]
        self.assertEqual(log1['command'], 'validate')
        self.assertEqual(log1['transform_type'], 'error')
        self.assertEqual(log1['transformation'],
                         'There are values greater than 3 years.')

    def test_validate_both_error(self):
        self.c.bound_lower = 2
        self.c.bound_upper = 3
        with self.assertRaises(ValueError):
            self.c.validate(self.map_)

        self.assertEqual(len(self.c.log), 2)

        log1 = self.c.log[1]
        self.assertEqual(log1['command'], 'validate')
        self.assertEqual(log1['transform_type'], 'error')
        self.assertEqual(log1['transformation'],
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

    def test_to_dict(self):
        known = {'name': self.name,
                 'description': self.description,
                 'units': self.units,
                 'dtype': self.dtype,
                 'limits': [1, None],
                 'outliers': [None, 5],
                 'clean_name': 'Years On Team'
                 }
        type_, test = self.c.to_dict()
        self.assertEqual(test, known)
        self.assertEqual('continous', type_)

    def test_to_series(self):
        self.c.missing = {"lacross"}
        self.c.frogs = ['foxtrot', 'whiskey', 'tango']
        known = pd.Series({'name': self.name,
                           'description': self.description,
                           'units': self.units,
                           'dtype': 'int',
                           'type': 'Continous',
                           'order': '1 | None',
                           'ambiguous': 'None | 5',
                           'missing': 'lacross',
                           'frogs': 'foxtrot | whiskey | tango',
                           'clean_name': 'Years On Team',
                           })
        known = known[['name', 'description', 'dtype', 'type', 
                       'clean_name', 'missing', 'units', 'frogs',
                       'order', 'ambiguous']]

        test = self.c._to_series()
        pdt.assert_series_equal(known, test)

if __name__ == '__main__':
    main()
