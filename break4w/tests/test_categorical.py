from unittest import TestCase, main

import pandas as pd
import numpy as np
import numpy.testing as npt
import pandas.util.testing as pdt

from break4w._categorical import Categorical

class CategoricalTest(TestCase):

    def setUp(self):

        self.map_ = pd.DataFrame([['1', '2', '2', '4'],
                                  ['TBD', 'True', 'True', 'False'],
                                  ['Striker', 'D-man', 'D-man', 'Goalie']],
                                 index=['years_on_team',
                                        'team_captain', 'position'],
                                 columns=['Bitty', 'Ransom', 'Holster',
                                          'Johnson'],
                                 ).T
        self.name = 'position'
        self.description = 'Where the player can normally be found on the ice'
        self.dtype = str
        self.order = ["Striker", "D-man", "Goalie"]

        self.c = Categorical(
            name=self.name,
            description=self.description,
            dtype=self.dtype,
            order=self.order,
            )

    def test_categorical_init(self):
        test = Categorical(self.name,
                           self.description,
                           self.dtype,
                           self.order)

        self.assertEqual(self.order, test.order)
        self.assertEqual(test.type, 'Categorical')
        self.assertEqual(test.frequency_cutoff, None)
        self.assertEqual(test.ref_value, self.order[0])
        self.assertEqual(test.var_numeric, None)
        self.assertEqual(test.var_labels, None)
        self.assertEqual(test.ambiguous, None)

    def test_categorical_init_error(self):
        with self.assertRaises(ValueError):
            Categorical(self.name, self.description, order=self.order,
                        dtype=ValueError)

    def test_categorical_numeric_ambig_str(self):
        test = Categorical(self.name,
                           self.description,
                           int,
                           self.order,
                           ambiguous='manager',
                           var_labels='1=Striker | 2=D-man | 3=Goalie',
                           ref_value='coach',
                           )
        self.assertEqual(test.var_numeric, 
                         {'Striker': 1, 'D-man': 2, 'Goalie': 3})
        self.assertEqual(test.var_labels, 
                         {1: 'Striker', 2: 'D-man', 3: 'Goalie'})
        self.assertEqual(test.ambiguous, set(['manager']))
        self.assertEqual(test.ref_value, 'coach')

    def test_categorical_dict_var_labels(self):
        test = Categorical(self.name,
                           self.description,
                           int,
                           self.order,
                           var_labels={1: 'Striker', 2: 'D-man', 3: 'Goalie'},
                           )


    def test_update_order(self):
        # Checks the current order
        self.assertEqual(self.c.order, ["Striker", "D-man", "Goalie"])

        # Sets up a function to adjust the data
        def remap_(x):
            if x in {"D-man", "Goalie"}:
                return "Defense"
            elif x in {"Striker"}:
                return "Offense"
            else:
                return "Not on the team!"

        # # updates the data
        self.c._update_order(remap_)

        # Checks the updated order
        self.assertEqual(self.c.order, ["Offense", "Defense"])

    def test_validate_dtype_fail(self):
        self.c.dtype = bool
        with self.assertRaises(TypeError):
            self.c.validate(self.map_)
        log_entry = self.c.log[0]
        self.assertEqual(log_entry['command'], 'validate')
        self.assertEqual(log_entry['transform_type'], 'error')
        self.assertEqual(log_entry['transformation'],
                         'the data cannot be cast to bool')

    def test_validate_fail_dtype(self):
        self.c.dtype = int
        with self.assertRaises(TypeError):
            self.c.validate(self.map_)
        log_entry = self.c.log[0]
        self.assertEqual(log_entry['command'], 'validate')
        self.assertEqual(log_entry['transform_type'], 'error')
        self.assertEqual(log_entry['transformation'],
                         'the data cannot be cast to int'
                         )

    def test_validate_fail_values(self):
        self.c.name = 'years_on_team'
        with self.assertRaises(ValueError):
            self.c.validate(self.map_)
        log_entry = self.c.log[1]
        self.assertEqual(log_entry['command'], 'validate')
        self.assertEqual(log_entry['transform_type'], 'error')
        self.assertEqual(log_entry['transformation'],
                         'The following are not valid values: 1 | 2 | 4'
                         )

    def test_validate_pass(self):
        self.c.validate(self.map_)
        log_entry = self.c.log[0]
        self.assertEqual(log_entry['command'], 'validate')
        self.assertEqual(log_entry['transform_type'], 'pass')
        self.assertEqual(log_entry['transformation'],
                         'the data can be cast to str'
                         )
        log_entry2 = self.c.log[1]
        self.assertEqual(log_entry2['command'], 'validate')
        self.assertEqual(log_entry2['transform_type'], 'pass')
        self.assertEqual(log_entry2['transformation'],
                         'all values were valid'
                         )

    def test_to_series(self):
        known = pd.Series({'name': self.name,
                           'description': self.description,
                           'dtype': 'str',
                           'type': 'Categorical',
                           'clean_name': 'Position',
                           'order': 'Striker | D-man | Goalie',
                           'ref_value': 'Striker',
                            })
        test_ = self.c._to_series()
        pdt.assert_series_equal(known, test_)

    def test_read_series(self):
        var_ = pd.Series({'name': self.name,
                          'description': self.description,
                          'dtype': 'str',
                          'order': 'Striker | D-man | Goalie',
                          })
        c = Categorical._read_series(var_)
        self.assertTrue(isinstance(c, Categorical))
        self.assertEqual(c.order, self.order)
        self.assertEqual(c.name, self.name)
        self.assertEqual(c.description, self.description)
        self.assertEqual(c.dtype, str)

    def test_round_trip(self):
        var_ = self.c._to_series()
        new_ = Categorical._read_series(var_)
        self.assertEqual(self.c.__dict__, new_.__dict__)

if __name__ == '__main__':
    main()
