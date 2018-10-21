from unittest import TestCase, main

import pandas as pd
import numpy as np
# import numpy.testing as npt
import pandas.util.testing as pdt

from break4w.bool import Bool


class BoolTest(TestCase):
    def setUp(self):

        self.map_ = pd.DataFrame([['1', '2', '2', '4'],
                                  ['TBD', 'True', 'True', 'False'],
                                  ['Striker', 'D-man', 'D-man', 'Goalie']],
                                 index=['years_on_team',
                                        'team_captain', 'position'],
                                 columns=['Bitty', 'Ransom', 'Holster',
                                          'Johnson'],
                                 ).T
        self.name = 'team_captain'
        self.description = 'Has the player been given a C or AC?'
        self.bool = ['True', 'False']

        self.b = Bool(
            name=self.name,
            description=self.description,
            bool_format=self.bool,
            dtype=str,
            ambiguous='TBD',
            )

    def test_bool_init_default(self):
        b = Bool(name=self.name,
                 description=self.description,
                 )
        self.assertEqual(b.order, ['false', 'true'])
        self.assertEqual(b.ambiguous, None)
        self.assertEqual(self.b.type, 'Bool')

    def test_bool_init_ambiguous_and_order(self):
        self.assertEqual(self.b.order, ['False', 'True'])
        self.assertEqual(self.b.ambiguous, {'TBD'})

    def test_validate_pass(self):
        self.b.validate(self.map_)

    def test_to_series(self):
        known = pd.Series({'name': self.name,
                           'description': self.description,
                           'dtype': 'str',
                           'type': 'Bool',
                           'clean_name': 'Team Captain',
                           'order': 'False | True',
                           'ref_value': 'False',
                           'ambiguous': 'TBD',
                           })
        test = self.b._to_series()
        pdt.assert_series_equal(known, test)

    def test_read_series(self):
        series = pd.Series({'name': self.name,
                           'description': self.description,
                           'dtype': 'bool',
                           'type': 'Bool',
                           'clean_name': 'Team Captain',
                           'order': 'False | True',
                           'ref_value': 'False',
                           })
        b = Bool._read_series(series)
        
        self.assertTrue(isinstance(b, Bool))
        self.assertEqual(b.name, self.name)
        self.assertEqual(b.description, self.description)
        self.assertEqual(b.clean_name, 'Team Captain')
        self.assertEqual(b.type, 'Bool')
        self.assertEqual(b.dtype, bool)
        self.assertEqual(b.order, [False, True])
        self.assertFalse(b.ref_value)

    def test_roundtrip(self):
        var_ = self.b._to_series()
        new_ = Bool._read_series(var_)
        self.assertEqual(self.b.__dict__, new_.__dict__)



if __name__ == '__main__':
    main()
