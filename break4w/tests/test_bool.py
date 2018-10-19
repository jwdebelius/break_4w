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
            ambiguous='TBD',
            )

    def test_bool_init_default(self):
        b = Bool(name=self.name,
                 description=self.description,
                 )
        self.assertEqual(b.order, ['true', 'false'])
        self.assertEqual(b.ambiguous, None)

    def test_bool_init_ambiguous_and_order(self):
        self.assertEqual(self.b.type, 'Bool')
        self.assertEqual(self.b.order, ['True', 'False'])
        self.assertEqual(self.b.ambiguous, {'TBD'})

    def test_validate_pass(self):
        self.b.validate(self.map_)

    def test_to_dict(self):
        known = {'name': self.name,
                 'description': self.description,
                 'dtype': bool,
                 'ambiguous': {'TBD'},
                 'order': self.bool,
                 'clean_name': 'Team Captain',
                 'ref_val': 'False',
                 }
        type_, test = self.b.to_dict()
        self.assertEqual(type_, 'bool')
        self.assertEqual(test, known)


if __name__ == '__main__':
    main()
