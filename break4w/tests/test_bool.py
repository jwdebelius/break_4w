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
            ambiguous_values='TBD',
            )

    def test_bool_init_default(self):
        b = Bool(name=self.name,
                 description=self.description,
                 )
        self.assertEqual(b.order, ['true', 'false'])
        self.assertEqual(b.ambiguous_values, None)

    def test_bool_init_ambigious_and_order(self):
        self.assertEqual(self.b.type, 'Bool')
        self.assertEqual(self.b.order, ['True', 'False', 'TBD'])
        self.assertEqual(self.b.ambiguous_values, {'TBD'})

    def test_analysis_convert_to_word(self):
        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series(['TBD', 'True', 'True', 'False'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )
        self.b.analyis_remove_ambiguious(self.map_)
        self.b.analysis_remap_dtype(self.map_)
        self.b.analysis_convert_to_word(self.map_)

        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series([np.nan, 'yes', 'yes', 'no'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )

if __name__ == '__main__':
    main()
