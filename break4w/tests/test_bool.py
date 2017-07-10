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

    def test_analysis_convert_to_word_pass(self):
        self.b.analysis_remap_dtype(self.map_)
        self.b.analysis_convert_to_word(self.map_)

        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series(['TBD', 'yes', 'yes', 'no'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )
        log_entry = self.b.log[1]
        self.assertEqual(log_entry['command'], 'convert boolean')
        self.assertEqual(log_entry['transform_type'], 'replace')
        self.assertEqual(log_entry['transformation'], 'standarize to yes/no')

    def test_analysis_convert_to_word_fail(self):
        with self.assertRaises(ValueError):
            self.b.analysis_convert_to_word(self.map_)

        log_entry = self.b.log[0]
        self.assertEqual(log_entry['command'], 'convert boolean')
        self.assertEqual(log_entry['transform_type'], 'replace')
        self.assertEqual(log_entry['transformation'],
                         'data could not be standardized')

    def test_validate_pass(self):
        self.b.validate(self.map_)

if __name__ == '__main__':
    main()
