from unittest import TestCase, main

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from break4w.continous import Continous


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
        self.unit = 'years'
        self.range = [0, 4]

        self.c = Continous(self.name,
                           self.description,
                           unit=self.unit,
                           dtype=self.dtype,
                           limits=self.range)

    def test_init_default(self):
        test = Continous(
            name=self.name,
            description=self.description,
            )
        self.assertEqual('Continous', test.type)
        self.assertEqual(test.dtype, float)
        self.assertEqual(test.lower, None)
        self.assertEqual(test.upper, None)

    def test_init(self):
        self.assertEqual(self.c.unit, self.unit)
        self.assertEqual(self.c.lower, 0)
        self.assertEqual(self.c.upper, 4)
        self.assertEqual(self.c.dtype, int)

    def test_init_error(self):
        with self.assertRaises(ValueError):
            Continous('TEST_COLUMN',
                      'Testing. 1, 2, 3.\nAnything but that!',
                      limits=[12, 5])

    def test_analysis_drop_outliers(self):
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series(['1', '2', '2', '4'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )
        self.c.lower = 2
        self.c.analysis_drop_outliers(self.map_)
        pdt.assert_series_equal(
            self.map_['years_on_team'],
            pd.Series([np.nan, 2, 2, 4],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='years_on_team')
            )


if __name__ == '__main__':
    main()