from unittest import TestCase, main

import pandas as pd
import numpy as np
# import numpy.testing as npt
import pandas.util.testing as pdt

from break4w.categorical import Categorical


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
        self.extremes = ["Striker", "Goalie"]

        self.c = Categorical(
            name=self.name,
            description=self.description,
            dtype=self.dtype,
            order=self.order,
            extremes=self.extremes,
            )

    def test_categorical_init(self):
        test = Categorical(self.name,
                           self.description,
                           self.dtype,
                           self.order)

        self.assertEqual(self.order, test.order)
        self.assertEqual(test.extremes, self.extremes)
        self.assertEqual(test.type, 'Categorical')
        self.assertEqual(test.frequency_cutoff, None)

    def test_categorical_init_error(self):
        with self.assertRaises(ValueError):
            Categorical(self.name, self.description, order=self.order,
                        dtype=ValueError)

    def test_update_order(self):

        # Checks the current order and extremes
        self.assertEqual(self.c.order, ["Striker", "D-man", "Goalie"])
        self.assertEqual(self.c.extremes, ["Striker", "Goalie"])

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
        self.assertEqual(self.c.extremes, ["Offense", 'Defense'])

    def test_analysis_drop_infrequent(self):
        self.assertTrue(set(self.map_[self.c.name].unique()),
                        {'Striker', 'D-man'})
        self.c.frequency_cutoff = 1
        self.c.analysis_drop_infrequent(self.map_)
        self.assertEqual(set(self.map_[self.c.name].unique()),
                         {'D-man', np.nan})

    def test_analysis_remove_ambigious_str(self):
        c = Categorical(name='team_captain',
                        description='who has the C or AC',
                        dtype=str,
                        order=['True', 'False'],
                        ambiguous_values='TBD'
                        )
        # Checks the current status
        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series(['TBD', 'True', 'True', 'False'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )

        c.analyis_remove_ambiguious(self.map_)

        # Checks the remapping
        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series([np.nan, 'True', 'True', 'False'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )

    def test_analysis_remove_ambigious_list(self):
        c = Categorical(name='team_captain',
                        description='who has the C or AC',
                        dtype=str,
                        order=['True', 'False'],
                        ambiguous_values=['TBD']
                        )
        # Checks the current status
        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series(['TBD', 'True', 'True', 'False'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )

        c.analyis_remove_ambiguious(self.map_)

        # Checks the remapping
        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series([np.nan, 'True', 'True', 'False'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )

    def test_analysis_remove_ambigious_nan(self):
        # Sets the ambigious value as goalie
        self.c.ambiguous_values = 'Goalie'
        # Drops the striker value
        self.map_.loc['Bitty', 'position'] = np.nan

        self.c.analyis_remove_ambiguious(self.map_)

        # Checks the remapping
        pdt.assert_series_equal(
            self.map_['position'],
            pd.Series([np.nan, 'D-man', 'D-man', np.nan],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='position')
            )

    def test_analysis_convert_to_numeric(self):
        self.c.order = ["Striker", "D-man"]
        self.c.analysis_convert_to_numeric(self.map_)
        self.assertEqual(self.c.order, [0, 1])
        pdt.assert_series_equal(
            self.map_['position'],
            pd.Series(data=[0, 1, 1, np.nan],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='position',
                      )
            )
        self.assertEqual(self.c.log[0]['transformation'],
                         'Striker >>> 0 | D-man >>> 1')

    def test_analysis_convert_to_numeric_number(self):
        c = Categorical(name='years_on_team',
                        description="Time in SMH in Bitty's frog year",
                        order=['1', '2', '3', '4'],
                        dtype=int)
        # Converts the data to numeric
        c.analysis_remap_dtype(self.map_)
        c.analysis_convert_to_numeric(self.map_)
        pdt.assert_series_equal(self.map_['years_on_team'],
                                pd.Series([1, 2, 2, 4], name='years_on_team',
                                          index=['Bitty', 'Ransom', 'Holster',
                                                 'Johnson']))

    def test_analysis_label_order(self):
        self.c.order = ["Striker", "D-man"]
        self.c.analysis_label_order(self.map_)
        self.assertEqual(self.c.order,
                         ["(0) Striker", "(1) D-man"]
                         )
        pdt.assert_series_equal(self.map_['position'],
                                pd.Series(data=["(0) Striker", "(1) D-man",
                                                "(1) D-man", np.nan],
                                          index=['Bitty', 'Ransom', 'Holster',
                                                 'Johnson'],
                                          name='position',
                                          )
                                )
        self.assertEqual(self.c.log[0]['transformation'],
                         'Striker >>> (0) Striker | D-man >>> (1) D-man')

    def test_analysis_label_order_number(self):
        c = Categorical(name='years_on_team',
                        description="Time in SMH in Bitty's frog year",
                        order=['1', '2', '3', '4'],
                        dtype=int)
        # Converts the data to numeric
        c.analysis_remap_dtype(self.map_)
        c.analysis_label_order(self.map_)
        pdt.assert_series_equal(self.map_['years_on_team'],
                                pd.Series([1, 2, 2, 4],
                                          name='years_on_team',
                                          index=['Bitty', 'Ransom', 'Holster',
                                                 'Johnson']))

    def test_analysis_remap_null(self):
        self.c.missing = ['Striker']
        self.assertEqual(self.c.order, ["Striker", "D-man", "Goalie"])
        self.c.analysis_remap_null(self.map_)
        self.assertEqual(self.c.order, ['D-man', 'Goalie'])
        pdt.assert_series_equal(self.map_['position'],
                                pd.Series([np.nan, 'D-man', 'D-man', 'Goalie'],
                                          index=['Bitty', 'Ransom', 'Holster',
                                                 'Johnson'],
                                          name='position'))

    def test_analysis_remap_values_function(self):
        pdt.assert_series_equal(
            self.map_['position'],
            pd.Series(['Striker', 'D-man', 'D-man', 'Goalie'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='position')
            )

        def remap_(x):
            if x in {"D-man", "Goalie"}:
                return "Defense"
            elif x in {"Striker"}:
                return "Offense"
            else:
                return "Not on the team!"

        self.c.analysis_remap_values(self.map_, remap_)
        pdt.assert_series_equal(
            self.map_['position'],
            pd.Series(['Offense', 'Defense', 'Defense', 'Defense'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='position')
            )

    def test_analysis_remap_values_dict(self):
        pdt.assert_series_equal(
            self.map_['position'],
            pd.Series(['Striker', 'D-man', 'D-man', 'Goalie'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='position')
            )

        remap_ = {'D-man': 'Defense',
                  'Striker': "Offense"}

        self.c.analysis_remap_values(self.map_, remap_)
        pdt.assert_series_equal(
            self.map_['position'],
            pd.Series(['Offense', 'Defense', 'Defense', 'Goalie'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='position')
            )

    def test_validate_pass(self):
        self.c.validate(self.map_[self.name])

    def test_validate_fail(self):
        with self.assertRaises(ValueError):
            self.c.validate(self.map_['years_on_team'])


if __name__ == '__main__':
    main()
