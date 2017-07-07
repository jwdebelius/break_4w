from unittest import TestCase, main

import pandas as pd
import numpy as np
import numpy.testing as npt
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

    def test_analysis_apply_conversion_logged(self):
        kseries = pd.Series(data=['Offense', 'Defense', 'Defense', 'Defense'],
                            index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                            name='position')

        # Sets up a function to adjust the data
        def remap_(x):
            if x in {"D-man", "Goalie"}:
                return "Defense"
            elif x in {"Striker"}:
                return "Offense"
            else:
                return "Not on the team!"

        self.c.analysis_apply_conversion(self.map_, remap_, 'condense groups')
        pdt.assert_series_equal(self.map_['position'], kseries)
        self.assertEqual(self.c.order, ["Offense", "Defense"])
        log_ = self.c.log[0]
        self.assertEqual(log_['command'], 'transformation')
        self.assertEqual(log_['transform_type'], 'condense groups')
        self.assertEqual(log_['transformation'],
                         'Striker >>> Offense | D-man >>> Defense | '
                         'Goalie >>> Defense')

    def test_analysis_apply_conversion_dict_no_log(self):
        kseries = pd.Series(data=['Not yet', 'AC', 'AC', np.nan],
                            index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                            name='team_captain'
                            )
        c = Categorical(name='team_captain',
                        description='Has the player been given a C or AC?',
                        order=['True', 'False', 'TBD'],
                        dtype=bool,
                        )
        mapping = {'TBD': 'Not yet',
                   'True': 'AC',
                   'False': np.nan}

        c.analysis_apply_conversion(self.map_, mapping, None, False)
        pdt.assert_series_equal(self.map_['team_captain'], kseries)
        self.assertEqual(c.order, ['AC', 'Not yet'])
        self.assertEqual(len(c.log), 0)

    def test_analysis_convert_to_label(self):
        name_mapping = {1: 'Freshman',
                        2: 'Sophomore',
                        3: 'Junior',
                        4: 'Senior'}
        c = Categorical(name='years_on_team',
                        description="Time in SMH in Bitty's frog year",
                        order=['1', '2', '3', '4'],
                        dtype=int,
                        name_mapping=name_mapping,
                        )
        kseries = pd.Series(['Freshman', 'Sophomore', 'Sophomore', 'Senior'],
                            index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                            name='years_on_team')

        c.analysis_remap_dtype(self.map_)
        c.analysis_convert_to_label(self.map_)

        self.assertEqual(c.order, ['Freshman', 'Sophomore',
                                   'Junior', 'Senior'])
        pdt.assert_series_equal(self.map_['years_on_team'], kseries)

        log_ = c.log[1]
        self.assertEqual(log_['command'], 'transformation')
        self.assertEqual(log_['transform_type'], 'convert code to label')
        self.assertEqual(log_['transformation'],
                         '1 >>> Freshman | 2 >>> Sophomore | 3 >>> Junior | '
                         '4 >>> Senior')
        self.assertEqual(c.numeric_mapping,
                         {'Freshman': 1, 'Sophomore': 2,
                          'Junior': 3, 'Senior': 4}
                         )

    def test_analysis_convert_to_numeric(self):
        self.assertEqual(self.c.name_mapping, None)
        self.assertEqual(self.c.numeric_mapping, None)

        self.c.analysis_convert_to_numeric(self.map_)
        known = pd.Series(data=[0, 1, 1, 2],
                          index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                          name='position',
                          )
        pdt.assert_series_equal(
            self.map_['position'], known)
        self.assertEqual(self.c.order, [0, 1, 2])
        self.assertEqual(self.c.log[0]['transformation'],
                         'Striker >>> 0 | D-man >>> 1 | Goalie >>> 2')
        self.assertEqual(self.c.name_mapping,
                         {0: 'Striker', 1: 'D-man', 2: 'Goalie'})
        self.assertEqual(self.c.numeric_mapping,
                         {'Striker': 0, 'D-man': 1, 'Goalie': 2})

    def test_analysis_convert_to_numeric_number(self):
        c = Categorical(name='years_on_team',
                        description="Time in SMH in Bitty's frog year",
                        order=['1', '2', '3', '4'],
                        dtype=int
                        )

        c.analysis_remap_dtype(self.map_)
        c.analysis_convert_to_numeric(self.map_)
        pdt.assert_series_equal(self.map_['years_on_team'],
                                pd.Series([1, 2, 2, 4], name='years_on_team',
                                          index=['Bitty', 'Ransom', 'Holster',
                                                 'Johnson']))
        self.assertEqual(c.order, [1, 2, 3, 4])

    def test_analysis_drop_infrequent(self):
        self.assertTrue(set(self.map_[self.c.name].unique()),
                        {'Striker', 'D-man'})
        self.c.frequency_cutoff = 1
        self.c.analysis_drop_infrequent(self.map_)
        self.assertEqual(set(self.map_[self.c.name].unique()),
                         {'D-man', np.nan})
        self.assertEqual(self.c.order, ['D-man'])
        log_ = self.c.log[0]
        self.assertEqual(log_['command'], 'drop infrequent values')
        self.assertEqual(log_['transform_type'], 'drop')
        self.assertEqual(log_['transformation'], 'below 1: Goalie | Striker')

    def test_analysis_remap_dtype_error(self):
        c = Categorical(name='team_captain',
                        description='Has the player been given a C or AC?',
                        order=['True', 'False'],
                        dtype=bool,
                        )
        with self.assertRaises(TypeError):
            c.analysis_remap_dtype(self.map_)
        log_ = c.log[0]
        self.assertEqual(log_['command'], 'transformation')
        self.assertEqual(log_['transform_type'], 'cast data type')
        self.assertEqual(log_['transformation'], 'could not convert to bool')

    def test_analysis_remap_dtype(self):
        self.c.analysis_remap_dtype(self.map_)
        log_ = self.c.log[0]
        self.assertEqual(log_['command'], 'transformation')
        self.assertEqual(log_['transform_type'], 'cast data type')
        self.assertEqual(log_['transformation'], 'convert to str')

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
        log_ = self.c.log[0]
        self.assertEqual(log_['command'], 'correct null values')
        self.assertEqual(log_['transform_type'], 'drop')
        self.assertEqual(log_['transformation'], 'Striker')

    def test_analysis_remove_ambigious_str(self):
        c = Categorical(name='team_captain',
                        description='who has the C or AC',
                        dtype=str,
                        order=['True', 'False'],
                        ambiguous_values='TBD'
                        )
        c.analyis_remove_ambiguious(self.map_)

        # Checks the remapping
        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series([np.nan, 'True', 'True', 'False'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )
        self.assertEqual(c.order, ['True', 'False'])
        log_ = c.log[0]
        self.assertEqual(log_['command'], 'remove ambigious values')
        self.assertEqual(log_['transform_type'], 'drop')
        self.assertEqual(log_['transformation'], 'TBD')

    def test_analysis_remove_ambigious_list(self):
        c = Categorical(name='team_captain',
                        description='who has the C or AC',
                        dtype=str,
                        order=['True', 'False'],
                        ambiguous_values=['TBD']
                        )
        c.analyis_remove_ambiguious(self.map_)
        # Checks the remapping
        pdt.assert_series_equal(
            self.map_['team_captain'],
            pd.Series([np.nan, 'True', 'True', 'False'],
                      index=['Bitty', 'Ransom', 'Holster', 'Johnson'],
                      name='team_captain')
            )
        self.assertEqual(c.order, ['True', 'False'])
        log_ = c.log[0]
        self.assertEqual(log_['command'], 'remove ambigious values')
        self.assertEqual(log_['transform_type'], 'drop')
        self.assertEqual(log_['transformation'], 'TBD')

    def test_analysis_remove_ambigious_nan(self):
        # Sets the ambigious value as goalie
        self.c.ambiguous_values = {'Goalie'}
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
        self.assertEqual(self.c.order, ['Striker', 'D-man'])
        log_ = self.c.log[0]
        self.assertEqual(log_['command'], 'remove ambigious values')
        self.assertEqual(log_['transform_type'], 'drop')
        self.assertEqual(log_['transformation'], 'Goalie')

    def test_validate_dtype_fail(self):
        self.c.dtype = bool
        with self.assertRaises(TypeError):
            self.c.validate(self.map_)
        log_entry = self.c.log[0]
        self.assertEqual(log_entry['command'], 'validate')
        self.assertEqual(log_entry['transform_type'], 'error')
        self.assertEqual(log_entry['transformation'],
                         'the data cannot be cast to bool')

    def test_validate_fail(self):
        self.c.name = 'years_on_team'
        with self.assertRaises(ValueError):
            self.c.validate(self.map_)
        log_entry = self.c.log[0]
        self.assertEqual(log_entry['command'], 'validate')
        self.assertEqual(log_entry['transform_type'], 'error')
        self.assertEqual(log_entry['transformation'],
                         'Data can be cast to str\n'
                         'The following are not valid values: 1 | 2 | 4'
                         )

    def test_validate_pass(self):
        self.c.validate(self.map_)
        log_entry = self.c.log[0]
        self.assertEqual(log_entry['command'], 'validate')
        self.assertEqual(log_entry['transform_type'], 'pass')
        self.assertEqual(log_entry['transformation'],
                         'Data can be cast to str\n'
                         'All values in the column were valid.'
                         )

if __name__ == '__main__':
    main()
