from unittest import TestCase, main

from collections import OrderedDict

import datetime

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt

from pandas.api.types import CategoricalDtype

from break4w.data_dictionary import DataDictionary
from break4w.question import Question
from break4w.categorical import Categorical
from break4w.bool import Bool
from break4w.continous import Continous


class DictionaryTest(TestCase):

    def setUp(self):
        self.map_ = pd.DataFrame([['1', '2', '2', '4'],
                                  ['TBD', 'True', 'True', 'False'],
                                  ['Striker', 'D-man', 'D-man', 'Goalie'],
                                  ['Eric', 'Adam', 'Justin', 'John']],
                                 index=['years_on_team', 'team_captain',
                                        'position', 'nickname'],
                                 columns=['Bitty', 'Ransom', 'Holster',
                                          'Johnson'],
                                 ).T
        self.map_['team_captain'] = self.map_['team_captain'].astype(str)
        self.map_['nickname'] = self.map_['nickname'].astype(str)
        self.map_['years_on_team'] = self.map_['years_on_team'].astype(float)
        cat_type = CategoricalDtype(["Striker", "D-man", "Goalie"], True)
        self.map_['position'] = self.map_['position'].astype(cat_type)

        self.columns = [
            {
                'name': 'years_on_team',
                'description': ("How many years the player has been on SMH "
                                "during Bitty's frog year"),
                'dtype': int,
                'units': 'years',
                'limits': [1, None],
            },
            {
                'name': 'team_captain',
                'dtype': bool,
                'description': 'Has the player been given a C or AC?',
                'missing': 'TBD',
            },
            {
                'name': 'position',
                'description': 'Where the player can normally be found on the'
                               ' ice',
                'dtype': str,
                'order': ["Striker", "D-man", "Goalie"],
            },
            {
                'name': 'nickname',
                'description': "the character's actual first name",
                'dtype': str,
            },
            ]
        self.types = ['continous', 'bool', 'categorical', 'question']
        self.empty = DataDictionary([], [])
        self.desc = 'Johnson doesnt know man, this is a weird study.'
        self.d = DataDictionary(self.columns, self.types, 
                                description=self.desc)

        self.var_desc = {
            'years_on_team': ("How many years the player has been on SMH "
                              "during Bitty's frog year"),
            'team_captain': ('Has the player been given a C or AC?'),
            'position': ('Where the player can normally be found on the ice'),
            'nickname': "the character's actual first name",
            }

    def test_init_real_data(self):
        self.assertTrue(isinstance(self.d, OrderedDict))
        self.assertEqual(list(self.d.keys()),
                         ['years_on_team', 'team_captain', 'position',
                          'nickname'])
        # Checks the question types. We're assuming here the question objects
        # have already been tested and it does its job... these are obstensibly
        # unit tests and not integration tests.
        self.assertTrue(isinstance(self.d['years_on_team'], Continous))
        self.assertTrue(isinstance(self.d['position'], Categorical))
        self.assertTrue(isinstance(self.d['team_captain'], Bool))
        self.assertTrue(isinstance(self.d['nickname'], Question))
        # Checks the log
        self.assertTrue(isinstance(self.d.log, list))
        self.assertEqual(len(self.d.log), 0)

        # Checks the description
        self.assertEqual(self.d.description, self.desc)

    def test_init_no_desc(self):
        d = DataDictionary(self.columns, self.types)
        self.assertEqual(d.description, '')

    def test_init_desc_length(self):
        d = ('Check, Please! is a 2013 webcomic written and '
             'illustrated by Ngozi Ukazu. The webcomic follows '
             'vlogger and figure-turned-ice hockey skater Eric '
             '"Bitty" Bittle as he deals with hockey culture in '
             'college, as well as his identity as a gay man.')
        with self.assertRaises(ValueError):
            DataDictionary(self.columns, self.types, d)

    def test_str_(self):
        known = ('Data Dictionary with 4 columns\n'
                 '\tJohnson doesnt know man, this is a weird study.\n'
                 '----------------------------------------------------------'
                 '----------------------\n'
                 'years_on_team (Continous)\n'
                 'team_captain (Bool)\n'
                 'position (Categorical)\n'
                 'nickname (Question)\n'
                 '----------------------------------------------------------'
                 '----------------------')
        test = self.d.__str__()

        self.assertEqual(known, test)

    def test_update_log(self):
        self.assertEqual(len(self.empty.log), 0)
        self.empty._update_log(command='check the dict',
                               column='all the columns')
        self.assertEqual(len(self.empty.log), 1)
        log_ = self.empty.log[0]
        self.assertTrue(isinstance(log_, dict))
        self.assertTrue(isinstance(log_['timestamp'], datetime.datetime))
        self.assertEqual(log_['command'], 'check the dict')
        self.assertEqual(log_['column'], 'all the columns')
        self.assertEqual(log_['transform_type'], None)
        self.assertEqual(log_['transformation'], None)

    def test_pull_question_log(self):
        with self.assertRaises(NotImplementedError):
            self.d._pull_question_log()

    def test_add_question_default(self):
        # Adds the `years_on_team` question.
        self.empty.add_question(self.columns[0], self.types[0])
        # Checks the question has been added and is a continous question.
        self.assertEqual(list(self.empty.keys()), ['years_on_team'])
        self.assertTrue(isinstance(self.empty['years_on_team'],
                                   Continous))
        # Checks the log
        self.assertEqual(len(self.empty.log), 1)
        log_ = self.empty.log[0]
        self.assertEqual(log_['column'], 'years_on_team')
        self.assertEqual(log_['command'], 'add column')
        self.assertEqual(log_['transform_type'], None)
        self.assertEqual(log_['transformation'],
                         'years_on_team was added to the dictionary')

    def test_add_question_series(self):
        var_ = pd.Series({
            'name': 'years_on_team',
            'description': ("How many years the player has been on SMH "
                            "during Bitty's frog year"),
            'units': 'years',
            'dtype': 'int',
            'order': '1 | None',
            })
        self.d.drop_question('years_on_team')
        self.d.add_question(var_, question_type='continous')
        test_ = self.d['years_on_team']
        self.assertTrue(isinstance(test_, Continous))
        self.assertTrue(test_.limits, [1, None])
        self.assertFalse('order' in test_.__dict__)

    def test_add_question_object_no_record(self):
        # Adds `years_on_team` as a continous quesiton.
        self.empty.add_question(Continous(**self.columns[0]), record=False)
        # Checks the record
        self.assertEqual(list(self.empty.keys()), ['years_on_team'])
        self.assertTrue(isinstance(self.empty['years_on_team'],
                        Continous))
        # Checks the log
        self.assertEqual(len(self.empty.log), 0)

    def test_add_question_default_error(self):
        # Checks for the error
        with self.assertRaises(ValueError):
            self.d.add_question(self.columns[0], self.types[0])
        # Checks the logging
        self.assertEqual(len(self.d.log), 1)
        log_ = self.d.log[0]
        self.assertEqual(log_['column'], 'years_on_team')
        self.assertEqual(log_['command'], 'add column')
        self.assertEqual(log_['transform_type'], 'error')
        self.assertEqual(log_['transformation'],
                         'years_on_team already has a dictionary entry')

    def test_add_question_no_check(self):
        self.d.add_question(self.columns[0], self.types[0],
                                     check=False)
        # Checks the log
        self.assertEqual(len(self.d.log), 1)
        log_ = self.d.log[0]
        self.assertEqual(log_['column'], 'years_on_team')
        self.assertEqual(log_['command'], 'add column')
        self.assertEqual(log_['transform_type'], None)
        self.assertEqual(log_['transformation'],
                         'years_on_team was added to the dictionary')

    def test_add_question_error(self):
        with self.assertRaises(ValueError):
            self.d.add_question('')

    def test_get_question(self):
        test = self.d.get_question('years_on_team')
        # Checks the returned value
        self.assertTrue(isinstance(test, Continous))
        self.assertEqual(test.name, 'years_on_team')
        # Checks the log
        self.assertEqual(len(self.d.log), 1)
        log_ = self.d.log[0]
        self.assertEqual(log_['column'], 'years_on_team')
        self.assertEqual(log_['command'], 'get question')
        self.assertEqual(log_['transform_type'], None)
        self.assertEqual(log_['transformation'], None)

    def test_get_question_error(self):
        with self.assertRaises(ValueError):
            self.empty.get_question('years_on_team')
        self.assertEqual(len(self.empty.log), 1)
        log_ = self.empty.log[0]
        self.assertEqual(log_['column'], 'years_on_team')
        self.assertEqual(log_['command'], 'get question')
        self.assertEqual(log_['transform_type'], 'error')
        self.assertEqual(log_['transformation'],
                         'There is no entry for years_on_team')

    def test_remove_question(self):
        self.assertEqual(
            list(self.d.keys()),
            ['years_on_team', 'team_captain', 'position', 'nickname']
            )
        self.d.drop_question('years_on_team')
        self.assertEqual(
            list(self.d.keys()),
            ['team_captain', 'position', 'nickname']
            )
        self.assertEqual(len(self.d.log), 1)
        log_ = self.d.log[0]
        self.assertEqual(log_['column'], 'years_on_team')
        self.assertEqual(log_['command'], 'remove question')
        self.assertEqual(log_['transform_type'], None)
        self.assertEqual(log_['transformation'], None)

    def test_update_question_error(self):
        self.assertFalse('years on team' in self.empty.keys())
        # Checks the error
        with self.assertRaises(ValueError):
            self.empty.update_question(update=Question(**self.columns[0]))
        # Checks the log
        self.assertEqual(len(self.empty.log), 1)
        log_ = self.empty.log[0]
        self.assertEqual(log_['command'], 'update question')
        self.assertEqual(log_['column'], 'years_on_team')
        self.assertEqual(log_['transform_type'], 'error')
        self.assertEqual(log_['transformation'],
                         'years_on_team is not a question in the current '
                         'dictionary.\nHave you tried adding the question?')

    def test_update_question(self):
        # Checks the current state of the value
        self.assertEqual(self.d['years_on_team'].blanks, None)
        with self.assertRaises(AttributeError):
            self.d.semester_conversion

        # Update the dictionary
        update = {'blanks': 'not applicable', 'semester_conversion': 2,
                  'log': ['this is a test']}
        self.d.update_question(update, name='years_on_team')
        self.assertEqual(self.d['years_on_team'].blanks,
                         'not applicable')
        self.assertEqual(self.d['years_on_team'].semester_conversion,
                         2)
        # Checks the log
        self.assertEqual(len(self.d.log), 1)
        log_ = self.d.log[0]
        self.assertEqual(log_['command'], 'update question')
        self.assertEqual(log_['column'], 'years_on_team')
        self.assertEqual(log_['transform_type'], 'update dictionary values')
        self.assertEqual(
            log_['transformation'],
            'blanks : None > not applicable | semester_conversion : add > 2'
            )

    def test_validate_question_order_pass(self):
        self.d._validate_question_order(self.map_)
        # Checks the log
        self.assertEqual(len(self.d.log), 1)
        log_ = self.d.log[0]
        self.assertEqual(log_['command'], 'validate')
        self.assertEqual(log_['column'], None)
        self.assertEqual(log_['transform_type'], 'pass')
        self.assertEqual(
            log_['transformation'],
            'The columns in the mapping file match the columns in '
            'the data dictionary.'
            )

    def test_validate_question_order_different_cols_error(self):
        self.d.drop_question('nickname')
        with self.assertRaises(ValueError):
            self.d._validate_question_order(self.map_)
        self.assertEqual(len(self.d.log), 2)
        log_ = self.d.log[1]
        self.assertEqual(log_['command'], 'validate')
        self.assertEqual(log_['column'], None)
        # self.assertEqual(log_['transform_type'], 'fail')
        self.assertEqual(
            log_['transformation'],
            'There are 0 columns in the data dictionary '
            'not in the mapping file, and 1 from the mapping'
            ' file not in the data dictionary.'
            )

    def test_validate_question_order_different_cols_error_verbose(self):
        self.d.drop_question('nickname')
        with self.assertRaises(ValueError):
            self.d._validate_question_order(self.map_, verbose=True)
        log_ = self.d.log[1]
        self.assertEqual(
            log_['transformation'],
            'There are 0 columns in the data dictionary '
            'not in the mapping file, and 1 from the mapping'
            ' file not in the data dictionary.\nIn the map but not in the '
            'dictionary:\n\tnickname\n'
            )

    def test_validate_question_check_order_error_true(self):
        self.map_ = self.map_[['nickname', 'years_on_team',
                               'team_captain', 'position']]
        with self.assertRaises(ValueError):
            self.d._validate_question_order(self.map_, verbose=True)
        log_ = self.d.log[0]
        self.assertEqual(
            log_['transformation'],
            'The columns in the dictionary and map are not in the same order.'
            )

    def test_validate_question_check_order_pass(self):
        self.map_ = self.map_[['nickname', 'years_on_team',
                               'team_captain', 'position']]
        self.d._validate_question_order(self.map_, check_order=False)
        log_ = self.d.log[0]
        self.assertEqual(log_['command'], 'validate')
        self.assertEqual(log_['column'], None)
        self.assertEqual(log_['transform_type'], 'pass')
        self.assertEqual(
            log_['transformation'],
            'The columns in the mapping file match the columns in '
            'the data dictionary.'
            )

    def test_validate_question_error_not_record(self):
        self.map_ = self.map_[['nickname', 'years_on_team',
                               'team_captain', 'position']]
        with self.assertRaises(ValueError):
            self.d._validate_question_order(self.map_, record=False)
        self.assertEqual(len(self.d.log), 0)

    def test_validate_pass(self):
        kcolumns = pd.Series([None, 'years_on_team','team_captain', 
                              'position', None],
                             name='column')
        self.d.validate(self.map_)
        # Checks the log
        self.assertEqual(len(self.d.log), 5)
        log_ = pd.DataFrame(self.d.log)
        self.assertTrue(np.all(log_['command'] == 'validate'))
        self.assertTrue(np.all(log_['transform_type'] == 'pass'))
        pdt.assert_series_equal(kcolumns, log_['column'])
        self.assertEqual(log_.iloc[-1]['transformation'], 
                         'All columns passed')

    def test_validate_error(self):
        # Sets up known series
        kcolumns = pd.Series([None, 'years_on_team', 'team_captain',
                              'position', None],
                             name='column')
        kvalidate = pd.Series(['pass', 'error', 'pass', 'pass', 'error'],
                              name='transform_type')

        self.map_.loc['Johnson', 'years_on_team'] = \
            ('How do you measure a year? Is it really a year when it takes'
             ' 24 months to get an update?')
        # self.d.validate(self.map_)
        with self.assertRaises(ValueError):
            self.d.validate(self.map_)
        # Checks the log
        self.assertEqual(len(self.d.log), 5)
        log_ = pd.DataFrame(self.d.log)
        self.assertTrue(np.all(log_['command'] == 'validate'))
        pdt.assert_series_equal(kvalidate, log_['transform_type'])
        pdt.assert_series_equal(kcolumns, log_['column'])
        self.assertEqual(log_.iloc[-1]['transformation'], 
            'There were issues with the following columns:\nyears_on_team\n'
            'Please See the log for more details.')

    def test_to_pandas_stata(self):
        known_vars = {x['name']: x['description'] for x in self.columns}
        test_desc, test_vars = self.d.to_pandas_stata()
        self.assertEqual(test_desc, self.desc)
        self.assertEqual(known_vars, test_vars)

    def test_to_dataframe(self):
        self.d['nickname'].var_labels = {1: 'Bitty', 
                                                  2: 'Ransom', 
                                                  3: 'Holster'}


        columns = ['years_on_team', 'team_captain', 'position', 'nickname']
        known = pd.DataFrame(
            data=[columns,
                  [c['description'] for c in self.columns],
                  ['int', 'bool', 'str', 'str'],
                  ['Continous', 'Bool', 'Categorical', 'Question'],
                  ['Years On Team', 'Team Captain', 'Position', 'Nickname'],
                  ['years', np.nan, np.nan, np.nan],
                  [np.nan, 'TBD', np.nan, np.nan],
                  ['1 | None', 'false | true', 'Striker | D-man | Goalie', 
                   '1=Bitty | 2=Ransom | 3=Holster'],
                  [np.nan, 'false', 'Striker', np.nan],
                  ],
            index=['name', 'description', 'dtype', 'type', 'clean_name', 
                   'units', 'missing', 'order', 'ref_value']
            ).T
        known.set_index('name', inplace=True)
        known = known[['description', 'dtype', 'type', 'clean_name', 'order', 
                       'units', 'missing', 'ref_value']]
        test = self.d.to_dataframe()

        pdt.assert_frame_equal(known, self.d.to_dataframe())

    def test_to_dataframe_clean(self):
        self.d['nickname'].var_labels = {1: 'Bitty', 
                                         2: 'Ransom', 
                                         3: 'Holster'}

        columns = ['years_on_team', 'team_captain', 'position', 'nickname']
        known = pd.DataFrame(
            data=[columns,
                  [c['description'] for c in self.columns],
                  ['int', 'bool', 'str', 'str'],
                  ['Continous', 'Bool', 'Categorical', 'Question'],
                  ['years', np.nan, np.nan, np.nan],
                  [np.nan, 'TBD', np.nan, np.nan],
                  ['1 | None', 'false | true', 'Striker | D-man | Goalie', 
                   '1=Bitty | 2=Ransom | 3=Holster'],
                  ],
            index=['name', 'description', 'dtype', 'type',
                   'units', 'missing', 'order']
            ).T
        known.set_index('name', inplace=True)
        known = known[['description', 'type', 'dtype', 'order', 'units', 
                       'missing']]
        test = self.d.to_dataframe(True, True)
        pdt.assert_frame_equal(known, test)

    def test_read_dataframe(self):
        columns = ['years_on_team', 'team_captain', 'position', 'nickname']
        df_ = pd.DataFrame(
            data=[columns,
                  [c['description'] for c in self.columns],
                  ['int', 'str', 'str', 'str'],
                  ['Continous', 'Bool', 'Categorical', 'Question'],
                  ['Years On Team', 'Team Captain', 'Position', 'Nickname'],
                  ['years', np.nan, np.nan, np.nan],
                  [np.nan, 'TBD', np.nan, np.nan],
                  ['1 | None', 'false | true', 'Striker | D-man | Goalie', 
                   np.nan],
                  [np.nan, np.nan, np.nan, '1=Bitty | 2=Ransom | 3=Holster'],
                  [np.nan, 'false', 'Striker', np.nan],
                  ],
            index=['name', 'description', 'dtype', 'type', 'clean_name', 
                   'units', 'missing', 'order', 'var_labels', 'ref_value']
            ).T
        df_.set_index('name', inplace=True)
        test_ = DataDictionary.read_dataframe(df_, description='test?')

        # Checks the initiation
        self.assertEqual(test_.description, 'test?')
        # print(test_)
        self.assertEqual(list(test_.keys()),
                         ['years_on_team', 'team_captain', 'position',
                          'nickname'])
        # Checks the question types. We're assuming here the question objects
        # have already been tested and it does its job... these are 
        # obstensibly unit tests and not integration tests.
        self.assertTrue(isinstance(test_['years_on_team'], Continous))
        self.assertTrue(isinstance(test_['position'], Categorical))
        self.assertTrue(isinstance(test_['team_captain'], Bool))
        self.assertTrue(isinstance(test_['nickname'], Question))

    def test_roundtrip(self):
        known_ = self.d
        known_['team_captain'].dtype = str
        var_ = known_.to_dataframe()
        desc = self.d.description
        test_ = self.d.read_dataframe(var_, description=desc)

        self.assertEqual(known_.description, test_.description)
        self.assertEqual(known_.keys(), test_.keys())
        self.assertEqual(
            {k: v for k, v in known_.__dict__.items()},
            {k: v for k, v in test_.__dict__.items()}
            )
        for name_, col_ in known_.items():
            self.assertEqual(col_.__dict__, test_[name_].__dict__)

    # def test_parse_col_props(self):
    #     ser_ = self.map_['team_captain'].astype(str)
    #     (dtype, values, counts, true_check, false_check, other_vals) = \
    #         self.d._parse_col_props(ser_)
    #     self.assertEqual(dtype, str)
    #     npt.assert_array_equal(np.array(values), 
    #                            np.array(['TBD', 'True', 'False']))
    #     self.assertEqual(counts, 3)
    #     self.assertTrue(true_check)
    #     self.assertTrue(false_check)
    #     npt.assert_array_equal(np.array(other_vals), np.array(['TBD']))

    # def test_parse_col_mixed_bool(self):
    #     ser_ = pd.Series(['TBD', True, True, False])
    #     (dtype, values, counts, true_check, false_check, other_vals) = \
    #         self.d._parse_col_props(ser_)
    #     self.assertEqual(dtype, bool)
    #     self.assertTrue(np.all([t == v for t, v in 
    #                             zip(*(values, ['TBD', True, False]))]))
    #     self.assertEqual(counts, 3)
    #     self.assertFalse(true_check)
    #     self.assertFalse(false_check)

    # def test_parse_col_mixed(self):
    #     ser_ = pd.Series(['TBD', 1, 1, 0])
    #     (dtype, values, counts, true_check, false_check, other_vals) = \
    #         self.d._parse_col_props(ser_)
    #     self.assertEqual(dtype, object)
    #     self.assertTrue(np.all([t == v for t, v in 
    #                             zip(*(values, ['TBD', 1, 0]))]))
    #     self.assertEqual(counts, 3)
    #     self.assertFalse(true_check)
    #     self.assertFalse(false_check)

    # def test_part_col_dtype(self):
    #     ser_ = pd.to_datetime(['2013-4-7', '2013-4-12', '2013-6-12'])
    #     (dtype, values, counts, true_check, false_check, other_vals) = \
    #         self.d._parse_col_props(ser_)
    #     self.assertEqual(dtype, datetime.datetime)
    #     self.assertTrue(np.all([t == v for t, v in 
    #                             zip(*(values, ser_.values))]))
    #     self.assertEqual(counts, 3)
    #     self.assertFalse(true_check)
    #     self.assertFalse(false_check)

    # def test_infer_type_cat_given(self):
    #     types = {'position': 'categorical', 'years_on_team': 'continous'}
    #     self.assertEqual(
    #         self.d._infer_type('position', self.map_['position'], 
    #                            categories=types),
    #         'categorical'
    #          )

    # def test_infer_type_no_infer_no_cat(self):
    #     self.assertEqual(
    #         self.d._infer_type('position', self.map_['position']),
    #         'question'
    #         )

    # def test_infer_type_categorical_dtype(self):
    #     self.assertEqual(
    #         self.d._infer_type('position', self.map_['position'], infer=True),
    #         'categorical'
    #         )

    # def test_infer_type_bool_type(self):
    #     team_captain = \
    #         self.map_['team_captain'].replace({'TBD': np.nan, 
    #                                            'True': True, 
    #                                            'False': False})
    #     self.assertEqual(
    #         self.d._infer_type('team_captain', team_captain, infer=True),
    #         'bool'
    #         )

    # def test_infer_type_bool_str(self):
    #     self.assertEqual(
    #         'bool',
    #         self.d._infer_type('team_captain', self.map_['team_captain'], 
    #                            infer=True)
    #         )
    
    # def test_infer_type_int_categorical(self):
    #     years_on_team = self.map_['years_on_team'].astype(int)
    #     self.assertEqual(
    #         self.d._infer_type('years_on_team', years_on_team, True),
    #         'categorical'
    #         )

    # def test_infer_type_continous(self):
    #     years_on_team = self.map_['years_on_team'].astype(float)
    #     self.assertEqual(
    #         self.d._infer_type('years_on_team', years_on_team, True, 
    #                            max_cats=1),
    #         'continous'
    #         )

    # def test_infer_type_no_hits(self):
    #     self.assertEqual(
    #         self.d._infer_type('nickname', self.map_['nickname'], 
    #                             True, max_cats=1),
    #         'question'
    #         )

    # def test_infer_type_dates(self):
    #     pub_dates = pd.to_datetime(['2013-4-7', '2013-4-12', '2013-6-12'])
    #     self.assertEqual(
    #         self.d._infer_type('pub_date', pub_dates, True), 'continous'
    #         )

    # def test_describe_stata_col_ordered(self):
    #     test = self.d._describe_stata_col(col_='position', 
    #                                       ser_=self.map_['position'],
    #                                       var_desc=self.var_desc,
    #                                       col_type='categorical'
    #                                       )
    #     self.assertEqual(
    #         list(test.keys()), 
    #         ['name', 'description', 'dtype', 'order', 'var_labels']
    #         )
    #     self.assertEqual(test['name'], 'position')
    #     self.assertEqual(
    #         test['description'], 
    #         'Where the player can normally be found on the ice'
    #         )
    #     self.assertEqual(test['dtype'], str)
    #     self.assertEqual(
    #         test['var_labels'], 
    #         {1: 'Striker', 2: 'D-man', 3: 'Goalie'})
    #     npt.assert_array_equal(np.array(test['order']),
    #         np.array(["Striker", "D-man", "Goalie"])
    #         )



if __name__ == '__main__':
    main()
