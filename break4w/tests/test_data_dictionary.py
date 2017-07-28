from unittest import TestCase, main

from collections import OrderedDict

import datetime

import pandas as pd
import numpy as np
import numpy.testing as npt
import pandas.util.testing as pdt

# import statsmodels.api as sm

from break4w.data_dictionary import DataDictionary
from break4w.question import Question
from break4w.categorical import Categorical
from break4w.bool import Bool
from break4w.continous import Continous

# data_ = pd.DataFrame(sm.datasets.anes96.load().data)


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
        self.columns = [
            {
                'name': 'years_on_team',
                'description': ("How many years the player has been on SMH "
                                "during Bitty's frog year"),
                'dtype': int,
                'units': 'years',
            },
            {
                'name': 'team_captain',
                'description': 'Has the player been given a C or AC?',
                'missing': 'TBD',
            },
            {
                'name': 'position',
                'description': 'Where the player can normally be found on the'
                               ' ice',
                'dtype': str,
                'order': ["Striker", "D-man", "Goalie"],
                'extremes': ["Striker", "Goalie"],
            },
            {
                'name': 'nickname',
                'description': "the character's actual first name",
                'dtype': str,
            },
            ]
        self.types = ['continous', 'bool', 'categorical', 'question']
        self.empty = DataDictionary([], [])
        self.dictionary = DataDictionary(self.columns, self.types)

    # def test_init_real_data(self):
    #     self.assertTrue(isinstance(self.dictionary, OrderedDict))
    #     self.assertEqual(list(self.dictionary.keys()),
    #                      ['years_on_team', 'team_captain', 'position',
    #                       'nickname'])
    #     # Checks the question types. We're assuming here the question objects
    #     # have already been tested and it does its job... these are obstensibly
    #     # unit tests and not integration tests.
    #     self.assertTrue(isinstance(self.dictionary['years_on_team'],
    #                                Continous))
    #     self.assertTrue(isinstance(self.dictionary['position'],
    #                                Categorical))
    #     self.assertTrue(isinstance(self.dictionary['team_captain'],
    #                                Bool))
    #     self.assertTrue(isinstance(self.dictionary['nickname'],
    #                                Question))
    #     # Checks the log
    #     self.assertEqual(len(self.dictionary.log), 1)
    #     self.assertEqual(self.dictionary.log[0]['column'], None)
    #     self.assertEqual(self.dictionary.log[0]['command'],
    #                      'initialize the dictionary')
    #     self.assertEqual(self.dictionary.log[0]['transform_type'], None)
    #     self.assertEqual(self.dictionary.log[0]['transformation'], None)

    # def test_update_log(self):
    #     self.assertEqual(len(self.empty.log), 1)
    #     self.empty._update_log(command='check the dict',
    #                            column='all the columns')
    #     self.assertEqual(len(self.empty.log), 2)
    #     self.assertTrue(isinstance(self.empty.log[1], dict))
    #     self.assertTrue(isinstance(self.empty.log[1]['timestamp'],
    #                     datetime.datetime))
    #     self.assertEqual(self.empty.log[1]['command'], 'check the dict')
    #     self.assertEqual(self.empty.log[1]['column'], 'all the columns')
    #     self.assertEqual(self.empty.log[1]['transform_type'], None)
    #     self.assertEqual(self.empty.log[1]['transformation'], None)

    # def test_add_question_default(self):
    #     # Adds the `years_on_team` question.
    #     self.empty.add_question(self.columns[0], self.types[0])
    #     # Checks the question has been added and is a continous question.
    #     self.assertEqual(list(self.empty.keys()), ['years_on_team'])
    #     self.assertTrue(isinstance(self.empty['years_on_team'],
    #                                Continous))
    #     # Checks the log
    #     self.assertEqual(len(self.empty.log), 2)
    #     self.assertEqual(self.empty.log[1]['column'], 'years_on_team')
    #     self.assertEqual(self.empty.log[1]['command'], 'add column')
    #     self.assertEqual(self.empty.log[1]['transform_type'], None)
    #     self.assertEqual(self.empty.log[1]['transformation'],
    #                      'years_on_team was added to the dictionary')

    # def test_add_question_object_no_record(self):
    #     # Adds `years_on_team` as a continous quesiton.
    #     self.empty.add_question(Continous(**self.columns[0]), record=False)
    #     # Checks the record
    #     self.assertEqual(list(self.empty.keys()), ['years_on_team'])
    #     self.assertTrue(isinstance(self.empty['years_on_team'],
    #                     Continous))
    #     # Checks the log
    #     self.assertEqual(len(self.empty.log), 1)

    # def test_add_question_default_error(self):
    #     # Checks for the error
    #     with self.assertRaises(ValueError):
    #         self.dictionary.add_question(self.columns[0], self.types[0])
    #     # Checks the logging
    #     self.assertEqual(len(self.dictionary.log), 2)
    #     self.assertEqual(self.dictionary.log[1]['column'], 'years_on_team')
    #     self.assertEqual(self.dictionary.log[1]['command'], 'add column')
    #     self.assertEqual(self.dictionary.log[1]['transform_type'], 'error')
    #     self.assertEqual(self.dictionary.log[1]['transformation'],
    #                      'years_on_team already has a dictionary entry')

    # def test_add_question_no_check(self):
    #     self.dictionary.add_question(self.columns[0], self.types[0],
    #                                  check=False)
    #     # Checks the log
    #     self.assertEqual(len(self.dictionary.log), 2)
    #     self.assertEqual(self.dictionary.log[1]['column'], 'years_on_team')
    #     self.assertEqual(self.dictionary.log[1]['command'], 'add column')
    #     self.assertEqual(self.dictionary.log[1]['transform_type'], None)
    #     self.assertEqual(self.dictionary.log[1]['transformation'],
    #                      'years_on_team was added to the dictionary')

    # def test_get_question(self):
    #     test = self.dictionary.get_question('years_on_team')
    #     # Checks the returned value
    #     self.assertTrue(isinstance(test, Continous))
    #     self.assertEqual(test.name, 'years_on_team')
    #     # Checks the log
    #     self.assertEqual(len(self.dictionary.log), 2)
    #     self.assertEqual(self.dictionary.log[1]['column'], 'years_on_team')
    #     self.assertEqual(self.dictionary.log[1]['command'], 'get question')
    #     self.assertEqual(self.dictionary.log[1]['transform_type'], None)
    #     self.assertEqual(self.dictionary.log[1]['transformation'], None)

    # def test_get_question_error(self):
    #     with self.assertRaises(ValueError):
    #         self.empty.get_question('years_on_team')
    #     self.assertEqual(len(self.empty.log), 2)
    #     self.assertEqual(self.empty.log[1]['column'], 'years_on_team')
    #     self.assertEqual(self.empty.log[1]['command'], 'get question')
    #     self.assertEqual(self.empty.log[1]['transform_type'], 'error')
    #     self.assertEqual(self.empty.log[1]['transformation'],
    #                      'There is no entry for years_on_team')

    # def test_remove_question(self):
    #     self.assertEqual(
    #         list(self.dictionary.keys()),
    #         ['years_on_team', 'team_captain', 'position', 'nickname']
    #         )
    #     self.dictionary.drop_question('years_on_team')
    #     self.assertEqual(
    #         list(self.dictionary.keys()),
    #         ['team_captain', 'position', 'nickname']
    #         )

    # def test_update_question_error(self):
    #     self.assertFalse('years on team' in self.empty.keys())
    #     # Checks the error
    #     with self.assertRaises(ValueError):
    #         self.empty.update_question(update=Question(**self.columns[0]))
    #     # Checks the log
    #     self.assertEqual(len(self.empty.log), 2)
    #     run_log = self.empty.log[1]
    #     self.assertEqual(run_log['command'], 'update question')
    #     self.assertEqual(run_log['column'], 'years_on_team')
    #     self.assertEqual(run_log['transform_type'], 'error')
    #     self.assertEqual(run_log['transformation'],
    #                      'years_on_team is not a question in the current '
    #                      'dictionary.\nHave you tried adding the question?')

    # def test_update_question(self):
    #     # Checks the current state of the value
    #     self.assertEqual(self.dictionary['years_on_team'].blanks, None)
    #     with self.assertRaises(AttributeError):
    #         self.dictionary.semester_conversion

    #     # Update the dictionary
    #     update = {'blanks': 'not applicable', 'semester_conversion': 2,
    #               'log': ['this is a test']}
    #     self.dictionary.update_question(update, name='years_on_team')
    #     self.assertEqual(self.dictionary['years_on_team'].blanks,
    #                      'not applicable')
    #     self.assertEqual(self.dictionary['years_on_team'].semester_conversion,
    #                      2)
    #     # Checks the log
    #     self.assertEqual(len(self.dictionary.log), 2)
    #     run_log = self.dictionary.log[1]
    #     self.assertEqual(run_log['command'], 'update question')
    #     self.assertEqual(run_log['column'], 'years_on_team')
    #     self.assertEqual(run_log['transform_type'], 'update dictionary values')
    #     self.assertEqual(
    #         run_log['transformation'],
    #         'blanks : None > not applicable | semester_conversion : add > 2'
    #         )

    # def test_validate_question_order_pass(self):
    #     self.dictionary.validate_question_order(self.map_)
    #     # Checks the log
    #     self.assertEqual(len(self.dictionary.log), 2)
    #     run_log = self.dictionary.log[1]
    #     self.assertEqual(run_log['command'], 'Check columns')
    #     self.assertEqual(run_log['column'], None)
    #     self.assertEqual(run_log['transform_type'], None)
    #     self.assertEqual(
    #         run_log['transformation'],
    #         'The columns in the mapping file match the columns in '
    #         'the data dictionary.'
    #         )

    # def test_validate_question_order_different_cols_error(self):
    #     self.dictionary.drop_question('nickname')
    #     with self.assertRaises(ValueError):
    #         self.dictionary.validate_question_order(self.map_)
    #     self.assertEqual(len(self.dictionary.log), 3)
    #     run_log = self.dictionary.log[2]
    #     self.assertEqual(run_log['command'], 'Check columns')
    #     self.assertEqual(run_log['column'], None)
    #     self.assertEqual(run_log['transform_type'], None)
    #     self.assertEqual(
    #         run_log['transformation'],
    #         'There are 0 columns in the data dictionary '
    #         'not in the mapping file, and 1 from the mapping'
    #         ' file not in the data dictionary.'
    #         )

    # def test_validate_question_order_different_cols_error_verbose(self):
    #     self.dictionary.drop_question('nickname')
    #     with self.assertRaises(ValueError):
    #         self.dictionary.validate_question_order(self.map_, verbose=True)
    #     run_log = self.dictionary.log[2]
    #     self.assertEqual(
    #         run_log['transformation'],
    #         'There are 0 columns in the data dictionary '
    #         'not in the mapping file, and 1 from the mapping'
    #         ' file not in the data dictionary.\nIn the dictionary but not in '
    #         'the map: \n\n\nIn the map but not in the dictionary: \nnickname\n'
    #         '\n'
    #         )

    # def test_validate_question_check_order_error_true(self):
    #     self.map_ = self.map_[['nickname', 'years_on_team',
    #                            'team_captain', 'position']]
    #     with self.assertRaises(ValueError):
    #         self.dictionary.validate_question_order(self.map_, verbose=True)
    #     run_log = self.dictionary.log[1]
    #     self.assertEqual(
    #         run_log['transformation'],
    #         'The columns in the dictionary and map are not in the same order.'
    #         )

    # def test_validate_question_check_order_pass(self):
    #     self.map_ = self.map_[['nickname', 'years_on_team',
    #                            'team_captain', 'position']]
    #     self.dictionary.validate_question_order(self.map_, check_order=False)
    #     run_log = self.dictionary.log[1]
    #     self.assertEqual(run_log['command'], 'Check columns')
    #     self.assertEqual(run_log['column'], None)
    #     self.assertEqual(run_log['transform_type'], None)
    #     self.assertEqual(
    #         run_log['transformation'],
    #         'The columns in the mapping file match the columns in '
    #         'the data dictionary.'
    #         )

    # def test_validate_pass(self):
    #     kcolumns = pd.Series([None, 'years_on_team', 'team_captain',
    #                           'team_captain', 'position', 'position', None],
    #                          name='column')
    #     self.dictionary.validate(self.map_)
    #     # Checks the log
    #     self.assertEqual(len(self.dictionary.log), 7)
    #     log_ = pd.DataFrame(self.dictionary.log)
    #     self.assertTrue(np.all(log_['command'] == 'validate'))
    #     self.assertTrue(np.all(log_['transform_type'] == 'pass'))
    #     pdt.assert_series_equal(kcolumns, log_['column'])
    #     self.assertEqual(log_.loc[6, 'transformation'], 'All columns passed')

    def test_validate_error(self):
        # Sets up known series
        kcolumns = pd.Series([None, 'years_on_team', 'team_captain',
                              'team_captain', 'position', 'position', None],
                             name='column')
        kvalidate = pd.Series(['pass', 'error', 'pass', 'pass', 'pass',
                               'pass', 'error'],
                              name='transform_type')

        self.map_.loc['Johnson', 'years_on_team'] = \
            ('How do you measure a year? Is it really a year when it takes'
             ' 24 months to get an update?')
        self.dictionary.validate(self.map_)
        # with self.assertRaises(ValueError):
        #     self.dictionary.validate(self.map_)
        # # Checks the log
        # self.assertEqual(len(self.dictionary.log), 7)
        # log_ = pd.DataFrame(self.dictionary.log)
        # self.assertTrue(np.all(log_['command'] == 'validate'))
        # pdt.assert_series_equal(kvalidate, log_['transform_type'])
        # pdt.assert_series_equal(kcolumns, log_['column'])
        # self.assertEqual(log_.loc[6, 'transformation'], 'All columns passed')

if __name__ == '__main__':
    main()
