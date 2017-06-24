from unittest import TestCase, main

import datetime

import pandas as pd
import numpy as np
import numpy.testing as npt
import pandas.util.testing as pdt

from break4w.question import Question, _remap_dtype


class QuestionTest(TestCase):

    def setUp(self):
        self.name = 'player_name'
        self.description = 'Samwell Hockey Players'
        self.dtype = str

        self.map_ = pd.DataFrame([['Bitty', 'Ransom', 'Holster'],
                                  ['2', '4', '4'],
                                  ['False', 'True', 'True']],
                                 index=['player_name', 'years_on_team',
                                        'team_captain']).T
        self.q = Question(name=self.name,
                          description=self.description,
                          dtype=self.dtype,
                          free_response=True,
                          )

    def test_init_default(self):
        self.assertEqual(self.name, self.q.name)
        self.assertEqual(self.description, self.q.description)
        self.assertEqual(self.dtype, self.q.dtype)
        self.assertEqual('Player Name', self.q.clean_name)
        self.assertEqual('Question', self.q.type)
        self.assertTrue(self.q.free_response)
        self.assertFalse(self.q.mimarks)
        self.assertFalse(self.q.ontology)
        self.assertEqual(self.q.missing,
                         {'not applicable', 'missing: not provided',
                          'missing: not collected', 'missing: restricted',
                          'not provided', 'not collected', 'restricted'})
        self.assertEqual(self.q.colormap, None)
        self.assertEqual(self.q.blanks, None)
        self.assertEqual(self.q.log, [])
        self.assertEqual(self.q.source_columns, [])
        self.assertEqual(self.q.derivative_columns, [])
        self.assertEqual(self.q.notes, None)

    def test_init_source_derivative_list(self):
        q = Question(name=self.name,
                     description=self.description,
                     dtype=self.dtype,
                     source_columns=['SMH'],
                     derivative_columns=['next_step']
                     )
        self.assertEqual(q.source_columns, ['SMH'])
        self.assertEqual(q.derivative_columns, ['next_step'])

    def test_init_error_name(self):
        with self.assertRaises(TypeError):
            Question(name=1,
                     description=self.description,
                     dtype=self.dtype,
                     )

    def test_init_error_description(self):
        with self.assertRaises(TypeError):
            Question(name=self.name,
                     description=self.dtype,
                     dtype=self.dtype,
                     )

    def test_init_error_dtype(self):
        with self.assertRaises(TypeError):
            Question(name=self.name,
                     description=self.description,
                     dtype=self.description,
                     )

    def test_init_error_clean_name(self):
        with self.assertRaises(TypeError):
            Question(name=self.name,
                     description=self.description,
                     dtype=self.dtype,
                     clean_name=self.dtype
                     )

    def test_init_clean_name_missing_str(self):
        q = Question(name=self.name,
                     description=self.description,
                     dtype=self.dtype,
                     clean_name='Player',
                     missing='Bitty')
        self.assertEqual(q.clean_name, 'Player')
        self.assertEqual(q.missing, set(['Bitty']))

    def test_init_missing_list(self):
        q = Question(name=self.name,
                     description=self.description,
                     dtype=self.dtype,
                     missing=['Bitty'])
        self.assertEqual(q.missing, set(['Bitty']))

    def test_update_log(self):
        self.assertEqual(len(self.q.log), 0)
        self.q._update_log(
            command='dibs',
            transform_type='replace',
            transformation='metaphysical goalie johnson > Bitty'
            )
        self.assertEqual(len(self.q.log), 1)
        log_ = self.q.log[0]

        self.assertTrue(isinstance(log_, dict))
        self.assertEqual({'timestamp', 'column', 'command', 'transform_type',
                          'transformation'}, set(log_.keys()))
        self.assertTrue(isinstance(log_['timestamp'], datetime.datetime))
        self.assertEqual(log_['column'], 'player_name')
        self.assertEqual(log_['command'], 'dibs')
        self.assertEqual(log_['transform_type'], 'replace')
        self.assertEqual(log_['transformation'],
                         'metaphysical goalie johnson > Bitty')

    def test_analysis_mask_missing(self):
        self.q.missing = {'Bitty'}
        self.q.analysis_mask_missing(self.map_)
        pdt.assert_series_equal(pd.Series([np.nan, 'Ransom', 'Holster'],
                                          name='player_name'),
                                self.map_['player_name'])

    def test_analysis_remap_dtype_pass(self):
        known = pd.Series([False, True, True], name='team_captain')
        q = Question(name='team_captain',
                     description='Has the player been given a C or AC?',
                     dtype=bool)
        q.analysis_remap_dtype(self.map_)
        pdt.assert_series_equal(known, self.map_['team_captain'])
        run_log = q.log[0]
        self.assertEqual(run_log['command'], 'Cast data type')
        self.assertEqual(run_log['transform_type'], 'transformation')
        self.assertEqual(run_log['transformation'], 'to bool')

    def test_write_provenance(self):
        known_log = pd.DataFrame(
            np.array([[datetime.datetime.now(), 'Cast data type',
                       'team_captain', 'transformation', 'to bool'],
                      [datetime.datetime.now(), 'Write Log', 'team_captain',
                       'recording', '']]),
            columns=['timestamp', 'command', 'column',  'transform_type',
                     'transformation']
           )

        q = Question(name='team_captain',
                     description='who is has the C or AC',
                     dtype=bool
                     )
        q.analysis_remap_dtype(self.map_)
        log_ = q.write_provenance()
        self.assertEqual(known_log.shape, log_.shape)
        pdt.assert_index_equal(known_log.columns, log_.columns)
        pdt.assert_series_equal(known_log['column'], log_['column'])
        pdt.assert_series_equal(known_log['command'], log_['command'])
        pdt.assert_series_equal(known_log['transform_type'],
                                log_['transform_type'])
        pdt.assert_series_equal(known_log['transformation'],
                                log_['transformation'])

    def test_read_provenance(self):
        self.q._read_provenance('dogs')

    def test_check_ontology(self):
        self.q._check_ontology()

    def test_remap_type_bool_placeholder(self):
        in_series = pd.Series(['True', 'true', 1, 'nope',
                               'False', 'false', 0, 0.0])
        # Sets the know values
        kseries = pd.Series([True, True, True, 'nope',
                             False, False, False, False])
        kmessage = 'to bool'

        # Gets the test values
        (tseries, tmessage, terror) = _remap_dtype(in_series, bool, {'nope'})
        self.assertEqual(kmessage, tmessage)
        self.assertFalse(terror)
        pdt.assert_series_equal(kseries, tseries)

    def test_remap_type_bool_placeholder_error(self):
        in_series = pd.Series(['True', 'true', 1, 'nope',
                               'False', 'false', 0, 0.0])
        # Sets the know values
        kseries = pd.Series([True, True, True, 'error',
                             False, False, False, False])
        kmessage = 'could not be cast to bool'

        # Gets the test values
        (tseries, tmessage, terror) = _remap_dtype(in_series, bool, {'cool '},
                                                   )
        self.assertEqual(kmessage, tmessage)
        self.assertTrue(terror)
        pdt.assert_series_equal(kseries, tseries)

    def test_remap_type_bool_error_no_placeholder_logging(self):
        in_series = self.map_['years_on_team']
        # Sets known values
        kseries = pd.Series(data=['error', 'error', 'error'],
                            name='years_on_team')
        kmessage = 'could not be cast to bool'
        (tseries, tmessage, terror) = _remap_dtype(in_series, bool)
        # Checks the function
        self.assertEqual(kmessage, tmessage)
        self.assertTrue(terror)
        pdt.assert_series_equal(kseries, tseries)

    def test_remap_type_str_pass(self):
        in_series = self.map_['player_name']
        # Sets the known values
        kseries = self.map_['player_name']
        kmessage = 'to str'
        (tseries, tmessage, terror) = _remap_dtype(in_series, str)
        # Checks the function
        self.assertEqual(kmessage, tmessage)
        self.assertFalse(terror)
        pdt.assert_series_equal(kseries, tseries)

    def test_remap_type_int_placeholder(self):
        in_series = pd.Series(data=['1', '2', '3', 'i dont skate'],
                              index=['Whiskey', 'Chowder', 'Bitty', 'Lardo'],
                              name='collegate_hockey_years')
        # Sets the known values
        kseries = pd.Series(data=[1, 2, 3, 'i dont skate'],
                            index=['Whiskey', 'Chowder', 'Bitty', 'Lardo'],
                            name='collegate_hockey_years')
        kmessage = 'to int'
        (tseries, tmessage, terror) = _remap_dtype(in_series, int,
                                                   {'i dont skate'})
        # Checks the function
        self.assertEqual(kmessage, tmessage)
        self.assertFalse(terror)
        pdt.assert_series_equal(kseries, tseries)

    def test_remap_type_float_log_error(self):
        in_series = pd.Series(data=['1', '2', '3', 'i dont skate'],
                              index=['Whiskey', 'Chowder', 'Bitty', 'Lardo'],
                              name='collegate_hockey_years')
        # Checks we get an error when logging is False
        with self.assertRaises(TypeError):
            _remap_dtype(in_series, float, loggable=False)

if __name__ == '__main__':
    main()
