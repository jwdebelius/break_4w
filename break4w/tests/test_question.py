from unittest import TestCase, main

import datetime

import pandas as pd
import numpy as np
import numpy.testing as npt
import pandas.util.testing as pdt

from break4w.question import Question


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

    def test_analysis_remap_dtype_bool_str(self):
        q = Question(name='team_captain',
                     description='who is has the C or AC',
                     dtype=bool
                     )
        self.assertTrue(
            self.map_['team_captain'].apply(lambda x: isinstance(x, str)).all()
            )
        q.analysis_remap_dtype(self.map_)
        npt.assert_array_equal(np.array([False, True, True]),
                               self.map_['team_captain'].values)

    def test_analysis_remap_dtype_bool(self):
        self.map_['team_captain'] = self.map_['team_captain'] == 'True'
        npt.assert_array_equal(np.array([False, True, True]),
                               self.map_['team_captain'].values)
        q = Question(name='team_captain',
                     description='who has the C or AC',
                     dtype=bool
                     )
        q.analysis_remap_dtype(self.map_)
        npt.assert_array_equal(np.array([False, True, True]),
                               self.map_['team_captain'].values)

    def test_analysis_remap_dtype_bool_int(self):
        self.map_['team_captain'] = \
            (self.map_['team_captain'] == 'True').astype(int)
        npt.assert_array_equal(np.array([0, 1, 1]),
                               self.map_['team_captain'].values)
        q = Question(name='team_captain',
                     description='who is has the C or AC',
                     dtype=bool
                     )
        q.analysis_remap_dtype(self.map_)
        npt.assert_array_equal(np.array([False, True, True]),
                               self.map_['team_captain'].values)

    def test_analysis_remap_dtype_bool_error(self):
        self.q.dtype = bool
        with self.assertRaises(TypeError):
            self.q.analysis_remap_dtype(self.map_)

    def test_analysis_remap_dtype_int(self):
        q = Question(name='years_on_team',
                     description=('The number of years a player has played '
                                  'hockey for Samwell'),
                     dtype=float)
        q.analysis_remap_dtype(self.map_)
        npt.assert_array_equal(np.array([2, 4, 4]),
                               self.map_['years_on_team'].values)

    def test_analysis_mask_missing(self):
        self.q.missing = {'Bitty'}
        self.q.analysis_mask_missing(self.map_)
        pdt.assert_series_equal(pd.Series([np.nan, 'Ransom', 'Holster'],
                                          name='player_name'),
                                self.map_['player_name'])

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


if __name__ == '__main__':
    main()
