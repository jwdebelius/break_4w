from unittest import TestCase, main

import pandas as pd

from break4w.question import Question


class QuestionTest(TestCase):

    def setUp(self):
        self.name = 'question_name'
        self.description = 'question description'
        self.dtype = str

        self.map_ = pd.DataFrame([['a', 'b', 'c']],
                                 index=['question_name']).T
        self.q = Question(name=self.name,
                          description=self.description,
                          dtype=self.dtype,
                          free_response=True,
                          )

    def test_init_default(self):
        self.assertEqual(self.name, self.q.name)
        self.assertEqual(self.description, self.q.description)
        self.assertEqual(self.dtype, self.q.dtype)
        self.assertEqual('Question Name', self.q.clean_name)
        self.assertEqual('Question', self.q.type)
        self.assertTrue(self.q.free_response)
        self.assertFalse(self.q.mimarks)
        self.assertFalse(self.q.ontology)
        self.assertFalse(self.q.ebi_required)
        self.assertFalse(self.q.qiita_required)
        self.assertEqual(self.q.missing, None)

    def test_init_error_name(self):
        with self.assertRaises(TypeError):
            self.q = Question(name=1,
                              description=self.description,
                              dtype=self.dtype,
                              )

    def test_init_error_description(self):
        with self.assertRaises(TypeError):
            self.q = Question(name=self.name,
                              description=self.dtype,
                              dtype=self.dtype,
                              )

    def test_init_error_dtype(self):
        with self.assertRaises(TypeError):
            self.q = Question(name=self.name,
                              description=self.description,
                              dtype=self.description,
                              )

    def test_init_error_clean_name(self):
        with self.assertRaises(TypeError):
            self.q = Question(name=self.name,
                              description=self.description,
                              dtype=self.dtype,
                              clean_name=self.dtype
                              )

    def test_check_map_pass(self):
        self.q.check_map(self.map_)

    def test_check_map_error(self):
        self.map_.rename(columns={'question_name': 'question'}, inplace=True)
        with self.assertRaises(ValueError):
            self.q.check_map(self.map_)


if __name__ == '__main__':
    main()
