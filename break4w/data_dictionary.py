from collections import OrderedDict
import datetime

import numpy as np
import pandas as pd

from break4w.question import Question
from break4w.categorical import Categorical
from break4w.bool import Bool
from break4w.continous import Continous

type_lookup = {'continous': Continous,
               'categorical': Categorical,
               'multiple choice': Categorical,
               'ordinal': Categorical,
               'bool': Bool,
               'boolean': Bool,
               'yes/no': Bool,
               }


class DataDictionary(OrderedDict):
    def __init__(self, columns, types, description=None):
        """Initializes the dictionary object

        This is a very basic prototype of the data dictionary object

        Parameters
        ----------
        columns: list of dicts
            A list of dictionaries representing each column in the metadata. 
            The dictionaries must contain a `name` key, describing the column 
            name. The values in the dictionary should the variables needed 
            for each type of question object in a data dictionary.
        types: list of strings
            A description of the type of question being asked. These come 
            from a relatively controlled vocabulary and include types such as 
            `"continous", "categorical", "bool"`. If the question type does 
            not conform to the controlled vocabulary, the column will be 
            read as a Question object with limited functionality.
        description: str
            The 
        """
        self.log = []
        if description is None:
            self.description = ''
        elif len(description) > 80:
            raise ValueError('The dictionary description cannot be more than '
                             '80 characters')
        else:
            self.description = description

        # Adds the question objects to the dictionary
        for col_, type_ in zip(*(columns, types)):
            self.add_question(question_data=col_,
                              question_type=type_,
                              record=False,
                              check=False)

    def __str__(self):
        """
        Generates printed summary
        """
        summary = ['Data Dictionary with %i columns'  % len(self)]
        if len(self.description) > 0:
            summary.append('\t%s' % self.description)
        summary.append('-----------------------------------------------------'
                       '------------------------')
                  
        for col in self.values():
            summary.append('%s (%s)' % (col.name, col.type))
        summary.append('-----------------------------------------------------'
                       '------------------------')
        return '\n'.join(summary)

    def _update_log(self, command, column=None,
        transform_type=None, transformation=None):
        """Used for internal tracking of the columns and data

        Every time a Question acts on data, a record should be made of
        the transformation. (See break4w.question.Question._update_log).
        However, this also tracks the transformation on the dictionary
        level.

        Parameters
        ----------
        command : str
            A short textual description of the command performed. This
            may be the function name in text format.
        column : str, optional
            The column in the metadata being explored.
        transform_type: str, optional
            A more general description of the type of action that was
            performed. Ideally, this comes for a preset list of possible
            actions, and the descriptions are consistent.
        transformation: str, optional
            Explains exactly how values were changed.

        """
        self.log.append({
            'timestamp': datetime.datetime.now(),
            'column': column,
            'command': command,
            'transform_type': transform_type,
            'transformation': transformation,
            })

    def _pull_question_log(self, column=None):
        """Adds information from the specified column to the log."""
        raise NotImplementedError

    def add_question(self, question_data, question_type=None,
        check=True, record=True):
        """
        Adds a new question object to the data dictionary

        Parameters
        ----------
        question_data: Dict, Question
            Describes the data dictionary entry for the question. This can
            be a break4w question object created directly, or a dictionary
            objecting with information like the name in the metadata
            representation, data type, a description, and specific information
            for the type of question. For instance, `question_type` specified
            the qustion was `"continous"`, the `question_data` must also
            describe units for the question.
        question_type: str, optional
            Describes the type of question object that should be selected
            for the question. If `question_data` is a `Question` object, then
            no `question_type` is needed.
        check: bool, optional
            Checks whether a name already exists in the question name space.
            If this is true, then the function will check if the column 
            already exists in the dictionary. If the column does exist and 
            check is true, an error will be raised. If check is not true, the
            data dictionary entry for the column will be overwritten and any
            information in that column will be lost.
        record, bool, optional
            Indicates where the addition should be logged.

        Raises
        ------
        ValueError
            When the function is checking for the column and the column name
            is already in the dictionary. If this is the case, the dictionary
            entry should be adjusted using `update_question`, not
            `add_question`, since this function will otherwise over write the
            existing column.

        """
        error = False

        # Converts the dict data to a Question object
        if isinstance(question_data, dict):
            question_object = type_lookup.get(question_type.lower(), Question)
            question_data = question_object(**question_data)
        name = question_data.name

        # Checks if the question is in the dictionary
        if (name in self.keys()) and check:
            error = True
            message = '%s already has a dictionary entry' % name
            transform_type = 'error'
        else:
            message = '%s was added to the dictionary' % name
            transform_type = None

        # Updates the log
        if record:
            self._update_log('add column', 
                             column=name, 
                             transformation=message,
                             transform_type=transform_type)

        # Raises an error or updates the dictionary, as appropriate
        if error:
            raise ValueError(message)
        else:
            self[name] = question_data

    def get_question(self, name):
        """
        Returns the data dictionary column

        Parameters
        ----------
        name: str
            The name of the dictionary column to be returned

        Returns
        -------
        Question
            The question object for the appropriate dictionary
            object

        Raises
        ------
        ValueError
            When the column being asked for does not exist.
        """
        if name not in self.keys():
            message = 'There is no entry for %s' % name
            self._update_log(column=name,
                             command='get question',
                             transform_type='error',
                             transformation=message)
            raise ValueError(message)
        self._update_log(column=name, command='get question')
        return self[name]

    def drop_question(self, name):
        """
        Removes a dictionary entry for the specified column.

        Parameters
        ----------
        name: str
            The name of the dictionary column to be returned
        """
        if name in self.keys():
            del self[name]
            self._update_log(command='remove question', column=name)

    def update_question(self, update, name=None):
        """
        Updates dictionary entry for the data

        Parameters
        ----------
        update: Dict, Question
            Describes the data dictionary entry for the question. This can
            be a break4w question object created directly, or a dictionary
            objecting with information like the name in the metadata
            representation, data type, a description, and specific information
            for the type of question. For instance, `question_type` specified
            the qustion was `"continous"`, the `question_data` must also
            describe units for the question.
        name: str, optional
            The name of the dictionary column to be returned. If `update` is
            a Question object, this can be infered from the question.
        """
        
        # Gets the dictionary of the new column and column name
        if isinstance(update, Question):
            update = vars(update)

        if name is None:
            name = update['name']

        # Checks if the data is already in the dictionary
        if name not in self.keys():
            message = ('%s is not a question in the current dictionary.\n'
                       'Have you tried adding the question?') % name
            self._update_log(command='update question',
                             column=name,
                             transform_type='error',
                             transformation=message)
            raise ValueError(message)
        current = vars(self[name])
        diff = {k: v for k, v in update.items()
                if (((k not in current) or (v != current[k])) and
                    (k not in {'log'}))
                }
        change_keys = {}
        for k, v in diff.items():
            if k in current:
                change_keys[k] = (current[k], v)
            else:
                change_keys[k] = ('add', v)
            setattr(self[name], k, v)
        if 'log' in update:
            self[name].log.extend(update['log'])
        self._update_log(
            command='update question',
            column=name,
            transform_type='update dictionary values',
            transformation=' | '.join(['%s : %s > %s' % (k, v[0], v[1])
                                       for k, v in change_keys.items()]))

    def validate(self, map_, check_order=True):
        """
        Checks columns appear in the mapping file in the appropriate order
        and conform to the standards set in the data dictionary.

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the metadata being analyzed.
        check_order: bool, optional
            Do the order of columns in the data dictionary and metadata have
            to match?
        """
        pass_ = True
        failures = []
        validation_messages = []
        self._validate_question_order(map_, check_order)
        for name, question in self.items():
            if question.type == 'Question':
                continue
            try:
                question.validate(map_)
            except:
                pass_ = False
                failures.append(name)
            if question.type in {'Categorical', 'Bool'}:
                validation_messages.append(question.log[-2])
            validation_messages.append(question.log[-1])

        self.log.extend(validation_messages)
        if pass_:
            self._update_log('validate', transform_type='pass',
                             transformation='All columns passed')
        else:
            message = ('There were issues with the following columns:\n%s'
                       % '\n'.join(failures))
            self._update_log('validate', transform_type='error',
                             transformation=message)
            raise ValueError(message)

    def _validate_question_order(self, map_, check_order=True, record=True,
        verbose=False):
        """
        Checks all the required questions are present in the mapping file
        and that they are in the correct order.

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the metadata being analyzed.
        check_order: bool, optional
            Do the order of columns in the data dictionary and metadata have
            to match?
        record: bool, optional
            Indicates where the addition should be logged.
        verbose: bool, optional
            Provides more detailed information about the error

        Raises
        ------
        ValueError

        """
        pass_ = True
        message = ('The columns in the mapping file match the columns in '
                   'the data dictionary.')
        map_columns = list(map_.columns)
        dict_columns = list(self.keys())

        if not set(map_columns) == set(dict_columns):
            pass_ = False
            in_map = list(set(map_columns) - set(dict_columns))
            in_dict = list(set(dict_columns) - set(map_columns))
            text = ('There are %i columns in the data dictionary '
                    'not in the mapping file, and %i from the mapping'
                    ' file not in the data dictionary.'
                    % (len(in_dict), len(in_map)))
            if verbose:
                message = '%s\n%s' % (
                    text,
                    'In the dictionary but not in the map: \n%s\n\n'
                    'In the map but not in the dictionary: \n%s\n\n'
                    % ('\n'.join(in_dict), '\n'.join(in_map))
                    )
            else:
                message = text

        elif not (map_columns == dict_columns) and check_order:
            pass_ = False
            message = ('The columns in the dictionary and map are not in'
                       ' the same order.')

        if record and pass_:
            self._update_log(command='validate', transform_type='pass',
                             transformation=message)
        elif record and not pass_:
            self._update_log(command='validate', transform_type='fail',
                             transformation=message)
            raise ValueError(message)
        elif not pass_:
            raise ValueError(message)

    def to_dataframe(self):
        u"""Converts data dictionary to a pandas dataframe

        Returns
        -------
        DataFrame
            A dataframe summary of the contents of the data dictionary. It
            will include the following columns:
                * "name": The name of the variable
                * "description": A description of the variable of no more than
                             80 characters
                * "dtype": a string representation of python data types (i.e 
                           str, int, bool, etc)
                * "clean_name": the cleaned up column name
            It may also contain columns describing the variable order,
            limits on the data, units, etc.
        """
        df_ = pd.concat(axis=1, sort=False, objs=[
            col_._to_series() for col_ in self.values()
            ])
        return df_.T.set_index('name')

    def to_pandas_stata(self):
        """
        Generates strings and dictionary compatible with writing to stata

        Returns
        -------
        str
            A stata-compatible dataset description for `pandas.write_stata`
        dictionary
            A stata-compatible description for each variable, compatible with
            `pandas.write_stata`.
        """

        variable_desc = {k: v.description for k,v in self.items()}

        return self.description, variable_desc

