import datetime
import inspect

import numpy as np
import pandas as pd


class Question:
    """A base object class for handling American Gut Data dictionary entries"""
    true_values = {'yes', 'true', 1, 1.0, True}
    false_values = {'no', 'false', 0, 0.0, False}
    ebi_null = {'not applicable',
                'missing: not provided',
                'missing: not collected',
                'missing: restricted',
                'not provided',
                'not collected',
                'restricted'}

    def __init__(self, name, description, dtype, clean_name=None,
                 free_response=False, mimarks=False, ontology=None,
                 ebi_required=False, qiita_required=False, missing=None):
        """A base object for describing single question outputs

        The Question Object is somewhat limited in its functionality. For most
        questions in the dataset, it is better to use a child object with the
        appropriate question type.

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        dtype : object
            The datatype in which the responses should be represented. (i.e.
            float, int, str).
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        free_response: bool, optional
            Whether the question is a free response question or controlled
            vocabulary
        mimarks : bool, optional
            If the question was a mimarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.
        ebi_required : bool, optional
            Describes whether the question is required by EBI
        qiita_required : bool, optional
            Is the question required by qiita
        missing : str, list, optional
            Acceptable missing values
        """

        # Checks the arguments
        if not isinstance(name, str):
            raise TypeError('name must be a string.')
        if not isinstance(description, str):
            raise TypeError('description must be a string')
        if not inspect.isclass(dtype):
            raise TypeError('dtype must be a class')
        if not isinstance(clean_name, str) and clean_name is not None:
            raise TypeError('If supplied, clean_name must be a string')

        # Handles the main information about the data
        self.name = name
        self.description = description
        self.dtype = dtype

        self.type = 'Question'
        if clean_name is None:
            self.clean_name = name.replace('_', ' ').title()
        else:
            self.clean_name = clean_name

        # Sets up
        self.free_response = free_response
        self.mimarks = mimarks
        self.ontology = ontology
        self.ebi_required = ebi_required
        self.qiita_required = qiita_required
        if missing is None:
            self.missing = self.ebi_null
        else:
            self.missing = missing

        self.log = []

    def _update_log(self, command, transform_type, transformation):
        """Updates the in-object documentation"""
        self.log.append({
            'timestamp': datetime.datetime.now(),
            'column': self.name,
            'command': command,
            'transform_type': transform_type,
            'transformation': transformation,
            })

    def check_map(self, map_):
        """Checks the group exists in the metadata

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        Raises
        ------
        ValueError
            If the column identified by the question object is not part of the
            supplied DataFrame.

        """
        if self.name not in map_.columns:
            raise ValueError('%s is not a column in the supplied map!'
                             % self.name)

    def remap_dtype(self, map_):
        """Makes sure the target column in map_ has the correct datatype

        map_ : DataFrame
            A pandas dataframe containing the column described by the question
            name.

        """
        if self.dtype == bool:
            def convert_dtype(x):
                if pd.isnull(x):
                    return x
                elif isinstance(x, str):
                    return x.lower()
                else:
                    return x

            map_[self.name] = map_[self.name].apply(convert_dtype)

            if not (set(map_[self.name].dropna()).issubset(
                    self.true_values.union(self.false_values))):
                raise TypeError('%s cannot be cast to a bool value.'
                                % self.name)

            def remap_(x):
                if pd.isnull(x):
                    return x
                if isinstance(x, str) and x.lower() in self.true_values:
                    return True
                elif isinstance(x, str) and x.lower() in self.false_values:
                    return False
                else:
                    return bool(x)

        else:
            def remap_(x):
                return self.dtype(x)

        map_[self.name] = map_[self.name].apply(remap_)
        map_.replace('nan', np.nan, inplace=True)

        self._update_log('Cast data type', 'transformation',
                         'to %s' % self.dtype)

    def write_providence(self):
        """Writes the question provinence to a string

        To be added!
        """
        pass

    def read_providence(self, fp_):
        """Reads the existing question provenance
        """
        pass

    def check_ontology(self):
        """
        Checks the ontology associated with the question

        To be added!
        """
        pass

    def check_required(self):
        """Checks whether or not the question is a required question

        To be added!
        """
        pass

    def check_missing(self):
        """Checks whether the missing values are appropriate

        To be added!

        """
        pass
