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

        Raises
        ------
        TypeError
            The name is not a string
        TypeError
            The description is not a string
        TypeError
            The dtype is not a class
        TypeError
            The `clean_name` is not a string.
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
        elif isinstance(missing, str):
            self.missing = set([missing])
        else:
            self.missing = set(missing)

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
            self._update_log('column check', 'validation',
                             'column does not exist!')
            raise ValueError('%s is not a column in the supplied map!'
                             % self.name)

    def analysis_remap_dtype(self, map_):
        """Makes sure the target column in map_ has the correct datatype

        Parameters
        ----------
        map_ : DataFrame
            A pandas dataframe containing the column described by the question
            name.

        Raises
        ------
        ValueError
            If the column identified by the question object is not part of the
            supplied DataFrame.

        """
        self.check_map(map_)
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

        type_str = 'to %s' % self.dtype
        type_str = type_str.replace("'", "")
        type_str = type_str.replace("<class ", "").replace(">", "")
        self._update_log('Cast data type', 'transformation', type_str)

    def write_providence(self):
        """Writes the question provinence to a string

        Returns
        -------
        DataFrame
            A pandas dataframe describing the time, column, action taken,
            action type, and changes in the data.
        """
        self._update_log('Write Log', 'recording', '')
        return pd.DataFrame(self.log)[['timestamp', 'column', 'command',
                                       'transform_type', 'transformation']]

    def _read_providence(self, fp_):
        """Reads the existing question provenance
        """
        pass

    def _check_ontology(self):
        """
        Checks the ontology associated with the question

        To be added!
        """
        pass
