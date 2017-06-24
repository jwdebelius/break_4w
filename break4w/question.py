import datetime
import inspect

import numpy as np
import pandas as pd

true_values = {'yes', 'true', 1, 1.0, True}
false_values = {'no', 'false', 0, 0.0, False}
ebi_null = {'not applicable',
            'missing: not provided',
            'missing: not collected',
            'missing: restricted',
            'not provided',
            'not collected',
            'restricted',
            }


class Question:
    u"""A base object class for handling American Gut Data dictionary entries
    """
    true_values = true_values
    false_values = false_values
    ebi_null = ebi_null

    def __init__(self, name, description, dtype, clean_name=None,
        free_response=False, mimarks=False, ontology=None,
        missing=None, blanks=None, colormap=None, original_name=None,
        source_columns=None, derivative_columns=None, notes=None,
        **other_properties):
        u"""A base object for describing single question outputs

        The Question Object is somewhat limited in its functionality. For most
        questions in the dataset, it is better to use a child object with the
        appropriate question type (i.e. Categorical, Bool, Continous, Dates).

        Parameters
        ----------
        name : str
            The name of a column in a microbiome mapping file where metadata
            describing a clincial or enviromental factor is stored.
        description : str
            A brief description of the biological relevance of the information
            in the column. This can also be used to clarify acronyms or
            definations.
        dtype : object
            The datatype in which the responses should be represented. (i.e.
            `float`, `int`, `str`).
        clean_name : str, optional
            A nicer version of the way the column should be named. This can be
            used for display in figures. If nothing is provided, the column
            name will be coverted to a title by replacing an underscores with
            spaces and converting to title case.
        mimarks : bool, optional
            If the question was a mimarks standard field
        ontology : str, optional
            The type of ontology, if any, used to answer the question. An
            ontology provides a consistent, structured vocabulary. A list
            of ontologies can be found at https://www.ebi.ac.uk/ols/ontologies
        missing : str, list, optional
            Acceptable missing values. Missing values will be used to validate
            all values in the column. Specified missing values can also be
            ignored during analysis if correctly specified.
        blanks: str, list, optional
            Value to represent experimental blanks, if relevent.
        colormap: str, iterable, optional
            The colors to use when plotting the data. This can be a matplotlib
            colormap object, a string describing a matplotlib compatable
            colormap (i.e. `'RdBu'`), or an iterable of matplotlib compatable
            color values.
        original_name: str, optional
            The name of the column in a previous iteration of the metadata
            (often the version of the metadata provided by the collaborator).
        source_columns: list, optional
            Other columns in the mapping file used to create this column.
        derivative_columns: list, optional
            Any columns whose data is derived from the data in this column.
        notes: str, optional
            Any additional notes about the column, such as information
            about the data source, manual correction if it happened, etc.
            Basically any free text information someone should know about
            the column.

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

        # Checks the core arguments
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
        if missing is None:
            self.missing = self.ebi_null
        elif isinstance(missing, str):
            self.missing = set([missing])
        else:
            self.missing = set(missing)
        self.blanks = blanks
        self.colormap = colormap
        self.original_name = original_name

        if source_columns is None:
            self.source_columns = []
        else:
            self.source_columns = source_columns

        if derivative_columns is None:
            self.derivative_columns = []
        else:
            self.derivative_columns = derivative_columns

        self.notes = notes
        for k, v in other_properties.items():
            setattr(self, k, v)

        self.log = []

    def _update_log(self, command, transform_type, transformation):
        u"""A helper function to update the in-object documentation object

        Every time a Question acts on data, a record should be made of
        the transformation. This function standardized the format of that
        recording by tracking the time, location, and command.
        Examples of how the function is used can be found in all functions
        which operate on a `map_` variable.

        Parameters
        ----------
        command : str
            A short textual description of the command performed. This
            may be the function name in text format.
        transform_type: str
            A more general description of the type of action that was
            performed. Ideally, this comes for a preset list of possible
            actions, and the descriptions are consistent.
        transformation: str
            Explains exactly how values were changed.

        """
        self.log.append({
            'timestamp': datetime.datetime.now(),
            'column': self.name,
            'command': command,
            'transform_type': transform_type,
            'transformation': transformation,
            })

    def analysis_remap_dtype(self, map_):
        """Converts values in the question column to the correct datatype

        Parameters
        ----------
        map_ : DataFrame
            A pandas DataFrame containing the metadata being analyzed. The
            question object describes a column within the `map_`.

        Raises
        ------
        TypeError
            The question is assumed to be a Boolean, but the value cannot
            be cast to a boolean value.

        """
        if self.blanks is None:
            blanks = set([])
        if hasattr(self, 'ambigious') and self.ambigious is not None:
            ambigious = self.ambigious
        else:
            ambigious = set([])

        placeholders = self.missing.union(blanks).union(ambigious)

        (series, message, error) = _remap_dtype(series=map_[self.name].copy(),
                                                dtype=self.dtype,
                                                placeholders=placeholders,
                                                true_values=self.true_values,
                                                false_values=self.false_values,
                                                loggable=True
                                                )
        self._update_log('Cast data type', 'transformation', message)
        if error:
            raise TypeError(message)
        map_[self.name] = series

    def analysis_mask_missing(self, map_):
        """
        Remaps known missing values with pandas friendly nans

        Parameters
        ----------
        map_ : DataFrame
            A pandas DataFrame containing the metadata being analyzed. The
            question object describes a column within the `map_`.
        """

        def remap_(x):
            if x in self.missing:
                return np.nan
            else:
                return x

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_log("Mask missing values", "replace", '%s > np.nan'
                         % ';'.join(list(self.missing)))

    def write_provenance(self):
        """Writes the question provenance to a string

        Returns
        -------
        DataFrame
            A pandas dataframe describing the time, column, action taken,
            action type, and changes in the data.
        """
        self._update_log('Write Log', 'recording', '')
        return pd.DataFrame(self.log)[['timestamp', 'command', 'column',
                                       'transform_type', 'transformation']]

    def _read_provenance(self, fp_):
        """Reads the existing question provenance
        """
        pass

    def _check_ontology(self):
        """
        Checks the ontology associated with the question

        To be added!
        """
        pass


def _remap_dtype(series, dtype, placeholders=None, true_values=true_values,
                 false_values=false_values, loggable=True):
    """
    Converts values from strings to the specified data type

    Parameters
    ----------
    series : Series
        The data being cast
    dtype : object
        The datatype in which the responses should be represented. (i.e.
        `float`, `int`, `str`).
    placeholders : set, optional
        Acceptable values to be ignored representing either placeholder values
        such as text for missing values, blanks, or ambigious measurements.

    Returns
    -------
    Series
        The data cast to the appropriate datatype, preserving the placeholder
        values in the correct format.

    Raises
    ------
    TypeError
        The question is assumed to be a Boolean, but the value cannot
        be cast to a boolean value.

    """
    type_str = '%s' % dtype
    type_str = type_str.replace("'", "")
    type_str = type_str.replace("<class ", "").replace(">", "")

    if dtype == bool:
        #  Converts any non-placeholder string values to lowercase
        if placeholders is not None:
            def clean_up_strings(x):
                if isinstance(x, str) and (x not in placeholders):
                    return x.lower()
                else:
                    return x

            def remap_(x):
                if (x in placeholders) or (pd.isnull(x)):
                    return x
                elif x in true_values:
                    return True
                elif x in false_values:
                    return False
                else:
                    return 'error'
        else:
            def clean_up_strings(x):
                if isinstance(x, str):
                    return x.lower()
                else:
                    return x

            def remap_(x):
                if (pd.isnull(x)):
                    return x
                elif x in true_values:
                    return True
                elif x in false_values:
                    return False
                else:
                    return 'error'

        series = series.apply(clean_up_strings)

    else:
        # Defines a function to clean up all other datatypes
        if placeholders is not None:
            def remap_(x):
                if (x in placeholders) or pd.isnull(x):
                    return x
                else:
                    try:
                        return dtype(x)
                    except:
                        return 'error'
        else:
            def remap_(x):
                if pd.isnull(x):
                    return x
                else:
                    try:
                        return dtype(x)
                    except:
                        return 'error'

    series = series.apply(remap_)

    error = np.any(series.apply(lambda x: x == 'error'))
    print(error)
    message = 'to %s' % type_str
    if error:
        message = 'could not be cast to %s' % type_str

    if loggable:
        returns = (series, message, error)
    else:
        returns = (series, message, error)

    if error and not loggable:
        raise TypeError(message)

    return returns
