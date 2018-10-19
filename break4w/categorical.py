import copy

import numpy as np
import pandas as pd

from break4w.question import Question


class Categorical(Question):

    def __init__(self, name, description, dtype, order, reference_val=None,
        ambiguous=None, frequency_cutoff=None, numeric_mapping=None,
        ordinal=False, code_delim='=', **kwargs):
        u"""
        A question object for categorical or ordinal questions

        Parameters
        ----------
        name : str
            The name of a column in a microbiome mapping file where metadata
            describing a clincial or enviromental factor is stored.
        description : str
            A brief description of the biological relevance of the information
            in the column. This can also be used to clarify acronyms or
            definations.
        dtype : {str, bool, int, float, tuple}
            The datatype in which the responses should be represented.
        order : list
            The list of all possible responses to the question which may be
            used for analysis. Ambigious responses (i.e. "I don't know") can
            be supplied in `ambiguous`; missing values are given in
            `missing`, and experimental blanks in `blanks`.
            In ordinal variables, this dictates the expected order for the
            values, even if they  do not map to a clear order in a string.
            (i.e. "Infant", "Toddler", "Preschooler", "Child") have a clear
            order, but do not map nicely into a well known order.
        reference_val: float, str
            The value from the field (must be an element in `order`) which
            should serve as the reference or null state. If no value is
            provided, its assumed that the first value is the reference value.
        ambiguous : str, list, optional
            A list of values which are considered ambiguous responses.
            For example, a response of "Not Sure" might be valid and useful to
            maintain for validation, but should be ignored during analysis.
            The ambiguous values can be cast to null values using the
            `analyis_remove_ambiguous` function.
        missing : str, list, optional
            Acceptable missing values. Missing values will be used to validate
            all values in the column. Specified missing values can also be
            ignored during analysis if correctly specified.
        frequency_cutoff : float, optional
            The minimum number of observations required to keep a sample group
            in an analysis. For example, if a value is only represented twice
            in a question, that value may not be appropriate for most
            standard statistical tests.
        name_mapping: dict, optional
            A dictionary of values which map the name of the values to a
            numeric code (i.e. if female is coded as 0, male is coded as 1,
            and other is coded as 2, then the dictionary would be
            `{0: "female", 1: "male", 2: "other"}`).
        ordinal : bool, optional
            Whether the data should be treated as ordinal, or not
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
            The dtype is not a str, bool, int, float, or tuple Python class.
        TypeError
            The `clean_name` is not a string.
        """

        if dtype not in {str, bool, int, float, tuple}:
            raise ValueError('%s is not a supported datatype for a '
                             'categorical variable.' % dtype)

        # Initializes the question
        Question.__init__(self, name, description, dtype,
                          **kwargs
                          )

        self.type = 'Categorical'

        self.order = order
        if reference_val is None:
            self.ref_val = order[0]
        else:
            self.ref_val = reference_val

        if numeric_mapping is not None:
            self.var_numeric = numeric_mapping
            self.var_label = {g: i for g, i in numeric_mapping.items()}
        else:
            self.var_label = None
            self.var_numeric = None

        self.frequency_cutoff = frequency_cutoff

        if ambiguous == None:
            self.ambiguous = None
        else:
            ambig = self._split_numeric_mapping(ambiguous)
            if isinstance(ambig, list):
                ambig = set(ambig)
            self.ambiguous = ambig

    def _update_order(self, remap_):
        """Updates the order and earlier order arguments

        Parameters
        ----------
        remap_: function
            A function to update the data in the order.
        """

        order = copy.copy(self.order)
        self.order = []
        for o in order:
            new_o = remap_(o)
            if new_o not in self.order and not pd.isnull(new_o):
                self.order.append(new_o)

    def analysis_apply_conversion(self, map_, remap_, command_name,
        loggable=True):
        """Applies and logs an arbitrary data remapping function

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.
        remap_: function, dict
            A function to update the data in the order or a dictionary
            describing the values to replace
        command_name: str, optional
            A description of the remapping function
        loggable: float, optional
            Whether the function should be updated
        """
        if isinstance(remap_, dict):
            mapping = remap_.copy()
            def remap_(x):
                if x in mapping:
                    return mapping[x]
                else:
                    return x

        message = ' | '.join(['%s >>> %s' % (o, remap_(o))
                              for o in self.order])
        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)
        if loggable:
            self._update_log('transformation', command_name, message)

    def analysis_convert_to_label(self, map_):
        """
        Converts integer group values to string names

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        if self.name_mapping is None:
            self._update_log('convert to label', 'error',
                             'There is no way to map codes to labels')
            raise ValueError('There is no way to map codes to labels')

        if self.numeric_mapping is None:
            self.numeric_mapping = {g: i for (i, g)
                                    in self.name_mapping.items()}

        def remap_(x):
            if x in self.name_mapping:
                return self.name_mapping[x]
            else:
                return np.nan

        self.analysis_apply_conversion(map_, remap_,
                                       command_name='convert code to label')

    def validate(self, map_):
        """Checks the values in the mapping file are correct

         Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        Raises
        ------
        ValueError
            If the values in the mapping file are not acceptable values
            for the question (given by order) or acceptable missing values.

        """
        # Gets the data to check
        iseries = map_[self.name].copy()
        message = []

        # Attempts to remap the data
        if self.blanks is None:
            blanks = set([])
        if hasattr(self, 'ambiguous') and self.ambiguous is not None:
            ambiguous = self.ambiguous
        else:
            ambiguous = set([])

        placeholders = self.missing.union(blanks).union(ambiguous)
        f_ = self._identify_remap_function(dtype=self.dtype,
                                           placeholders=placeholders,
                                           true_values=self.true_values,
                                           false_values=self.false_values,
                                          )
        dseries = iseries.apply(f_)
        new_order = [f_(o) for o in self.order]

        if dseries.apply(lambda x: x == 'error').any():
            message = (
                'the data cannot be cast to %s'
                % (str(self.dtype).replace("<class '", '').replace("'>", ''))
                )
            self._update_log('validate', 'error', message)
            raise TypeError(message)
        else:
            self._update_log(
                'validate', 'pass', 'the data can be cast to %s'
                % (str(self.dtype).replace("<class '", '').replace("'>", ''))
                )

        acceptable_values = placeholders.union(set(new_order))
        actual_values = set(dseries.unique()) - {np.nan}

        if not acceptable_values.issuperset(actual_values):
            descriptor = ['%s' % v
                          for v in sorted((actual_values - acceptable_values))]
            m_ = 'The following are not valid values: %s' \
                % (' | '.join(descriptor))
            message.append(m_)
            self._update_log('validate', 'error', '\n'.join(message))
            raise ValueError(m_)
        else:
            self._update_log('validate', 'pass', 'all values were valid')

