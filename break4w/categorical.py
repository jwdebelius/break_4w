import copy

import numpy as np
import pandas as pd

from break4w.question import Question, _identify_remap_function


class Categorical(Question):

    def __init__(self, name, description, dtype, order, extremes=None,
        frequency_cutoff=None, ambiguous_values=None,
        clean_name=None, mimarks=False, ontology=None,
        missing=None, blanks=None, colormap=None,
        name_mapping=None, source_columns=None,
        derivative_columns=None, notes=None, **other_properties):
        r"""A question object for categorical or ordinal questions

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
            be supplied in `ambiguous_values`; missing values are given in
            `missing`, and experimental blanks in `blanks`.
            In ordinal variables, this dictates the expected order for the
            values, even if they  do not map to a clear order in a string.
            (i.e. "Infant", "Toddler", "Preschooler", "Child") have a clear
            order, but do not map nicely into a well known order.
        clean_name : str, optional
            A nicer version of the way the column should be named. This can be
            used for display in figures. If nothing is provided, the column
            name will be coverted to a title by replacing an underscores with
            spaces and converting to title case.
        frequency_cutoff : float, optional
            The minimum number of observations required to keep a sample group
            in an analysis. For example, if a value is only represented twice
            in a question, that value may not be appropriate for most
            standard statistical tests.
        ambiguous_values : str, list, optional
            A list of values which are considered ambiguous responses.
            For example, a response of "Not Sure" might be valid and useful to
            maintain for validation, but should be ignored during analysis.
            The ambigious values can be cast to null values using the
            `analyis_remove_ambiguious` function.
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
        name_mapping: dict, optional
            A dictionary of values which map the name of the values to a
            numeric code (i.e. if female is coded as 0, male is coded as 1,
            and other is coded as 2, then the dictionary would be
            `{0: "female", 1: "male", 2: "other"}`).
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
                          clean_name=clean_name,
                          mimarks=mimarks,
                          ontology=ontology,
                          missing=missing,
                          blanks=blanks,
                          colormap=colormap,
                          source_columns=source_columns,
                          derivative_columns=derivative_columns,
                          notes=notes,
                          **other_properties
                          )

        self.type = 'Categorical'

        self.order = order
        if extremes is not None:
            self.extremes = extremes
        else:
            self.extremes = [order[0], order[-1]]

        self.frequency_cutoff = frequency_cutoff
        if isinstance(ambiguous_values, str):
            self.ambiguous_values = set([ambiguous_values])
        elif ambiguous_values is None:
            self.ambiguous_values = None
        else:
            self.ambiguous_values = set(ambiguous_values)

        self.name_mapping = name_mapping
        self.numeric_mapping = None

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
        self.extremes = [remap_(e) for e in self.extremes]

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
            elif x in self.name_mapping.values():
                return x
            else:
                return np.nan

        self.analysis_apply_conversion(map_, remap_,
                                       command_name='convert code to label')

    def analysis_convert_to_numeric(self, map_):
        """
        Converts the values in each group into integers based on `order`

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        mapping = {g: i for i, g in enumerate(self.order)}
        if self.numeric_mapping is None:
            self.numeric_mapping = mapping
        self.name_mapping = {i: g for (i, g) in enumerate(self.order)}

        def remap_(x):
            if isinstance(x, (int, float)):
                return x
            elif x in mapping:
                return mapping[x]
            else:
                return np.nan

        self.analysis_apply_conversion(map_, remap_,
                                       command_name='convert label to code')

    def analysis_drop_infrequent(self, map_):
        """
        Replaces any value from a group below the frequency cutoff with null

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """

        counts = map_.groupby(self.name).count().max(1)[self.order]
        below_locs = (counts <= self.frequency_cutoff) | pd.isnull(counts)
        below = counts.loc[below_locs].index

        def remap_(x):
            if x in below:
                return np.nan
            else:
                return x

        self.analysis_apply_conversion(map_, remap_,
                                       command_name=False,
                                       loggable=False)
        self._update_log('drop infrequent values', 'drop', 'below %i: %s'
                         % (self.frequency_cutoff, ' | '.join(sorted(below))))

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
        message = []

        # Checks the data type
        if self.blanks is None:
            blanks = set([])
        if hasattr(self, 'ambigious') and self.ambigious is not None:
            ambigious = self.ambigious
        else:
            ambigious = set([])

        placeholders = self.missing.union(blanks).union(ambigious)

        remap_ = _identify_remap_function(dtype=self.dtype,
                                          placeholders=placeholders,
                                          true_values=self.true_values,
                                          false_values=self.false_values,
                                          )
        iseries = map_[self.name].copy()
        oseries = iseries.apply(remap_)

        if np.any(oseries.apply(lambda x: x == 'error')):
            message = (
                'could not convert to %s'
                % (str(self.dtype).replace("<class '", '').replace("'>", ''))
                )
            self._update_log('transformation', 'cast data type', message)
            raise TypeError(message)

        map_[self.name] = oseries
        self._update_order(remap_)

        message = (
            'convert to %s'
            % (str(self.dtype).replace("<class '", '').replace("'>", ''))
            )
        self._update_log('transformation', 'cast data type', message)

    def analysis_remap_null(self, map_):
        """Converts approved null values to nans for analysis

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        map_[self.name].replace(self.missing, np.nan, inplace=True)
        for m in self.missing:
            if m in self.order:
                self.order.remove(m)
        self._update_log('correct null values', 'drop',
                         ' | '.join(list(self.missing)))

    def analyis_remove_ambiguious(self, map_):
        """
        Replaces ambigious values in the metadata with nulls

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """

        def _remap(x):
            if pd.isnull(x) or (x is None):
                return np.nan
            elif x in self.ambiguous_values:
                return np.nan
            else:
                return x

        if self.ambiguous_values is not None:
            map_[self.name] = map_[self.name].apply(_remap)
            self._update_order(_remap)
        self._update_log('remove ambigious values', 'drop',
                         ' | '.join([v for v in self.ambiguous_values]))

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
        if hasattr(self, 'ambigious') and self.ambigious is not None:
            ambigious = self.ambigious
        else:
            ambigious = set([])

        placeholders = self.missing.union(blanks).union(ambigious)
        f_ = _identify_remap_function(dtype=self.dtype,
                                      placeholders=placeholders,
                                      true_values=self.true_values,
                                      false_values=self.false_values,
                                      )
        dseries = iseries.apply(f_)

        if dseries.apply(lambda x: x == 'error').any():
            message = (
                'the data cannot be cast to %s'
                % (str(self.dtype).replace("<class '", '').replace("'>", ''))
                )
            self._update_log('validate', 'error', message)
            raise TypeError(message)
        else:
            message.append(
                'Data can be cast to %s'
                % str(self.dtype).replace("<class '", '').replace("'>", '')
                )

        acceptable_values = placeholders.union(set(self.order))
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
            message.append('All values in the column were valid.')

        self._update_log('validate', 'pass', '\n'.join(message))
