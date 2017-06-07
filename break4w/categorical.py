import copy

import numpy as np
import pandas as pd

from break4w.question import Question


class Categorical(Question):

    def __init__(self, name, description, dtype, order, extremes=None,
                 frequency_cutoff=None, ambiguous_values=None,
                 clean_name=None, mimarks=False, ontology=None,
                 missing=None, blanks=None, colormap=None,
                 numeric_mapping=None, source_columns=None,
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

    def _update_order(self, remap_):
        """Updates the order and earlier order arguments

        Parameters
        ----------
            remap_: function
                A function to update the data in the order.
            command : str
                The name of the function being run command
            transform_type : str
                The type of transformation being performed
        """

        order = copy.copy(self.order)
        self.order = []
        for o in order:
            new_o = remap_(o)
            if new_o not in self.order and not pd.isnull(new_o):
                self.order.append(new_o)
        self.extremes = [remap_(e) for e in self.extremes]

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
                         self.ambiguous_values)

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

        def _remap(x):
            if x in below:
                return np.nan
            else:
                return x

        index = map_[self.name].dropna().index
        map_[self.name] = map_.loc[index, self.name].apply(_remap)
        self._update_order(_remap)
        self._update_log('drop infrequent values', 'drop', below)

    def analysis_convert_to_numeric(self, map_):
        """
        Converts the values in each group into integers based on `order`

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        order = {g: i for i, g in enumerate(self.order)}

        def remap_(x):
            if isinstance(x, (int, float)):
                return x
            elif x in order:
                return order[x]
            else:
                return np.nan

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)
        self._update_log('convert to numeric', 'replace',
                         ' | '.join(['%s >>> %s' % (g, i)
                                     for (g, i) in order.items()]))

    def analysis_label_order(self, map_):
        """Prefixes the data with an ordinal integer

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        order = {g: '(%i) %s' % (i, g) for i, g in enumerate(self.order)}

        def remap_(x):
            if isinstance(x, (int, float)):
                return x
            elif x in order:
                return order[x]
            else:
                return np.nan

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)
        self._update_log('convert to numeric', 'replace',
                         ' | '.join(['%s >>> %s' % (g, i)
                                     for (g, i) in order.items()]))

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
        self._update_log('correct null values', 'drop', list(self.missing))

    def analysis_remap_values(self, map_, remap):
        """Remaps the values in the mapping file

        Paramters
        ---------
        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.
        remap: function, dict
            Describes how the data should be remapped. If a dictionary is
            passed, the values will be replaced as keys. Otherwise, the
            the function will be used directly.
        """
        if isinstance(remap, dict):
            def remap_(x):
                if x in remap:
                    return remap[x]
                else:
                    return x
        else:
            remap_ = remap

        values = map_[self.name].unique()
        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)
        self._update_log('remap values', 'replace',
                         ' | '.join(['%s >>> %s' % (value, remap_(value))
                                     for value in values])
                         )

    def validate_map(self, map_):
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
        acceptable_values = set(self.missing).union(set(self.order))
        actual_values = set(map_[self.name])

        if not acceptable_values.issuperset(actual_values):
            descriptor = ['\t%s' % v
                          for v in (actual_values - acceptable_values)]
            self._update_log('Validate the mapping file', 'error',
                             'The following are not valid values: %s'
                             % ' | '.join(descriptor))
            raise ValueError('The following are not valid values: %s'
                             % ('\n'.join(descriptor)))
        else:
            self._update_log('Validate the mapping file', 'pass',
                             'The column meets requirements.')
