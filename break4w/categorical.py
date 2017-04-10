import copy

import numpy as np
import pandas as pd

from break4w.question import Question


class Categorical(Question):

    def __init__(self, name, description, dtype, order, extremes=None,
                 frequency_cutoff=None, ambiguous_values=None,
                 clean_name=None, mimarks=False, ontology=None,
                 ebi_required=False, qiita_required=False,
                 missing=None):
        """A question object for categorical or ordinal questions

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        descrip tion : str
            A brief description of the data contained in the question
        dtype : object
            The datatype in which the responses should be represented. (i.e.
            float, int, str).
        order : list
            The list of responses to the question
        clean_name : str, optional
            A nicer version of the way the column should be named.
        frequency_cutoff : float, optional
            The minimum number of observations required to keep a sample group.
        ambigigous_values : str, list, optional
            A list of values which are considered ambiguous and removed when
            `drop_ambiguous` is `True`.
        free_response: bool, optional
            Whether the question is a free response question or controlled
            vocabulary
        mimmarks : bool, optional
            If the question was a mimmarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.
        ebi_required : bool, optional
            Describes whether the question is required by EBI
        qiita_required : bool, optional
            Is the question required by qiita
        missing : str, list, optional
            Acceptable missing values

        """
        if dtype not in {str, bool, int, float, tuple}:
            raise ValueError('%s is not a supported datatype for a '
                             'categorical variable.' % dtype)

        # Initializes the question
        Question.__init__(self, name, description, dtype,
                          clean_name=clean_name,
                          free_response=False,
                          mimarks=mimarks,
                          ontology=ontology,
                          ebi_required=False,
                          qiita_required=False,
                          missing=None,
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
        """Removes ambiguous groups from the mapping file

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        self.check_map(map_)

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
        """Removes groups below a frequency cutoff

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        self.check_map(map_)

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
        """Converts the data to integer values

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        self.check_map(map_)
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
        self.check_map(map_)
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
            If the column is not in the mapping file
        ValueError
            If the values in the mapping file are not acceptable values
            for the question (given by order) or acceptable missing values.

        """
        self.check_map(map_)
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
