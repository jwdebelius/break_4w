import pandas as pd

from break4w.categorical import Categorical


class Bool(Categorical):
    def __init__(self, name, description, clean_name=None, bool_format=None,
        ambiguous=None, mimarks=False, ontology=None,
        missing=None, blanks=None, colormap=None):
        """A question object for boolean question

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        clean_name : str, optional
            A nicer version of the way the column should be named.
        bool_format: list, optional
            The format expected for the boolean values. For example, `['True',
            'False']`, `[0, 1]`, `[True, False]`. By default, data is assumed
            to be a lower case text string (`['true', 'false']`).
        ambiguous: str, set, optional
            A value indicating whether the respondant was unsure about the
            response. This is a different value than the participant failing
            to answer the question. (For instance, in response to a question
            like, "Are you pregnant?", the participant may not yet known but
            may suspect it.)
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

        """

        if bool_format is not None:
            t_format = bool_format[0]
            f_format = bool_format[1]
        else:
            t_format = 'true'
            f_format = 'false'

        Categorical.__init__(self,
                             name=name,
                             description=description,
                             dtype=bool,
                             order=[t_format, f_format],
                             extremes=[t_format, f_format],
                             clean_name=clean_name,
                             ambiguous=ambiguous,
                             mimarks=mimarks,
                             ontology=ontology,
                             missing=missing,
                             blanks=blanks,
                             colormap=colormap,
                             )
        self.type = 'Bool'

    def analysis_convert_to_word(self, map_):
        """Converts boolean values to 'yes' and 'no'

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """

        def remap_(x):
            if pd.isnull(x):
                return x
            elif (x in self.ambiguous) and (self.ambiguous is not None):
                return x
            if isinstance(x, bool) and x:
                return 'yes'
            elif isinstance(x, bool) and (not x):
                return 'no'
            else:
                return 'error'

        self.analysis_apply_conversion(map_, remap_, None, False)

        if map_[self.name].apply(lambda x: x == 'error').any():
            self._update_log('convert boolean', 'replace',
                             'data could not be standardized')
            raise ValueError('data could not be standardized')
        self._update_log('convert boolean', 'replace', 'standarize to yes/no')
