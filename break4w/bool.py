import numpy as np
import pandas as pd

from break4w.categorical import Categorical


class Bool(Categorical):
    def __init__(self, name, description, clean_name=None, bool_format=None,
                 ambiguous_values=None, mimarks=False, ontology=None,
                 ebi_required=False, qiita_required=False,
                 missing=None):
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
        ambigious_values: str, optional
            A value indicating whether the respondant was unsure about the
            response. This is a different value than the participant failing
            to answer the question. (For instance, in response to a question
            like, "Are you pregnant?", the participant may not yet known but
            may suspect it.)
        mimmarks : bool, optional
            If the question was a mimmarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.

        """

        if bool_format is not None:
            t_format = bool_format[0]
            f_format = bool_format[1]
        else:
            t_format = 'true'
            f_format = 'false'

        if ambiguous_values is not None:
            adj_order = [t_format, f_format, ambiguous_values]
        else:
            adj_order = [t_format, f_format]


        Categorical.__init__(self, name=name,
                             description=description,
                             dtype=bool,
                             order=adj_order,
                             extremes=[t_format, f_format],
                             clean_name=clean_name,
                             ambiguous_values=ambiguous_values,
                             mimarks=mimarks,
                             ontology=ontology,
                             ebi_required=ebi_required,
                             qiita_required=qiita_required,
                             missing=missing
                             )
        self.type = 'Bool'

    def analysis_convert_to_word(self, map_):
        """Converts boolean values to 'yes' and 'no'

         map_ : DataFrame
            A pandas dataframe containing the column described by the question
            name.
        """
        self.check_map(map_)
        self.remap_dtype(map_)

        def remap_(x):
            if pd.isnull(x):
                return x
            if x:
                return 'yes'
            elif not x:
                return 'no'
            else:
                return np.nan

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)
        self._update_log('convert boolean', 'replace', 'standarize to yes/no')
