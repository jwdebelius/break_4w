import pandas as pd

from break4w._categorical import Categorical


class Bool(Categorical):
    def __init__(self, name, description, dtype=bool, clean_name=None, 
                  bool_format=None, ref_value=None, **kwargs):
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

        if ref_value is None:
            ref_value = f_format

        if 'dtype' not in kwargs:
            kwargs['dtype'] = dtype
        if 'order' not in kwargs:
            kwargs['order'] = [f_format, t_format]
        if name not in kwargs:
            kwargs['name'] = name
        if 'description' not in description:
            kwargs['description'] = description
        if 'clean_name' not in kwargs:
            kwargs['clean_name'] = clean_name
        if 'ref_value' not in kwargs:
            kwargs['ref_value'] = ref_value

        Categorical.__init__(self, **kwargs)
        self.type = 'Bool'

