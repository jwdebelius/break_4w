import numpy as np

from break4w.question import Question


class Continous(Question):

    def __init__(self, name, description, dtype=None, unit=None, limits=None,
                 rounding=None, clean_name=None, mimarks=False, ontology=None,
                 ebi_required=False, qiita_required=False, missing=None):
        """A Question object with continous responses

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        unit : str, optional
            The unit of measure for the data type, if relevant. Units
            are awesome, and should be included whenever relevant.
        limits : two-elemant iterable
            The range of values pertinant to analysis.
        rounding : int, optional
            The number of digits to which results should be rounded.
        clean_name : str, optional
            A nicer version of the way the column should be named.
        mimarks : bool, optional
            If the question was a mimarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.
        ebi_required : bool, optional
            Describes whether the question is required by EBI
        qiita_required : bool, optional
            Is the question req,uired by qiita
        missing : str, list, optional
            Acceptable missing values

        Raises
        ------
        ValueError
            When the values in `limits` are not expressed in order (i.e.
            the first value is larger than the second)
        """

        if dtype is None:
            dtype = float

        Question.__init__(self,
                          name=name,
                          description=description,
                          dtype=dtype,
                          clean_name=clean_name,
                          free_response=False,
                          mimarks=mimarks,
                          ontology=ontology,
                          qiita_required=qiita_required,
                          ebi_required=ebi_required,
                          missing=missing
                          )
        self.unit = unit
        if limits is not None:
            lower, upper = limits
            if lower > upper:
                raise ValueError('The lower limit cannot be greater than '
                                 'the upper!')
        else:
            lower, upper = (None, None)

        self.lower = lower
        self.upper = upper
        self.type = 'Continous'
        self.rounding = rounding

    def analysis_drop_outliers(self, map_):
        """Removes datapoints outside of the default limits.

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
        self.analysis_remap_dtype(map_)

        if self.lower is not None:
            def remap_(x):
                if x < self.lower:
                    return np.nan
                elif x > self.upper:
                    return np.nan
                else:
                    return x

            map_[self.name] = map_[self.name].apply(remap_)
            self._update_log('drop outliers', 'drop',
                             'values outside [%f, %f]'
                             % (self.lower, self.upper))
