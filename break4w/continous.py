import numpy as np

from break4w.question import Question


class Continous(Question):

    def __init__(self, name, description, dtype=None, unit=None, limits=None,
                 rounding=None, clean_name=None, mimarks=False, ontology=None,
                 missing=None, blanks=None, colormap=None):
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
            colormap object or a string describing a matplotlib compatable
            colormap (i.e. `'RdBu'`).

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
                          mimarks=mimarks,
                          ontology=ontology,
                          missing=missing,
                          blanks=blanks,
                          colormap=colormap,
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
        """
        Removes datapoints outside of the specified limits

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
