import numpy as np
import pandas as pd

from break4w.question import Question


class Continous(Question):

    def __init__(self, name, description, units, dtype=float, limits=None,
        outliers=None, sig_figs=None, magnitude=1, clean_name=None,
        mimarks=False, ontology=None, missing=None, blanks=None,
        colormap=None):
        """A Question object with continous responses

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        units : str
            The units of measure for the data type. Unitless quantities can be
            denoted with an empty string (`""`; i.e. reporting absorbance
            values from a spectrophotometer). This is required, even if the
            unitss are already recorded in the name. Units are awesome; without
            them your data becomes absurd and meaningless. Public serivce
            announcement over, back to your reguarly scheduled doc-string.
        dtype : {int, float}
            The datatype in which the responses should be represented.
        limits: two element iterable of numbers, optional
            Physical sonstraints on the value being measured. Essentially,
            these describe values which it is not possible for the measurement
            to take, due to constraints in the natural world.
            For instance,  absloute concentrations cannot be negative. So, a
            solution cannot contain -3 mM NaCl.
            Limits can be expressed in both directions, or in a single
            direction, with `None` replacing the missing value. So, for
            concentration, the limit could be represented as `[0, None]`.
        outliers : two element iterable of numbers, optional
            The range of values pertinant to analysis. This is seperate form
            the range of physical values *possible* for the data (provided in)
            limits. In the same way as `limits`, bounds can be skipped using
            `None` for the position.
        sig_figs : int, optional
            The number of signfiicant figures appropriate for the measurement.
            This should be expressed as a decimal (i.e. `sig_figs = 0.1`),
            specifying the units of error on the value.
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
        ValueError
            When the values in `outliers` are not expressed in order (i.e.
            the first value is larger than the second)
        """

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

        o_lower, o_upper = _check_limits(outliers, 'outliers')
        a_lower, a_upper = _check_limits(limits, 'limits')

        self.units = units
        self.bound_lower = a_lower
        self.bound_upper = a_upper
        self.outlier_lower = o_lower
        self.outlier_upper = o_upper
        self.type = 'Continous'
        self.sig_figs = sig_figs

    def dictionary_update_outliers(self, outliers):
        """
        Updates the bounds for outliers in the metadata column.

        Parameters
        ----------
        outliers : two element iterable of numbers, optional
            The range of values pertinant to analysis. This is seperate form
            the range of physical values *possible* for the data (provided in)
            limits.
            Outlires can be expressed in both directions, or in a single
            direction, with `None` replacing the missing value. So, for
            concentration, the limit could be represented as `[0, None]`.

        Raises
        ------
        ValueError
            When the values in `outliers` are not expressed in order (i.e.
            the first value is larger than the second)

        """
        ori_lower = self.outlier_lower
        ori_upper = self.outlier_upper
        lower, upper = _check_limits(outliers, 'outliers')
        self.outlier_lower = lower
        self.outlier_upper = upper
        self._update_log('update outlier values', 'update dictionary',
                         'Outlier values have been updated: %s > %s '
                         'and %s > %s.' % (ori_lower, lower, ori_upper, upper)
                         )

    def validate(self, map_):
        """
        Checks values fall into the acceptable range for this type of data.

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        Raises
        ------
        ValueError
            If the data in `map_` falls outside the specified limits
        """

        iseries = map_[self.name].copy()
        message = []

        # Attempts to remap the data
        if self.blanks is None:
            blanks = set([])
        elif isinstance(self.blanks, str):
            blanks = set([self.blanks])
        else:
            blanks = set(self.blanks)

        if hasattr(self, 'ambiguous') and self.ambiguous is not None:
            if isinstance(self.ambiguous, str):
                ambiguous = set([self.ambiguous])
            else:
                ambiguous = set(self.ambiguous)
        else:
            ambiguous = set([])

        placeholders = self.missing.union(blanks).union(ambiguous)
        f_ = self._identify_remap_function(dtype=self.dtype,
                                           placeholders=placeholders,
                                           true_values=self.true_values,
                                           false_values=self.false_values,
                                           )
        iseries = iseries.apply(f_)
        if np.any(iseries.apply(lambda x: x == 'error')):
            message = (
                'the data cannot be cast to %s'
                % (str(self.dtype).replace("<class '", '').replace("'>", ''))
                )
            self._update_log('validate', 'error', message)
            raise TypeError(message)
        else:
            self._update_log(
                'validate', 'pass', 'the data can be cast to %s'
                % (str(self.dtype).replace("<class '", '').replace("'>", '')))

        iseries = iseries.replace(list(placeholders), np.nan).dropna()

        # Defines the text based on the bounding values
        if (self.bound_upper is not None) and (self.bound_lower is not None):
            update_text = ('The values were between %s and %s %s.'
                           % (self.bound_lower, self.bound_upper, self.units))
        elif self.bound_upper is not None:
            update_text = ('The values were less than or equal to %s %s.'
                           % (self.bound_upper, self.units))
        elif self.bound_lower is not None:
            update_text = ('The values were greater than or equal to %s %s.'
                           % (self.bound_lower, self.units))
        else:
            update_text = 'there were no limits specified'

        lower_text = ''
        lower_issue = False
        if (self.bound_lower is not None):
            if np.any(iseries < self.bound_lower):
                lower_text = ('less than %s' % self.bound_lower)
                lower_issue = True

        upper_text = ''
        upper_issue = False
        if (self.bound_upper is not None):
            if np.any(iseries > self.bound_upper):
                upper_text = ('greater than %s'
                              % (self.bound_upper))
                upper_issue = True

        error = False
        if lower_issue and upper_issue:
            error = True
            error_string = ('There are values %s and %s %s.'
                            % (lower_text, upper_text, self.units))
        elif lower_issue:
            error = True
            error_string = ('There are values %s %s.'
                            % (lower_text, self.units))
        elif upper_issue:
            error = True
            error_string = ('There are values %s %s.'
                            % (upper_text, self.units))

        if error:
            self._update_log('validate', 'error',
                             error_string)
            raise ValueError(error_string)
        else:
            self._update_log('validate', 'pass',
                             update_text)

    def to_dict(self):
        """Converts the question column to a dictionary
        """
        tent_dict = self.__dict__.items()

        def _check_dict(k, v):
            if k in {'log', 'type', 'bound_lower', 'bound_upper', 
                     'outlier_lower', 'outlier_upper'}:
                return False
            if v is None:
                return False
            elif isinstance(v, list) and (len(v) == 0):
                return False
            elif ((k in self.defaults) and 
                (self.defaults[k] == v)):
                return False
            else:
                return True

        summ_dict = {k:v for k, v in tent_dict if _check_dict(k, v)}
        if ((self.bound_lower is not None) or (self.bound_upper is not None)):
            summ_dict['limits'] = [self.bound_lower, self.bound_upper]

        if ((self.outlier_lower is not None) or 
            (self.outlier_upper is not None)):
            summ_dict['outliers'] = [self.outlier_lower, self.outlier_upper]

        return (self.type.lower(), summ_dict)

    def _to_series(self):
        """Formats data to be written to tsv"""
        tent_dict = self.__dict__.items()

        def _check_dict(k, v):
            if k in {'log', 'bound_lower', 'bound_upper', 
                     'outlier_lower', 'outlier_upper'}:
                return False
            if v is None:
                return False
            elif isinstance(v, list) and (len(v) == 0):
                return False
            elif ((k in self.defaults) and 
                (self.defaults[k] == v)):
                return False
            else:
                return True

        def _format_value(v):
            if isinstance(v, list):
                return ' | '.join(v)
            if isinstance(v, (set, tuple)):
                return ' | '.join(list(v))
            else:
                str_ = str(v)
                str_1 = str_.replace("<class '", '').replace("'>", "")
                return str_1

        dict_ = {k: _format_value(v) for k, v in tent_dict 
                 if _check_dict(k, v)}

        if ((self.bound_lower is not None) or (self.bound_upper is not None)):
            dict_['order'] = \
                ('%s | %s' % (self.bound_lower, self.bound_upper))

        if ((self.outlier_lower is not None) or 
            (self.outlier_upper is not None)):
            dict_['ambiguous'] = \
                ('%s | %s') % (self.outlier_lower, self.outlier_upper)

        return pd.Series(dict_)


def _check_limits(limits, var_name):
    """
    A helper function that checks whether the limit value meets the criteria
    """
    if limits is not None:
            lower, upper = limits
            if ((lower is not None) and (upper is not None) and
                    (lower > upper)):
                raise ValueError('The lower limit cannot be greater than '
                                 'the upper for %s.')
    else:
        lower, upper = (None, None)

    return lower, upper


