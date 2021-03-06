import numpy as np
import pandas as pd

from break4w.question import Question


class Continous(Question):

    def __init__(self, name, description, units=np.nan, dtype=float, 
        limits=None, 
        sig_figs=None, magnitude=1, order=None, **kwargs):
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
        sig_figs : int, optional
            The number of signfiicant figures appropriate for the measurement.
            This should be expressed as a decimal (i.e. `sig_figs = 0.1`),
            specifying the units of error on the value.
        magnitude: int, optional
            Describes the order of magnitude for the value. For instance

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
                          **kwargs
                          )
        if isinstance(order, list) and limits is None:
            limits = order
        self.limits = self._check_limits(limits, var_name='limits')

        self.units = units
        self.magnitude = magnitude

        self.type = 'Continous'
        self.sig_figs = sig_figs

    def __str__(self):
        """
        Prints a nice summary of the object
        """
        s_ = """
------------------------------------------------------------------------------------
{name} (Continous {dtype_str})
    {description}
------------------------------------------------------------------------------------
limits      {limits}
units       {units}
missing     {missing}
blanks      {blanks}
------------------------------------------------------------------------------------
        """
        def _check_limits(limits):
            if limits == [None, None]:
                return 'None'
            else: 
                return self._iterable_to_str(
                    limits,  
                    var_str=self.var_str_format.get(self.dtype, '%s')
                    )

        def _check_units(units, magnitude):
            if (((magnitude is None) | (magnitude == 1)) & pd.isnull(units)):
                return '{name} is unitless'.format(name=self.name)
            elif ((magnitude is None) | (magnitude == 1)):
                return units
            elif pd.isnull(units):
                return '{mag} (unitless)'.format(mag=magnitude)
            else:
                return '{mag} {units}'.format(mag=magnitude, units=units)

        def _check_missing(missing):
            if missing == self.ebi_null:
                return 'default'
            else:
                return self._iterable_to_str(
                    missing,  
                    var_str='%s',
                    )
        return s_.format(name=self.name, 
                         dtype_str=self._iterable_to_str(self.dtype),
                         description=self.description,
                         limits=_check_limits(self.limits),
                         units=_check_units(self.units, self.magnitude),
                         missing=_check_missing(self.missing),
                         blanks=self._iterable_to_str(self.blanks)
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

        if pd.isnull(self.units):
            unit_str = ''
        else:
            unit_str = self.units

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
        [lower_, upper_] = self.limits

        # Defines the text based on the bounding values
        if (lower_ is not None) and (upper_ is not None):
            update_text = ('The values were between %s and %s %s'
                           % (lower_, upper_, unit_str))
        elif upper_ is not None:
            update_text = ('The values were less than or equal to %s %s'
                           % (upper_, unit_str))
        elif lower_ is not None:
            update_text = ('The values were greater than or equal to %s %s'
                           % (lower_, unit_str))
        else:
            update_text = 'there were no limits specified'

        lower_text = ''
        lower_issue = False
        if (lower_ is not None):
            if np.any(iseries < lower_):
                lower_text = ('less than %s' % lower_)
                lower_issue = True

        upper_text = ''
        upper_issue = False
        if (upper_ is not None):
            if np.any(iseries > upper_):
                upper_text = ('greater than %s' % (upper_))
                upper_issue = True

        error = False
        if lower_issue and upper_issue:
            error = True
            error_string = ('There are values %s and %s %s'
                            % (lower_text, upper_text, unit_str))
        elif lower_issue:
            error = True
            error_string = ('There are values %s %s'
                            % (lower_text, unit_str))
        elif upper_issue:
            error = True
            error_string = ('There are values %s %s'
                            % (upper_text, unit_str))

        if error:
            self._update_log('validate', 'error',
                             error_string)
            raise ValueError(error_string)
        else:
            self._update_log('validate', 'pass',
                             update_text)

    @staticmethod
    def _check_limits(limits, var_name):
        """
        A helper function that checks whether the limit value meets the criteria
        """
        if limits is not None:
                lower, upper = limits
                if ((lower is not None) and (upper is not None) and
                        (lower > upper)):
                    raise ValueError('The lower limit cannot be greater than '
                                     'the upper for %s')
        else:
            lower, upper = (None, None)

        return [lower, upper]

    @staticmethod
    def _check_cmap(x, num_colors=None):
        """
        Checks that a read object qualifies as a colormap
        """
        if isinstance(x, str):
            if colors.is_color_like(x):
                c = np.array([colors.to_rgba(x)])
            elif x in cm.cmap_d.keys():
                c = x
            else:
                raise ValueError('%s is not an acceptable colormap' % x)
        elif (isinstance(x, list) | isinstance(x, np.ndarray)):
            if np.all([colors.is_color_like(c) for c in x]):
                c = np.array([colors.to_rgba(c) for c in x])




def _check_cmap(x):
    """Checks that a read object qualifies as a colormap

    To be developed further!
    """
    pass
    # if isinstance(x, str):
    #     if colors.is_color_like(x):
    #         c = np.array([colors.to_rgba(x)])
    #     elif x in cm.cmap_d.keys():
    #         c = x
    #     else:
    #         raise ValueError('%s is not an acceptable colormap' % x)



    # return x