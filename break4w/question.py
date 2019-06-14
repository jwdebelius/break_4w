import datetime
from functools import partial
import inspect
import pydoc

import numpy as np
import pandas as pd

import break4w._defaults as b4wdefaults


class Question:
    u"""A base object class for handling Data dictionary entries
    """
    true_values = b4wdefaults.true_values
    false_values = b4wdefaults.false_values
    ebi_null = b4wdefaults.ebi_null
    defaults = b4wdefaults.defaults
    var_str_format = {str: '%s', int: '%i', float: '%1.5f', bool: '%s'}

    def __init__(self, name, description, dtype, clean_name=None,
        question=None, format=None,
        free_response=False, mimarks=False, ontology=None,
        missing=None, blanks=None, colormap=None, original_name=None,
        source_columns=None, derivative_columns=None, notes=None,
        **other_properties):
        u"""A base object for describing single question outputs

        The Question Object is somewhat limited in its functionality. For most
        questions in the dataset, it is better to use a child object with the
        appropriate question type (i.e. Categorical, Bool, Continous, Dates).

        Parameters
        ----------
        name : str
            The name of a column in a microbiome mapping file where metadata
            describing a clincial or enviromental factor is stored.
        description : str
            A brief description of the biological relevance of the information
            in the column. This can also be used to clarify acronyms or
            definations.
        dtype : object
            The datatype in which the responses should be represented. (i.e.
            `float`, `int`, `str`).
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
            colormap object, a string describing a matplotlib compatable
            colormap (i.e. `'RdBu'`), or an iterable of matplotlib compatable
            color values.
        original_name: str, optional
            The name of the column in a previous iteration of the metadata
            (often the version of the metadata provided by the collaborator).
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
            The dtype is not a class
        TypeError
            The `clean_name` is not a string.

        """

        # Checks the core arguments
        if not isinstance(name, str):
            raise TypeError('name must be a string.')
        if not isinstance(description, str):
            raise TypeError('description must be a string')
        elif len(description) > 80:
            raise ValueError('The description must be less than 80 '
                             'characters')
        if not inspect.isclass(dtype):
            raise TypeError('dtype must be a class')
        if not isinstance(clean_name, str) and clean_name is not None:
            raise TypeError('If supplied, clean_name must be a string')


        # Handles the main information about the data
        self.name = name
        self.description = description
        self.dtype = dtype

        self.type = 'Question'
        if clean_name is None:
            self.clean_name = name.replace('_', ' ').title()
        else:
            self.clean_name = clean_name

        # Sets up
        self.free_response = free_response
        self.mimarks = mimarks
        self.ontology = ontology
        if missing is None:
            self.missing = self.ebi_null
        elif isinstance(missing, str):
            self.missing = set([missing])
        else:
            self.missing = set(missing)
        self.blanks = blanks
        self.colormap = colormap
        self.original_name = original_name

        if source_columns is None:
            self.source_columns = []
        else:
            self.source_columns = source_columns

        if derivative_columns is None:
            self.derivative_columns = []
        else:
            self.derivative_columns = derivative_columns

        self.notes = notes
        for k, v in other_properties.items():
            setattr(self, k, v)

        self.log = []

    def __str__(self):
        u"""Prints a nice summary of the object"""
        return '%s (%s)\n\t%s' % (self.name, self.type, self.description)

    def _update_log(self, command, transform_type, transformation):
        u"""A helper function to update the in-object documentation object

        Every time a Question acts on data, a record should be made of
        the transformation. This function standardized the format of that
        recording by tracking the time, location, and command.
        Examples of how the function is used can be found in all functions
        which operate on a `map_` variable.

        Parameters
        ----------
        command : str
            A short textual description of the command performed. This
            may be the function name in text format.
        transform_type: str
            A more general description of the type of action that was
            performed. Ideally, this comes for a preset list of possible
            actions, and the descriptions are consistent.
        transformation: str
            Explains exactly how values were changed.

        """
        self.log.append({
            'timestamp': datetime.datetime.now(),
            'column': self.name,
            'command': command,
            'transform_type': transform_type,
            'transformation': transformation,
            })
    
    def write_provenance(self):
        """Writes the question provenance to a string

        Returns
        -------
        DataFrame
            A pandas dataframe describing the time, column, action taken,
            action type, and changes in the data.
        """
        self._update_log('Write Log', 'recording', '')
        return pd.DataFrame(self.log)[['timestamp', 'command', 'column',
                                       'transform_type', 'transformation']]

    def _read_provenance(self, fp_):
        """
        Reads the existing question provenance

        to be added!

        """
        raise NotImplementedError

    def _check_ontology(self):
        """
        Checks the ontology associated with the question

        To be added!
        """
        raise NotImplementedError

    @staticmethod
    def _iterable_to_str(val_, code_delim='=', var_delim=' | ', var_str='%s',
        code_str='%s', null_value='None'):
        """
        Converts a list or dict into a delimited string for reading
        """
        def _to_str(x):
            if pd.isnull(x):
                return null_value
            else:
                return var_str % x
        if (isinstance(val_, (list, set, tuple, np.ndarray, dict)) and 
             len(val_) == 0):
            return null_value
        if isinstance(val_, (list, set, tuple, np.ndarray)):
            return var_delim.join([_to_str(v) for v in val_])
        elif isinstance(val_, dict):
            return var_delim.join([
                ('%s%s%s' % (var_str, code_delim, code_str )) % (k, v)
                 for k, v in val_.items()
                ])
        elif val_ is None:
            return null_value
        else:
            return str(val_).replace("<class '", '').replace("'>", "")

    @staticmethod
    def _iterable_from_str(val_, code_delim='=', var_delim=' | ', 
        var_type=str, code_type=str, null_value=np.nan, return_type=set):
        """
        Converts a delimited string into a list or dict
        """ 
        def check_null(x):
            if (x in {null_value, 'None', None}):
                return None
            else:
                return var_type(x)
        if val_ in {null_value, 'None', None}:
            return None
        elif code_delim in val_:
            def get_k(v):
                return var_type(v.split(code_delim)[0])
            def get_v(v):
                return code_type(v.split(code_delim)[1])
            return {get_k(x): get_v(x) for x in val_.split(var_delim)}
        elif var_delim in val_:
            return return_type([check_null(x) for x in val_.split(var_delim)])
        # elif val_ in {null_value, 'None', None}:
        #     return None
        else:
            return return_type([check_null(val_)])

    def _to_series(self, code_delim='=', var_delim=' | ', 
        var_str=None, code_str='%s', null_value='None'):
        """Formats data as a series of text values"""

        tent_dict = self.__dict__.items()

        def _check_dict(k, v):
            if k in {'log'}:
                return False
            elif ((v is None) or 
                (isinstance(v, (list, set, dict)) and (len(v) == 0))):
                return False
            elif ((k in self.defaults) and (self.defaults[k] == v)):
                return False
            else:
                return True

        if var_str is None:
            var_str = self.var_str_format.get(self.dtype, '%s') 

        f_ = partial(self._iterable_to_str, code_delim=code_delim, 
                     var_delim=var_delim, var_str=var_str, code_str=code_str, 
                     null_value=null_value)

        return pd.Series({k: f_(v) for k, v in tent_dict 
                         if _check_dict(k, v)})

    def _to_usgs(self):
        """Converts question object to usgs xml format

        see: https://www.usgs.gov/products/data-and-tools/data-management/data-dictionaries
        """
        pass


    @classmethod
    def _read_series(cls, var_, var_delim=' | ', code_delim='=', null_value='None'):
        """
        Builds a question object off a series

        Parameters
        ----------
        var_: Series
            The series containing the parameters
        var_delim: str, optional
            The seperator between values in the "order" column.
        code_delim: str, optional
            The delimiter between a numericly coded categorical variable and
            the value it maps to.

        Returns
        -------
        Question

        """
        # Drops out type, if necessary
        if 'type' in var_:
            var_.drop('type', inplace=True)

        # Extracts the datatype
        dtype_ = pydoc.locate(var_['dtype'])
        var_['dtype'] = dtype_

        i_param = {'code_delim': code_delim, 
                    'var_delim': var_delim, 
                    'null_value': null_value,
                    'var_type': dtype_}

        def _handle_col(k, v):
            if pd.isnull(v) or v == str(null_value):
                return None
            if k == 'colormap':
                return _check_cmap(v)
            elif (k == 'ref_value') and (dtype_ is bool):
                return pydoc.locate(v.title())
            elif (k == 'ref_value'):
                return dtype_(v)
            elif (k in {'order', 'limits'}) and (dtype_ is bool):
                s_ = cls._iterable_from_str(
                    v, return_type=list, code_delim=code_delim, 
                    var_delim=var_delim, null_value=null_value)
                return [pydoc.locate(v_.title()) for v_ in s_]
            elif k in {'order', 'limits'}:
                return cls._iterable_from_str(v, return_type=list, **i_param)
            elif k in b4wdefaults.properties_num:
                return float(v)
            elif k in b4wdefaults.properties_bin:
                return pydoc.locate(v.title())
            elif k in b4wdefaults.properties_set:
                return cls._iterable_from_str(v, **i_param)
            else:
                return v

        dict_ = {k: _handle_col(k, v) for k, v in var_.iteritems()
                 if (not (pd.isnull(v) or v == str(null_value)) or 
                     not k in {'type', 'dtype'})}

        if ('order' in dict_) and isinstance(dict_['order'], dict):
            part_ = dict_['order']
            dict_['order'] = [dtype_(k) for k in part_.keys()] 
            dict_['var_labels'] = part_

        return cls(**dict_)

    @staticmethod
    def _identify_remap_function(dtype, placeholders=None, 
        true_values=true_values, false_values=false_values):
        """
        Selects an appropriate function to convert data from str to dtype

        Parameters
        ----------
        dtype : object
            The datatype in which the responses should be represented. (i.e.
            `float`, `int`, `str`).
        placeholders : set, optional
            Acceptable values to be ignored representing either placeholder 
            values such as text for missing values, blanks, or ambigious 
            measurements.
        true_values : set, optional
            Acceptable values for true values for boolean data
        false_values : set, optional
            Acceptable values for false values for boolean data

        Returns
        -------
        Function
            A function to convert the strings to the correct data type. The
            function will return "error" if the value cannot be cast 
            appropriately.
        """
        if placeholders is None:
            placeholders = []
        if dtype == bool:
            #  Converts any non-placeholder string values to lowercase
            def clean_up_strings(x):
                if isinstance(x, str) and (x not in placeholders):
                    return x.lower()
                else:
                    return x

            def remap_(x):
                if (x in placeholders) or (pd.isnull(x)):
                    return x
                x = clean_up_strings(x)

                if x in true_values:
                    return True
                elif x in false_values:
                    return False
                else:
                    return 'error'
        else:
            # Defines a function to clean up all other datatypes
            def remap_(x):
                if (x in placeholders) or pd.isnull(x):
                    return x
                else:
                    try:
                        return dtype(x)
                    except:
                        return 'error'

        return remap_


def _check_cmap(x):
    """Checks that a read object qualifies as a colormap

    To be developed further!
    """
    return x

