import inspect


class Question:
    """A base object class for handling American Gut Data dictionary entries"""
    true_values = {'yes', 'Yes', 'YES', 'true', 'True', 'TRUE',
                   1, 1.0, True}
    false_values = {'no', 'No', 'NO', 'false', 'False', 'FALSE', 0, 0.0, False}

    def __init__(self, name, description, dtype, clean_name=None,
                 free_response=False, mimarks=False, ontology=None,
                 ebi_required=False, qiita_required=False, missing=None):
        """A base object for describing single question outputs

        The Question Object is somewhat limited in its functionality. For most
        questions in the dataset, it is better to use a child object with the
        appropriate question type.

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        dtype : object
            The datatype in which the responses should be represented. (i.e.
            float, int, str).
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        free_response: bool, optional
            Whether the question is a free response question or controlled
            vocabulary
        mimarks : bool, optional
            If the question was a mimarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.

        """

        # Checks the arguments
        if not isinstance(name, str):
            raise TypeError('name must be a string.')
        if not isinstance(description, str):
            raise TypeError('description must be a string')
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
        self.ebi_required = ebi_required
        self.qiita_required = qiita_required
        self.missing = missing

    def check_map(self, map_):
        """Checks the group exists in the metadata

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        Raises
        ------
        ValueError
            If the column identified by the question object is not part of the
            supplied DataFrame.

        """
        if self.name not in map_.columns:
            raise ValueError('%s is not a column in the supplied map!'
                             % self.name)

    def check_ontology(self):
        """
        Checks the ontology associated with the question

        To be added!
        """
        pass

    def check_requried(self):
        """Checks whether or not the question is a required question

        To be added!
        """
        pass

    def check_missing(self):
        """Checks whether the missing values are appropriate

        To be added!

        """
        pass
