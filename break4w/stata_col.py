def _categorize_stata_column(col_, ser_, stype, var_desc, qtypes=None, 
    infer=False):
    """
    Converts a stata column into something like a data dictionary result

    Parameters
    ----------
    col_: str
        The name of the column being handled
    ser_: Series
        The column from the metadata
    stype: str, optional
        The datatype specified by stata for the column
    var_desc: dictionary
        The variable descriptions from the pandas stata iterator object
    qtypes: dict, optional
        The question types which should be used for the questions, if known.
        If none are provided, then the dictionary will either try to infer
        the type, or return all objects as Question type.
    inter: bool, optional
        Whether question type should be infered from the column

    Returns
    -------
    str
        The type of question
    Question
        The generated question object
    """

    # Extracts the basic column information
    col_args = {'name': col_,
                'description': var_desc[col_]}

    clean_type = 

    ### Determines the datatype
    # If there is no question type given and we are not to infer, then
    # no assumptions are made and everything is a question
    if (qtypes is None) and not infer:
        question_type = 'question'
        dtype = clean_type
    # If there is no inference, then questions are pulled from the dictionary
    # if they exist and otherwise designated as a question
    elif not infer:
        question_type = qtypes.get(col_, 'question')
        dtype = clean_type
    # If the data is a categorical variable, then we assume it to be
    # categorical
    elif infer and isinstance(stype, CategoricalDtype):
        question_type = 'categorical'
        col_args['order'] = stype.categories
        if stype.ordered:
            col_args['var_labels'] = \
                {i + 1: g for i, g in enumerate(stype.categories)}
        dtype = str
    elif isinstance()
    

def _clean_dtype(type_):
    if isinstance(type_, CategoricalDtype):
        return 