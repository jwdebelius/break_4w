true_values = {'yes', 'true', 1, 1.0, True}
false_values = {'no', 'false', 0, 0.0, False}
ebi_null = {'not applicable',
            'missing: not provided',
            'missing: not collected',
            'missing: restricted',
            'not provided',
            'not collected',
            'restricted',
            }
properties_str = {'name', 'description', 'clean_name', 'notes', 'units',
                      'original_name', 'var_labels', 'var_numbers', 'type'}
properties_num = {'frequency_cutoff', 'sig_figs', 'magnitude'}
properties_bin = {'free_response', 'mimarks', 'ordinal'}
properties_set = {'source_columns', 'ontology', 'derivative_columns', 
                  'blanks', 'ambigious'}
defaults = {'missing': ebi_null,
                'true_values': true_values,
                'false_values': false_values,
                'mimarks': False,
                'free_response': False,
                'magnitude': 1,
                }

{str: '%s', int: '%i', float: '%1.5f', bool: '%s'}