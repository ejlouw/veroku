
class FactorTemplate:
    """
    A base class for factor templates.
    """

    def __init__(self, var_templates=None, conditioning_var_templates=None, conditional_var_templates=None):
        """
        Constructor
        """
        self._var_templates = None
        self._conditioning_var_templates = None
        self._conditional_var_templates = None
        if (conditioning_var_templates is None) and (conditional_var_templates is None):
            self._var_templates = var_templates
        elif (conditioning_var_templates is None) or (conditional_var_templates is None):
            error_msg = 'neither or both of conditioning_var_templates and conditional_var_templates should be None'
            raise ValueError(error_msg)
        else:
            self._conditioning_var_templates = conditioning_var_templates
            self._conditional_var_templates = conditional_var_templates

    def formattable(self, format_dict):
        try:
            if self._var_templates is not None:
                [vt.format(**format_dict) for vt in self._var_templates]
            else:
                [vt.format(**format_dict) for vt in self._conditioning_var_templates]
                [vt.format(**format_dict) for vt in self._conditional_var_templates]
            return True
        except KeyError:
            return False
