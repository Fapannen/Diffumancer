import copy


class GridInferenceSettings:
    """A helper class that builds all combinations of
    settings to be evaluated from passed kwargs or a
    dictionary representing the kwargs.

    Consider a simple example, our pipeline contains
    hyperparameters 'a', 'b' and 'c' and we want to
    iterate over all comnbinations of settings stemming,
    from the following domains

    eg. a ∈ [1, 2]
        b ∈ [-1, -2, -3]
        c = "constant"

    In essence, we want to run the pipeline with settings

    (a = 1, b = -1, c = "constant")
    (a = 1, b = -2, c = "constant")
    (a = 1, b = -3, c = "constant")
    (a = 2, b = -1, c = "constant")
    ...

    This class builds a list of dictionaries defining the
    individual settings, that can then be passed into the
    pipeline as kwargs.

    If you have a hyperparameter at which you want to explore
    different values, define the values as a list, eg.
    a = [1, 2, "something", ClassWhatever(...)]

    If you have a hyperparameter which you want to keep constant,
    define it either in a standard way, or as a list of a single
    element, eg.
    a = 1
    a = [1]
    """

    def __init__(self, *args, **kwargs):
        """Initialize the dictionary from args and kwargs.
        If a single argument is provided, it is expected to be a valid
        settings dictionary. If pure keyword arguments are provided,
        the dictionary is built on the fly.

        Thus, you can initialize this helper class equivalently

        a = GridInferenceSettings(a=1, b=2)
        a = GridInferenceSettings({"a":1,"b":2})
        """
        self.settings = {}

        if len(args) > 1:
            raise ValueError(
                "Invalid setting provided. Either pass kwargs direcly or "
                "via a (single) dictionary"
            )

        if len(args) == 1 and isinstance(args[0], dict):
            self.settings = args[0]

        else:
            for key, value in kwargs.items():
                self.settings[key] = value if isinstance(value, list) else [value]

    def get_settings(self) -> list[dict[str, list]]:
        """Build all the individual setting combinations
        that can be passed as kwargs to some other funtions

        Returns
        -------
        list[dict[str, list]]
            list of kwargs representing the various settings, eg.
            [
                {"a":1, "b":2},
                {"a":1, "b":3},
                ...
            ]
        """
        settings_list = []

        for key, vals in self.settings.items():
            if len(settings_list) == 0:
                settings_list = [{key: val} for val in vals]
            else:
                new_settings = []
                for existing_setting in settings_list:
                    for val in vals:
                        new_setting = copy.deepcopy(existing_setting)
                        new_setting[key] = val
                        new_settings.append(new_setting)
                settings_list = new_settings

        return settings_list
