""" Extends functools """


# Allows user to `import functools_ext` without having to also `import functools`
from functools import *


def compose(*args):
    """
    :param args: a list of functions
    :return: composition of the functions
    """
    def compose2(f1, f2):
        def composed(*args_c, **kwargs_c):
            return f1(f2(*args_c, **kwargs_c))
        return composed

    return reduce(compose2, args)

