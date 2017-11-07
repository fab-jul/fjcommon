""" Extends functools """


import functools


def compose(*args):
    """
    :param args: a list of functions
    :return: composition of the functions
    """
    def compose2(f1, f2):
        def composed(*args_c, **kwargs_c):
            return f1(f2(*args_c, **kwargs_c))
        return composed

    return functools.reduce(compose2, args)


def unzip(gen):
    return zip(*list(gen))


def fst(t):
    return t[0]


def snd(t):
    return t[1]


def identity(*args):
    if len(args) == 0:
        return None
    if len(args) == 1:
        return args[0]
    return args

