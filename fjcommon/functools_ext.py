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


def return_list(f):
    """ Can be used to decorate generator functions. """
    return compose(list, f)


def return_tuple(f):
    """ Can be used to decorate generator functions. """
    return compose(tuple, f)


def unzip(gen):
    return zip(*list(gen))


def const(val):
    def _inside(*args, **kwargs):
        return val
    return _inside


def catcher(exc_cls, handler, f):
    def _f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except exc_cls as e:
            return handler(e)
    return _f


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

