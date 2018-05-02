""" Extends functools """


import functools


def assert_post_cond(cond):
    """
    Decorator. Example:
    >>> @assert_post_cond(lambda a: a > 0)
    >>> def foo(bar):
    >>>     return max(bar, 0)
    :param cond: function mapping return value to bool
    """
    def decorator(f):
        def wrapped_f(*args, **kwargs):
            ret = f(*args, **kwargs)
            assert cond(ret)
            return ret
        return wrapped_f
    return decorator


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


def partial(f):
    """
    Enables creating of partial functions using the ellipsis constant. E.g.

    >>> import operator
    >>> append_world = partial(operator.add)(..., 'world')
    >>> append_world('hello')
    """
    def create_f(*create_args):
        def new_f(*passed_args):
            passed_args_it = iter(passed_args)
            try:
                args_for_f = [
                    next(passed_args_it) if create_arg is ... else create_arg
                    for create_arg in create_args]
            except StopIteration:
                num_expected = sum(1 for create_arg in create_args if create_arg is ...)
                num_actual = len(passed_args)
                create_args_str = tuple(create_arg if create_arg is not ... else '...'
                                        for create_arg in create_args)
                raise ValueError('Not enough arguments, expected {} ({}) got {} ({})'.format(
                        create_args_str, num_expected, passed_args, num_actual))
            return f(*args_for_f)
        return new_f
    return create_f


def return_list(f):
    """ Can be used to decorate generator functions. """
    return compose(list, f)


def return_tuple(f):
    """ Can be used to decorate generator functions. """
    return compose(tuple, f)


def unzip(gen):
    return zip(*list(gen))


def const(val):
    """ :returns a function f that returns val no matter with which arguments it is called. """
    def _inside(*args, **kwargs):
        return val
    return _inside


def catcher(exc_cls, handler, f):
    """ :returns a function f' that behaves like f except when an exception e of type exc_cls is caught, in which case
    handler(e) is returned """
    def _f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except exc_cls as e:
            return handler(e)
    return _f


def catch(exc_cls, handler):
    """ Like catcher but can be used as a decorator. """
    def decorator(f):
        return catcher(exc_cls, handler, f)
    return decorator


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

