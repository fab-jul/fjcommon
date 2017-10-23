
class _NoOp(object):
    """
    Class that silently ignores calls to any function.
    """
    def __getattr__(self, attr):
        return _no_op

    def __call__(self, *args, **kwargs):
        """ Means NoOp() works, but also NoOp()()()... """
        return NoOp


NoOp = _NoOp()


def _no_op(*args, **kwargs):
    return NoOp  # return NoOp s.t. NoOp.foo().bar().baz() works.

