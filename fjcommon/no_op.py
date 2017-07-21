
class NoOp(object):
    """
    Class that silently ignores calls to any function.
    """
    def __getattr__(self, attr):
        return _no_op


def _no_op(*args, **kwargs):
    return None
