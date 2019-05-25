
class _NoOp(object):
    """
    Class that silently ignores calls to any function. Use the singleton instance defined below, i.e.,
        from fjcommon.no_op import NoOp
    (See README.md)
    """
    def __getattr__(self, attr):
        return _no_op

    def __call__(self, *args, **kwargs):
        """ Means NoOp() works, but also NoOp()()()... """
        return NoOp

    def __getitem__(self, item):
        return NoOp

    def __setitem__(self, key, value):
        pass

    # support for context manager calls
    def __enter__(self):
        return NoOp

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Singleton instance that should be used
NoOp = _NoOp()


def _no_op(*args, **kwargs):
    return NoOp  # return NoOp s.t. NoOp.foo().bar().baz() works.


def test_no_op():
    n = NoOp
    n.foo().bar().baz()

    with n.some_ctx_mgr() as t:
        print(t.foo())

    import pytest
    with pytest.raises(ValueError):
        with n.sommmm():
            raise ValueError()


def test_dict():
    n = NoOp
    n['foo']
    n['bar'] = 123
