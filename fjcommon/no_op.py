
class _NoOp(object):
    """
    Class that silently ignores calls to any function.
    """
    def __getattr__(self, attr):
        return _no_op

    def __call__(self, *args, **kwargs):
        """ Means NoOp() works, but also NoOp()()()... """
        return NoOp

    def __enter__(self):
        return NoOp

    def __exit__(self, exc_type, exc_val, exc_tb):
        return NoOp


NoOp = _NoOp()


def _no_op(*args, **kwargs):
    return NoOp  # return NoOp s.t. NoOp.foo().bar().baz() works.


def test_no_op():
    n = NoOp()
    n.foo().bar().baz()

    with n.some_ctx_mgr() as t:
        print(t.foo())
