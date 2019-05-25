
def assert_exc(cond, msg=None, exc=ValueError):
    """ replacement for assert that throws the specified exception instead of AssertionError """
    if not cond:
        raise exc(msg)


