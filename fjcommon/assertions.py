

def assert_exc(cond, msg=None, exc=ValueError):
    if not cond:
        raise exc(msg)
