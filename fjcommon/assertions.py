

def assertExc(cond, msg, exc=ValueError):
    if not cond:
        raise exc(msg)
