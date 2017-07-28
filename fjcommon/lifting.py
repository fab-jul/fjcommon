def lift1d(n):
    """ Convert the first n arguments of the decorated function to 1d lists if they are not passed as lists,
    then unpack lists before returning """
    def decorator(fn):
        def new_fn(*args, **kwargs):
            assert len(args) >= n, 'Expected first {} arguments to be non-keyword'
            args_to_lift, args_remaining = args[:n], args[n:]
            all_are_1d = all(not isinstance(arg, list) for arg in args_to_lift)
            none_is_1d = all(isinstance(arg, list) for arg in args_to_lift)
            all_or_none_1d = all_are_1d ^ none_is_1d
            assert all_or_none_1d, 'Expected first {} arguments either all to be lists, or none to be lists: {}'.format(
                n, args_to_lift)
            if all_are_1d:
                args_to_lift = tuple([arg] for arg in args_to_lift)
            args_lifted = args_to_lift + args_remaining
            ret = fn(*args_lifted, **kwargs)
            if not ret or not all_are_1d:
                return ret
            if isinstance(ret, tuple):  # unpack lists again
                return tuple(r[0] for r in ret)
            return ret[0]
        return new_fn
    return decorator
