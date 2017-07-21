from os import path
import pickle


def cached(cache_path, generator):
    """
    Takes a function `generator` that takes no arguments and returns any pickle-able object. Checks if the output of
    a previous call to that function has been stored at `cache_path`. If so, returns that output, otherwise calls
    `generator` to generate the output and save it at `cache_path`.
    """
    if path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    output = generator()
    with open(cache_path, 'wb+') as f:
        pickle.dump(output, f)
    return output


def cache(cache_path):
    """
    Decorator version of `cached`. Useful for when `cache_path` is static.
    The following two are equivalent:

    @cache('cache_path')
    def foo():
        return 42

    def foo():
        def _get():
            return 42
        return cached('cache_path', _get)

    """
    def cache_decorator(generator):
        def wrapper():
            return cached(cache_path, generator)
        return wrapper
    return cache_decorator


