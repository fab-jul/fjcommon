from os import path
import pickle


def cached(cache_path, generator):
    if path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    output = generator()
    with open(cache_path, 'wb+') as f:
        pickle.dump(output, f)
    return output


def cache(cache_path):
    """
    Decorator. Decorates a function that takes no arguments and returns any pickle-able object. Useful for when
    `cache_path` is static.
    """
    def cache_decorator(generator):
        def wrapper():
            return cached(cache_path, generator)
        return wrapper
    return cache_decorator


