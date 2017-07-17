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
