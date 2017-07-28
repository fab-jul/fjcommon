import math


def chunks(l, num_chunks):
    max_per_chunk = math.ceil(len(l) / num_chunks)
    for c in range(num_chunks - 1):
        yield l[c * max_per_chunk:(c+1) * max_per_chunk]
    yield l[(num_chunks - 1) * max_per_chunk:]  # last slice may be shorter


def get_element_at(element_idx, it):
    try:
        return it[element_idx]
    except TypeError:
        pass
    for i, el in enumerate(it):
        if i == element_idx:
            return el
    raise IndexError('Iterator does not have {} elements'.format(element_idx))
