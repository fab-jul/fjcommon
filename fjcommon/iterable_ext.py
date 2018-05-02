import math


def chunks(l, num_chunks):
    """ :returns iterator going over list l in num_chunks chunks. the last chunk may be shorter """
    max_per_chunk = math.ceil(len(l) / num_chunks)
    for c in range(num_chunks - 1):
        yield l[c * max_per_chunk:(c+1) * max_per_chunk]
    yield l[(num_chunks - 1) * max_per_chunk:]  # last chunk may be shorter


def flag_first_iter(it):
    """ :returns an iterator yielding tuples (first, el), where first is True for the first element and False
    otherwise. """
    it = iter(it)
    try:
        first_el = next(it)
        yield True, first_el
    except StopIteration:
        return
    for el in it:
        yield False, el


def sliced_iter(it, slice_len, allow_smaller_final_slice=True):
    """ :returns iterator going over it in slices of length slice_len. """
    assert slice_len > 0
    current_slice = []
    for el in it:
        current_slice.append(el)
        if len(current_slice) == slice_len:
            yield current_slice
            current_slice = []
    if allow_smaller_final_slice and current_slice:
        yield current_slice


def printing_iterator(it, first_n=5):
    """ :returns it but as a side effect prints the first first_n elements in it. """
    for i, el in enumerate(it):
        if i < first_n:
            print(el)
        yield el


def iter_with_sliding_window(it, window_size):
    """ :returns iterator going over it with a moving window of size window_size.
    Example:
    >>> it = iter_with_sliding_window(range(5), 3)
    >>> next(it) -> (0, 1, 2)
    >>> next(it) -> (1, 2, 3)
    >>> next(it) -> (2, 3, 4)
    """
    it = iter(it)
    current_window = []
    for el in it:
        current_window.append(el)
        if len(current_window) == window_size:
            yield tuple(current_window)
            break
    for el in it:
        current_window.pop(0)
        current_window.append(el)
        yield tuple(current_window)


def get_element_at(element_idx, it):
    try:
        return it[element_idx]
    except TypeError:  # it is not a list
        pass
    for i, el in enumerate(it):
        if i == element_idx:
            return el
    raise IndexError('Iterator does not have {} elements'.format(element_idx))
