import math
import numpy as np


def reshape_into_tiles(a, axis, filler_mat=None):
    assert a.ndim == 3, 'Expected 3D array, got {}'.format(a.shape)

    def get_tile_from_a(tile_i):
        return a[get_idx_slice(axis, tile_i, ndim=3)]

    C = a.shape[axis]
    if filler_mat is None:
        filler_mat = np.zeros_like(get_tile_from_a(0))

    tile_h = int(math.ceil(math.sqrt(C)))
    tile_w = int(math.ceil(C / tile_h))
    assert tile_w * tile_h >= C

    return np.vstack([
        np.hstack([get_tile_from_a(c) if c < C else filler_mat
                   for c in range(row_i * tile_w, (row_i+1) * tile_w)])
        for row_i in range(tile_h)
    ])


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_idx_slice(axis, i, ndim):
    """
    Returns a slice s where

    mat[s] == mat[:, ---, :, i, :, ---, :]
                  |_______|     |_______|
                  axis          ndim - axis - 1

    This allows indexing into an arbitrary dimension of an n-dimensional array.
    """
    idx = [slice(None)] * ndim
    idx[axis] = i
    return idx

