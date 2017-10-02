""" Extends os """


import os


def chmodr(root, mode, upto=''):
    """ recursive chmod until root == upto, where it stops, i.e., upto is not modified. """
    if not root or os.path.normpath(root) == os.path.normpath(upto):
        return
    os.chmod(root, mode)
    chmodr(os.path.dirname(root), mode, upto)


def listdir_paths(p):
    """
    Like os.listdir, but yield full paths
    """
    return (os.path.join(p, sub_dir) for sub_dir in os.listdir(p))

