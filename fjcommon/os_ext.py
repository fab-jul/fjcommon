""" Extends os """


import os


def chmodr(root, mode, upto=''):
    """ recursive chmod until root == upto, where it stops, i.e., upto is not modified. """
    if not root or os.path.normpath(root) == os.path.normpath(upto):
        return
    os.chmod(root, mode)
    chmodr(os.path.dirname(root), mode, upto)

