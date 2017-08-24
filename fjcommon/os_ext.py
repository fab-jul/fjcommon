""" Extends os """


# Allows user to `import os_ext` without having to also `import os`
from os import *


def chmodr(root, mode, upto=''):
    """ recursive chmod until root == upto, where it stops, i.e., upto is not modified. """
    if not root or path.normpath(root) == path.normpath(upto):
        return
    chmod(root, mode)
    chmodr(path.dirname(root), mode, upto)

