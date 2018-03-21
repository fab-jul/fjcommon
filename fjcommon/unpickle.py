"""
Helps with messing around with pickles in the REPL. Basically keeps around path for easy saving and backup.
Example:

>>> from fjcommon.unpickle import *
>>> c = unpickle('some_rick.pkl')
>>> new_c = [e for e in c if ...]
>>> make_backup()
>>> write(new_c)
"""

import os
import shutil
import pickle


_path = None


def _set_path(p):
    global _path
    _path = p


def _get_path():
    if not _path:
        raise ValueError('No path set. Call unpickle(p) first!')
    return _path


def unpickle(p):
    content = pickle.load(open(p, 'rb'))
    _set_path(p)
    return content


def write(content):
    pickle.dump(content, open(_get_path(), 'wb'))


def make_backup():
    p_head, p_ext = os.path.splitext(_get_path())
    p_bak = p_head + '_bak' + p_ext
    shutil.copy(_get_path(), p_bak)

