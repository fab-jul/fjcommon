#!/usr/bin/env python

import os
import sys
#import shutil
import subprocess
#import fasteners
import argparse


def unique_checkout(unique_id, git_url, git_checkout):
    """
    Flow:
    if exists unique_id:
        cd unique_id
        git fetch
    else:
        git clone git_rul unique_id
    cd uinque_id
    git checkout git_checkout

    Note that the caller is responsible for making sure only one process calls this with a certain unique_id.
    """
    assert isinstance(unique_id, str)
    _get_repo(git_url, unique_id)
    _checkout_in_copy_repo(unique_id, git_checkout)


def _get_repo(git_url, unique_id):
    if os.path.isdir(unique_id):
        print('Repo already exists, fetching...')
        _call_and_check(['git', 'fetch'], cwd=unique_id, stderr=subprocess.DEVNULL)
    else:
        _call_and_check(['git', 'clone', git_url, unique_id])


def _checkout_in_copy_repo(unique_id, git_checkout):
    _call_and_check(['git', 'checkout', git_checkout], cwd=unique_id, stderr=subprocess.DEVNULL)


def _call_and_check(*args, **kwargs):
    cmd = args[0]
    print(' '.join(cmd))
    ret_code = subprocess.call(*args, **kwargs)
    if ret_code != 0:
        print('{} failed: {}'.format(' '.join(cmd), ret_code))
        sys.exit(ret_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('unique_id', type=str, help='Used to create copy of repository. Should be unique for '
                                                    'concurrent runs.')
    parser.add_argument('git_url', type=str, help='URL to base repository to git clone from.')
    parser.add_argument('git_checkout', type=str, help='What to git checkout, e.g., origin/master, 3cab6479.')
    flags = parser.parse_args()
    unique_checkout(flags.unique_id, flags.git_url, flags.git_checkout)



if __name__ == '__main__':
    main()



