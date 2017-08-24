#!/usr/bin/env python

"""
This is still very much work in progress!
"""

import argparse
import re
import os
import time
import glob
import subprocess
from contextlib import contextmanager



def lift_argument_parser(parser):
    """ Adds --job_id and --num_jobs to parser. These get used by job_id_iterator. """
    try:
        job_id = int(os.environ.get('SGE_TASK_ID', 1))
    except ValueError:
        job_id = 1
    parser.add_argument('--job_id', type=int, default=job_id)
    parser.add_argument('--num_jobs', type=int, default=1)


def job_id_iterator(flags):
    """ Given flags returned by a argparse.ArgumentParser's parse_args() method, where the parser was previously
    lifted with lift_argument_parser, this method return an a function that takes an iterator and returns a new
    iterator which only yields elements that should be processed by the current job.

    Usage:

    p = argparse.ArgumentParser()
    ... add arguments
    qsuba.lift_argument_parser(p)
    flags = p.parse_args()
    it = qsuba.job_id_iterator(flags)

    for el in it(element_generator()):
        # process el
    """
    assert flags.num_jobs >= flags.job_id >= 1

    def _iterator(it):
        for i, el in enumerate(it):
            if i % flags.num_jobs != (flags.job_id - 1):  # job_ids start at 1
                continue
            yield el
    return _iterator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('script', type=str)
    p.add_argument('--num_jobs', type=int, help='If given, this becomes an array job.')
    p.add_argument('--out_dir', type=str, default='out')
    p.add_argument('--hours', type=str, default=4)
    p.add_argument('--gpu', type=bool, default=False)
    p.add_argument('--skip_tailf', '-q', action='store_const', const=True)
    p.add_argument('--mem', type=int, default=5)
    p.add_argument('--num_jobs_flag', type=str, default='--num_jobs')
    p.add_argument('--interpreter', type=str, default='python -u')
    p.add_argument('--pre_run_cmds', type=str,
                   default='source ~/cudarc; source ~/pyenvrc; pyenv activate ma_env_tf1_2')

    flags, other_args = p.parse_known_args()
    os.makedirs(flags.out_dir, exist_ok=True)
    assert os.path.isdir(flags.out_dir)

    follow_file = not flags.skip_tailf
    with tmp_run_file(flags.pre_run_cmds, flags.script, flags.interpreter,
                      remove=flags.skip_tailf) as run_file:
        try:
            h_rt_flag = 'h_rt={:02d}:00:00'.format(int(flags.hours))
        except ValueError:  # flags.hours is not int
            assert ':' in flags.hours
            h_rt_flag = 'h_rt={}'.format(flags.hours)

        qsub_call = [
            'qsub',
            '-o', flags.out_dir,
            '-l', h_rt_flag,
            '-l', 'h_vmem={}G'.format(flags.mem),
            '-l', 'gpu={}'.format(int(flags.gpu)),
            '-cwd', '-j', 'y',
        ]

        if flags.num_jobs:
            qsub_call += ['-t', '1-{}'.format(flags.num_jobs)]
            other_args += [
                flags.num_jobs_flag, str(flags.num_jobs)]

        qsub_call += [run_file]

        print(' '.join(qsub_call))
        if other_args:
            print('   ' + ' '.join(other_args))
            qsub_call += other_args

        otp = subprocess.check_output(qsub_call).decode()
        print(otp)
        job_id = re.search(r'(\d+)(\.|\s)', otp).group(1)  # match (1234).1 if array job, else (1234)

        if follow_file:
            output_glob = os.path.join(flags.out_dir, '*{}*'.format(job_id))
            try:
                wait_for_output(output_glob)
                subprocess.call('tail -f ' + output_glob, shell=True)
            except KeyboardInterrupt:
                ask_for_kill(job_id)


@contextmanager
def tmp_run_file(pre_run_cmds, script_name, interpreter, remove):
    fn = '{}_sub.sh'.format(os.path.splitext(script_name)[0])
    if not os.path.exists(fn):
        with open(fn, 'w+') as f:
            f.write('#!/bin/bash\n')
            f.write('uname -n; echo "Job ID: $JOB_ID"; echo "GPU: $SGE_GPU"\n')
            f.write('{}\n'.format(pre_run_cmds))
            f.write('CUDA_VISIBLE_DEVICES=$SGE_GPU {} {} "$@"\n'.format(
                interpreter, script_name))
    yield fn
    if remove:
        os.remove(fn)


def wait_for_output(output_glob):
    while True:
        if len(glob.glob(output_glob)) > 0:
            break
        time.sleep(1)


def ask_for_kill(job_id):
    qstat = subprocess.check_output(['qstat']).decode()
    if job_id not in qstat:
        return
    if input('Kill job? (y/n)') == 'y':
        subprocess.call(['qdel', job_id])


if __name__ == '__main__':
    main()


