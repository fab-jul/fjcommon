#!/usr/bin/env python

"""
This is still very much work in progress!
"""

import argparse
import re
import os
import sys
import time
import glob
import subprocess
from contextlib import contextmanager

import shutil


def lift_argument_parser(parser):
    """ Adds --job_id, --num_jobs and --qsuba_dry_run to parser. These get used by job_id_iterator. """
    try:
        job_id = int(os.environ.get('SGE_TASK_ID', 1))
    except ValueError:
        job_id = 1
    parser.add_argument('--job_id', type=int, default=job_id)
    parser.add_argument('--num_jobs', type=int, default=1)
    parser.add_argument('--qsuba_dry_run', action='store_const', const=True)
    parser.add_argument('--qsuba_rerun', type=str)


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

    def _iterator_for_job_id(job_id):
        if flags.qsuba_rerun:
            rerun_job_ids = list(map(int, flags.qsuba_rerun.split(',')))
            if job_id not in rerun_job_ids:
                print('Skipping job {}, only running {}'.format(job_id, rerun_job_ids))
                sys.exit(1)

        def _iterator(it):
            for i, el in enumerate(it):
                if i % flags.num_jobs != (job_id - 1):  # job_ids start at 1
                    continue
                yield el
        return _iterator

    if flags.qsuba_dry_run:
        def _dry_run_iterator(it):
            data = list(it)
            for job_id in range(flags.num_jobs):
                print(job_id)
                print('\n'.join(_iterator_for_job_id(job_id)(data)))
                print('---')
            print('--qsuba_dry_run, will exit now')
            sys.exit(0)
        return _dry_run_iterator

    return _iterator_for_job_id(flags.job_id)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('script', type=str, help='The script to run.')
    p.add_argument('--num_jobs', type=str,
                   help='If given, this becomes an array job. '
                        'If int, submit 1-`num_jobs`. If str, expected to be START_ID-END_ID.')
    p.add_argument('--out_dir', type=str, default='out', help='Path to directory for  of log files')
    p.add_argument('--hosts', type=str, help='Passed to qsub as -l h= flag')
    p.add_argument('--duration', type=str, default='4',
                   help='If int: hours to run job. If str, expected to be HH:MM:SS.')
    p.add_argument('--gpu', type=bool, default=False, help='Whether to use gpu')
    p.add_argument('--skip_tailf', '-q', action='store_const', const=True,
                   help='If given, do not tail -f relevant log file.')
    p.add_argument('--mem', type=int, default=5, help='Memory, in GB, to allocate for job.')
    p.add_argument('--interpreter', type=str, default='python -u',
                   help='What to put on the line before `script` in the auto-generated run file.')
    p.add_argument('--pre_run_cmds', type=str,
                   default=os.environ.get('QSUBA_PRE_RUN_CMDS', ''),
                   help='Value given by QSUBA_PRE_RUN_CMDS')
    p.add_argument('--dry_run', action='store_const', const=True,
                   help='Does not create array job. Passes --qsuba_dry_run to job. If job calls job_id_iterator, '
                        'an info about which job consumes which input is printed and then the job is cancelled.')
    p.add_argument('--rerun', type=str,
                   help='Comma separated list of job_ids to run again.')

    flags, other_args = p.parse_known_args()
    os.makedirs(flags.out_dir, exist_ok=True)
    assert os.path.isdir(flags.out_dir)

    follow_file = not flags.skip_tailf
    with tmp_run_file(flags.pre_run_cmds, flags.script, flags.interpreter,
                      remove=flags.skip_tailf) as run_file:
        if _is_int(flags.duration):
            h_rt_flag = 'h_rt={:02d}:00:00'.format(int(flags.duration))
        else:
            assert ':' in flags.duration
            h_rt_flag = 'h_rt={}'.format(flags.duration)

        qsub_call = [
            'qsub',
            '-o', flags.out_dir,
            '-l', h_rt_flag,
            '-l', 'h_vmem={}G'.format(flags.mem),
            '-cwd', '-j', 'y',
        ]
        if flags.hosts:
            qsub_call += ['-l', 'h={}'.format(flags.hosts)]

        if flags.num_jobs:
            # Create array job
            if _is_int(flags.num_jobs):  # just a number given
                num_jobs = flags.num_jobs
                job_id_range = '1-{}'.format(num_jobs)
            else:  # a range given, i.e., 1-30
                job_id_range = flags.num_jobs
                start_id, end_id = flags.num_jobs.split('-')  # raises if incorrect str given
                num_jobs = end_id.split()
            other_args += ['--num_jobs', num_jobs]

            if flags.rerun:
                # if rerun, only submit range MIN(rerun) - MAX(rerun) to reduce unnecessary job dispatch
                rerun_job_ids = list(map(int, flags.rerun.split(',')))
                job_id_range = '{}-{}'.format(min(rerun_job_ids), max(rerun_job_ids))
                print('--rerun given, only submitting {}'.format(job_id_range))
                other_args += ['--qsuba_rerun', flags.rerun]

            if not flags.dry_run:
                qsub_call += ['-t', job_id_range]
            else:
                other_args += ['--qsuba_dry_run']

        if flags.gpu:
            qsub_call += ['-l', 'gpu=1']

        qsub_call += [run_file]

        print(' '.join(qsub_call))
        if other_args:
            print(other_args)
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


def _is_int(v):
    try:
        int(v)
        return True
    except ValueError:
        return False


@contextmanager
def tmp_run_file(pre_run_cmds, script_name, interpreter, remove):
    fn = '{}_sub.sh'.format(os.path.splitext(script_name)[0])
    fn_tmp = '{}_sub_tmp.sh'.format(os.path.splitext(script_name)[0])
    print(fn)
    with open(fn_tmp, 'w+') as f:
        f.write('#!/bin/bash\n')
        f.write('uname -n; echo "Job ID: $JOB_ID"; echo "GPU: $SGE_GPU"\n')
        f.write('{}\n'.format(pre_run_cmds))
        f.write('CUDA_VISIBLE_DEVICES=$SGE_GPU {} {} "$@"\n'.format(
            interpreter, script_name))
    overwrite_if_changed(fn, fn_tmp)
    yield fn
    if remove:
        os.remove(fn)


def overwrite_if_changed(pold, pnew):
    with open(pnew, 'r') as fnew:
        fnew_content = fnew.read()
    if os.path.exists(pold):
        with open(pold, 'r') as fold:
            fold_content = fold.read()
    else:
        fold_content = None
    if fold_content == fnew_content:
        return
    shutil.move(pnew, pold)



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


