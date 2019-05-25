#!/usr/bin/env python

"""
This is still very much work in progress!
"""

import fasteners
import argparse
import re
import os
import sys
import time
import glob
import subprocess
from contextlib import contextmanager
from collections import namedtuple

import shutil


QSUBA_GIT_REF = 'QSUBA_GIT_REF'


GitManager = namedtuple('GitManager', ['git_root_dir', 'git_url', 'git_checkout', 'qsuba_git_helper_path'])


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                usage="Note that qsuba will consume all arguments it can. To pass an argument "
                                      "supported by qsuba to your script, separate the script's argument with --, "
                                      "e.g., qsuba --out_dir=out myscript.py -- --out_dir=script_out")

    p.add_argument('script', type=str, help='The script to run.')
    p.add_argument('--num_jobs', type=str,
                   help='If given, this becomes an array job. '
                        'If int, submit 1-`num_jobs`. If str, expected to be START_ID-END_ID.')
    p.add_argument('--out_dir', type=str, default='out', help='Path to directory for  of log files')
    p.add_argument('--log_file', type=str, default='qsuba_log',
                   help='Path to file where submission log is stored. Submission log: job_id followed by all '
                        'arguments.')
    p.add_argument('--hosts', type=str, help='Passed to qsub as -l h= flag')
    p.add_argument('--duration', type=str,
                   help='If int: hours to run job. If str, expected to be HH:MM:SS.')

    p.add_argument('--gpu', type=int, help='Whether to use gpu')

    p.add_argument('--skip_tailf', '-q', action='store_const', const=True,
                   help='If given, do not tail -f relevant log file.')
    p.add_argument('--mem', type=int, default=5, help='Memory, in GB, to allocate for job.')
    p.add_argument('--wait_for', type=str, help='Passed to --hold_jid')
    p.add_argument('--queue', type=str, help='Queue to submit to. If given, -q QUEUE is passed to qsub')
    p.add_argument('--interpreter', type=str, default='python -u',
                   help='What to put on the line before `script` in the auto-generated run file.')
    p.add_argument('--pre_run_cmds', type=str,
                   default=os.environ.get('QSUBA_PRE_RUN_CMDS', ''),
                   help='Value given by QSUBA_PRE_RUN_CMDS')
    p.add_argument('--dry_run', action='store_const', const=True,
                   help='Does not create array job. Passes --qsuba_dry_run to job. If job calls job_id_iterator, '
                        'an info about which job consumes which input is printed and then the job is cancelled. '
                        'If --num_jobs is not given, this simply prints the qsub call that would be made and exits.')
    p.add_argument('--rerun', type=str,
                   help='Comma separated list of job_ids to run again.')

    p.add_argument('--copy_PATH', dest='copy_PATH', action='store_true')
    p.add_argument('--no-copy_PATH', dest='copy_PATH', action='store_false',
                   help='If given, do not copy $PATH to target env')
    p.set_defaults(copy_PATH=True)

    p.add_argument('--git_qsuba_helper', type=str, default='qsuba_git_helper.py',
                   help='Path to executable qsuba_git_helper.py, expected to be in $PATH by default.')
    p.add_argument('--git_repo', type=str, nargs=2,
                   metavar=('GIT_ROOT_DIR_LOCAL', 'GIT_URL'))
    p.add_argument('--git_cd', type=str, help='Run code from $GIT_ROOT/GIT_CD.', default='')
    p.add_argument('--git_checkout', '-c', type=str, metavar='GIT_REF',
                   help='This is passed to git checkout if --git_repo is given. This will be exported as {} for any '
                        'script to check.'.format(QSUBA_GIT_REF))

    p.add_argument('--var', '-v', type=str, action='append', default=[],
                   help='Variables to pass to qsub. Use multiple flags for multiple variables. Example:\n'
                        '-v ENV1=1 -v ENVX=x')

    # TODO: add --vars or sth to pass env variables to qsub

    flags, other_args = p.parse_known_args()
    if '--' in other_args:  # user passed -- should be ignored because we add it later
        other_args.remove('--')
    run(flags, other_args)


def run(flags, other_args):
    os.makedirs(flags.out_dir, exist_ok=True)
    assert os.path.isdir(flags.out_dir)

    follow_file = not flags.skip_tailf
    with tmp_run_file(flags.pre_run_cmds, flags.script, flags.interpreter,
                      remove=flags.skip_tailf) as run_file:
        qsub_call = [
            'qsub',
            '-o', flags.out_dir,
            '-l', 'h_vmem={}G'.format(flags.mem),
            '-cwd', '-j', 'y',
        ]
        if flags.duration:
            if _is_int(flags.duration):
                h_rt_flag = 'h_rt={:02d}:00:00'.format(int(flags.duration))
            else:
                assert ':' in flags.duration
                h_rt_flag = 'h_rt={}'.format(flags.duration)
            qsub_call += ['-l', h_rt_flag]

        if flags.queue:
            qsub_call += ['-q', flags.queue]
        if flags.wait_for:
            qsub_call += ['-hold_jid', flags.wait_for]

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
            qsub_call += ['-l', 'gpu={}'.format(flags.gpu)]

        # Set up env vars to pass
        env = get_envs(flags)

        for var, value in env:
            qsub_call.extend(['-v', '{}={}'.format(var, value)])

        qsub_call += [run_file]

        print(' '.join(qsub_call))
        if other_args:
            print('  ' + '\n  '.join(other_args))
            qsub_call += ['--'] + other_args

        if not flags.num_jobs and flags.dry_run:
            print('--dry_run given, stopping...')
            sys.exit(0)

        otp = subprocess.check_output(qsub_call).decode()
        print(otp)
        job_id = re.search(r'(\d+)(\.|\s)', otp).group(1)  # match (1234).1 if array job, else (1234)

        if flags.log_file:
            log_submission(flags.log_file, job_id, qsub_call)

        if follow_file:
            output_glob = os.path.join(flags.out_dir, '*{}*'.format(job_id))
            try:
                wait_for_output(output_glob)
                subprocess.call('tail -f ' + output_glob, shell=True)
            except KeyboardInterrupt:
                ask_for_kill(job_id)


def get_envs(flags):
    """ Set up array of tuples (env_var_name, env_var_value) to pass to qsub. """
    env = []
    if flags.copy_PATH:
        env.append(('PATH', os.environ['PATH']))
    if flags.git_repo:
        git_root_dir_local, git_url = flags.git_repo
        git_qsuba_helper = flags.git_qsuba_helper
        git_cd = flags.git_cd
        env.extend([
            (_GitEnvVars.root, git_root_dir_local),
            (_GitEnvVars.url, git_url),
            (_GitEnvVars.qsuba_git_helper, git_qsuba_helper),
            (_GitEnvVars.cd, git_cd),
        ])
        if flags.git_checkout:
            env.append((_GitEnvVars.ref, flags.git_checkout))
    else:
        if flags.git_checkout:
            print('--git_checkout invalid without --git_repo!')
            sys.exit(1)
    for v in flags.var:
        try:
            name, value = v.split('=')
            env.append((name, value))
        except ValueError:
            print('Invalid --var, expected --var NAME=VALUE, got `{}`'.format(v))
            sys.exit(1)
    return env


def _is_int(v):
    try:
        int(v)
        return True
    except ValueError:
        return False


class _GitEnvVars(object):
    ref = 'GIT_ENV_REF'
    root = 'GIT_ENV_ROOT_DIR'
    url = 'GIT_ENV_URL'
    cd = 'GIT_CD'
    qsuba_git_helper = 'GIT_ENV_QSUBA_HELPER_P'


@contextmanager
def tmp_run_file(pre_run_cmds, script_name, interpreter, remove):
    # make sure only one qsuba process changes the submission script at a time
    #
    with fasteners.InterProcessLock('.sub_creation_lock'.format(script_name)):
        fn = '{}_sub.sh'.format(os.path.splitext(script_name)[0])
        fn_tmp = '{}_sub_tmp.sh'.format(os.path.splitext(script_name)[0])
        with open(fn_tmp, 'w+') as f:
            f.write('#!/bin/bash\n')
            f.write('uname -n; echo "Job ID: $JOB_ID"; echo "GPU: $CUDA_VISIBLE_DEVICES"\n')
            f.write('{}\n'.format(pre_run_cmds))

            # the idea here is that a submission file should be the same for the same script, independent of whether
            # --git_repo or --git_checkout was passed. So we set up the following using environment variables which get
            # set by passing -v flags to qsub, depending on the flags to qsuba.
            gev = _GitEnvVars()
            f.write('if [[ -n ${gev.ref} ]]; then\n'.format(gev=gev))
            f.write(' mkdir -p ${gev.root} && cd "$_" && pwd\n'.format(gev=gev))
            f.write(' '
                    'if [[ -n $CUDA_VISIBLE_DEVICES ]]; '
                    'then UNIQUE_ID=$CUDA_VISIBLE_DEVICES; '
                    'else UNIQUE_ID="cpu"; '
                    'fi\n'.format(gev=gev))
            f.write(' ${gev.qsuba_git_helper} $UNIQUE_ID ${gev.url} ${gev.ref}\n'.format(gev=gev))
            f.write(' rc=$?; if [[ $rc != 0 ]]; then echo "Error $rc"; exit $rc; fi\n')
            f.write(' cd $UNIQUE_ID/${gev.cd} && pwd  # where the repo has been cloned into\n'.format(gev=gev))
            f.write(' export {qsuba_git_ref}=${gev.ref}\n'.format(qsuba_git_ref=QSUBA_GIT_REF, gev=gev))
            f.write('fi\n')

            # Run script
            f.write('shift # removes initial -- argument, which is added automatically by qsuba\n')
            f.write('{} {} "$@"\n'.format(interpreter, script_name))
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


def log_submission(log_file, job_id, qsub_cmd):
    log_file_lock_p = get_log_file_lock_p(log_file)

    log = '{}\n{}\n{}\n'.format(job_id, pretty_print_call(qsub_cmd), '-' * 80)

    with fasteners.InterProcessLock(log_file_lock_p):
        with open(log_file, 'a') as f:
            f.write(log)


def get_log_file_lock_p(log_file):
    assert not log_file.endswith(os.path.sep)
    log_file_dir, log_file_name = os.path.split(log_file)
    log_file_lock_fn = '.{}_lock'.format(log_file_name)
    return os.path.join(log_file_dir, log_file_lock_fn)


def pretty_print_call(call):
    joined_args_call = _join_args(call, nargs={'omg': 2})
    return _join_lines(joined_args_call, max_len=80)


def _join_args(call, nargs=None):
    """
    Join arguments in call together.
    :param call:
    :param nargs:  dict from arguments names to how many args it takes, default 1 for all.
    :return:
    """
    if not nargs:
        nargs = {}
    assert all(not k.startswith('-') for k in nargs.keys())
    out = []
    current_arg = []
    for c in call:
        if len(current_arg) > 0:
            if c.startswith('-') or (nargs.get(current_arg[0].strip('--'), 1) + 1 == len(current_arg)):
                out.append(' '.join(current_arg))
                current_arg = []
        current_arg.append(c)
    if current_arg:
        out.append(' '.join(current_arg))
    return out


def _join_lines(lines, max_len):
    out = ''
    current_l = ''
    for l in lines:
        if len(current_l) + len(l) <= max_len:
            current_l += l + ' '
        else:
            out += current_l.strip() + '\n'
            current_l = l + ' '
    if current_l:
        out += current_l.strip() + '\n'
    return out




# ------------------------------------------------------------------------------


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



if __name__ == '__main__':
    main()
