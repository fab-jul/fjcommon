"""

"""


import tensorflow as tf
from os import path
import os
import argparse
import glob
from fjcommon import printing
from fjcommon import iterable_tools


_JOB_SUBDIR_PREFIX = 'job_'
_TF_RECORD_EXT = 'tfrecord'


def create_images_records_distributed(image_dir, job_id, num_jobs, out_dir, num_per_shard):
    image_paths = _get_image_paths(image_dir)
    image_paths_per_job = iterable_tools.chunks(image_paths, num_chunks=num_jobs)
    image_paths_current_job = iterable_tools.get_element_at(job_id - 1, image_paths_per_job)
    img_it = (open(image_path, 'rb').read() for image_path in image_paths_current_job)
    create_records(img_it, num_per_shard, out_dir=path.join(out_dir, '{}{}'.format(_JOB_SUBDIR_PREFIX, job_id)))


def join_created_images_records(out_dir, num_jobs):
    jobs_dirs_glob = path.join(out_dir, '{}*'.format(_JOB_SUBDIR_PREFIX))
    jobs_dirs = glob.glob(jobs_dirs_glob)
    assert len(jobs_dirs) == num_jobs, 'Expected {} subdirs, got {}'.format(num_jobs, jobs_dirs)

    records = glob.glob(path.join(jobs_dirs_glob, '*.{}'.format(_TF_RECORD_EXT)))
    assert len(records) > 0, 'Did not find any records in {}/{}_*'.format(out_dir, _JOB_SUBDIR_PREFIX)

    base_records_file_name = path.basename(records[0]).split('_')[0]  # get SHARD from out_dir/job_x/SHARD_xxx.ext
    for shard_number, records_p in enumerate(printing.ProgressPrinter('Moving records...', iter_list=records)):
        target_p = path.join(out_dir, _records_file_name(base_records_file_name, shard_number))
        os.rename(records_p, target_p)

    list(map(os.removedirs, jobs_dirs))


def _get_image_paths(image_dir):
    return sorted(glob.glob(path.join(image_dir, '*.png')))


def create_records(feature_bytes_it, num_per_shard, out_dir, file_name='shard'):
    """
    :param feature_bytes_it: iterator yielding bytes to encode as features
    :param out_dir:
    :param num_per_shard:
    :param file_name:
    :return:
    """
    os.makedirs(out_dir, exist_ok=True)
    writer = None
    for count, b in enumerate(feature_bytes_it):
        if count % num_per_shard == 0:
            if count > 0:
                print()  # to finish progress line
            if writer:
                writer.close()
            shard_number = count // num_per_shard
            record_p = path.join(out_dir, _records_file_name(file_name, shard_number))
            assert not path.exists(record_p), 'Record already exists! {}'.format(record_p)
            print('Creating {}...'.format(record_p))
            writer = tf.python_io.TFRecordWriter(record_p)
        bytes_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
        feature = {'M': bytes_feature}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        printing.progress_print((count % num_per_shard) / num_per_shard)


def _records_file_name(base_filename, shard_number):
    return '{}_{:08d}.{}'.format(base_filename, shard_number, _TF_RECORD_EXT)


def read_image_records(records_dir, shuffle):
    return tf.image.decode_image(read_records(records_dir, shuffle), channels=3)


def read_records(records_dir, shuffle):
    reader = tf.TFRecordReader()
    records_paths = glob.glob(path.join(records_dir, '*.tfrecord'))
    if not shuffle:
        records_paths = sorted(records_paths)
    filename_queue = tf.train.string_input_producer(records_paths, shuffle=shuffle)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'M': tf.FixedLenFeature([], tf.string),
        })
    return features['M']


def _test(out_dir):
    img = read_image_records(out_dir, shuffle=True)
    from fjcommon import tf_helpers
    with tf_helpers.start_queues_in_sess() as (s, _):
        print(s.run(img).shape)

def main(args):
    parser = argparse.ArgumentParser()
    mode_subparsers = parser.add_subparsers(dest='mode', title='Mode')
    parser_make = mode_subparsers.add_parser('mk_img_recs')
    parser_make.add_argument('out_dir', type=str)
    parser_make.add_argument('image_dir', type=str)
    parser_make.add_argument('--job_id', type=int, required=True)
    parser_make.add_argument('--num_jobs', type=int, required=True)
    parser_make.add_argument('--num_per_shard', type=int, required=True)
    parser_join = mode_subparsers.add_parser('join')
    parser_join.add_argument('out_dir', type=str)
    parser_join.add_argument('--num_jobs', type=int, required=True)
    _parser_test = mode_subparsers.add_parser('test')
    _parser_test.add_argument('out_dir', type=str)
    flags = parser.parse_args(args)
    if flags.mode == 'mk_img_recs':
        create_images_records_distributed(
            flags.image_dir, flags.job_id, flags.num_jobs, flags.out_dir, flags.num_per_shard)
    elif flags.mode == 'join':
        join_created_images_records(flags.out_dir, flags.num_jobs)
    elif flags.mode == 'test':
        _test(flags.out_dir)
    else:
        parser.print_usage()

