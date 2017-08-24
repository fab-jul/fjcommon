"""

"""
import itertools
import tensorflow as tf
from os import path
import os
import random
import argparse
import glob
from fjcommon import tf_helpers
from fjcommon import printing
from fjcommon import iterable_tools


_JOB_SUBDIR_PREFIX = 'job_'
_TF_RECORD_EXT = 'tfrecord'
_DEFAULT_FEATURE_KEY = 'M'


def create_images_records_distributed(image_dir, job_id, num_jobs, out_dir, num_per_shard):
    assert 1 <= job_id <= num_jobs, 'Invalid job_id: {}'.format(job_id)
    assert num_jobs >= 1, 'Invalid num_jobs: {}'.format(num_jobs)
    image_paths = _get_image_paths(image_dir)
    image_paths_per_job = iterable_tools.chunks(image_paths, num_chunks=num_jobs)
    image_paths_current_job = iterable_tools.get_element_at(job_id - 1, image_paths_per_job)
    img_it = (open(image_path, 'rb').read() for image_path in image_paths_current_job)
    out_dir_job = out_dir if num_jobs == 1 else path.join(out_dir, '{}{}'.format(_JOB_SUBDIR_PREFIX, job_id))
    create_records(img_it, out_dir_job, num_per_shard)


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

    print('Removing empty job dirs...')
    list(map(os.removedirs, jobs_dirs))  # remove all job dirs, which are now empty

    print('Counting...')
    all_records_glob = path.join(out_dir, '*.{}'.format(_TF_RECORD_EXT))
    printing.print_join('{}: {}'.format(path.basename(p), _number_of_examples_in_record(p))
                        for p in sorted(glob.glob(all_records_glob)))


def _number_of_examples_in_record(p):
    return sum(1 for _ in tf.python_io.tf_record_iterator(p))


def _get_image_paths(image_dir):
    """ Shuffled list of all .pngs in `image_dir` """
    paths = sorted(glob.glob(path.join(image_dir, '*.png')))
    random.Random(6).shuffle(paths)  # shuffle deterministically, so that the returned list is consistent between jobs
    return paths


def create_records(feature_bytes_it, out_dir, num_per_shard,
                   max_shards=None, file_name='shard', feature_key=_DEFAULT_FEATURE_KEY):
    def _feature_dicts():
        for b in feature_bytes_it:
            yield {feature_key: bytes_feature(b)}
    return create_records_with_feature_dict(_feature_dicts(), out_dir, num_per_shard, max_shards, file_name)


def bytes_feature(b):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def int64_feature(i):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))


def create_records_with_feature_dict(feature_dicts, out_dir, num_per_shard, max_shards=None, file_name='shard'):
    """
    :param feature_bytes_it: iterator yielding bytes to encode as features
    :param out_dir:
    :param num_per_shard:
    :param file_name:
    :return:
    """
    os.makedirs(out_dir, exist_ok=True)
    writer = None
    with printing.ProgressPrinter() as progress_printer:
        for count, feature in enumerate(feature_dicts):
            if count % num_per_shard == 0:
                progress_printer.finish_line()
                if writer:
                    writer.close()
                shard_number = count // num_per_shard
                if max_shards is not None and shard_number == max_shards:
                    print('Created {} shards...'.format(max_shards))
                    return
                record_p = path.join(out_dir, _records_file_name(file_name, shard_number))
                assert not path.exists(record_p), 'Record already exists! {}'.format(record_p)
                print('Creating {}...'.format(record_p))
                writer = tf.python_io.TFRecordWriter(record_p)
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            progress_printer.update((count % num_per_shard) / num_per_shard)
    if writer:
        writer.close()
    else:
        print('Nothing written...')


def _records_file_name(base_filename, shard_number):
    return '{}_{:08d}.{}'.format(base_filename, shard_number, _TF_RECORD_EXT)


def feature_to_image(feature):
    """ Use case: feature_to_img(read_records(...)) """
    im = tf.image.decode_image(feature, channels=3)
    im.set_shape((None, None, 3))
    return im


def extract_images(records_glob, max_images, out_dir, feature_key=_DEFAULT_FEATURE_KEY):
    tf.logging.set_verbosity(tf.logging.INFO)
    image = feature_to_image(read_records(records_glob, num_epochs=1, shuffle=False, feature_key=feature_key))
    image = tf.expand_dims(image, axis=0)  # make 'batched'
    index_iterator = range(max_images) if max_images else itertools.count()
    img_names_iterator = map('img_{:010d}'.format, index_iterator)
    img_saver = tf_helpers.ImageSaver(out_dir)
    with tf_helpers.start_queues_in_sess() as (sess, coord):
        img_fetcher = sess.make_callable(img_saver.get_fetch_dict(image))
        for img_name in img_names_iterator:
            tf.logging.info('Saving {}...'.format(img_name))
            img_saver.save(img_fetcher(), img_names=[img_name])


def read_records(records_glob, num_epochs=None, shuffle=True, feature_key=_DEFAULT_FEATURE_KEY):
    features_dict = {feature_key: tf.FixedLenFeature([], tf.string)}
    features = read_records_with_features_dict(records_glob, features_dict, num_epochs, shuffle)
    return features[feature_key]


def read_records_with_features_dict(records_glob, features_dict, num_epochs=None, shuffle=True):
    reader = tf.TFRecordReader()
    records_paths = glob.glob(records_glob)
    assert records_paths, 'Did not find any records matching {}'.format(records_glob)
    if not shuffle:
        records_paths = sorted(records_paths)

    filename_queue = tf.train.string_input_producer(records_paths, num_epochs=num_epochs, shuffle=shuffle)
    _, serialized_example = reader.read(filename_queue)
    return tf.parse_single_example(serialized_example, features=features_dict)


def main(args):
    parser = argparse.ArgumentParser()
    mode_subparsers = parser.add_subparsers(dest='mode', title='Mode')
    # Make image records ---
    parser_make = mode_subparsers.add_parser('mk_img_recs')
    parser_make.add_argument('out_dir', type=str)
    parser_make.add_argument('image_dir', type=str)
    parser_make.add_argument('--num_per_shard', type=int, required=True)
    # Make image records, distributed ---
    parser_make_dist = mode_subparsers.add_parser('mk_img_recs_dist')
    parser_make_dist.add_argument('out_dir', type=str)
    parser_make_dist.add_argument('image_dir', type=str)
    parser_make_dist.add_argument('--job_id', type=int, required=True)
    parser_make_dist.add_argument('--num_jobs', type=int, required=True)
    parser_make_dist.add_argument('--num_per_shard', type=int, required=True)
    # Join image records ---
    parser_join = mode_subparsers.add_parser('join')
    parser_join.add_argument('out_dir', type=str)
    parser_join.add_argument('--num_jobs', type=int, required=True)
    # Extract image Records ---
    parser_extract = mode_subparsers.add_parser('extract')
    parser_extract.add_argument('records_glob', type=str)
    parser_extract.add_argument('out_dir', type=str)
    parser_extract.add_argument('max_imgs', type=int)
    parser_extract.add_argument('--feature_key', type=str, default=_DEFAULT_FEATURE_KEY)
    # ---
    flags = parser.parse_args(args)
    if flags.mode == 'mk_img_recs':
        create_images_records_distributed(
            flags.image_dir, job_id=1, num_jobs=1, out_dir=flags.out_dir, num_per_shard=flags.num_per_shard)
    elif flags.mode == 'mk_img_recs_dist':
        create_images_records_distributed(
            flags.image_dir, flags.job_id, flags.num_jobs, flags.out_dir, flags.num_per_shard)
    elif flags.mode == 'join':
        join_created_images_records(flags.out_dir, flags.num_jobs)
    elif flags.mode == 'extract':
        extract_images(flags.records_glob, flags.max_imgs, flags.out_dir, flags.feature_key)
    else:
        parser.print_usage()

