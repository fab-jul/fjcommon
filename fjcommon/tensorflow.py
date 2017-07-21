import tensorflow as tf
import numpy as np
from contextlib import contextmanager
import functools
from collections import OrderedDict
import os
from os import path
import pickle
from . import lifting

# Session ----------------------------------------------------------------------

@contextmanager
def start_queues_in_sess(init_vars=True, name=None):
    # TODO: allow only starting a subset of queues
    with create_session() as sess:
        tf.logging.info(' '.join(filter(None, ['Session', name, 'Started'])))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if init_vars:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
            yield sess, coord
        except tf.errors.OutOfRangeError:
            tf.logging.info('Queues exhausted...')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


def create_session(graph=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True
    return tf.Session(config=config, graph=graph)


# Ops --------------------------------------------------------------------------


def set_backwards_pass(op, backwards):
    """
    Returns new operation which behaves like `op` in the forward pass but
    like `backwards` in the backwards pass.
    """
    return backwards + tf.stop_gradient(op - backwards)


def logb(x, b):
    return tf.log(x) / tf.log(tf.constant(b, dtype=x.dtype))


log2 = functools.partial(logb, b=2)
log10 = functools.partial(logb, b=10)


get_variable_zeros = functools.partial(tf.get_variable, initializer=tf.zeros_initializer(), trainable=False)
get_variable_ones = functools.partial(tf.get_variable, initializer=tf.ones_initializer(), trainable=False)


# Caching ----------------------------------------------------------------------


@lifting.lift1d(2)
def cache_some(tensors, np_dtyes, cache_size, cache_per_batch=2):
    """
    Caches the output of running `tensors` enough times to get a cache `cache_size`.
    """
    all_shapes = [t.get_shape().as_list() for t in tensors]
    assert all(None not in s and len(s) >= 2 for s in all_shapes), \
        'All shapes must contain batch_size plus at least one additional dimension + no unknown: {}'.format(all_shapes)

    batch_size = all_shapes[0][0]
    num_batches = cache_size // cache_per_batch
    caches = [np.zeros((cache_size,) + tuple(t_shape[1:]), np_dtype)
              for t_shape, np_dtype in zip(all_shapes, np_dtyes)]
    tensor_names = ','.join(t.name for t in tensors)
    #TODO take sess as parameter
    with start_queues_in_sess(name='Caching {}'.format(tensor_names)) as (sess, _):
        for i in range(num_batches):
            tensors_out = sess.run(tensors)
            for t_i, t_out in enumerate(tensors_out):
                t_cache = caches[t_i]
                for j in range(cache_per_batch):
                    t_cache[cache_per_batch * i + j, :] = t_out[j * (batch_size // cache_per_batch), :]
    return caches


# Logging ----------------------------------------------------------------------


class _Loggable(object):
    def __init__(self, operation, format_str):
        self.operation = operation
        self.format_str = format_str


class Logger(object):
    def __init__(self):
        self.loggables = OrderedDict()
        self.fetcher = None

    def add_loggable(self, name, operation, format_str=None):
        if not format_str:
            format_str = self._get_default_format_str(operation)
        self.loggables[name] = _Loggable(operation, format_str)

    def finalize_with_session(self, sess):
        """ """
        assert self.fetcher is None, 'Already finalized!'
        fetches = {name: loggable.operation for name, loggable in self.loggables.items()}
        self.fetcher = sess.make_callable(fetches)

    def log(self, joiner=', '):
        assert self.fetcher is not None, 'Call finalize_with_session()'

        def _iter_formatted_strs():
            fetched = self.fetcher()
            for name, fetched_val in fetched.items():
                fetched_val_formatted = self.loggables[name].format_str.format(fetched_val)
                yield '{}: {}'.format(name, fetched_val_formatted)
        return joiner.join(_iter_formatted_strs())

    @staticmethod
    def _get_default_format_str(operation):
        if tf.float32.is_compatible_with(operation.dtype):
            return '{:.4f}'
        return '{}'


_default_logger = Logger()


def get_default_logger():
    return _default_logger


# Saving -----------------------------------------------------------------------


class VersionAwareSaver(object):
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = path.join(save_dir, 'ckpt')
        self.var_names_fn = path.join(save_dir, 'var_names.pkl')

        current_vars = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
                        tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
        if path.exists(self.var_names_fn):
            restorable_var_names = self._get_restorable_var_names()
            var_list = [var for var in current_vars if var.name in restorable_var_names]
            if len(var_list) != len(current_vars):
                tf.logging.warn('Graph holds {} variables, restoring {}...'.format(
                    len(current_vars), len(var_list)))
            # TODO; init un-restored vars
        else:
            var_list = current_vars
            self._set_restorable_var_names([var.name for var in current_vars])

        self.saver = tf.train.Saver(var_list=var_list)

    def save(self, sess, global_step):
        self.saver.save(sess, self.save_path, global_step)

    def restore(self, sess):
        latest_ckpt = tf.train.latest_checkpoint(path.dirname(self.save_path))
        self.saver.restore(sess, latest_ckpt)

    def _get_restorable_var_names(self):
        assert path.exists(self.var_names_fn)
        with open(self.var_names_fn, 'rb') as f:
            return pickle.load(f)

    def _set_restorable_var_names(self, var_names):
        assert not path.exists(self.var_names_fn)
        with open(self.var_names_fn, 'wb') as f:
            return pickle.dump(var_names, f)



