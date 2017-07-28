import tensorflow as tf
import numpy as np
from contextlib import contextmanager
import functools
from collections import OrderedDict, namedtuple
import os
import sys
from os import path
import pickle
from fjcommon import lifting


# Session ----------------------------------------------------------------------

@contextmanager
def start_queues_in_sess(init_vars=True, name=None):
    # TODO: allow only starting a subset of queues
    with create_session() as sess:
        tf.logging.info('Session Started' + (': {}'.format(name) if name else ''))
        if init_vars:
            tf.logging.info('Initializing variables...')
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            yield sess, coord
        except tf.errors.OutOfRangeError:
            tf.logging.info('Queues exhausted...')
        finally:
            coord.request_stop()
        coord.join(threads)


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
    def __init__(self, fetches, to_value, format_str, add_summary=None):
        """
        :param fetches: anything accepted by tf.Session.run
        :param to_value: function taking the result of tf.Session.run(fetches) and returns a value that can be
        formatted with format_str
        :param format_str:
        """
        self.fetches = fetches
        self.to_value = to_value
        self.format_str = format_str
        self.add_summary = add_summary


class LoggerOutput(object):
    def __init__(self, log_str, tags_and_values):
        self.log_str = log_str
        self.tags_and_values = tags_and_values

    def add_to_tensorboard(self, filewriter, itr):
        if self.tags_and_values:
            log_values(filewriter, self.tags_and_values, iteration=itr)
        return self

    def add_to_console(self, itr, avg_time=None):
        log_str = '{}: {}'.format(itr, self.log_str)
        if avg_time:
            log_str += ' (s/step: {:.3f})'.format(avg_time)
        tf.logging.info(log_str)
        return self


_LOGGER_DEFAULT_FORMAT_STR = '{:.4f}'


class Logger(object):
    def __init__(self):
        self._loggables = OrderedDict()
        self._fetcher = None

    def add_loggable(self, name, operation, format_str=_LOGGER_DEFAULT_FORMAT_STR, add_summary=None):
        """
        :param name: name to print
        :param operation: tensorflow operation to run
        :param format_str: string formatting the result
        :param add_summary: if not None, output of operation is logged as summary.
        """
        if add_summary is not None and add_summary is True:
            add_summary = name
        self._loggables[name] = _Loggable(
            operation, to_value=lambda x: x, format_str=format_str, add_summary=add_summary)

    def add_loggable_function(self, name, f, arg_tensors, format_str=_LOGGER_DEFAULT_FORMAT_STR, add_summary=None):
        """
        :param name: name to print
        :param f: function with keyword arguments == arg_tensors.keys(), returning a value that can be formatted with
        format_str and, if add_summary is not None, can be logged as a summary.
        :param arg_tensors: dictionary str -> Tensor
        :param format_str:
        :param add_summary: if not None, log value returned by f as summary named `add_summary`.
        :return:
        """
        assert isinstance(arg_tensors, dict), 'Expected dict, not {}'.format(arg_tensors)

        if add_summary is not None and add_summary is True:
            add_summary = name

        def to_value(arg_tensors_output):
            return f(**arg_tensors_output)
        self._loggables[name] = _Loggable(arg_tensors, to_value, format_str, add_summary)

    def required_fetches(self):
        return {name: loggable.fetches for name, loggable in self._loggables.items()}

    def finalize_with_session(self, sess):
        assert self._fetcher is None
        self._fetcher = sess.make_callable(self.required_fetches())

    def log(self, fetched=None, joiner=', '):
        """
        :param fetched: dict, containing at least the result of tf.Session.run(required_fetches()) or None,
        if `finalize_with_session` was previously called
        :return: LoggerOutput
        """
        if fetched is None:
            assert self._fetcher is not None
            fetched = self._fetcher()
        fetched = self._filter_required(fetched)
        log_str, log_tags_and_values = self._get_log_str_and_values(fetched, joiner)
        return LoggerOutput(log_str=log_str, tags_and_values=log_tags_and_values)

    def _filter_required(self, fetched):
        assert isinstance(fetched, dict)
        return {name: val for name, val in fetched.items() if name in self._loggables}

    def _get_log_str_and_values(self, fetched, joiner):
        formatted_strs = []
        log_tags_and_values = []
        for name, fetched in fetched.items():
            loggable = self._loggables[name]
            fetched_val = loggable.to_value(fetched)
            if loggable.add_summary:
                log_tags_and_values.append((loggable.add_summary, fetched_val))
            fetched_val_formatted = loggable.format_str.format(fetched_val)
            formatted_strs.append('{}: {}'.format(name, fetched_val_formatted))
        return joiner.join(formatted_strs), log_tags_and_values


_default_logger = Logger()


def get_default_logger():
    return _default_logger


def log_values(summary_writer, tags_and_values, iteration):
    summary_values = [
        tf.Summary.Value(tag=tag, simple_value=value)
        for tag, value in tags_and_values
    ]
    summary_writer.add_summary(tf.Summary(value=summary_values), global_step=iteration)


# Saving -----------------------------------------------------------------------


class VersionAwareSaver(object):
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = path.join(save_dir, 'ckpt')
        self.var_names_fn = path.join(save_dir, 'var_names.pkl')
        self.init_unrestored_op = None

        current_vars = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
                        tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
        if path.exists(self.var_names_fn):
            restorable_var_names = self._get_restorable_var_names()
            var_list = [var for var in current_vars if var.name in restorable_var_names]
            if len(var_list) != len(current_vars):
                tf.logging.warn('Graph holds {} variables, restoring {}...'.format(len(current_vars), len(var_list)))
                unrestored = [var for var in current_vars if var.name not in restorable_var_names]
                if unrestored:
                    tf.logging.warn('Not restored: {}'.format(unrestored))
                    self.init_unrestored_op = tf.initialize_variables(unrestored)
        else:
            var_list = current_vars
            self._set_restorable_var_names([var.name for var in current_vars])

        self.saver = tf.train.Saver(var_list=var_list)

    def save(self, sess, global_step):
        self.saver.save(sess, self.save_path, global_step)

    def restore(self, sess):
        """ Restores variables and initialized un-restored variables. """
        latest_ckpt = tf.train.latest_checkpoint(path.dirname(self.save_path))
        assert latest_ckpt is not None, 'No checkpoints at {}'.format(self.save_path)
        self.saver.restore(sess, latest_ckpt)
        if self.init_unrestored_op is not None:
            sess.run(self.init_unrestored_op)

    def _get_restorable_var_names(self):
        assert path.exists(self.var_names_fn)
        with open(self.var_names_fn, 'rb') as f:
            return pickle.load(f)

    def _set_restorable_var_names(self, var_names):
        assert not path.exists(self.var_names_fn)
        with open(self.var_names_fn, 'wb') as f:
            return pickle.dump(var_names, f)


# Histogram --------------------------------------------------------------------


def histogram_nd(name, values, L, num_rows=100):
    """
    :param name: name of the histogram variable
    :param values: tensor of dtype int64. Will do histogram over last dimension (channel dimension).
    :param L: number of possible values in `values`.
    :param num_rows: number of rows of the histogram
    :return: (histogram, update_op), where
        histogram: tensor of dimension (num_rows, C, L), where C = values.shape[-1]
        update_op: operation to run to update the histogram from the `values` tensor
    """
    assert tf.int64.is_compatible_with(values.dtype), 'values must be int64, not {}'.format(values.dtype)

    C = values.get_shape().as_list()[-1]
    histogram = get_variable_histogram(name, num_rows, C, L)
    histogram_current_idx = get_variable_zeros(name + '_idx', shape=(), dtype=tf.int64)
    histogram_current_idx_inc = tf.assign(histogram_current_idx,
                                          tf.mod(histogram_current_idx + 1, num_rows))
    # one row of the histogram, (C, L) dimensional
    histo_slice = _histogram_slice(values, C, L)

    with tf.control_dependencies([histogram_current_idx_inc]):
        # write histo_slice to the current index in the histogram
        update_op = tf.scatter_update(histogram, histogram_current_idx, histo_slice)
        
    return histogram, update_op


def get_variable_histogram(name, num_rows, C, L):
    return tf.get_variable(
        name, shape=(num_rows, C, L), dtype=tf.int64, initializer=tf.ones_initializer(), trainable=False)


def _histogram_slice(values, C, L):
    # necessary to work around TF expecting floats for histogram_fixed_width,
    # see https://github.com/tensorflow/tensorflow/issues/6650
    values = tf.to_float(values)
    histo_vals = tf.constant([0, L - 1], dtype=tf.float32)
    # (C, L)
    histo_slice = tf.stack([
        tf.histogram_fixed_width(values=values[..., channel], value_range=histo_vals, nbins=L)
        for channel in range(C)], axis=0)
    return tf.cast(histo_slice, tf.int64)


def _test_histogram_nd():
    a = [[0, 1, 2, 3],
         [0, 1, 2, 3],
         [4, 4, 4, 4]]
    L = np.max(a) + 1
    a = tf.constant(a, dtype=tf.int64)
    histogram, update_op = histogram_nd('histo', a, L, num_rows=4)

    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())
    for _ in range(10):
        histo_, _ = s.run([histogram, update_op])
        print(np.sum(histo_, 0))


# ImageSaver -------------------------------------------------------------------


class ImageSaver(object):
    """
    Use case:

        s = ImageSaver(out_dir)
        fetch_dict = ..  # some dictionary used to fetch your tensors
        s.augment_fetch_dict(fetch_dict, image_batch_tensor)
        ...
        fetched = sess.run(fetch_dict)
        s.save(fetched, [img_name1, img_name2, ...])

    or, if you don't fetch a lot of other stuff:

        s = ImageSaver(out_dir)
        fetch_dict = s.get_fetch_dict(image_batch_tensor)
        s.save(sess.run(fetch_dict), [img_name1, img_name2, ...])

    """
    def __init__(self, img_dir):
        tf.logging.info('Saving images at {}...'.format(img_dir))
        self.img_dir = img_dir
        os.makedirs(img_dir, exist_ok=True)
        self.fetch_dict_key = 'output_{}'.format(self.img_dir.replace(os.sep, '_'))
        try:
            import scipy.misc
            self.save_img = scipy.misc.imsave
        except ImportError:
            print('Need scipy to save images!')
            sys.exit(1)

    def augment_fetch_dict(self, fetch_dict, output_tensor):
        """
        :param fetch_dict: dictionary that will later be passed to a session.run() call. Will add output_tensor to the
        dict such that it will be fetched. Must be shape NHWC, with C == 3.
        """
        assert tf.uint8.is_compatible_with(output_tensor.dtype)
        assert self.fetch_dict_key not in fetch_dict
        output_tensor_shape = output_tensor.shape.as_list()
        assert len(output_tensor_shape) == 4 and output_tensor_shape[3] == 3, 'Expected NHWC with C == 3'
        fetch_dict[self.fetch_dict_key] = output_tensor

    def get_fetch_dict(self, output_tensor):
        """
        :returns: a dictionary that can be passed to sess.run(). The output can be passed to self.save()
        """
        fetch_dict = {}
        self.augment_fetch_dict(fetch_dict, output_tensor)
        return fetch_dict

    def save(self, fetched_tensors, img_names):
        """
        Saves fetched images.
        :param fetched_tensors: Result of a call to session.run(fetches) where previously,
        augment_fetch_dict(fetches, output) was called.
        :param img_names: list of lenght batch_size
        """
        assert self.fetch_dict_key in fetched_tensors, 'Use augment_fetch_dict'
        img_out = fetched_tensors[self.fetch_dict_key]
        num_batches = img_out.shape[0]
        assert len(img_names) == num_batches

        for batch in range(num_batches):
            img_out_p = path.join(self.img_dir, img_names[batch])
            if not img_out_p.endswith('.png'):
                img_out_p += '.png'
            img_out = img_out[batch, ...]
            self.save_img(name=img_out_p, arr=img_out)

