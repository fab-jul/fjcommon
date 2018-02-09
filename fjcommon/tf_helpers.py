import glob

import tensorflow as tf
import numpy as np

from contextlib import contextmanager
import functools
import itertools
from collections import OrderedDict, namedtuple
import os
import re
import sys
from os import path
import pickle
from fjcommon import lifting
from fjcommon import os_ext
from fjcommon.numpy_ext import is_int


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


# Static Asserts ---------------------------------------------------------------


def assert_dtype(x, dtype):
    assert dtype.is_compatible_with(x.dtype), 'Expected {} == {}'.format(dtype.name, x.dtype.name)


def assert_ndims(x, ndims):
    assert x.shape.ndims == ndims, 'Expected ndims == {}, got shape {}'.format(ndims, x.shape)


def assert_dim(x, x_dim, target):
    assert_dim_is_specified(x, x_dim)
    assert int(x.shape[x_dim]) == target, 'Expected {}[{}] == {}'.format(x.shape, x_dim, target)


def assert_equal_shape(x, y):
    assert x.shape == y.shape, 'Expected {} == {}'.format(x.shape, y.shape)


def assert_equal_dims(x, x_dim, y, y_dim):
    assert_dim_is_specified(x, x_dim)
    assert_dim_is_specified(y, y_dim)
    assert int(x.shape[x_dim]) == int(y.shape[y_dim]), 'Expected {}[{}] == {}[{}]'.format(
        x.shape, x_dim, y.shape, y_dim)


def assert_shape_is_fully_defined(x):
    assert x.shape.is_fully_defined(), 'Expected {} to be fully defined'.format(x.shape)


def assert_dim_is_specified(x, x_dim):
    assert x.shape[x_dim] is not None, 'Expected {}-th entry of {} to be specified!'.format(
        x_dim, x.shape)


# Ops --------------------------------------------------------------------------


def transpose_NCHW_to_NHWC(t):
    return tf.transpose(t, (0, 2, 3, 1), name='to_NHWC')


def transpose_NHWC_to_NCHW(t):
    return tf.transpose(t, (0, 3, 1, 2), name='to_NCHW')


def mse_psnr(inp, otp, add_mse_to_total_loss=True):
    """ NOTE: doesn't matter if inp and otp are NCHW or NHWC, as long as it's the same for both. """
    with tf.name_scope('mse_psnr'):
        assert tf.float32.is_compatible_with(inp.dtype) and tf.float32.is_compatible_with(otp.dtype)

        # Calculate MSE using floats for differentiability
        mse_per_image_float = tf.reduce_mean(tf.square(otp - inp), axis=[1, 2, 3])
        mse_float = tf.reduce_mean(mse_per_image_float)
        if add_mse_to_total_loss:
            tf.losses.add_loss(mse_float)

        # Calculate PSNR using quantized int values, bc that's the real world.
        # Values are expected to be in 0...255, i.e., uint8, but tf.square does not support uint8's
        otp_int32, inp_int32 = tf.cast(otp, tf.int32), tf.cast(inp, tf.int32)
        squared_error_int32 = tf.square(otp_int32 - inp_int32)
        squared_error_float = tf.to_float(squared_error_int32)
        mse_per_image = tf.reduce_mean(squared_error_float, axis=[1, 2, 3])
        psnr_per_image = 10 * log10(255.0 * 255.0 / mse_per_image)
        psnr = tf.reduce_mean(psnr_per_image)
        return mse_float, psnr


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


def reverse_every_other_row(inp, batch_axis=0, seq_axis=1):
    """
    Given some tensor `input`, reverse every other row of the `batch_axis` in the direction of the `seq_axis`.
    Example:

        inp = [[1,  2,  3],
               [4,  5,  6],
               [7,  8,  9],
               [10, 11, 12]]

        reverse_every_other_row(inp, 0, 1) ->
              [[1,  2,  3],
               [6,  5,  4],   # reversed
               [7,  8,  9],
               [12, 11, 10]]  # reversed

    """
    assert inp.shape.ndims >= max(batch_axis, seq_axis) + 1
    inp_shape = tf.shape(inp)
    h, w = inp_shape[batch_axis], inp_shape[seq_axis]

    # Create seq_lengths == [0, w, 0, w, ...], where len(seq_lengths) == h
    seq_lengths = tf.stack((
        tf.zeros([h//2 + 1], dtype=tf.int32),
        tf.fill([h//2 + 1], w)), axis=1)
    seq_lengths = tf.reshape(seq_lengths, [-1])[:h]  # flatten and trim

    return tf.reverse_sequence(inp, seq_lengths=seq_lengths, seq_dim=seq_axis, batch_axis=batch_axis)


# Helpers ----------------------------------------------------------------------


def list_without_None(*args):
    """ Basically list(filter(None, args)) but this wouldn't work for tensors because they cannot compare to None. """
    return [arg for arg in args if arg is not None]


# Variable Learning Rates ------------------------------------------------------


_GRADIENT_SUMMARY_NAME_SCOPE = 'grads'


def create_train_op_with_different_lrs(total_loss, optimizer_default, special_optimizers_and_vars,
                                       summarize_gradients=False, gradient_clipping=None):
    """
    :param total_loss: loss to minimize
    :param optimizer_default: optimizer to use for all variables not assigned to one of the special optimizers. Note:
                this may be None, in which case only special_optimizers_and_vars is used.
    :param special_optimizers_and_vars: list of tuples (Optimizer, [variables]). Note: this also works if
                len(special_optimizers_and_vars) == 0
    :return: tf.group of training steps
    """
    trainable_vars = tf.trainable_variables()
    all_vars_special = list(itertools.chain.from_iterable((vs for _, vs in special_optimizers_and_vars)))
    trainable_vars_without_special = [var for var in trainable_vars if var not in all_vars_special]
    assert len(trainable_vars) == len(trainable_vars_without_special) + len(all_vars_special), \
        "{} does not contain each {} exactly once".format(trainable_vars, all_vars_special)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    all_vars_sorted = trainable_vars_without_special + all_vars_special
    with tf.control_dependencies(update_ops):
        grads = tf.gradients(total_loss, all_vars_sorted)

    if gradient_clipping is not None:
        tf.logging.info('Gradient clipping to global norm {}'.format(gradient_clipping))
        grads, global_norm = tf.clip_by_global_norm(grads, gradient_clipping)
        with tf.name_scope(_GRADIENT_SUMMARY_NAME_SCOPE):
            tf.summary.scalar('global_norm', global_norm)

    global_step = tf.contrib.slim.get_or_create_global_step()
    grads_special_start_idx = len(trainable_vars_without_special)
    grads_default = grads[:grads_special_start_idx]
    has_default_optimizer = optimizer_default is not None
    train_steps = []
    if has_default_optimizer:
        train_steps.append(
            optimizer_default.apply_gradients(
                zip(grads_default, trainable_vars_without_special),
                global_step=global_step))
    for optimizer_special, vars_special in special_optimizers_and_vars:
        grads_end_idx = grads_special_start_idx + len(vars_special)
        grads_special = grads[grads_special_start_idx:grads_end_idx]
        grads_special_start_idx = grads_end_idx
        train_steps.append(
            optimizer_special.apply_gradients(
                zip(grads_special, vars_special),
                # exactly one call to apply_gradients must contain global_step
                global_step=None if has_default_optimizer else global_step))
    assert grads_special_start_idx == len(all_vars_sorted), '{} != {}'.format(grads_special_start_idx, len(all_vars_sorted))

    if summarize_gradients:
        add_gradients_summaries(zip(grads, all_vars_sorted), histograms=False, excludes=['BatchNorm'])

    return tf.group(*train_steps)


def add_gradients_summaries(grads_and_vars, histograms=True, norms=True, excludes=None):
    assert histograms or norms
    if not excludes:
        excludes = []
    with tf.name_scope(_GRADIENT_SUMMARY_NAME_SCOPE):
        for grad, var in grads_and_vars:
            if any(exclude in var.op.name for exclude in excludes):
                continue
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    grad_values = grad.values
                else:
                    grad_values = grad
                if histograms:
                    tf.summary.histogram(var.op.name + '/gradient', grad_values)
                if norms:
                    tf.summary.scalar(var.op.name + '/gradient_norm', tf.global_norm([grad_values]))
            else:
                tf.logging.info('Var {} has no gradient'.format(var.op.name))


# Mutable Variables ------------------------------------------------------------

class MutableVar(object):
    """
    Behaves like a normal tf.Variable but sets up setter using a placeholder
    """
    def __init__(self, name, shape, dtype, initializer, allowed_range=None):
        self.allowed_range = allowed_range
        self._var = tf.get_variable(
                name, shape=shape, dtype=dtype, initializer=initializer, trainable=False)
        self._var_setter_ph = tf.placeholder(dtype, shape, name=name + '_ph')
        self._var_setter = tf.assign(self._var, self._var_setter_ph, validate_shape=True)

    def set(self, sess, new_value):
        if self.allowed_range:
            allowed_min, allowed_max = self.allowed_range
            assert allowed_min <= new_value <= allowed_max, 'Not in range: {}, {}'.format(new_value, self.allowed_range)
        sess.run(self._var_setter, {self._var_setter_ph: new_value})

    def get(self):  # -> Tensor
        return self._var


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
    """
    Esample Usage:

        logger = Logger()
        ... # setup logger
        logger.log().add_to_tensorboard(fw, itr).add_to_console(itr)

    or

        log_output = logger.log()
        ... # do something with log_output
        log_output.add_to_tensorboard(fw, itr).add_to_console(itr)

    """
    def __init__(self, log_str, tags_and_values):
        self.log_str = log_str
        self.tags_and_values = tags_and_values

    def add_to_tensorboard(self, filewriter, itr):
        if self.tags_and_values:
            log_values(filewriter, self.tags_and_values, iteration=itr)
        return self

    def add_to_console(self, itr, append=''):
        log_str = '{}: {}'.format(itr, self.log_str)
        if append:
            log_str += ' ' + append
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

    # TODO Evaluate if this should be implemented with tf.py_func
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
        fetched = self._remove_non_loggables(fetched)
        log_str, log_tags_and_values = self._get_log_str_and_values(fetched, joiner)
        return LoggerOutput(log_str=log_str, tags_and_values=log_tags_and_values)

    def _remove_non_loggables(self, fetched):
        assert isinstance(fetched, dict)
        return {name: val for name, val in fetched.items() if name in self._loggables}

    def _get_log_str_and_values(self, fetched, joiner):
        formatted_strs = []
        log_tags_and_values = []
        for name, fetched in sorted(fetched.items()):
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


def add_scalar_summaries_with_prefix(prefix, summaries):
    for loss_name, loss_tensor in summaries.items():
        summary_name = '{}/{}'.format(prefix, loss_name)
        tf.summary.scalar(summary_name, loss_tensor)


def prep_for_image_summary(t, n=3, autoscale=False, name='img'):
    """ given tensor t of shape NCHW, return t[:n, ...] transposed to NHWC, cast to uint8 if not autoscale """
    assert int(t.shape[1]) == 3, 'Expected N3HW, got {}'.format(t)
    with tf.name_scope('prep_' + name):
        t = transpose_NCHW_to_NHWC(t[:n, ...])
        if autoscale:  # if t is float32, tf.summary.image will automatically rescale
            assert tf.float32.is_compatible_with(t.dtype)
            return t
        else:  # if t is uint8, tf.summary.image will NOT automatically rescale
            return tf.cast(t, tf.uint8, 'uint8')


def prep_for_grayscale_image_summary(t, n=3, autoscale=False, name='img'):
    """ given tensor t of shape NHW, return t[:n, ...] reshaped to NHW1, cast to uint8 if not autoscale """
    assert len(t.shape) == 3
    with tf.name_scope('prep_' + name):
        t = t[:n, ...]
        t = tf.expand_dims(t, -1)  # NHW1
        if autoscale:
            assert tf.float32.is_compatible_with(t.dtype)
            return t
        else:
            return tf.cast(t, tf.uint8, name='uint8')



# Saving -----------------------------------------------------------------------


def all_saveable_objects(scope=None):
    """ Copied private function in TF source. This is what tf.train.Saver saves if var_list=None is passed. """
    return (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope) +
            tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS, scope))


def clean_checkpoints(ckpt_dirs, always_keep=None):

    if '-q' in ckpt_dirs:
        quiet = True
        ckpt_dirs.remove('-q')
    else:
        quiet = False
    if not always_keep:
        always_keep = []
    for ckpt_dir in sorted(ckpt_dirs):
        if any(always_keep_el in ckpt_dir for always_keep_el in always_keep):
            continue
        all_ckpts = VersionAwareSaver.all_ckpts_with_iterations(ckpt_dir)
        for _, ckpt_file_base in all_ckpts[:-1]:
            related_files = glob.glob(ckpt_file_base + '.*')
            assert 1 <= len(related_files) <= 3, related_files
            for f in related_files:
                if quiet:
                    print(f)
                else:
                    os.remove(f)


class VersionAwareSaver(object):
    _CKPT_FN = 'ckpt'

    def __init__(self, save_dir, **kwargs_saver):
        """
        :param save_dir: where to save data
        :param kwargs_saver: Passed on to the tf.train.Saver that will be created
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpt_base_file_path = path.join(save_dir, VersionAwareSaver._CKPT_FN)
        self.var_names_fn = path.join(save_dir, 'var_names.pkl')
        self.init_unrestored_op = None

        var_list = kwargs_saver.get('var_list', None)
        if 'var_list' in kwargs_saver:
            del kwargs_saver['var_list']

        current_vars = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
                        tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
        if var_list is None:
            if path.exists(self.var_names_fn):
                restorable_var_names = self._get_restorable_var_names()
                var_list = [var for var in current_vars if var.name in restorable_var_names]
                if len(var_list) != len(current_vars):
                    tf.logging.warn('Graph holds {} variables, restoring {}...'.format(len(current_vars), len(var_list)))
                    unrestored = [var for var in current_vars if var.name not in restorable_var_names]
                    if unrestored:
                        tf.logging.warn('Not restored: {}'.format(unrestored))
                        self.init_unrestored_op = tf.variables_initializer(unrestored)
            else:
                var_list = current_vars
                self._set_restorable_var_names([var.name for var in current_vars])
        else:
            pass
            #var_list_names = [var.name for var in var_list]
            #self.init_unrestored_op = tf.variables_initializer(
                #[var for var in current_vars if var in var_list])
        self.saver = tf.train.Saver(var_list=var_list, **kwargs_saver)

    def save(self, sess, global_step):
        self.saver.save(sess, self.ckpt_base_file_path, global_step)

    def iter_ckpts(self, sess, restore_itr_spec):
        assert isinstance(restore_itr_spec, str)
        if is_int(restore_itr_spec):
            ckpt_itr = self.restore(sess, int(restore_itr_spec))
            yield from [ckpt_itr]
        elif restore_itr_spec == 'all':
            yield from self.restore_all_ckpts_iterator(sess)
        else:
            raise ValueError('restore_itr_spec should be either int or "all", got {}'.format(restore_itr_spec))

    def restore(self, sess, restore_itr=-1):
        """ Restores variables and initialized un-restored variables. """
        ckpt_to_restore_itr, ckpt_to_restore = self.get_checkpoint_path(restore_itr)
        assert ckpt_to_restore is not None
        self.saver.restore(sess, ckpt_to_restore)
        if self.init_unrestored_op is not None:
            sess.run(self.init_unrestored_op)
        return ckpt_to_restore_itr

    def restore_all_ckpts_iterator(self, sess):
        """ Restores one chkpt after the other, yielding the iteration each time """
        for ckpt_itr, ckpt_path in VersionAwareSaver.all_ckpts_with_iterations(self.save_dir):
            self.saver.restore(sess, ckpt_path)
            if self.init_unrestored_op is not None:
                sess.run(self.init_unrestored_op)
            yield ckpt_itr

    def get_checkpoint_path(self, restore_itr):
        all_ckpts_with_iterations = VersionAwareSaver.all_ckpts_with_iterations(self.save_dir)
        ckpt_to_restore_idx = -1 if restore_itr == -1 else VersionAwareSaver.index_of_ckpt_with_iter(
            all_ckpts_with_iterations, restore_itr)
        ckpt_to_restore_itr, ckpt_to_restore = all_ckpts_with_iterations[ckpt_to_restore_idx]
        assert ckpt_to_restore is not None
        return ckpt_to_restore_itr, ckpt_to_restore

    @staticmethod
    def all_ckpts_with_iterations(save_dir):
        return sorted(
            (VersionAwareSaver.iteration_of_checkpoint(ckpt_path), ckpt_path)
            for ckpt_path in VersionAwareSaver.all_ckpts_in(save_dir))

    @staticmethod
    def index_of_ckpt_with_iter(ckpts_with_iterations, target_ckpt_itr):
        """ given a sorted list `ckpts_with_iterations` of (ckpt_iter, ckpt_path), returns the smallest index i in that
        list where target_ckpt_itr >= ckpt_iter """
        for i, (ckpt_iter, _) in reversed(list(enumerate(ckpts_with_iterations))):
            if target_ckpt_itr >= ckpt_iter:
                return i
        raise ValueError('*** Cannot find ckpt with iter >= {} in {}'.format(
            target_ckpt_itr, ckpts_with_iterations))

    @staticmethod
    def iteration_of_checkpoint(ckpt_path):
        ckpt_file_name = os.path.basename(ckpt_path)
        m = re.search(r'-(\d+)', ckpt_file_name)
        assert m is not None, 'Expected -(\\d+), got {}'.format(ckpt_path)
        return int(m.group(1))

    @staticmethod
    def all_ckpts_in(save_dir):
        return set(
            os.path.join(save_dir, os.path.splitext(fn)[0])
            for fn in os.listdir(save_dir)
            if VersionAwareSaver._CKPT_FN in fn)

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
    :param values: tensor of dtype int64. Will do histogram over last dimension!
    :param L: number of possible values in `values`.
    :param num_rows: number of rows of the histogram
    :return: (histogram, update_op), where
        histogram: tensor of dimension (num_rows, C, L), where C = values.shape[-1]
        update_op: operation to run to update the histogram from the `values` tensor
    """
    with tf.name_scope(name, 'histo'):
        assert tf.int64.is_compatible_with(values.dtype), 'values must be int64, not {}'.format(values.dtype)

        C = values.get_shape().as_list()[-1]
        histogram = get_variable_histogram(name, num_rows, C, L)
        tf.logging.info('Creating histogram of shape {}...'.format(histogram.shape.as_list()))

        histogram_current_idx = get_variable_zeros(name + '_idx', shape=(), dtype=tf.int64)
        histogram_current_idx_inc = tf.assign(histogram_current_idx,
                                              tf.mod(histogram_current_idx + 1, num_rows))
        # one row of the histogram, which is stored at histogram[histogram_current_idx, :, :]
        histo_slice = _histogram_slice(values, C, L)  # (C, L)

        with tf.control_dependencies([histogram_current_idx_inc]):
            # same as
            #   histogram[histogram_current_idx, :, :] = histo_slice
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

    def all_exist(self, img_names):
        for img_name in img_names:
            img_out_p = self._to_full_path(img_name)
            if not path.exists(img_out_p):
                return False
        return True

    def _to_full_path(self, rel_img_name):
        img_out_p = path.join(self.img_dir, rel_img_name)
        if not img_out_p.endswith('.png'):
            return img_out_p + '.png'
        return img_out_p

    def save(self, fetched_tensors, img_names, chmod=0o755, exists_ok=True):
        """
        Saves fetched images.
        :param fetched_tensors: Result of a call to session.run(fetches) where previously,
        augment_fetch_dict(fetches, output) was called.
        :param img_names: list of length batch_size. Image will be saved at 'self.img_dir/img_names[i]', i.e.,
        names may contain slashes. The corresponding dirs will be created if necessary.
        :param chmod: if not None, chmod the resulting image, up to self.img_dir
        """
        assert self.fetch_dict_key in fetched_tensors, 'Use augment_fetch_dict'
        img_out = fetched_tensors[self.fetch_dict_key]
        num_batches = img_out.shape[0]
        assert len(img_names) == num_batches

        for batch in range(num_batches):
            img_out_p = self._to_full_path(img_names[batch])
            if not exists_ok and path.exists(img_out_p):
                raise FileExistsError('Image already exists: {}'.format(img_out_p))
            img_out_b = img_out[batch, ...]
            os.makedirs(path.dirname(img_out_p), exist_ok=True)
            self.save_img(name=img_out_p, arr=img_out_b)
            if chmod is not None:
                os_ext.chmodr(img_out_p, chmod, upto=path.dirname(self.img_dir))

        return img_out



def random_flip(mat, axis=1, seed=None):
    """Randomly flip mat horizontally (left to right).

    With a 1 in 2 chance, outputs the contents of `mat` flipped along the
    axis dimension.  Otherwise output the mat as-is.

    Args:
      mat: tensor
      seed: A Python integer. Used to create a random seed. See
        @{tf.set_random_seed}
        for behavior.

    Returns:
      A 3-D tensor of the same type and shape as `mat`.
    """
    mat = tf.convert_to_tensor(mat, name='mat')
    uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = tf.less(uniform_random, .5)
    result = tf.cond(mirror_cond,
                     lambda: tf.reverse(mat, [axis]),
                     lambda: mat)
    return result


if __name__ == '__main__':
    clean_checkpoints(sys.argv[1:])





