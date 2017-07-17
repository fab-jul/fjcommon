import tensorflow as tf
import numpy as np
from contextlib import contextmanager
import functools


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


def log2(x):
    return tf.log(x) / tf.log(tf.constant(2, dtype=x.dtype))


def cache_some(tensors, np_dtyes, cache_size, cache_per_batch=2):
    all_shapes = [t.get_shape().as_list() for t in tensors]
    assert all(None not in s and len(s) >= 2 for s in all_shapes), \
        'All shapes must contain batch_size plus at least one additional dimension + no unknown: {}'.format(all_shapes)

    batch_size = all_shapes[0][0]
    num_batches = cache_size // cache_per_batch
    caches = [np.zeros((cache_size,) + tuple(t_shape[1:]), np_dtype)
              for t_shape, np_dtype in zip(all_shapes, np_dtyes)]
    tensor_names = ','.join(t.name for t in tensors)
    with start_queues_in_sess(name='Caching {}'.format(tensor_names)) as (sess, _):
        for i in range(num_batches):
            tensors_out = sess.run(tensors)
            for t_i, t_out in enumerate(tensors_out):
                t_cache = caches[t_i]
                for j in range(cache_per_batch):
                    t_cache[cache_per_batch * i + j, :] = t_out[j * (batch_size // cache_per_batch), :]
    return caches


get_variable_zeros = functools.partial(tf.get_variable, initializer=tf.zeros_initializer(), trainable=False)
get_variable_ones = functools.partial(tf.get_variable, initializer=tf.ones_initializer(), trainable=False)
