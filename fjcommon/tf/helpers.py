import tensorflow as tf
from contextlib import contextmanager


@contextmanager
def start_queues_in_sess(init_vars=True, name=None):
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
