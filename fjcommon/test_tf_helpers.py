from unittest import TestCase
from . import tf_helpers
import tensorflow as tf
import numpy as np


class TestTFRecords(TestCase):
    def test_random_flip(self):
        with tf.Session() as sess:
            mat = [[0, 1, 2],
                   [2, 1, 0]]
            mat_flipped_expected = [[2, 1, 0],
                                    [0, 1, 2]]
            mat_flipped = tf_helpers.random_flip(mat, axis=1, seed=1)
            o = sess.run(mat_flipped)
            np.testing.assert_equal(o, mat_flipped_expected)

            shape = [2, None, None, 3]
            p = tf.placeholder(tf.uint8, shape)
            p_flipped = tf_helpers.random_flip(p)
            self.assertEqual(shape, p_flipped.shape.as_list())

    def test_version_aware_saver(self):
        with tf.variable_scope('net'):
            a = tf.get_variable('a', shape=())
            b = tf.get_variable('b', shape=())
            c = a + b
        with tf.variable_scope('prob'):
            a = tf.get_variable('c', shape=())
            d = c + a
        s = tf_helpers.VersionAwareSaver('test', skip_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                             scope='prob'))

