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
