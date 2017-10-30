from unittest import TestCase
from . import iterable_tools as it


class TestIterableTools(TestCase):
    def test_sliced_iter(self):
        otp = list(it.sliced_iter(range(5), slice_len=2))
        self.assertEqual(otp, [[0, 1], [2, 3], [4]])

        otp = list(it.sliced_iter(range(5), slice_len=2, allow_smaller_final_slice=False))
        self.assertEqual(otp, [[0, 1], [2, 3]])

        otp = list(it.sliced_iter(range(5), slice_len=1))
        self.assertEqual(otp, list([[el] for el in range(5)]))

    def test_flag_first_iter(self):
        otp = list(it.flag_first_iter([]))
        self.assertEqual(otp, [])

        otp = list(it.flag_first_iter(['LJ']))
        self.assertEqual(otp, [(True, 'LJ')])

        otp = list(it.flag_first_iter(range(5)))
        self.assertEqual(otp, list(zip((i == 0 for i in range(5)), range(5))))


    def test_iter_with_sliding_window(self):
        opt = list(it.iter_with_sliding_window(range(5), window_size=2))
        self.assertEqual(opt, [(0, 1), (1, 2), (2, 3), (3, 4)])

        opt = list(it.iter_with_sliding_window(range(5), window_size=3))
        self.assertEqual(opt, [(0, 1, 2), (1, 2, 3), (2, 3, 4)])

        opt = list(it.iter_with_sliding_window(list(range(5)), window_size=3))
        self.assertEqual(opt, [(0, 1, 2), (1, 2, 3), (2, 3, 4)])

        opt = list(it.iter_with_sliding_window(list(), window_size=3))
        self.assertEqual(opt, [])

        opt = list(it.iter_with_sliding_window(range(5), window_size=1))
        self.assertEqual(opt, [(i,) for i in range(5)])

