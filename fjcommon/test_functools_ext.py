from unittest import TestCase
from . import functools_ext as ft


class TestMoreFunctools(TestCase):
    def test_compose(self):
        def f1(x):
            return [x]

        def f2(l):
            return l + l

        def f3(l):
            return sum(l)

        f12 = ft.compose(f2, f1)
        self.assertEqual(f12(10), [10, 10])

        f123 = ft.compose(f3, f2, f1)
        self.assertEqual(f123(10), 20)
