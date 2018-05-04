from unittest import TestCase
from . import functools_ext as ft

import io
import sys
from contextlib import contextmanager



class TestMoreFunctools(TestCase):
    @contextmanager
    def assert_output(self, expected):
        capturedOutput = io.StringIO()
        stdout = sys.stdout
        sys.stdout = capturedOutput
        yield
        sys.stdout = stdout
        self.assertEqual(capturedOutput.getvalue(), expected)

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

    def test_assert_post_cond(self):
        @ft.assert_post_cond(lambda a: a > 0)
        def foo(bar):
            return max(bar, 0)

        try:
            foo(5)
        except AssertionError:
            self.fail('Unexpected assertion')


        @ft.assert_post_cond(lambda a_b: a_b[0] is not None)
        def faulty():
            return None, 'ups'

        with self.assertRaises(AssertionError):
            faulty()

    def test_print_generator(self):
        @ft.print_generator('---')
        def foo(els):
            for el in els:
                yield 'foo'
                yield el
                yield None

        with self.assert_output('foo\na\n'):
            foo(['a'])

        with self.assert_output('foo\na\n---\nfoo\nb\n'):
            foo(['a', 'b'])
