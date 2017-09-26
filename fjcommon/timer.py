import time
from contextlib import contextmanager


class TimeAccumulator(object):
    """
    Usage:

    t = TimeAccumulator()
    for i, x in enumerate(it):
        with t.execute():
            expensive_operation()
        if i % 50 == 0:
            print('Average: {}'.format(t.mean_time_spent()))

    """
    _EPS = 1e-8

    def __init__(self):
        self.times = []

    @contextmanager
    def execute(self):
        prev = time.time()
        yield
        self.times.append(time.time() - prev)

    def mean_time_spent(self):
        """ :returns mean time spent and resets cached times. """
        total_time_spent = sum(self.times)
        count = len(self.times)
        self.times = []
        return total_time_spent / (count + self._EPS)  # prevent div by zero errors


@contextmanager
def execute(name=''):
    start = time.time()
    yield
    duration = time.time() - start
    print('{}{:.5f}'.format(name + ': ' if name else '', duration))

