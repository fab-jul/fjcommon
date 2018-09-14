"""
config_diff configA configB [--same]

shows the differences
"""

import argparse

from fjcommon.config_parser import _Config as Config
from fjcommon.config_parser import parse_configs
from fjcommon.iterable_ext import filter_split


def diff(config_a, config_b, show_same):
    cs = sorted(_compare(config_a, config_b), key=lambda c: c.k)
    same, different = filter_split(lambda c: c.same(), cs)
    if show_same and len(same) > 0:
        print('* Same **********')
        for c in same:
            print(c)

    if len(different) > 0:
        print('* Different *****')
        for c in different:
            print(c)


class _Comparison(object):
    def __init__(self, k, value_a, value_b):
        self.k = k
        self.value_a = value_a
        self.value_b = value_b

    def same(self):
        return self.value_a == self.value_b

    def __str__(self):
        len_k = len(self.k)
        return '{} = {}\n' \
               '{} = {}'.format(self.k, self.value_a, ' ' * len_k, self.value_b)


def _compare(config_a: Config, config_b: Config):
    config_b_params = {k: v for k, v in config_b.all_params_and_values()}
    for k_a, v_a in config_a.all_params_and_values():
        if k_a in config_b_params:
            yield _Comparison(k_a, v_a, config_b_params[k_a])
        else:  # not in config b
            yield _Comparison(k_a, v_a, None)

    # find the ones in b but not in a
    config_a_params = {k: v for k, v in config_a.all_params_and_values()}
    for k_b, v_b in config_b_params.items():
        if k_b not in config_a_params:
            yield _Comparison(k_b, None, v_b)



def main():
    p = argparse.ArgumentParser()
    p.add_argument('config_a')
    p.add_argument('config_b')
    p.add_argument('--same', action='store_true')
    flags = p.parse_args()
    (c_a, c_b), _ = parse_configs(flags.config_a, flags.config_b)
    diff(c_a, c_b, flags.same)



if __name__ == '__main__':
    main()