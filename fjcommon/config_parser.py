"""
Simple and Re-usable Configuration File Framework for tracking unique configuration files.

## Example

File `base.cf`:
```
constrain network_type :: LINEAR, DNN
network_type = LINEAR

lr = 1e-5
batch_size_train = 25
batch_size_val = 0.5 * batch_size_train
conv_params = {'f': 5,
 'pad': 'zeros'}
```

File `lr_sweep/lr_1e-6.cf`:
```
use ../base
lr = 1e-6
```

File `lr_sweep/lr_1e-4.cf`:
```
use ../base
lr = 1e-4
```

## Syntax

Uses configuration files with the following syntax:

###   `use` statment
The first line may contain a use statement, specifying a path to another config file, relative to the containing
file. The specified file is parsed first. If a parameter is redefined in some file, the last definition is used.

```
use <RELATIVE_PATH>
```

The following lines may contain

###  `constrain` statement
```
 constrain <PARAM_NAME> :: <CONSTRAIN_VAL_1>, <CONSTRAIN_VAL_2>, ...
```

###  `parameter` statement
```
 <PARAM_NAME> = <PARAM_VALUE>
```
where `<PARAM_VALUE>` is a python expression that can reference any previously defined parameters (see note below about this). Can also be a multi-line statement by enclosing it in round brackets
```
value = (123+
		 456)
```
or a multi-line dictionary or list definition.

###  Comments
```
 # <COMMENT>
```
is ignored.

## Note on using previously defined variables

These variables should not be treated as placeholders. Example:

File `base.cf`:
```
batch_size_train = 25
batch_size_val = 0.5 * batch_size_train
```
File `bigger_batches.cf`:
```
use base.cf
batch_size_train = 50
```
In this case, when using `bigger_batches.cf`, `batch_size_val = 0.5 * 25` still, because it simply uses the value defined
in `base.cf`.
"""

# TODO:
# - some support for per-module parameters. could be automatically generated? or unique syntax?
# - allow setting config.foo.bar = 10

import itertools
import os
import re
from os import path

from fjcommon import functools_ext as ft
from fjcommon.assertions import assert_exc


_PAT_CONSTRAIN = re.compile(r'^constrain\s+([^\s]+?)\s*::\s*(.+)$')
_PAT_PARAM = re.compile(r'^([^\s]+?)\s*=\s*(.+)$')


_SUB_SEP = os.environ.get('FJCOMMON_CONFIGP_SUBSEP', '.')


def parse_configs(*configs):
    """ Parse multiple configs """
    return ft.unzip(map(parse, configs))


def parse(config_p):
    """
    Parses a configuration file at `config_p`.
    :returns tuple (config, rel_path), where rel_path is the relative path of config to the dir of the root config file,
    where thre root config file is the one without a 'use' statement.
    """
    config, root_path = _parse(config_p)
    rel_path = path.abspath(config_p).replace(path.dirname(root_path), '').strip(path.sep)
    return config, rel_path


# ------------------------------------------------------------------------------


class _ParseError(Exception):
    pass


def _parse(config_p):
    with open(config_p, 'r') as f:
        lines = f.read().split('\n')
        if len(lines) == 0:
            raise ValueError('Invalid config file, not enough lines: {}'.format(config_p))
        if 'use' in lines[0]:  # import other config
            config_p_dir = path.dirname(config_p)
            import_config_path = lines[0].replace('use', '').strip()
            config, root_path = _parse(path.join(config_p_dir, import_config_path))
            return _update_config(config, lines[1:]), root_path
        else:
            return _update_config(_Config(), lines), path.abspath(config_p)


class _BracketPairings(object):
    """ Helper class for matching brackets """
    def __init__(self, left, right):
        self.l = left
        self.r = right
        self.num_left = 0

    def match(self, c):
        if c == self.l:
            self.num_left += 1
        elif c == self.r:
            if self.num_left == 0:
                raise SyntaxError('Found {} without left {}'.format(c, self.l))
            self.num_left -= 1

    def balanced(self):
        return self.num_left == 0


def _merge_multiline_statements(lines):
    """
    Merge multiline statements into single line statements. Checks (), [], {}. Doesn't care about wrongly mixing
    parenthesis, e.g., ([)], but this will be caught by the eval later.
    """
    cur = ''
    pairings = [_BracketPairings('(', ')'),
                _BracketPairings('[', ']'),
                _BracketPairings('{', '}')]
    for l in lines:
        cur += l.strip()
        for c in l:
            for p in pairings:
                p.match(c)

        if all(p.balanced() for p in pairings):
            yield cur
            cur = ''
        else:  # at least one is unbalanced
            continue

    for p in pairings:
        if not p.balanced():
            raise SyntaxError('Missing {} in expression!'.format(p.r))


def _update_config(config, lines):
    for line in _merge_multiline_statements(lines):
        if not line or line.startswith('#'):
            continue

        constrain_match = _PAT_CONSTRAIN.match(line)
        if constrain_match:
            constrain_name, constrain_vals = constrain_match.group(1, 2)
            constrain_vals = [val.strip() for val in constrain_vals.split(',')]
            config.add_constraint(constrain_name, constrain_vals)
            continue

        param_match = _PAT_PARAM.match(line)
        if not param_match:
            raise ValueError('*** Invalid line: `{}`'.format(line))
        var_name, var_value = param_match.group(1, 2)
        # construct a dict with all attributes of the config plus all constraints. adding the constraints allows
        # us to write param = CONSTRAINT instead of param = 'CONSTRAINT'
        globals_dict = dict(config.__dict__, **{val: val for val in config.all_constraint_values()})
        try:
            var_value = eval(var_value, globals_dict)  # pass current config as globals dict
        except SyntaxError:
            raise SyntaxError('Cannot parse line: {}'.format(line))
        config.assert_fullfills_constraint(var_name, var_value)
        try:
            config.set_attr(var_name, var_value)
        except _ParseError as e:
            raise SyntaxError('Cannot parse line: {} ({})'.format(line, e))
    return config


class ConstraintViolationException(Exception):
    pass


class _Config(object):  # placeholder object filled with setattr
    def __init__(self):
        self._constraints = {}

    def add_constraint(self, var_name, allowed_var_values):
        if var_name in self._constraints:
            raise ValueError('Already have constraint for {}, not overwriting!'.format(var_name))
        self._constraints[var_name] = allowed_var_values

    def all_constraint_values(self):
        return set(itertools.chain.from_iterable(self._constraints.values()))

    def assert_fullfills_constraint(self, var_name, var_value):
        if var_name not in self._constraints:
            return
        allowed_var_values = self._constraints[var_name]
        if var_value not in allowed_var_values:
            raise ConstraintViolationException('{} does not fullfill constraint {} :: {}'.format(
                var_value, var_name, allowed_var_values))

    def all_params_and_values(self):
        return ((k, v) for k, v in sorted(self.__dict__.items())
                if self.is_valid_key(k))

    @staticmethod
    def is_valid_key(k):
        return re.match(r'[A-Za-z]+', k) is not None

    def __str__(self):
        def _lines():
            for k, v in self.all_params_and_values():
                yield '{} = {}'.format(k, v)
        return '\n'.join(_lines())

    def __getitem__(self, items):
        """
        Allows usage as
        lr, L, num_layers = config['lr', 'L', 'num_layer']
        """
        if not isinstance(items, tuple):
            items = [items]
        for item in items:
            assert_exc(item in self.__dict__, 'Invalid parameter: {}'.format(item), AttributeError)
            yield self.__dict__[item]

    # TODO: overwrite __setitem__ with tuple support

    def set_attr(self, k, v):
        if not self.is_valid_key(k):
            raise _ParseError('Invalid key: {}'.format(k))
        setattr(self, k, v)

    def __setattr__(self, key, value):
        super(_Config, self).__setattr__(key, value)

    def __getattr__(self, item):
        """
        TODO: this is WIP. see top of file
        This function returns config[item] if it exists.
        if it doesn't exist, it checks for module parameters, and returnes a filtered config. Example:
        some_config:
            lr = 1e-4
            ae.x = 1
            ae.y = 2
        some_config.ae -> {x: 1, y: 2}
        """
        if item in self.__dict__:
            return self.__dict__[item]
        prefix = item + _SUB_SEP
        filtered_dict = {k.replace(prefix, '', 1): v for k, v in self.__dict__.items() if k.startswith(prefix)}
        if len(filtered_dict) == 0:
            raise AttributeError('{} has no attribute {} and no attributes starting with {}'.format(
                    self.__class__.__name__, item, prefix))
        # create filtered config
        return _Config()._setattr_from_dict(filtered_dict)

    def _setattr_from_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        return self


# Tests ------------------------------------------------------------------------


def test_config(tmpdir):
    import pytest
    spec = '\n'.join([
        "lr = 1e-4",
        "L = 7",
        "c = L // 2",
        "ae.c = 123",
        "ae.ae.d = 124",
        "test = {'a': 123,",
        "        'b': 222}"
        ])

    print(spec)
    p = str(tmpdir.join('config.cf'))
    with open(p, 'w') as f:
        f.write(spec)

    config, _ = parse(p)

    assert config.lr == 1e-4
    assert config.L == 7
    assert config.c == 3
    assert config.test == {'a': 123, 'b': 222}

    config_ae = config.ae
    assert config_ae.c == 123
    assert config_ae.ae.d == 124

    print(config.ae)

    # error
    with pytest.raises(SyntaxError):
        _update_config(_Config(), ['_illegal = 5'])
    # assert does not raise
    _update_config(_Config(), ['i = 5'])

    # update values
    config.c = 10
    assert config.c == 10
    config.__dict__['ae.c'] = 12
    assert config.ae.c == 12


def test_merger():
    for i, o in (
            (['a', 'b'], ['a', 'b']),
            (['(a', 'b)'], ['(ab)']),
            (['(a = {c )', 'b}'], ['(a = {c )b}']),  # invalid python but valid paren
    ):
        assert list(_merge_multiline_statements(i)) == o
    import pytest
    with pytest.raises(SyntaxError):
        list(_merge_multiline_statements(['(a))']))
    with pytest.raises(SyntaxError):
        list(_merge_multiline_statements(['(a']))
