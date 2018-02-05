"""
Simple and Re-usable Configuration File Framework for tracking unique machine experiment setups.

Uses configuration files with the following syntax:

The first line may contain a use statement, specifying a path to another config file, relative to the containing
file. The specified file is parsed first. If a parameter is redefined in some file, the last definition is used
-   use statment
        use <RELATIVE_PATH>

The following lines may contain

-   constrain statement
        constrain <PARAM_NAME> :: <CONSTRAIN_VAL_1>, <CONSTRAIN_VAL_2>, ...

-   parameter statement
        <PARAM_NAME> = <PARAM_VALUE>

-   comment
        # <COMMENT>

Example:

base:
    constrain network_type :: LINEAR, DNN
    network_type = LINEAR

    lr = 1e-5
    batch_size_train = 25
    batch_size_val = 0.5 * batch_size_train

lr_sweep/lr_1e-6:
    use ../base
    lr = 1e-6

lr_sweep/lr_1e-4:
    use ../base
    lr = 1e-4

"""

from os import path
import os
import sys
import re
import json
import itertools


_PAT_CONSTRAIN = re.compile(r'^constrain\s+([^\s]+?)\s*::\s*(.+)$')
_PAT_PARAM = re.compile(r'^([^\s]+?)\s*=\s*(.+)$')


def parse(config_p):
    """
    Parses a configuration file at `config_p`.
    :returns tuple (config, rel_path), where rel_path is the relative path of config to the dir of the root config file,
    where thre root config file is the one without a 'use' statement.
    """
    config, root_path = _parse(config_p)
    rel_path = path.abspath(config_p).replace(path.dirname(root_path), '').strip(path.sep)
    return config, rel_path


def _gen_grid_search_configs(grid_spec, base_config_p, outdir):
    """
    Generates each possible combination of parameters
    :param grid_spec: Path to a JSON or JSON string containing a dictionary mapping parameter names to values. Example:
        {"lr": [1e-06, 1e-07], "normalization": ["OFF", "FIXED"]}
    :param base_config_p: config to import into each grid spec file
    :param outdir: where to save the configs
    """
    base_config_dir, base_config_name = path.split(base_config_p)
    outdir_to_base_dir = path.relpath(base_config_dir, outdir)  # to get nice paths like ../base
    os.makedirs(outdir, exist_ok=True)

    grid_spec = _parse_grid_spec(grid_spec)
    grid_spec_its = [
        [(param, val) for val in grid_spec[param]]
        for param in sorted(grid_spec.keys())]

    for config in itertools.product(*grid_spec_its):
        unique_name = '_'.join(param[0] + str(val) for param, val in config)
        p = path.join(outdir, unique_name)
        print(p)
        with open(p, 'w+') as f:
            f.write('use {}\n'.format(path.join(outdir_to_base_dir, base_config_name)))
            f.write('\n'.join('{} = {}'.format(param, val) for param, val in config))
            f.write('\n')


def _parse_grid_spec(grid_spec_p):
    if path.exists(grid_spec_p):
        return json.load(open(grid_spec_p, 'r'))
    try:
        return json.loads(grid_spec_p)
    except json.JSONDecodeError as e:
        print('Neither valid path, nor valid JSON: {}\n{}'.format(grid_spec_p, e))
        sys.exit(1)


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


def _update_config(config, lines):
    for line in lines:
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
        setattr(config, var_name, var_value)
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
                if re.match(r'[A-Za-z][A-Za-z_]+', k))

    def __str__(self):
        def _lines():
            for k, v in self.all_params_and_values():
                yield '{} = {}'.format(k, v)
        return '\n'.join(_lines())


def compare(c1, c2):
    print('Comparing {}, {}'.format(os.path.basename(c1), os.path.basename(c2)))

    c1, _ = parse(c1)
    c2, _ = parse(c2)

    for k, v1 in c1.all_params_and_values():
        v2 = c2.__dict__[k]
        if v2 != v1:
            print('{}: {} != {}'.format(k, v1, v2))


if __name__ == '__main__':
    # compare(sys.argv[1], sys.argv[2])
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('grid_spec', type=str, help='Path or inline JSON')
    parser.add_argument('base_config_p', type=str)
    parser.add_argument('outdir', type=str)
    flags = parser.parse_args()
    _gen_grid_search_configs(flags.grid_spec, flags.base_config_p, flags.outdir)
