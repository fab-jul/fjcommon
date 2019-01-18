# fjcommon

Some python helper libraries.

# no_op.py

Class that silently ignores calls to any function and acts as a no-op context manager. Useful for quickly disabling 
some part of your code without 
re-writing/commenting out many parts.

Usage

```python
from fjcommon.no_op import NoOp

foo = NoOp

foo.do_this().do_that()  # no effect

a = foo.get_me_this(18)  # returns NoOp, arguments ignored
a.more(spam=True)  # no effect, keyboard arguments ignored

with foo.enter_context() as t:  # t is NoOp
    t.run()  # no effect                                
```

# configparser.py

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