# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A few things commonly used across A LOT of config files."""

import string

import ml_collections as mlc


def input_for_quicktest(config_input, quicktest):
  if quicktest:
    config_input.batch_size = 8
    config_input.shuffle_buffer_size = 10
    config_input.cache_raw = False


def parse_arg(arg, lazy=False, **spec):
  """Makes ConfigDict's get_config single-string argument more usable.

  Example use in the config file:

    import big_vision.configs.common as bvcc
    def get_config(arg):
      arg = bvcc.parse_arg(arg,
          res=(224, int),
          runlocal=False,
          schedule='short',
      )

      # ...

      config.shuffle_buffer = 250_000 if not arg.runlocal else 50

  Ways that values can be passed when launching:

    --config amazing.py:runlocal,schedule=long,res=128
    --config amazing.py:res=128
    --config amazing.py:runlocal  # A boolean needs no value for "true".
    --config amazing.py:runlocal=False  # Explicit false boolean.
    --config amazing.py:128  # The first spec entry may be passed unnamed alone.

  Uses strict bool conversion (converting 'True', 'true' to True, and 'False',
    'false', '' to False).

  Args:
    arg: the string argument that's passed to get_config.
    lazy: allow lazy parsing of arguments, which are not in spec. For these,
      the type is auto-extracted in dependence of most complex possible type.
    **spec: the name and default values of the expected options.
      If the value is a tuple, the value's first element is the default value,
      and the second element is a function called to convert the string.
      Otherwise the type is automatically extracted from the default value.

  Returns:
    ConfigDict object with extracted type-converted values.
  """
  # Normalize arg and spec layout.
  arg = arg or ''  # Normalize None to empty string
  spec = {k: get_type_with_default(v) for k, v in spec.items()}

  result = mlc.ConfigDict(type_safe=False)  # For convenient dot-access only.

  # Expand convenience-cases for a single parameter without = sign.
  if arg and ',' not in arg and '=' not in arg:
    # (think :runlocal) If it's the name of sth in the spec (or there is no
    # spec), it's that in bool.
    if arg in spec or not spec:
      arg = f'{arg}=True'
    # Otherwise, it is the value for the first entry in the spec.
    else:
      arg = f'{list(spec.keys())[0]}={arg}'
      # Yes, we rely on Py3.7 insertion order!

  # Now, expand the `arg` string into a dict of keys and values:
  raw_kv = {raw_arg.split('=')[0]:
                raw_arg.split('=', 1)[-1] if '=' in raw_arg else 'True'
            for raw_arg in arg.split(',') if raw_arg}

  # And go through the spec, using provided or default value for each:
  for name, (default, type_fn) in spec.items():
    val = raw_kv.pop(name, None)
    result[name] = type_fn(val) if val is not None else default

  if raw_kv:
    if lazy:  # Process args which are not in spec.
      for k, v in raw_kv.items():
        result[k] = autotype(v)
    else:
      raise ValueError(f'Unhandled config args remain: {raw_kv}')

  return result


def get_type_with_default(v):
  """Returns (v, string_to_v_type) with lenient bool parsing."""
  # For bool, do safe string conversion.
  if isinstance(v, bool):
    def strict_bool(x):
      assert x.lower() in {'true', 'false', ''}
      return x.lower() == 'true'
    return (v, strict_bool)
  # If already a (default, type) tuple, use that.
  if isinstance(v, (tuple, list)):
    assert len(v) == 2 and isinstance(v[1], type), (
        'List or tuple types are currently not supported because we use `,` as'
        ' dumb delimiter. Contributions (probably using ast) welcome. You can'
        ' unblock by using a string with eval(s.replace(";", ",")) or similar')
    return (v[0], v[1])
  # Otherwise, derive the type from the default value.
  return (v, type(v))


def autotype(x):
  """Auto-converts string to bool/int/float if possible."""
  assert isinstance(x, str)
  if x.lower() in {'true', 'false'}:
    return x.lower() == 'true'  # Returns as bool.
  try:
    return int(x)  # Returns as int.
  except ValueError:
    try:
      return float(x)  # Returns as float.
    except ValueError:
      return x  # Returns as str.


def pack_arg(**kw):
  """Packs key-word args as a string to be parsed by `parse_arg()`."""
  for v in kw.values():
    assert ',' not in f'{v}', f"Can't use `,` in config_arg value: {v}"
  return ','.join([f'{k}={v}' for k, v in kw.items()])


def arg(**kw):
  """Use like `add(**bvcc.arg(res=256, foo=bar), lr=0.1)` to pass config_arg."""
  return {'config_arg': pack_arg(**kw), **kw}


def _get_field_ref(config_dict, field_name):
  path = field_name.split('.')
  for field in path[:-1]:
    config_dict = getattr(config_dict, field)
  return config_dict.get_ref(path[-1])


def format_str(format_string, config):
  """Format string with reference fields from config.

  This makes it easy to build preprocess strings that contain references to
  fields tha are edited after. E.g.:

  ```
  config = mlc.ConficDict()
  config.res = (256, 256)
  config.pp = bvcc.format_str('resize({res})', config)
  ...
  # if config.res is modified (e.g. via sweeps) it will propagate to pp field:
  config.res = (512, 512)
  assert config.pp == 'resize((512, 512))'
  ```

  Args:
    format_string: string to format with references.
    config: ConfigDict to get references to format the string.

  Returns:
    A reference field which renders a string using references to config fields.
  """
  output = ''
  parts = string.Formatter().parse(format_string)
  for (literal_text, field_name, format_spec, conversion) in parts:
    assert not format_spec and not conversion
    output += literal_text
    if field_name:
      output += _get_field_ref(config, field_name).to_str()
  return output
