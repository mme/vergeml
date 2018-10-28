import re
import yaml.reader
import yaml.scanner
import yaml
from typing import Union
from copy import deepcopy
from vergeml.utils import VergeMLError, did_you_mean
from vergeml.utils import dict_set_path, dict_has_path, dict_get_path, dict_del_path, dict_merge, dict_paths
from vergeml.option import option, Option
from vergeml.plugins import PLUGINS
from vergeml.io import Source
from vergeml.operation import Operation


def _invalid_option(key, help_topic=None, suggestion=None, kind='value'):
    label = "Invalid value for option" if kind == 'value' else "Invalid option"
    return VergeMLError(f"{label} '{key}'.", suggestion, help_topic=help_topic, hint_type=kind, hint_key=key)

def _normalize(raw, validators):
    raw = deepcopy(raw)
    res = {}

    for _, conf in validators.items():
        options = Option.discover(conf)
        aliases = [opt for opt in options if opt.alias]
        for alias in aliases:
            if dict_has_path(raw, alias.name):
                v = dict_get_path(raw, alias.name)
                if not alias.type or isinstance(v, alias.type):
                    dict_set_path(res, alias.alias, v)
                    dict_del_path(raw, alias.name)

    for k, v in deepcopy(raw).items():
        if "." in k:
            dict_set_path(res, k, v)
            del raw[k]

    return dict_merge(res, raw)


def apply_config(raw, validators):
    raw = _normalize(raw, validators)

    # find invalid options
    invalid = deepcopy(raw)

    for k, config in validators.items():
        options = [opt for opt in config.options() if opt.alias is None]

        for option in options:
            if dict_has_path(invalid, option.name):
                dict_del_path(invalid, option.name)

        if k in invalid or (not k and invalid):
            first = dict_paths(invalid, k)[0]
            candidates = [opt.name for opt in options]
            suggestion = did_you_mean(candidates, first)
            raise _invalid_option(first, help_topic=config.name, suggestion=suggestion, kind='key')
    
        # validate and cast options
        for option in options:
            if dict_has_path(raw, option.name):
                value = dict_get_path(raw, option.name)
                dict_del_path(raw, option.name)
                config.configure(option, value)
    
    return raw

        
class Validate:

    def __init__(self, name=None, plugins=PLUGINS):
        self._values = {}
        self._options = Option.discover(self, plugins=plugins)
        self.plugins = plugins
        self.name = name
        for option in self._options:
            if option.alias is None:
                dict_set_path(self._values, option.name, option.default)
    
    def configure(self, option, value):
        try:
            value = option.cast_value(value)
            value = option.transform_value(value)
            option.validate_value(value)

            if isinstance(value, dict):
                if dict_has_path(self._values, option.name):
                    dict_merge(dict_get_path(self._values, option.name), value)
            else:
                dict_set_path(self._values, option.name, value)
        except VergeMLError as err:
            # set help topic
            err.help_topic = self.name 
            raise err

    def options(self):
        return self._options

    @property
    def values(self):
        return self._values


@option('data.input', type=dict, default={'type': None}, yaml_only=True)
@option('data.output', type=dict, default={'type': None}, yaml_only=True)
@option('data.preprocess', type=list, default=[], yaml_only=True)
@option('data.cache', type=str, validate=('none', 'mem', 'disk', 'mem-in', 'disk-in', '*auto*'), default='*auto*')

# mem = mem-in
# disk = disk-in
# mem-out = mem
# disk-out = disk
# default = *auto*

class ValidateData(Validate):
    
    def configure(self, option, value):
        if option.name == 'data.input' and 'type' in value:
            self._validate_source(value, 'input')
        elif option.name == 'data.output' and 'type' in value:
            self._validate_source(value, 'output')
        elif option.name == 'data.preprocess':
            self._validate_preprocess(value)

        super().configure(option, value)
    
    
    def _validate_source(self, config, source_name):
        name = config['type']
        if not name:
            return
        plugin = self.plugins.get("vergeml.io", name)
        if not plugin:
            raise _invalid_option(f"data.{source_name}.type", 
                                  help_topic='data',
                                  suggestion=did_you_mean(self.plugins.keys('vergeml.io'), name))
        
        source = Source.discover(plugin)
        options = list(filter(lambda o: o.name != 'type', source.options))
        validator = ValidateOptions(options, source_name, self.plugins)
        config = {source_name: deepcopy(config)}
        dict_del_path(config, source_name + ".type")
        rest = apply_config(config, {name: validator})
        if rest:
            k = dict_paths(rest)[0]
            candidates = [opt.name for opt in source.options]
            raise _invalid_option(f"data.{source_name}.{k}",
                                    help_topic=name,
                                    suggestion=did_you_mean(candidates, k),
                                    kind='key')
        else:
            values = dict(data=validator.values)
            if source_name not in values['data']:
                values['data'][source_name] = {}
            values['data'][source_name]['type'] = name
            dict_merge(self.values, values)
    
    def _validate_preprocess(self, value):
        operations = []
        for ix, config in enumerate(value):
            if not isinstance(config, dict):
                raise VergeMLError(f"Invalid entry in preprocess - must be key value pairs.",
                                    "Please fix the entry in the project file.", 
                                    help_topic="preprocess", 
                                    hint_type='key', 
                                    hint_key='data.preprocess.' + str(ix))
            elif not 'op' in config:
                raise VergeMLError(f"Invalid entry in preprocess - missing 'op' key.",
                                    "Please fix the entry in the project file.", 
                                    help_topic="preprocess", 
                                    hint_type='key', 
                                    hint_key='data.preprocess.' + str(ix))
            op_name = config['op']
            plugin = self.plugins.get("vergeml.operation", op_name)
            if not plugin:
                raise VergeMLError(f"Invalid entry in preprocess - unknown operation '{op_name}'.",
                                    "Please fix the entry in the project file.", 
                                    help_topic="preprocess", 
                                    hint_type='value', 
                                    hint_key="data.preprocess.{ix}.op")

            op = Operation.discover(plugin)
            options = list(filter(lambda o: o.name != 'op', op.options))
            validator = ValidatePreprocess(options, op_name, self.plugins)
            config = deepcopy(config)
            del config['op']
            try:
                apply_config(config, {None: validator})
            except VergeMLError as err:
                err.hint_key = "data.preprocess.{ix}." + err.hint_key
                raise err
            validator.values['op'] = op_name
            operations.append(validator.values)
        dict_merge(self.values, dict(data=dict(preprocess=operations)))
         

def _validate_device_id(option, value):
    if not re.match(r"^(gpu:[0-9]+|gpu|cpu|auto)", value):
        raise _invalid_option(option.name, 'device')

def _validate_device_memory(option, value):
    if value == 'auto':
        return value
    if not re.match(r'^[0-9]+(\.[0-9]*)?%$', value):
        raise _invalid_option(option.name, 'device')
    value = float(value.rstrip('%'))
    if value < 0. or value > 100.:
        raise _invalid_option(option.name, 'device')


@option('device', type=str, alias='device.id')
@option('device.id', validate=_validate_device_id, type=str, default='auto', yaml_only=True)
@option('device.memory', validate=_validate_device_memory, type=str, default='auto')
@option('device.grow-memory', type=bool, default=False, yaml_only=True)
class ValidateDevice(Validate):

    def configure(self, option, value):
        super().configure(option, value)
        if option.name == 'device.id' and self._values['device']['id'] == 'gpu':
            self._values['device']['id'] = 'gpu:0'


class ValidateOptions(Validate):

    def __init__(self, options, name=None, plugins=PLUGINS):
        options = deepcopy(options)
        for opt in options:
            opt.name = name + "." + opt.name
            opt.plugins = plugins
        # CHEAT
        from vergeml.option import _OPTIONS_META_KEY
        setattr(self, _OPTIONS_META_KEY, options)
        super().__init__(name, plugins)


class ValidatePreprocess(Validate):

    def __init__(self, options, name=None, plugins=PLUGINS):
        options = deepcopy(options)
        for opt in options:
            opt.plugins = plugins
        from vergeml.option import _OPTIONS_META_KEY
        setattr(self, _OPTIONS_META_KEY, options)
        super().__init__(name, plugins)
 

def display_err_in_file(filename, line, column, message, length=1, nlines=3):
    with open(filename, "r") as fp:
        return _display_err(filename, line, column, message, length, nlines, fp.read())


def _display_err(filename, line, column, message, length, nlines, content):
    lines = content.splitlines()
    start = max(0, line+1-nlines)
    res = [f"File {filename}, line {line+1}:{column+1}"]
    res.append(str('-' * (len(res[0]) + 7)))
    res += lines[start:line+1]
    res += [(' ' * column) + ("^" * length), message]
    return "\n".join(res)
        
def load_yaml_file(filename, label='YAML file', loader=yaml.Loader):
    try:
        with open(filename, "r") as fp:
            res = yaml.load(fp.read(), Loader=loader) or {}
            if not isinstance(res, dict):
                raise VergeMLError(f"Invalid {label}: {filename}",
                                   f"Please ensure that the top level of the {label} consists of key value pairs.")
            return res
    except yaml.YAMLError as err:
        if hasattr(err, 'problem_mark'):
            mark = getattr(err, 'problem_mark')
            problem = getattr(err, 'problem')
            message = f"Could not read {label} {filename}:"
            message += "\n" + display_err_in_file(filename, mark.line, mark.column, problem)
        elif hasattr(err, 'problem'):
            problem = getattr(err, 'problem')
            message = f"Could not read {label} {filename}: {problem}"
        else:
            message = f"Could not read {label} {filename}: YAML Error"
        
        suggestion = f"There is a syntax error in your {label} - please fix it and try again."

        raise VergeMLError(message, suggestion)

    except OSError as err:
        raise VergeMLError(f"Could not open {label} {filename}: {err.strerror}",
                            "Please ensure the file exists and you have the required access priviledges.")


class _YAMLAnalyzer(yaml.reader.Reader, yaml.scanner.Scanner):
 
    def __init__(self, stream):
        yaml.reader.Reader.__init__(self, stream)
        yaml.scanner.Scanner.__init__(self)

def _get_location(an, key, kind):
    mark = an.get_mark()

    # try to mark the value
    if kind == 'value':
        if isinstance(an.peek_token(), yaml.ValueToken):
            # forward to the next token
            tk = an.get_token()
            if isinstance(an.peek_token(), yaml.ScalarToken):
                # forward again
                tk = an.get_token()
                length = len(tk.value)
                return (mark.line, mark.column + 1, length)
    
    length = len(key) + 1
    return (mark.line, max(0, mark.column - length), length)

def yaml_find_definition(stream, key, kind='key'):
    assert kind in ('key', 'value')
    keys = list(map(lambda k: int(k) if k.isdigit() else k, key.split(".")))
    level = -1
    matches = [False] * len(keys)
    indices = []

    an = _YAMLAnalyzer(stream)

    tk = an.get_token()
    while tk:
        if isinstance(tk, (yaml.BlockMappingStartToken, yaml.BlockSequenceStartToken)):
            level += 1
            indices.append(0)
        elif isinstance(tk, yaml.ValueToken) and isinstance(an.peek_token(), yaml.BlockEntryToken):
            level += 1
            indices.append(0)
            # this is a special case since no start and end tokens are emitted for a simple list
            tk = an.get_token()
            while tk:
                if isinstance(tk, yaml.BlockEntryToken):
                    if 0 <= level < len(keys) and isinstance(keys[level], int):
                        if keys[level] == indices[level] and all(matches[:level]):
                            matches[level] = True
                            if all(matches):
                                return _get_location(an, str(keys[-1]), kind)
                    indices[level] += 1
                elif not isinstance(tk, (yaml.ScalarToken, yaml.ValueToken)):
                    break

        elif isinstance(tk, yaml.BlockEndToken):
            level -= 1
            indices.pop()
        elif isinstance(tk, yaml.KeyToken) and 0 <= level < len(keys) and isinstance(keys[level], str):
            name = an.get_token().value
            
            if name == keys[level] and all(matches[:level]):
                matches[level] = True
                if all(matches):
                    return _get_location(an, keys[-1], kind)
        
        elif isinstance(tk, yaml.BlockEntryToken):
            
            if 0 <= level < len(keys) and isinstance(keys[level], int):
                if keys[level] == indices[level] and all(matches[:level]):
                    matches[level] = True
                    if all(matches):
                        return _get_location(an, keys[-1], kind)

            indices[level] += 1
        
        tk = an.get_token()
    return None
