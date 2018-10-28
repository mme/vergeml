from vergeml import VergeMLError
from vergeml.validate import ValidateDevice, ValidateData, apply_config, yaml_find_definition, _display_err, _normalize
from vergeml.plugins import _DictPluginManager
from vergeml.sources.image import ImageSource
from vergeml.operations.augment import AugmentOperation
import pytest

def test_normalize():
    VALIDATORS = {'device': ValidateDevice()}
    assert _normalize({
        'device.id': 'gpu',
        'device.memory': '20%',
        'some.thing.else': 1,
        'this': {
             'is': 'normal'
        }
    } , VALIDATORS) == {
        'device': {
            'id': 'gpu',
            'memory': '20%'
        },
        'some': {
            'thing': {
                'else': 1
            }
        },
        'this': {
             'is': 'normal'
        }
    }

def test_normalize_aliases():
    VALIDATORS = {'device': ValidateDevice()}
    assert _normalize({
        'device': 'gpu:1',
        'device.memory': '20%'
    } , VALIDATORS) == {
        'device': {
            'id': 'gpu:1',
            'memory': '20%'
        },
    }

def test_apply_empty_config():
    VALIDATORS = {'device': ValidateDevice()}
    assert apply_config({}, VALIDATORS) == {}
    assert VALIDATORS['device'].values == {
        'device': {
            'id': 'auto',
            'memory': 'auto',
            'grow-memory': False
        }
    }

def test_apply_config():
    VALIDATORS = {'device': ValidateDevice()}
    assert apply_config({'device': 'gpu', 'model': 'inception-v3'}, VALIDATORS) == {'model': 'inception-v3'}
    assert VALIDATORS['device'].values == {
        'device': {
            'id': 'gpu:0',
            'memory': 'auto',
            'grow-memory': False
        }
    }


def test_input_output():
    PLUGINS = _DictPluginManager()
    PLUGINS.set('vergeml.io', 'image', ImageSource)
    VALIDATORS = {'data': ValidateData('image', plugins=PLUGINS)}
    apply_config({
        'data': {
            'input': {
                'type': 'image'
            },
            'output': {
                'type': 'image'
            }
        }
    }, validators=VALIDATORS)
    assert VALIDATORS['data'].values['data']['input']['type'] == 'image'
    assert VALIDATORS['data'].values['data']['output']['type'] == 'image'


def test_validate_preprocess():
    PLUGINS = _DictPluginManager()
    PLUGINS.set('vergeml.operation', 'augment', AugmentOperation)
    VALIDATORS = {'data': ValidateData(plugins=PLUGINS)}
    apply_config({
        'data': {
            'preprocess': [
                {'op': 'augment',
                 'variants': 4}
            ]
        }
    }, VALIDATORS) 
    assert VALIDATORS['data'].values == {
        'data': {
            'cache': '*auto*',
            'input': {
                'type': None
            },
            'output': {
                'type': None
            },
            'preprocess': [
                {'op': 'augment',
                 'variants': 4}
            ]
        }
    }


def test_validate_preprocess_invalid():
    PLUGINS = _DictPluginManager()
    PLUGINS.set('vergeml.operation', 'augment', AugmentOperation)
    VALIDATORS = {'data': ValidateData(plugins=PLUGINS)}
    with pytest.raises(VergeMLError, match=r".*Did you mean 'variants'.*"):
        apply_config({
            'data': {
                'preprocess': [
                    {'op': 'augment',
                    'variantz': 4}
                ]
            }
        }, VALIDATORS)

def test_config_dict():
    VALIDATORS = {'device': ValidateDevice()}
    res = apply_config({'device': {'id': 'cpu'}}, VALIDATORS)
    assert(res == {})
    assert(VALIDATORS['device'].values['device']['id'] == 'cpu')


def test_config_invalid():
    VALIDATORS = {'device': ValidateDevice()}
    with pytest.raises(VergeMLError):
        apply_config({'device': {'id': 'cpu', 'invalid': 'true'}}, VALIDATORS)


TEST_YAML = """\
data:
    input:
        type: imagez
        
    preprocess:
        - op: center-crop
          width: 30
          height: 30

        - op: flip-horizontalz

        - op: rgb
"""


def test_find_definition_key():
    res = yaml_find_definition(TEST_YAML, 'data.input.type', 'key')
    assert res == (2, 8, 5)


def test_find_definition_val():
    res = yaml_find_definition(TEST_YAML, 'data.input.type', 'value')
    assert res == (2, 14, 6)


def test_find_definition_arr_key():
    res = yaml_find_definition(TEST_YAML, 'data.preprocess.1.op', 'key')
    assert res == (9, 10, 3)


def test_find_definition_arr_val():
    res = yaml_find_definition(TEST_YAML, 'data.preprocess.1.op', 'value')
    assert res == (9, 14, 16)


def test_display_err():
    line, column, length = yaml_find_definition(TEST_YAML, 'data.preprocess.1.op', 'value')
    res = _display_err("vergeml.yaml", 
                        line, 
                        column, 
                        "Invalid preprocessing operation 'flip-horizontalz'. Did you mean 'flip-horizontal'?", 
                        length, 
                        3, 
                        TEST_YAML)
    res = "Error! " + res
    
    assert res == """\
Error! File vergeml.yaml, line 10:15
------------------------------------
          height: 30

        - op: flip-horizontalz
              ^^^^^^^^^^^^^^^^
Invalid preprocessing operation 'flip-horizontalz'. Did you mean 'flip-horizontal'?"""


def test_apply_config_image():
    PLUGINS = _DictPluginManager()
    PLUGINS.set('vergeml.io', 'image', ImageSource)
    VALIDATORS = {'data': ValidateData(plugins=PLUGINS)}
    assert apply_config({'data': {'input': {'type': 'image', 'input-patterns': '*.jpg'}}}, VALIDATORS) == {}
    assert VALIDATORS['data'].values == {
        'data': {
            'input': {
                'type': 'image',
                'input-patterns': '*.jpg'
            },
            'output': {
                'type': None
            },
            'cache': '*auto*',
            'preprocess': []
        }
    }



def test_apply_config_image_invalid():
    PLUGINS = _DictPluginManager()
    PLUGINS.set('vergeml.io', 'image', ImageSource)
    VALIDATORS = {'data': ValidateData(plugins=PLUGINS)}
    with pytest.raises(VergeMLError):
        assert apply_config({'data': {'input': {'type': 'image', 'input-patternz': '*.jpg'}}}, VALIDATORS) == {}
