from vergeml.utils import dict_set_path, dict_get_path, dict_del_path, dict_has_path, dict_merge, dict_paths, parse_ai_names

def test_set_path():
    d = {}
    dict_set_path(d, "x.y.z", 1)
    assert d == {'x': {'y': {'z': 1}}}

def test_set_path_existing():
    d = {'x': {'y1': 1}}
    dict_set_path(d, "x.y.z", 1)
    assert d == {'x': {'y': {'z': 1}, 'y1': 1}}

def test_del_path():
    d = {'x': {'y': {'z': 1}}}
    dict_del_path(d, "x.y.z")
    assert d == {}

def test_del_path_non_empty():
    d = {'x': {'y': {'z': 1}, 'y1': 1}}
    dict_del_path(d, "x.y.z")
    assert d == {'x': {'y1': 1}}

def test_get_path():
    d = {'x': {'y': {'z': 1}}}
    assert dict_get_path(d, 'x.y.z') == 1

def test_get_path_top():
    d = {'x': {'y': {'z': 1}}}
    assert dict_get_path(d, 'x') == {'y': {'z': 1}}

def test_has_path():
    d = {'x': {'y': {'z': 1}}}
    assert dict_has_path(d, 'x.y.z') == True

def test_has_path_top():
    d = {'x': {'y': {'z': 1}}}
    assert dict_has_path(d, 'x') == True

def test_has_path_false():
    d = {'x': {'y': {'z': 1}}}
    assert dict_has_path(d, 'x.z.y') == False

def test_merge():
    d1 = {'device': {'id': 'gpu'}, 'something': 1}
    d2 = {'device': {'memory': '20%'}, 'else': 2}
    assert dict_merge(d1, d2) == {'device': {'id': 'gpu', 'memory': '20%'}, 'something': 1, 'else': 2}

def test_paths():
    d = {'x': {'y': {'z': 1}},
         'a': {'b': {'c1': 1, 'd1': 2}}}
    assert dict_paths(d) == ['x.y.z', 'a.b.c1', 'a.b.d1']

def test_paths2():
    d = {'x': {'y': {'z': 1}},
         'a': {'b': {'c1': 1, 'd1': 2}}}
    assert dict_paths(d, 'a') == ['a.b.c1', 'a.b.d1']


def test_parse_ai_names():
    assert parse_ai_names(["@touchy-automaton", "train"]) == (["touchy-automaton"], ["train"])
    assert parse_ai_names(["@touchy-automaton", "@evil-skynet", "run", "tensorboard"]) == \
                         (["touchy-automaton", "evil-skynet"], ["run", "tensorboard"])
    assert parse_ai_names(["train", "--epochs=20"]) == \
                         ([], ["train", "--epochs=20"])
    