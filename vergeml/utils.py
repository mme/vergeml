import inspect
import tempfile
import os.path
import urllib.request
import textwrap
from collections import namedtuple
import random
import numpy as np
import uuid
from typing import List
import yaml
import re
import os
import sys

SPLITS = ('train', 'val', 'test')


class VergeMLError(Exception):

    def __init__(self, message, suggestion=None, help_topic=None, hint_type=None, hint_key=None):
        super().__init__(message)
        self.suggestion = suggestion
        self.message = message
        self.hint_type = hint_type
        self.hint_key = hint_key
        self.help_topic = help_topic

    def __str__(self):
        if self.suggestion:
            if len(self.message + self.suggestion) < 80:
                return self.message + " " + self.suggestion
            else:
                return self.message + "\n" + self.suggestion
        else:
            return self.message


def wrap_text(text):
    # TODO check terminal width
    res = []
    for para in text.split("\n\n"):
        if para.splitlines()[0].strip().endswith(":"):
            res.append(para)
        else:
            res.append(textwrap.fill(para, drop_whitespace=True, fix_sentence_endings=True))
    return "\n\n".join(res)


def print_text(text):
    print(wrap_text(text))


_Intro = namedtuple('_Intro', ['args', 'defaults', 'types'])


def introspect(call):
    spec = inspect.getfullargspec(call)
    args = spec.args
    defaults = dict(zip(reversed(spec.args), reversed(spec.defaults or [])))
    types = spec.annotations
    return _Intro(args, defaults, types)


# taken from here: https://www.python-course.eu/levenshtein_distance.php
def _iterative_levenshtein(s, t):
    """
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution

    return dist[row][col]

def did_you_mean(candidates, value, fmt="'{}'"):
    candidates = list(candidates)
    names = list(sorted(map(lambda n: (_iterative_levenshtein(value, n), n), candidates)))
    names = list(filter(lambda dn: dn[0] <= 2, names))
    return 'Did you mean ' + fmt.format(names[0][1]) + '?' if names else None


def dict_set_path(dic, path, value):
    cur = dic
    path = path.split(".")
    for key in path[:-1]:
        cur = cur.setdefault( key, {} )
    cur[path[-1]] = value

def dict_del_path(d, path):
    if isinstance(path, str):
        path = path.split(".")
    if len(path) == 1:
        del[d[path[0]]]
    else:
        p, *rest = path
        dict_del_path(d[p], rest)
        if not d[p]:
            del d[p]

def dict_has_path(d, path):
    c = d
    for p in path.split("."):
        if isinstance(c, dict) and p in c:
            c = c[p]
        else:
            return False
    return True

_DEFAULT = object()
def dict_get_path(d, path, default=_DEFAULT):
    c = d
    for p in path.split("."):
        if isinstance(c, dict) and p in c:
            c = c[p]
        elif default != _DEFAULT:
            return default
        else:
            raise KeyError(path)
    return c

def dict_merge(dict1, dict2):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = dict_merge(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1

def dict_paths(d, path=None):
    res = []
    if path:
        if not dict_has_path(d, path):
            return res
        value = dict_get_path(d, path)
    else:
        value = d
    if not isinstance(d, dict):
        return res
    def _collect_path(d, path):
        for k, v in d.items():
            npath = f"{path}.{k}" if path is not None else k
            if isinstance(v, dict):
                _collect_path(v, npath)
            else:
                res.append(npath)
    _collect_path(value, path)
    return res


def parse_trained_models(argv):
    names = []
    for part in argv:
        if re.match("^@[a-zA-Z0-9_-]+$", part):
            names.append(part[1:])
        else:
            break
    rest = argv[len(names):]
    return names, rest

def parse_split(value):
    """Decodes the split value.

    Returns a tuple (type, value) where type is either perc, num or dir set.
    """
    assert isinstance(value, (int, str))

    if isinstance(value, int):
        return ('num', value)
    elif value.endswith("%"):
        return ('perc', float(value.rstrip("%").strip()))
    elif value.isdigit():
        return ('num', int(value))
    else:
        return ('dir', value)

def format_info_text(text, indent=0, width=70):
    text = text.strip("\n")
    res = []
    for line in text.splitlines():
        if line.startswith("  "):
            res.append(line)
        elif line.strip() == "":
            res.append(line)
        else:
            res.extend(textwrap.wrap(line, width=width-indent))
    if indent:
        indstr = str(' ' * indent)
        res = list(map(lambda l: indstr + l, res))
    return "\n".join(res)
