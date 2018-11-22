from vergeml.views import ListView, IteratorView
from vergeml.loader import LiveLoader
from vergeml.io import SourcePlugin, source, Sample
import random
import itertools

def test_listview_default():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'train')
    assert list(map(lambda tp: tp[0], listview)) == list(range(100))

def test_listview_neg():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'train')
    assert listview[-1][0] == 99


def test_listview_slice1():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'val')
    assert list(map(lambda tp: tp[0], listview[-2:])) == [8, 9]

def test_listview_slice2():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'val')
    assert list(map(lambda tp: tp[0], listview[-2:-1])) == [8]

def test_listview_slice3():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'val')
    assert list(map(lambda tp: tp[0], listview[:2])) == [0, 1]

def test_listview_slice4():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'val')
    assert list(map(lambda tp: tp[0], listview[1:3])) == [1, 2]

def test_listview_random():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'train', randomize=True, fetch_size=1)
    assert list(map(lambda tp: tp[0], listview[:10])) == [92, 1, 43, 61, 35, 73, 48, 18, 98, 36]

def test_listview_random2():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'train', randomize=True, fetch_size=1)
    listview2 = ListView(loader, 'train', randomize=True, random_seed=2601, fetch_size=1)
    assert list(map(lambda tp: tp[0], listview2[:10])) \
        != list(map(lambda tp: tp[0], listview[:10]))

def test_listview_random_fetch_size():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'train', randomize=True, fetch_size=10)
    assert list(map(lambda tp: tp[0], listview[:10])) == list(range(70, 80))

def test_listview_transform():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'train', transform_x=lambda x: x + 10)
    assert list(map(lambda tp: tp[0], listview)) == list(range(10, 110))

def test_listview_meta():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'train', with_meta=True)
    assert listview[0] == (0, 5, dict(meta=0))
    assert listview[1] == (1, 6, dict(meta=1))
    assert listview[2] == (2, 7, dict(meta=2))

def test_listview_transform_y():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'train', transform_y=lambda _: 'transformed_y')
    assert listview[0][1] == 'transformed_y'

def test_listview_val():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'val')
    assert list(map(lambda tp: tp[0], listview)) == list(range(10))

def test_listview_test():
    loader = LiveLoader('.cache', SourceTest())
    listview = ListView(loader, 'test')
    assert list(map(lambda tp: tp[0], listview)) == list(range(20))

# IteratorView

def test_iterview_default():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'train')
    assert list(map(lambda tp: tp[0], iterview)) == list(range(100))

def test_iterview_infinite():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'train', infinite=True)
    assert list(map(lambda tp: tp[0], itertools.islice(iterview, 150))) == list(range(100)) + list(range(50))


def test_iterview_random():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'train', randomize=True, fetch_size=1)
    assert list(map(lambda tp: tp[0], itertools.islice(iterview, 10))) == [92, 1, 43, 61, 35, 73, 48, 18, 98, 36]

def test_iterview_random_fetch_size():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'train', randomize=True, fetch_size=10)
    assert list(map(lambda tp: tp[0], itertools.islice(iterview, 10))) == list(range(70, 80))

def test_iterview_transform():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'train', transform_x=lambda x: x + 10)
    assert list(map(lambda tp: tp[0], iterview)) == list(range(10, 110))

def test_iterview_meta():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'train', with_meta=True)
    assert next(iterview) == (0, 5, dict(meta=0))
    assert next(iterview) == (1, 6, dict(meta=1))
    assert next(iterview) == (2, 7, dict(meta=2))

def test_iterview_transform_y():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'train', transform_y=lambda _: 'transformed_y')
    assert next(iterview)[1] == 'transformed_y'

def test_iterview_val():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'val')
    assert list(map(lambda tp: tp[0], iterview)) == list(range(10))

def test_iterview_test():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'test')
    assert list(map(lambda tp: tp[0], iterview)) == list(range(20))

def test_iterview_random2():
    loader = LiveLoader('.cache', SourceTest())
    iterview = IteratorView(loader, 'train', randomize=True, fetch_size=1)
    iterview2 = IteratorView(loader, 'train', randomize=True, random_seed=2601, fetch_size=1)
    assert list(map(lambda tp: tp[0], itertools.islice(iterview2, 10))) \
        != list(map(lambda tp: tp[0], itertools.islice(iterview, 10)))


@source('test-source', 'A test source.')
class SourceTest(SourcePlugin):

    def __init__(self, args: dict={}):
        self.data = dict(
            train = list(range(100)),
            val = list(range(10)),
            test = list(range(20))
        )
        super().__init__(args)

    def num_samples(self, split: str) -> int:
        return len(self.data[split])

    def read_samples(self, split, index, n=1):
        items = self.data[split][index: index+n]
        return [Sample(item, item+5, {'meta': item}, random.Random(self.random_seed + item))
            for item in items]