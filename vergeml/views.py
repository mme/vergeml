import numpy as np
import random
import itertools
from operator import itemgetter
from vergeml.utils import VergeMLError

class ListView:

    def __init__(self, 
                 loader, 
                 split, 
                 with_meta=False,
                 randomize=False,
                 random_seed=2204,
                 fetch_size=8,
                 max_samples=None,
                 transform_x=lambda x: x,
                 transform_y=lambda y : y):
    
        self.loader = loader
        self.split = split
        self.with_meta = with_meta
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.loader.begin_read_samples()
        num_samples = self.loader.num_samples(self.split)
        self.num_samples = min(num_samples, max_samples if max_samples is not None else num_samples)
        self.loader.end_read_samples()
        self.ixs = None
        if randomize:
            ixs = list(range(0, self.num_samples))
            ixs = [ixs[i:i + fetch_size] for i in range(0, len(ixs), fetch_size)]
            rng = random.Random(random_seed)
            rng.shuffle(ixs)
            self.ixs = list(itertools.chain.from_iterable(ixs))

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, key):
        
        if isinstance(key, slice):
            slc = dict(start=key.start or 0, stop=key.stop or self.num_samples)
            for k, v in slc.items():
                if slc[k] < 0:
                    slc[k] = self.num_samples - abs(v)
                if slc[k] < 0:
                    raise IndexError("list index out of range")
            
            start, stop = slc['start'], slc['stop']

            if start >= self.num_samples or stop > self.num_samples:
                raise IndexError("list index out of range")

            if self.ixs:
                ixs = self.ixs[start: stop]
                self.loader.begin_read_samples()
                samples = [self.loader.read_samples(self.split, ix, 1)[0] for ix in ixs]
                self.loader.end_read_samples()
            else:
                self.loader.begin_read_samples()
                samples = self.loader.read_samples(self.split, start, stop - start)
                self.loader.end_read_samples()
            res = list(map(self._transform_sample, samples))

            return res
        else:
            if key < 0:
                key = self.num_samples - abs(key)
            if key < 0:
                raise IndexError("list index out of range")
            if key >= self.num_samples:
                raise IndexError("list index out of range")

            if self.ixs:
                key = self.ixs[key]
            self.loader.begin_read_samples()
            sample = self.loader.read_samples(self.split, key, 1)[0]
            self.loader.end_read_samples()
            return self._transform_sample(sample)
    
    def _transform_sample(self, sample):
        x, y = self.transform_x(sample.x), self.transform_y(sample.y)
        m = sample.meta
        res = (x, y, m) if self.with_meta else (x, y)
        
        return res


class BatchView:

    def __init__(self,
                 loader,
                 split,
                 batch_size,
                 randomize,
                 random_seed,
                 fetch_size,
                 infinite,
                 with_meta,
                 layout,
                 max_samples,
                 transform_x,
                 transform_y):

        self.loader = loader
        self.split = split

        self.loader.begin_read_samples()
        self.num_samples = self.loader.num_samples(self.split)
        if max_samples:
            self.num_samples = min(self.num_samples, max_samples)
        self.loader.end_read_samples()
        self.infinite = infinite
        self.with_meta = with_meta
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.fetch_size = fetch_size
        self.rng = None

        if randomize:
            self.rng = random.Random(random_seed)
        self.ixs = None
        self._shuffle()

        if batch_size > self.num_samples:
            # TODO issue warning
            batch_size = self.num_samples

        self.batch_size = batch_size
        self.num_batches = self.num_samples // self.batch_size
        self.current_batch = 0
        self.layout = layout
    
    def _shuffle(self):
        self.ixs = range(0, self.num_samples)
        if self.rng:
            self.ixs = [self.ixs[i:i + self.fetch_size] for i in range(0, len(self.ixs), self.fetch_size)]
            self.rng.shuffle(self.ixs)
            self.ixs = list(itertools.chain.from_iterable(self.ixs))

    def __iter__(self):
        self.current_batch = 0
        self._shuffle()
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):

        if self.current_batch >= self.num_batches:
            if not self.infinite:
                raise StopIteration
            else:
                self.current_batch = 0
                self._shuffle()
        
        start = self.current_batch*self.batch_size
        batch_ixs = self.ixs[start:start+self.batch_size]

        self.current_batch += 1

        res = []
        self.loader.begin_read_samples()

        # continuos indexes can be read efficiently from caches
        continuous_ixs = [list(map(itemgetter(1), g)) for k, g in itertools.groupby(enumerate(batch_ixs), lambda i_x :i_x[0]-i_x[1])]
        for ixs in continuous_ixs:
            ix = ixs[0]
            n = len(ixs)
            samples = self.loader.read_samples(self.split, ix, n)
            for sample in samples:
                x, y, m = sample.x, sample.y, sample.meta
                x, y = self.transform_x(x), self.transform_y(y)
                res.append((x, y, m) if self.with_meta else (x, y))

        self.loader.end_read_samples()

        if self.layout in ('lists', 'arrays'):
            res = tuple(map(list, zip(*res)))
        if self.layout == 'arrays':
            xs, ys, *meta = res
            res = tuple([np.array(xs), np.array(ys)] + meta)
        return res

class IteratorView:
    def __init__(self,
                 loader,
                 split,
                 randomize=False,
                 random_seed=2204,
                 fetch_size=8,
                 infinite=False,
                 with_meta=False,
                 max_samples=None,
                 transform_x=lambda x: x,
                 transform_y=lambda y: y):

        self.loader = loader
        self.split = split

        self.loader.begin_read_samples()
        num_samples = self.loader.num_samples(self.split)
        self.num_samples = min(num_samples, max_samples if max_samples is not None else num_samples)
        self.loader.end_read_samples()

        self.infinite = infinite
        self.with_meta = with_meta
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.fetch_size = fetch_size
        self.rng = random.Random(random_seed) if randomize else None
        self.ixs = None

        self.current_index = 0
        self._shuffle()
    
    def _shuffle(self):
        self.ixs = range(0, self.num_samples)
        if self.rng:
            self.ixs = [self.ixs[i:i + self.fetch_size] for i in range(0, len(self.ixs), self.fetch_size)]
            self.rng.shuffle(self.ixs)
            self.ixs = list(itertools.chain.from_iterable(self.ixs))

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):

        if self.current_index >= self.num_samples:
            self.current_index = 0
            self._shuffle()
            if not self.infinite:
                raise StopIteration
        
        ix = self.ixs[self.current_index]
        sample = self.loader.read_samples(self.split, ix, 1)[0]
        x, y, m = sample.x, sample.y, sample.meta
        x, y = self.transform_x(x), self.transform_y(y)
        res = (x, y, m) if self.with_meta else (x, y)

        self.current_index += 1
        return res
