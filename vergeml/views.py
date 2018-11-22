"""
This module implements the data structures returned by Data.load() to
support the unique data loading requirements of different deep learning
libraries.
"""

import random
import itertools
import numpy as np

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

def _rand_batch_ixs(num_samples: int, batch_size: int, fetch_size: int, random_seed: int):
    """A generator which yields a list of tuples (offset, size) in random order.

    This list will be used by the data loader to efficiently load samples and pass it to
    the model during training.

    :param num_samples: Number of available samples.
    :param batch_size: The size of the batch to fill.
    :param fetch_size: Desired fetch_size.
    :param random_seed: RNG seed.
    """
    rng = random.Random(random_seed)
    batch, batch_count = [], 0

    while True:
        if fetch_size * 3 < num_samples:
            # if the number of samples is too small, having a random offset
            # makes no sense
            offset = rng.randint(0, fetch_size)
        else:
            offset = 0

        ixs = list(range(offset, num_samples - offset, fetch_size))
        rng.shuffle(ixs)

        # collect enough samples to fill the batch

        while ixs:
            next_fetch = ixs.pop(0)

            # calculate the next fetch size depending on the samples remaining
            # and the number of samples required to fill the batch

            next_fetch_size = min(fetch_size,
                                  num_samples - next_fetch, batch_size - batch_count)

            batch.append((next_fetch, next_fetch_size))
            batch_count += next_fetch_size

            if batch_count == batch_size:
                yield batch
                batch, batch_count = [], 0

def _ser_batch_ixs(num_samples, batch_size):
    """A generator which yields a list of tuples (offset, size) in serial order.

    :param num_samples: Number of available samples.
    :param batch_size: The size of the batch to fill.
    """
    current_index = 0
    batch, batch_count = [], 0

    while True:
        next_fetch = current_index
        next_fetch_size = min(batch_size - batch_count, num_samples - next_fetch)

        batch.append((next_fetch, next_fetch_size))
        batch_count += next_fetch_size

        if batch_count == batch_size:

            # If we have enough samples to fill the batch size, yield
            # the indices and reset the batch count.
            yield batch
            batch, batch_count = [], 0

        current_index += next_fetch_size

        if current_index == num_samples:
            current_index = 0


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

        # TODO do we really need max_samples ?
        if max_samples:
            self.num_samples = min(self.num_samples, max_samples)

        self.loader.end_read_samples()

        self.infinite = infinite
        self.with_meta = with_meta
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.fetch_size = fetch_size
        self.batch_size = batch_size
        self.layout = layout
        self.num_batches = self.num_samples // self.batch_size
        self.current_batch = 0

        if randomize:
            ix_fn = lambda _: _rand_batch_ixs(self.num_samples, self.batch_size, self.fetch_size, random_seed)
        else:
            ix_fn = lambda _: _ser_batch_ixs(self.num_samples, self.batch_size)

        # two identical ix generators - one for this object, one for the loader
        self.ix_gen, ix_gen_ = map(ix_fn, range(2))

        def pumpfn():
            while True:
                yield from next(ix_gen_)

        self.loader.pump(self.split, pumpfn())

    def __iter__(self):
        self.current_batch = 0
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):

        if self.current_batch >= self.num_batches and not self.infinite:
            raise StopIteration

        # BEGIN loading samples from the data loader
        self.loader.begin_read_samples()

        res = []
        for ix, n in next(self.ix_gen):

            samples = self.loader.read_samples(self.split, ix, n)
            for sample in samples:
                x, y, m = sample.x, sample.y, sample.meta
                x, y = self.transform_x(x), self.transform_y(y)
                res.append((x, y, m) if self.with_meta else (x, y))

        self.loader.end_read_samples()
        # END loading samples

        self.current_batch += 1

        # rearrange the result according to the configured layout
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
