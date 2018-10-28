from vergeml.io import Sample
from vergeml.utils import SPLITS
import operator
from functools import reduce
from vergeml.cache import MemoryCache, SerializedFileCache
import io
import os.path

class Loader:

    def __init__(self, cache_dir, input, ops=None, output=None, progress_callback=lambda n, t: None):
        self.cache_dir = cache_dir
        self.input = input
        self.ops = ops or []
        self.output = output
        self.cache = {}
        self.progress_callback = progress_callback

    @property
    def meta(self):
        return self.input.meta
        
    def begin_read_samples(self):
        raise NotImplementedError

    def num_samples(self, split: str) -> int:
        return len(self.cache[split])
    
    def read_samples(self, split: str, index: int, n: int=1) -> Sample:
        samples = []
        for item in self.cache[split].read(index, n):
            x, y = item[0]
            meta, rng = item[1]
            samples.append(Sample(x,y,meta,rng))
        return samples
    
    def end_read_samples(self):
        pass
    
    def _multiplier(self, split, op):
        has_split = hasattr(op, 'apply') and op.apply.intersection(set(SPLITS))
        if has_split and split not in op.apply:
            return 1.0
        else:
            return op._multiplier()

    def _calculate_num_samples(self, split):
        """Calculate the total number of samples after applying ops"""
        num_samples = self.input.num_samples(split)
        if self.ops:
            multiplier = reduce(operator.mul, map(lambda op: self._multiplier(split, op), self.ops), 1)
        else:
            multiplier = 1
        return int(num_samples * multiplier)

    def _calculate_hashed_state(self):
        input_conf_str = str(sorted(self.input._configuration().items()))

        state = self.input.__class__.__name__ + input_conf_str
  
        if self.output:     
            ops_state = "-".join([str(sorted(op._configuration().items())) for op in self.ops])
            out_state = self.output.__class__.__name__ + str(sorted(self.output._configuration().items()))
            state = "-".join([state, ops_state, out_state])
        
        hashed_state = self.input.hash(state)

        return hashed_state
    
    def _iter_samples(self, split, raw=False):
        """Iterate samples possibly applying operations"""
        n = self.input.num_samples(split)
        if self.output and self.ops:
            op1, *oprest = self.ops

            for ix in range(n):
                sample = self.input.read_samples(split, ix)[0]
                for sample_ in op1.process(sample, oprest):
                    yield self.output.transform(sample_)
        elif self.output:
            for ix in range(n):
                sample_ = self.input.read_samples(split, ix)[0]
                yield self.output.transform(sample_)
        elif raw:
            for i in range(n):
                yield self.input.read_raw_samples(split, i)[0]
        else:
            for i in range(n):
                yield self.input.read_samples(split, i)[0]


class MemoryCachedLoader(Loader):

    def begin_read_samples(self):
        if self.cache:
            return

        self.input.begin_read_samples()
        # copy meta
        if self.output:
            self.output.meta = self.input.meta

        self.cache = {k:MemoryCache() for k in SPLITS}
        total = sum(map(lambda split: self._calculate_num_samples(split), SPLITS))
       
        i = 0
        self.progress_callback(-1, total)
        for split in SPLITS:
            cache = self.cache[split]
            for sample in self._iter_samples(split):
                cache.write((sample.x, sample.y), (sample.meta, sample.rng))
                self.progress_callback(i, total)
                i = i + 1
                 
        self.input.end_read_samples()

        

class FileCachedLoader(Loader):

    def begin_read_samples(self):
        if self.cache:
            return

        self.input.begin_read_samples()
        # copy meta
        if self.output:
            self.output.meta = self.input.meta
            
        hashed_state = self._calculate_hashed_state()

        paths = [(split, self._cache_path(split, hashed_state)) for split in SPLITS]
        total = 0
        for split, path in paths:
            if not os.path.exists(path):
                total += self._calculate_num_samples(split)
        
        if total != 0:
            i = 0
            cache = None

            try:
                self.progress_callback(-1, total)
                for split, path in paths:
                    if not os.path.exists(path):
                        # we compress output data since its likely to be numpy arrays
                        cache = SerializedFileCache(path, "w", compress=bool(self.output))
                        
                        for sample in self._iter_samples(split, raw=True):
                            cache.write((sample.x, sample.y), (sample.meta, sample.rng))
                            self.progress_callback(i, total)
                            i += 1
                        cache.close()
                        
                        cache = None
            except (KeyboardInterrupt, SystemExit, Exception) as e:
                # clean up in case of an exception

                # first, in case there is an open cache, close it
                if cache:
                    cache.close()

                # then, delete all cached files
                for _, path in paths:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                        except:
                            pass

                raise e

        self.cache = {split:SerializedFileCache(path, "r", compress=bool(self.output)) for split, path in paths}
        self.input.end_read_samples()

        

    def read_samples(self, split: str, index: int, n: int=1) -> Sample:
        samples = super().read_samples(split, index, n)
        if not self.output:
            samples = [self.input.recover_raw_sample(sample) for sample in samples]
        return samples

    def _cache_path(self, split, hashed_state):
        return os.path.join(self.cache_dir, "{}-{}.cache".format(hashed_state, split))


class LiveLoader(Loader):
    multipliers = None
    rngs = None

    def begin_read_samples(self):
        if self.cache:
            return
        
        self.input.begin_read_samples()
         # copy meta
        if self.output:
            self.output.meta = self.input.meta

        self.multipliers = {}
        self.rngs = {}
        
        for split in SPLITS:
            self.multipliers[split] = reduce(operator.mul, map(lambda op: self._multiplier(split, op), self.ops), 1)
            self.cache[split] = self._calculate_num_samples(split)
            self.rngs[split] = self.cache[split] * [None]
        
        self.input.end_read_samples()
        
       
        
    def num_samples(self, split: str) -> int:
        return self.cache[split]

    def read_samples(self, split: str, index: int, n: int=1) -> Sample:
        mul = self.multipliers[split]
        offset = int(index % mul)
        start_index = int(index/mul)
        end_index = int((index+n)/mul)
        read = max(1, int(n/mul) + int(min(1, index%mul)))

        res = []
        
        samples = self.input.read_samples(split, start_index, read)
        if self.output and self.ops:
            op1, *oprest = self.ops

            for sample in samples:
                res.extend(op1.process(sample, oprest))
                
        else:
            res = samples
        
        if self.output:
            res = [self.output.transform(sample) for sample in res]
        
        for s, i in zip(res, range(start_index, end_index)):
            if self.rngs[split][i] is None:
                self.rngs[split][i] = s.rng
            else:
                s.rng = self.rngs[split][i]
        
        return res[offset: offset+n]