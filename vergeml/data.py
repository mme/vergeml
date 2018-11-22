from vergeml.utils import VergeMLError, parse_split
from vergeml.views import ListView, BatchView, IteratorView
from vergeml.io import Sample, SourcePlugin
from vergeml.operation import BaseOperation
from vergeml.loader import FileCachedLoader, LiveLoader, MemoryCachedLoader
from vergeml.plugins import PLUGINS
from vergeml.utils import introspect
from vergeml.display import DISPLAY
from typing import List, Any, Union, Callable, Optional
import random
import numpy as np


class Labels(list):
    pass


class BoundingBoxes(list):
    pass


class BoundingBox:
    def __init__(self, label: str, x: int, y: int, width: int, height: int):
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class Data:
    """Handle loading, augmentation and caching of your sample data.

    The easiest way to set up Data is by passing it and a vergeml.Environment object::

        from vergeml import Data
        data = Data(env)
        xy_train = data.load()
        # xy_train is now [(x1, y1), (x2, y2), ...]

    This will automatically set up the needed configuration and it will be ready to use.
    Alternatively, it is possible to set it up manually by providing input, output and ops.
    """

    def __init__(self,
                 env: 'Environment' = None,
                 input: SourcePlugin = None,
                 output: SourcePlugin = None,
                 ops: List[BaseOperation] = [],
                 random_seed: int = 2204,
                 cache_dir: str = '.cache',
                 cache_input: Union[str, bool] = 'mem',
                 cache_output: Union[str, bool] = False,
                 plugins=PLUGINS):
        """For automatic configuration, pass in an env object. To manually setup the data class,
        you need to provide at least the input parameter. If you want to provide a list of
        preprocessing operations ops to be applied to the data, you need to provide an output
        object as well.

        :param env: an Environment object used to set up Data. If an environment is provided,
                    the other options are ignored.
        :param input: the input object
        :param output: the output object
        :param ops: a list of preprocessing operations
        :param random_seed: the random seed to use
        :param cache_input: how input data should be cached. defaults to False
                            possible values are 'mem', 'disk' or False
        :param cache_output: how output data should be cached. defaults to 'disk'
        """
        self.cache_dir = cache_dir
        self.env = env
        self.input = input
        self.output = output
        self.ops = ops.copy()
        self.random_seed = random_seed
        self.loader = None
        self.plugins = plugins
        self._progress_bar = None
        self.cache_input = cache_input
        self.cache_output = cache_output

        if env:
            self._setup_from_env(env)
        else:
            assert cache_input in ('mem', 'disk', False)
            assert cache_output in ('mem', 'disk', False)
            assert self.input is not None
            if self.output is None:
                self.output = self.input
            self.loader = self._get_loader(cache_input, cache_output)

    def _get_loader(self, cache_input, cache_output):
        if cache_input == 'disk':
            input_loader = FileCachedLoader(self.cache_dir, self.input)
            input_loader.progress_callback = self._progress_callback
        elif cache_input == 'mem':
            input_loader = MemoryCachedLoader(self.cache_dir, self.input)
            input_loader.progress_callback = self._progress_callback
        else:
            input_loader = self.input

        if cache_output == 'disk':
            loader = FileCachedLoader(self.cache_dir, input_loader, self.ops, self.output)
            loader.progress_callback = self._progress_callback
            return loader
        elif cache_output == 'mem':
            loader = MemoryCachedLoader(self.cache_dir, input_loader, self.ops, self.output)
            loader.progress_callback = self._progress_callback
            return loader
        else:
            return LiveLoader(self.cache_dir, input_loader, self.ops, self.output)

    @property
    def meta(self):
        self.loader.begin_read_samples()
        meta = self.loader.meta
        self.loader.end_read_samples()
        return meta

    def load(self,
             split:str='train',
             view:str='list',
             layout:str='tuples',
             batch_size:int=64,
             fetch_size:Optional[int]=None,
             stream:bool=True,
             infinite:bool=False,
             with_meta:bool=False,
             randomize:bool=False,
             transform_x:Callable[[Any], Any]=lambda x: x,
             transform_y:Callable[[Any], Any]=lambda y: y):

        """
        :param split: One of ("train", "val", "test"). Will return the split data as configured
                      via the env or via input. Defaults to "train".

        :param view: A view determines the **class** which will hold the sample data.
                     Possible values are:

                    - "list" (default): reads all data into memory and return it as **python list**
                      or optionally as numpy array (see the layout option).

                    - "lazy-list": returns a list that reads the data on demand.

                    - "batch": return a **generator**, splitting the data into batches of batch_size.
                      The returned object supports getting the length (number of batches) via the len()
                      function. The generator will stream the batches from disk if stream is set to
                      True. Otherwise it will read all data into RAM.

                    - "iter": return the data as an **iterator** object. Supports streaming from disk
                      or loading all data into memory (via the stream parameter).

        :param layout: Determines how x, y and (optionally meta) is returned.
                       Can be one of:

                       - "tuples" (default): the data will be returned as (x,y) pairs.

                         [(x1,y1), (x2,y2), ...]

                       - "lists": the data will be returned as lists [xs], [ys]

                         ([x1, x2, x3], [y1, y2, y3])

                       - "arrays": the data will be returned as numpy arrays [xs], [ys]

                         array([[1, 2, 3],
                                [4, 5, 6]])

        :param batch_size: Sample batch size. Applies to "batch" view only. Defaults to 64.

        :param fetch_size: Fetch samples in pairs of fetch_size. None means the system will automatically
                           set a fetch size.

        :param infinite: Applies to "batch" and "iter" views. If set to **True**, the returned
                         object will be an infinite generator object.

            NOTE for KERAS users: This setting is useful when used in with the
            model.fit_generator() of the keras framework. Since len() will return the number of
            steps per epoch, the steps_per_epoch of fit_generator() can be left unspecified.

        :param with_meta: If True, will return meta in addition to x and y.

        :param randomize: If True, the data will be returned in random order.

        :param transform_x: a function that takes x as an argument and returns a transformed version.
                            Defaults to None (No transformation)

        :param transform_y: a function that takes x as an argument and returns a transformed version.
                            Defaults to None (No transformation) """

        assert view in ('list', 'lazy-list', 'batch', 'iter')
        assert layout in ('tuples', 'lists', 'arrays')
        fetch_size = fetch_size or 8

        if view == 'list':
            res = []
            self.loader.begin_read_samples()

            num_samples = self.loader.num_samples(split)

            for sample in self.loader.read_samples(split, 0, num_samples):
                x, y, m = sample.x, sample.y, sample.meta
                x, y = transform_x(x), transform_y(y)
                res.append((x, y, m)) if with_meta else res.append((x, y))
            self.loader.end_read_samples()

            if randomize:
                random.Random(self.random_seed).shuffle(res)

            if layout in ('lists', 'arrays'):
                res = tuple(map(list, zip(*res)))
                if not res:
                    res = ([], [], []) if with_meta else ([], [])

            if layout == 'arrays':
                xs, ys, *meta = res
                res = tuple([np.array(xs), np.array(ys)] + meta)
            return res

        elif view == 'lazy-list':
            return ListView(self.loader,
                            split,
                            with_meta,
                            randomize,
                            self.random_seed,
                            fetch_size,
                            transform_x,
                            transform_y)

        elif view == 'batch':
            return BatchView(self.loader,
                             split,
                             batch_size,
                             randomize,
                             self.random_seed,
                             fetch_size,
                             infinite,
                             with_meta,
                             layout,
                             transform_x,
                             transform_y)

        elif view == 'iter':
            return IteratorView(self.loader,
                                split,
                                randomize,
                                self.random_seed,
                                fetch_size,
                                infinite,
                                with_meta,
                                transform_x,
                                transform_y)

    def _setup_from_env(self, env):
        # TODO type check arguments of input and output
        self.env = env
        self.random_seed = self.env.get('random-seed')
        self.cache_dir = self.env.cache_dir()

        # get base configuration
        config = {}
        config['val-split'] = env.get('val-split')
        config['test-split'] = env.get('test-split')
        config['samples-dir'] = self.env.get('samples-dir')
        config['random-seed'] = self.env.get('random-seed')
        config['trainings-dir'] = self.env.get('trainings-dir')

        # get the name of the input plugin
        input_name = self.env.get('data.input.type')
        if not input_name:
            raise VergeMLError("data.input.type is not defined.")

        # get input configuration and merge base config
        input_conf = self.env.get('data.input').copy()
        input_conf.update(config)

        # instantiate the input plugin
        input_class = self.plugins.get('vergeml.io', input_name)
        if not input_class:
            raise VergeMLError("input name not found: {}".format(input_name))

        # TODO validate configuration and set defaults
        del input_conf['type']

        self.input = input_class(input_conf)

        # set up preprocessing operations
        self.ops = []
        for conf in self.env.get('data.preprocess') or []:
            if isinstance(conf, str):
                conf = dict(name=conf)
            else:
                conf = conf.copy()

            # every preprocessing operations needs a name property
            name = conf.get('op', None)
            if not name:
                raise VergeMLError("Name missing in data.preprocess item.")
            del conf['op']

            # instantiate the preprocessing plugin
            plugin = self.plugins.get('vergeml.operation', name)
            if not plugin:
                raise VergeMLError("preprocess plugin not found: {}".format(name))

            # check arguments
            intro = introspect(plugin)
            mandatory = set(intro.args[1:]).difference(set(intro.defaults.keys()))
            missing = set(mandatory).difference(conf.keys())
            unknown = set(conf.keys()).difference(intro.args[1:])

            # TODO automatic type checking

            # report missing or unknown arguments
            if missing:
                raise VergeMLError("preprocess operation {} is missing argument(s): {}".format(
                    name, missing))

            if unknown:
                raise VergeMLError("preprocess operation {} received unknown argument(s): {}".format(
                    name, unknown))

            op = plugin(**conf)
            self.ops.append(op)

        # get the name of the output plugin or set it to input
        output_name = self.env.get('data.output.type') or input_name

        # get output configuration or set it to input configuration and merge
        # with base config
        output_conf = self.env.get('data.output') or self.env.get('data.input').copy()
        output_conf['name'] = output_name
        output_conf.update(config)

        # instantiate the output plugin
        output_class = self.plugins.get('vergeml.io', output_name)
        if not output_class:
            raise VergeMLError("output name not found: {}".format(output_name))

        self.output = output_class(output_conf)

        cache = env.get("data.cache")
        if cache == 'mem-in':
            self.cache_input, self.cache_output = 'mem', False
        elif cache == 'disk-in':
            self.cache_input, self.cache_output = 'disk', False
        elif cache == 'mem':
            self.cache_input, self.cache_output = False, 'mem'
        elif cache == 'disk' or cache == '*auto*':
            self.cache_input, self.cache_output = False, 'disk'
        elif cache == 'none':
            self.cache_input, self.cache_output = False, False

        self.loader = self._get_loader(self.cache_input, self.cache_output)

    def num_samples(self, split):
        self.loader.begin_read_samples()
        res = self.loader.num_samples(split)
        self.loader.end_read_samples()
        return res

    def _progress_callback(self, n, t):
        if t:
            if n == -1:
                if self.cache_input == 'mem':
                    DISPLAY.print("Caching input samples in memory ...")
                elif self.cache_output == 'mem':
                    DISPLAY.print("Caching output samples in memory ...")
                elif self.cache_input == 'disk':
                    DISPLAY.print("Caching input samples on disk ...")
                elif self.cache_output == 'disk':
                    DISPLAY.print("Caching output samples on disk ...")
            else:
                if not self._progress_bar:
                    self._progress_bar = DISPLAY.progressbar(steps=t, label="samples", keep=False)
                    self._progress_bar.start()
                self._progress_bar.update(n)
                if n+1 == t:
                    self._progress_bar.stop()
                    print("")
