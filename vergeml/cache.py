from vergeml import VergeMLError
import struct
import pickle
import mmap
import io
import gzip
import numpy as np
import lz4.frame

class Cache:
    
    def write(self, data, meta):
        raise NotImplementedError
    
    def read(self, ix, n):
        raise NotImplementedError

class MemoryCache(Cache):

    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def write(self, data, meta):
        self.data.append((data, meta))
    
    def read(self, ix, n):
        return self.data[ix:ix+n]


class FileCache(Cache):

    def __init__(self, path, mode):
        assert mode in ("r", "w")

        self.path = path
        self.fp = open(self.path, mode + "b")
        self.mode = mode
        self.index = []
        self.item_meta = []
        self.meta = None
        self.info = None
        self.mm = None

        if mode == "r":
            pos, = struct.unpack('<Q', self.fp.read(8))
            if pos == 0:
                raise VergeMLError("Invalid cache file: {}".format(self.path))
            self.fp.seek(pos)
            self.index, self.item_meta, self.meta, self.info = pickle.load(self.fp)
            self.mm = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.fp.write(struct.pack('<Q', 0))

    def __len__(self):
        return len(self.index)

    def write(self, data, meta):
        assert self.mode == "w"

        self.item_meta.append(meta)
        pos = self.fp.tell()
        entry = (pos, pos + len(data))
        self.index.append(entry)
        self.fp.write(data)

    def read(self, ix, n):
        assert self.mode == "r"
        
        # get the absolute start and end adresses of the whole chunk
        start, _ = self.index[ix]
        _, end = self.index[ix+n-1]

        # read the bytes and wrap in memory view to avoid copying
        chunk = memoryview(self.mm[start:end])

        res = []

        for i in range(n):
            s, e = self.index[ix+i]

            # convert addresses to be relative to the chunk we read
            s = s - start
            e = e - start

            data = chunk[s:e]
            res.append((data, self.item_meta[ix+i]))
        return res

    def close(self):
        if self.mode == "w":
            pos = self.fp.tell()
            pickle.dump((self.index, self.item_meta, self.meta, self.info), self.fp)
            self.fp.seek(0)
            self.fp.write(struct.pack('<Q', pos))
        self.fp.close()

_BYTES, _NUMPY, _PICKLE = range(3)

class SerializedFileCache(FileCache):

    def __init__(self, path, mode, compress=True):
        super().__init__(path, mode)
        self.info = self.info or []
        self.compress = compress
    
    def _serialize_data(self, data):
        type_ = _BYTES
        if isinstance(data, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, data)
            data = buf.getvalue()
            type_ = _NUMPY
        elif not isinstance(data, (bytearray, bytes)):
            data = pickle.dumps(data)
            type_ = _PICKLE
        if self.compress:
            data = lz4.frame.compress(data)
        return type_, data
    
    def _deserialize(self, data, type_):
        if self.compress:
            data = lz4.frame.decompress(data)
        if type_ == _NUMPY:
            buf = io.BytesIO(data)
            data = np.load(buf)
        elif type_ == _PICKLE:
            data = pickle.loads(data)
        return data
    
    def write(self, data, meta):
        if isinstance(data, tuple) and len(data) == 2:
            type1, data1 = self._serialize_data(data[0])
            type2, data2 = self._serialize_data(data[1])
            pos = len(data1)
            data = io.BytesIO()
            data.write(struct.pack('<Q', pos))
            data.write(data1)
            data.write(data2)
            data = data.getvalue()
            type_ = (type1, type2)
        else:
            type_, data = self._serialize_data(data)
        
        super().write(data, meta)
        self.info.append(type_)

    def read(self, ix, n):
        entries = super().read(ix, n)
        res = []
        for i, entry in enumerate(entries):
            data, meta = entry
            type_ = self.info[ix+i]
            if isinstance(type_, tuple):
                buf = io.BytesIO(data)
                pos, = struct.unpack('<Q', buf.read(8))
                data1 = buf.read(pos)
                data2 = buf.read()
                data1 = self._deserialize(data1, type_[0])
                data2 = self._deserialize(data2, type_[1])
                res.append(((data1, data2), meta))
            else:
                data = self._deserialize(data, type_)
                res.append((data, meta))
        return res
