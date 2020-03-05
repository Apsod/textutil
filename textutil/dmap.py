import numpy
import multiprocessing
from functools import partial
from textutil.text import read_file
from textutil.util import B
import mmap
import tqdm



class Growable(object):
    def __init__(self, capacity=1024, dtype=numpy.uint32, grow=2):
        self.grow = grow
        self.capacity=capacity
        self.dtype=dtype
        self.arr = numpy.empty((self.capacity,), dtype=self.dtype)
        self.size = 0

    def __grow_to__(self, total):
        if self.capacity >= total:
            return
        else:
            while self.capacity < total:
                self.capacity *= self.grow
            new = numpy.empty((self.capacity,), dtype=self.dtype)
            new[:self.size] = self.arr[:self.size]
            self.arr = new

    def __len__(self):
        return self.size


    def update(self, other):
        n = len(other)
        self.__grow_to__(self.size + n)
        self.arr[self.size : self.size+n] = other
        self.size += n

    def finalize(self):
        return self.arr[:self.size]


def ixifyfile(file, vocab=None):
    even = True
    arr = Growable()
    for sentence in read_file(file):
        six = numpy.array([vocab.get(word) for word in sentence], dtype=numpy.uint32)
        if not even:
            six |= B
        even = not even
        arr.update(six)
    return arr.finalize(), even


def ixifyfiles(ixfile, files, vocab):
    ixf = partial(ixifyfile, vocab=vocab)
    even = True
    files = list(files)
    with open(ixfile, 'wb') as ixhandle:
        with multiprocessing.Pool(8) as pool:
            for arr, i_even in tqdm.tqdm(pool.imap_unordered(ixf, files), total=len(files)):
                if even:
                    ixhandle.write(arr.tobytes())
                else:
                    ixhandle.write((arr ^ B).tobytes())
                even = not (i_even ^ even)
