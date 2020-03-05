import numpy
import random
import torch
import itertools

B = numpy.uint32(1 << 31)



def pseudopermutation(N):
    twos = 0
    m = 1
    while m <= N:
        twos += 1
        m <<= 1

    a = 4 * random.randrange(1, (1 << 28) + 1, 2) + 1

    c = random.randrange(1, (1 << 28) + 1, 2)

    x = random.randrange(N)

    while True:
        yield x
        x = (a * x + c) % m
        while x >= N:
            x = (a * x + c) % m

class WindowData(torch.utils.data.Dataset):
    def __init__(self, file, vocab, radius, memmap=True):
        self.pad_ix = vocab.pad_ix
        self.sos_ix = vocab.sos_ix
        self.eos_ix = vocab.eos_ix
        self.radius = radius
        self.memmap = memmap
        if self.memmap:
            self.ixmm = numpy.memmap(file, numpy.uint32, 'r')
        else:
            self.ixmm = numpy.fromfile(file, numpy.uint32)

        self.size = self.radius * 2 + 1
        self.len = len(self.ixmm)

    def __len__(self):
        return len(self.ixmm)

    def __getitem__(self, ix):
        vals = numpy.empty(
            self.size,
            dtype=numpy.uint32,
        )
        tmp = ix - self.radius
        lb = max(tmp, 0)
        lp = lb - tmp

        tmp = ix + self.radius + 1
        ub = min(tmp, self.len)
        up = tmp - ub

        sup = self.size - up
        vals[lp:sup] = self.ixmm[lb:ub]

        flags = vals & B
        flags ^= flags[self.radius]

        vals &= ~B

        i = self.radius - 1
        while i >= lp and not flags[i]:
            i -= 1

        if i >= 0:
            vals[i] = self.sos_ix
            vals[:i] = self.pad_ix

        i = self.radius + 1
        while i < sup and not flags[i]:
            i += 1

        if i < self.size:
            vals[i] = self.eos_ix
            vals[i+1:] = self.pad_ix

        return vals.astype(numpy.int64)

    def close(self):
        if self.memmap:
            del self.ixmm


class PseudoShuffle(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.n = len(data_source)

    def __iter__(self):
        return itertools.islice(pseudopermutation(self.n), self.n)

    def __len__(self):
        return self.n
