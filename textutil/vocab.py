import collections
import itertools
import enum
from textutil.text import read_count


@enum.unique
class Type(enum.Enum):
    TOKEN = 1
    SPECIAL = 2


token = Type.TOKEN
special = Type.SPECIAL


class Vocabulary(collections.abc.Mapping):
    def __init__(self, i2wf, sentences):
        self.i2w = [
            *self.specials,
            *[(token, x) for x, _ in i2wf]
        ]

        self.i2f = [
            *[0 for _ in self.specials],
            *[c for _, c in i2wf]
        ]

        self.w2i = {w: i for i, w in enumerate(self.i2w)}

        self.sentences = sentences

    @property
    def specials(self):
        return [self.pad, self.sos, self.eos, self.unk]

    def __getitem__(self, key):
        return self.w2i[key if type(key) is tuple else (token, key)]

    def __contains__(self, key):
        return (key if type(key) is tuple else (token, key)) in self.w2i

    def __iter__(self):
        yield from self.i2w

    def __len__(self):
        return len(self.i2w)

    def get(self, word):
        try:
            return self[word]
        except KeyError:
            return self[self.unk]

    @property
    def pad(self):
        return (special, 'PAD')

    @property
    def sos(self):
        return (special, 'SOS')

    @property
    def eos(self):
        return (special, 'EOS')

    @property
    def unk(self):
        return (special, 'UNK')

    @property
    def pad_ix(self):
        return self[(special, 'PAD')]

    @property
    def sos_ix(self):
        return self[(special, 'SOS')]

    @property
    def eos_ix(self):
        return self[(special, 'EOS')]

    @property
    def unk_ix(self):
        return self[(special, 'UNK')]

    @staticmethod
    def from_file(fp, num_words=None, min_freq=1):
        return Vocabulary(*read_count(fp, num_words, min_freq))

    def to_file(self, fp):
        with open(fp, 'wt') as handle:
            handle.write('{}\n'.format(self.sentences))
            for word, count in itertools.islice(zip(self.i2w, self.i2f), 4, None):
                handle.write('{} {}\n'.format(word[1], count))
