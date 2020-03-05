import argparse
import itertools
from textutil.text import count_files, write_count
from textutil.vocab import Vocabulary
from textutil.dmap import ixifyfiles
from textutil.util import WindowData, PseudoShuffle


def counter(sp):
    parser = sp.add_parser('vocab', help='Create vocabulary file.')
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--output', type=str, default='counter.txt')

    def go(args):
        write_count(args.output, count_files(args.files))

    parser.set_defaults(go=go)

def ixify(sp):
    parser = sp.add_parser('ixify', help='Flatten and ixify files.')
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--num_words', type=int, default=None)
    parser.add_argument('--min_freq', type=int, default=10)
    parser.add_argument('--prefix', type=str, default='ixified')
    parser.add_argument('--num_workers', type=int, default=8)

    def go(args):
        vocab = Vocabulary.from_file(args.vocab, num_words=args.num_words, min_freq=args.min_freq)
        vocab.to_file('{}.vocabulary'.format(args.prefix))
        ixifyfiles('{}.bin'.format(args.prefix), args.files, vocab)

    parser.set_defaults(go=go)

def sample(sp):
    parser = sp.add_parser('sample', help='Sample context windows.')
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--memmap', action='store_true')

    def go(args):
        vocab = Vocabulary.from_file('{}.vocabulary'.format(args.prefix))
        ds = WindowData('{}.bin'.format(args.prefix), vocab, args.radius, args.memmap)
        ps = PseudoShuffle(ds)
        for ix in itertools.islice(ps, args.samples):
            print(' '.join([vocab.i2w[i][1] for i in ds[ix]]))
    parser.set_defaults(go=go)

def main():
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()
    for f in [counter, ixify, sample]:
        f(sp)
    args = parser.parse_args()
    args.go(args)
