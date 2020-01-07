import collections
import itertools
import multiprocessing

"""
File -> [[Word]]
read words in sentences
"""
def read_file(path):
    with open(path, 'rt') as handle:
        for line in handle:
            words = line.strip().split()
            if words:
                yield words

"""
File -> (Counter Word, Int)
"""
def count_file(path):
    ctr = collections.Counter()
    sentences = 0
    for sentence in read_file(path):
        sentences += 1
        for word in sentence:
            ctr[word] += 1
    return ctr, sentences

"""
Files -> (Counter Word, Int)
Count words in all files
"""
def count_files(files):
    ctr = collections.Counter()
    sentences = 0
    with multiprocessing.Pool(8) as pool:
        for fc, fs in pool.imap_unordered(count_file, files):
            ctr += fc
            sentences += fs
    return ctr, sentences

"""
Filepath, (Counter Word, Int) -> IO ()
Write a word count to a file
"""
def write_count(fp, cs):
    (ctr, sentences) = cs
    with open(fp, 'wt') as handle:
        handle.write('{}\n'.format(sentences))
        for word, count in ctr.most_common():
            handle.write('{} {}\n'.format(word, count))

"""
Filepath -> [Word, Int], Int
Write a word count to a file
"""
def read_count(fp, num_words=None, min_freq=1):
    with open(fp, 'rt') as handle:
        sentences = int(next(handle))
        i2wf = []
        for line in itertools.islice(handle, num_words):
            w, f = line.split()
            f = int(f)
            if f < min_freq:
                break
            i2wf.append((w, f))
    return i2wf, sentences

