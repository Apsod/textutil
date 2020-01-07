# Textutil

A convenient library to convert a bunch of text files to a convenient pytorch
dataset (which admits sampling). Requires Numpy and Pytorch.

To install, run `python setup.py install`

```
usage: textutil [-h] {vocab,ixify,sample} ...

positional arguments:
  {vocab,ixify,sample}
    vocab               Create vocabulary file.
    ixify               Flatten and ixify files.
    sample              Sample context windows.

optional arguments:
  -h, --help            show this help message and exit
```

To create vocabulary, ixify text files, and then sample, run the following:

```
textutil vocab /path/to/text/files/* --output all.vocab
textutil ixify /path/to/text/files/* --vocab all.vocab --prefix ixified
textutil sample --prefix ixified
```

Or use the ixified file as a pytorch dataset:

```
import torch
from textutil.util import WindowData
from textutil.vocab import Vocabulary

radius = 3
vocab = Vocabulary.from_files('ixified.vocabulary')
dataset = WindowData('ixified.bin', vocab, radius)
```
