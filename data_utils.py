"""MNIST -> RGB JPEG file bytes; reproducible train/validation/test separation."""
import gzip
import hashlib
import io
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from prepare import ROOT

PAD_LENGTH = 2048
ENCODING = 'JPEG_RGB_28x28_quality100_subsampling0_pad2048_v1'


def read_mnist(train=True):
    prefix = 'train' if train else 't10k'
    raw = ROOT / 'data/MNIST/raw'
    with gzip.open(raw / f'{prefix}-images-idx3-ubyte.gz', 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051 and (rows, cols) == (28, 28)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).copy()
    with gzip.open(raw / f'{prefix}-labels-idx1-ubyte.gz', 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
    assert len(images) == len(labels) == n
    return images, labels


def image_to_bytes(image):
    image = image.convert('L').resize((28, 28), Image.Resampling.BILINEAR).convert('RGB')
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=100, subsampling=0, optimize=False)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    if len(data) > PAD_LENGTH:
        raise ValueError(f'JPEG is {len(data)} bytes, exceeds {PAD_LENGTH}; no silent truncation.')
    tokens = np.full(PAD_LENGTH, -1, dtype=np.int16)
    tokens[:len(data)] = data
    return tokens, len(data)


def split_indices(seed, train_samples, val_samples, test_samples):
    if not (1 <= train_samples <= 50000 and 1 <= val_samples <= 10000 and 1 <= test_samples <= 10000):
        raise ValueError('Use train_samples 1..50000, val_samples 1..10000, test_samples 1..10000')
    # Validation always belongs to a reserved pool, even when train_samples changes.
    order = np.random.default_rng(seed).permutation(60000)
    test_order = np.random.default_rng(seed + 1).permutation(10000)
    return order[:train_samples], order[50000:50000 + val_samples], test_order[:test_samples]


class ByteMNIST(Dataset):
    def __init__(self, indices, train=True):
        self.indices = np.asarray(indices, dtype=np.int64)
        all_images, all_labels = read_mnist(train)
        self.images = all_images[self.indices]
        self.labels = all_labels[self.indices]
        key = hashlib.sha256(self.indices.tobytes() + ENCODING.encode()).hexdigest()[:16]
        cache = ROOT / 'data/cache' / f'{"train" if train else "test"}_{key}.npz'
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            with np.load(cache, allow_pickle=False) as f:
                self.tokens, self.lengths = f['tokens'], f['lengths']
        else:
            encoded = [image_to_bytes(Image.fromarray(img)) for img in self.images]
            self.tokens = np.stack([x[0] for x in encoded])
            self.lengths = np.array([x[1] for x in encoded])
            np.savez_compressed(cache, tokens=self.tokens, lengths=self.lengths)
        print(f'[DATA] {"train pool" if train else "official test"}: {len(self)} images; JPEG bytes {self.lengths.min()}..{self.lengths.max()}', flush=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return torch.from_numpy(self.tokens[index].astype(np.int64)), int(self.labels[index])
