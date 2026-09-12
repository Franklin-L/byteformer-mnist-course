"""Build the balanced clean and medium-corruption MNIST course datasets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from course_corruption import MEDIUM_SCENARIOS, can_decode, corrupt_bytes
from data_utils import ENCODING, PAD_LENGTH, image_to_bytes, read_mnist
from prepare import ROOT


DATA_DIR = ROOT / 'data/course_1of10'


def balanced_indices(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    result = []
    for label in range(10):
        candidates = np.flatnonzero(labels == label)
        result.extend(rng.choice(candidates, per_class, replace=False).tolist())
    return np.asarray(result, dtype=np.int64)


def encode(images):
    pairs = [image_to_bytes(Image.fromarray(image)) for image in images]
    return (np.stack([pair[0] for pair in pairs]).astype(np.int16),
            np.asarray([pair[1] for pair in pairs], dtype=np.int16))


def clean_training_views(images):
    """Return valid, uncorrupted MNIST views for small-data regularization."""
    operations = [
        lambda image: image,
        lambda image: image.rotate(-10, resample=Image.Resampling.BILINEAR, fillcolor=0),
        lambda image: image.rotate(10, resample=Image.Resampling.BILINEAR, fillcolor=0),
        lambda image: image.transform(
            image.size, Image.Transform.AFFINE, (1, 0, -2, 0, 1, 0),
            resample=Image.Resampling.BILINEAR, fillcolor=0),
        lambda image: image.transform(
            image.size, Image.Transform.AFFINE, (1, 0, 2, 0, 1, 0),
            resample=Image.Resampling.BILINEAR, fillcolor=0),
    ]
    encoded = []
    for array in images:
        image = Image.fromarray(array)
        encoded.append(np.stack([image_to_bytes(operation(image))[0]
                                 for operation in operations]))
    return np.stack(encoded).astype(np.int16)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = DATA_DIR / 'mnist_clean_balanced.npz'
    corrupt_path = DATA_DIR / 'mnist_test_corrupted_medium.npz'
    manifest_path = DATA_DIR / 'manifest.json'
    if not args.force and all(path.exists() for path in
                              (clean_path, corrupt_path, manifest_path)):
        print(f'[READY] {DATA_DIR}')
        return

    train_images, train_labels = read_mnist(train=True)
    test_images, test_labels = read_mnist(train=False)
    # 600 examples/class from the official training portion, then a fixed
    # per-class 500/100 train/validation split.
    train_val = balanced_indices(train_labels, 600, args.seed)
    train_parts, val_parts = [], []
    for label in range(10):
        class_indices = train_val[train_labels[train_val] == label]
        train_parts.append(class_indices[:500])
        val_parts.append(class_indices[500:])
    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    test_indices = balanced_indices(test_labels, 100, args.seed + 1)
    for indices, labels, expected in [
        (train_indices, train_labels, 500),
        (val_indices, train_labels, 100),
        (test_indices, test_labels, 100),
    ]:
        counts = np.bincount(labels[indices], minlength=10)
        assert np.all(counts == expected), counts
    assert not set(train_indices) & set(val_indices)

    train_tokens, train_lengths = encode(train_images[train_indices])
    train_aug_tokens = clean_training_views(train_images[train_indices])
    val_tokens, val_lengths = encode(train_images[val_indices])
    test_tokens, test_lengths = encode(test_images[test_indices])
    np.savez_compressed(
        clean_path,
        train_tokens=train_tokens, train_lengths=train_lengths,
        train_aug_tokens=train_aug_tokens,
        train_labels=train_labels[train_indices], train_indices=train_indices,
        val_tokens=val_tokens, val_lengths=val_lengths,
        val_labels=train_labels[val_indices], val_indices=val_indices,
        test_tokens=test_tokens, test_lengths=test_lengths,
        test_labels=test_labels[test_indices], test_indices=test_indices,
    )

    corrupted = {}
    scenario_stats = {}
    for scenario_index, (name, params) in enumerate(MEDIUM_SCENARIOS.items()):
        scenario_tokens = np.full_like(test_tokens, -1)
        lengths, changed, decoded = [], [], 0
        for row, (tokens, length) in enumerate(zip(test_tokens, test_lengths)):
            raw = bytes(tokens[:int(length)].astype(np.uint8))
            damaged, stats = corrupt_bytes(
                raw, params, args.seed + 100000 * (scenario_index + 1) + row)
            length_out = min(len(damaged), PAD_LENGTH)
            scenario_tokens[row, :length_out] = np.frombuffer(
                damaged[:length_out], dtype=np.uint8).astype(np.int16)
            lengths.append(length_out)
            changed.append(stats['changed_fraction'])
            decoded += int(can_decode(damaged))
        key = name.lower().replace('-', '_')
        corrupted[key + '_tokens'] = scenario_tokens
        corrupted[key + '_lengths'] = np.asarray(lengths, dtype=np.int16)
        scenario_stats[name] = {
            'parameters': params.to_dict(),
            'mean_changed_fraction': float(np.mean(changed)),
            'decode_success_count': decoded,
            'decode_rate': decoded / len(test_indices),
        }
    np.savez_compressed(
        corrupt_path, labels=test_labels[test_indices], indices=test_indices,
        **corrupted)

    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed,
        'encoding': ENCODING,
        'pad_length': PAD_LENGTH,
        'clean_dataset': {
            'train': 5000, 'validation': 1000, 'test': 1000,
            'per_class': {'train': 500, 'validation': 100, 'test': 100},
            'source': 'MNIST official 60000 training and 10000 test archives',
            'training_views': 'original, rotate -10/+10 degrees, horizontal shift -2/+2 pixels',
        },
        'corrupted_test_dataset': {
            'source_samples': 1000,
            'scenarios': scenario_stats,
        },
    }
    manifest['files'] = {
        clean_path.name: {'sha256': sha256(clean_path), 'bytes': clean_path.stat().st_size},
        corrupt_path.name: {'sha256': sha256(corrupt_path), 'bytes': corrupt_path.stat().st_size},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
