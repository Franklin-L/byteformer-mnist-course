"""Evaluate a saved course checkpoint on the official MNIST test split.

The checkpoint's test sample count is used by default. To evaluate all 10,000
official test images, use --test-samples 10000 --output outputs/full_test.
This command writes evaluation.json and never rewrites training metrics.json.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from byteformer_model import ByteFormerTiny
from data_utils import ByteMNIST, ENCODING, split_indices
from prepare import digest
from train import choose_device, score, write_plots


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def test_count(value):
    number = positive_int(value)
    if number > 10000:
        raise argparse.ArgumentTypeError('official MNIST test set has at most 10000 images')
    return number


def load_course_checkpoint(path, device):
    """Restore the trained model without requiring the original Apple file."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not {'config', 'model'} <= checkpoint.keys():
        raise ValueError('Expected a course best.pt containing config and model.')
    config = checkpoint['config']
    if config.get('encoding') != ENCODING:
        raise ValueError(f'Checkpoint input encoding {config.get("encoding")!r} '
                         f'does not match this course version {ENCODING!r}.')
    model = ByteFormerTiny(num_classes=10,
                          corrected_masks=config.get('corrected_masks', True))
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), config, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trained course best.pt')
    parser.add_argument('--test-samples', type=test_count, help='Default: checkpoint configuration')
    parser.add_argument('--batch-size', type=positive_int, help='Default: checkpoint configuration')
    parser.add_argument('--threads', type=positive_int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, help='Default: checkpoint directory')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, config, checkpoint = load_course_checkpoint(args.checkpoint, device)
    count = args.test_samples if args.test_samples is not None else int(config['test_samples'])
    batch_size = args.batch_size if args.batch_size is not None else int(config['batch_size'])
    if batch_size < 1:
        raise ValueError('Checkpoint batch_size must be positive.')
    _, _, indices = split_indices(int(config['seed']), int(config['train_samples']),
                                  int(config['val_samples']), count)
    dataset = ByteMNIST(indices, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    start = time.perf_counter()
    result, predictions, labels = score(model, loader, device)
    elapsed = time.perf_counter() - start
    out = args.output if args.output is not None else args.checkpoint.parent
    out.mkdir(parents=True, exist_ok=True)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': digest(args.checkpoint),
        'best_epoch': checkpoint.get('best_epoch'),
        'best_validation_accuracy': checkpoint.get('best_val_accuracy'),
        'dataset': 'official MNIST test split; no training/validation images',
        'selection': 'seed+1 permutation prefix; identical rule to train.py',
        'seed': int(config['seed']),
        'encoding': ENCODING,
        'corrected_masks': config.get('corrected_masks', True),
        'test': result,
        'evaluation_seconds': elapsed,
        'device': str(device),
        'batch_size': batch_size,
    }
    np.savez_compressed(out / 'evaluation_predictions.npz', indices=indices,
                        predictions=predictions, labels=labels)
    report['predictions_sha256'] = digest(out / 'evaluation_predictions.npz')
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    write_plots(out, [], dataset, predictions, labels)
    print(f'[EVALUATION] accuracy={100*result["accuracy"]:.2f}% '
          f'({result["correct"]}/{result["samples"]}); '
          f'loss={result["loss"]:.4f}; time={elapsed:.1f}s', flush=True)
    print(f'[SAVED] {out.resolve()} / evaluation.json, evaluation_predictions.npz, '
          'predictions.png, confusion_matrix.png', flush=True)


if __name__ == '__main__':
    main()
