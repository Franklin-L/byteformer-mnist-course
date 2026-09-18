"""Evaluate a course-subset checkpoint on clean and medium-corrupted tests."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_course_dataset import DATA_DIR
from byteformer_model import ByteFormerTiny
from prepare import digest
from train import choose_device, score
from train_course_subset import ArrayDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--clean-data', type=Path,
                        default=DATA_DIR / 'mnist_clean_balanced.npz')
    parser.add_argument('--corrupt-data', type=Path,
                        default=DATA_DIR / 'mnist_test_corrupted_medium.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ByteFormerTiny(num_classes=10)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device).eval()
    with np.load(args.clean_data, allow_pickle=False) as clean:
        clean_tokens = clean['test_tokens']
        clean_labels = clean['test_labels']
        indices = clean['test_indices']
    with np.load(args.corrupt_data, allow_pickle=False) as corrupt:
        labels = corrupt['labels']
        assert np.array_equal(labels, clean_labels)
        scenarios = {
            'Clean': clean_tokens,
            'Medium-Flip': corrupt['medium_flip_tokens'],
            'Medium-Loss': corrupt['medium_loss_tokens'],
        }
    args.output.mkdir(parents=True, exist_ok=True)
    results, predictions = {}, {}
    start = time.perf_counter()
    for name, tokens in scenarios.items():
        value, preds, observed_labels = score(
            model,
            DataLoader(ArrayDataset(tokens, labels), batch_size=args.batch_size,
                       shuffle=False, num_workers=0),
            device)
        assert np.array_equal(observed_labels, labels)
        results[name] = value
        predictions[name] = preds
        print(f'[{name}] {100*value["accuracy"]:.2f}% '
              f'({value["correct"]}/{value["samples"]})', flush=True)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': digest(args.checkpoint),
        'clean_data_sha256': digest(args.clean_data),
        'corrupt_data_sha256': digest(args.corrupt_data),
        'best_epoch': checkpoint.get('best_epoch'),
        'training_method': checkpoint.get('config', {}).get('method'),
        'results': results,
        'evaluation_seconds': time.perf_counter() - start,
    }
    (args.output / 'evaluation.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'accuracy.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['scenario', 'accuracy', 'correct', 'samples'])
        for name, value in results.items():
            writer.writerow([name, value['accuracy'], value['correct'], value['samples']])
    np.savez_compressed(args.output / 'predictions.npz', indices=indices,
                        labels=labels, **{name.lower().replace('-', '_'): value
                                         for name, value in predictions.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(results)
    values = [100 * results[name]['accuracy'] for name in names]
    fig, axis = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    bars = axis.bar(names, values, color=['#176BA0', '#D9895B', '#C85A54'])
    axis.set_ylim(0, 100)
    axis.set_ylabel('Top-1 accuracy (%)')
    axis.set_title('Balanced MNIST test: clean and medium byte corruption')
    axis.grid(axis='y', alpha=.2)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width()/2, value + 1, f'{value:.1f}',
                  ha='center', va='bottom')
    fig.savefig(args.output / 'accuracy.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
