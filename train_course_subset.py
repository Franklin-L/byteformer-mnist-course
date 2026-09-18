"""Train ByteFormer on the balanced 1/10 MNIST course dataset."""

import argparse
import csv
from datetime import datetime, timezone
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from build_course_dataset import DATA_DIR
from byteformer_model import build_model
from data_utils import ENCODING
from prepare import ROOT, WEIGHT_NAME, digest
from train import choose_device


class ArrayDataset(Dataset):
    def __init__(self, tokens, labels, random_variant=False):
        self.tokens = tokens
        self.labels = labels
        self.random_variant = random_variant

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        if tokens.ndim == 2:
            variant = int(torch.randint(tokens.shape[0], (1,))) if self.random_variant else 0
            tokens = tokens[variant]
        return torch.from_numpy(tokens.astype(np.int64)), int(self.labels[index])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    count = correct = 0
    loss_sum = 0.0
    for step, (tokens, labels) in enumerate(loader):
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        loss_sum += nn.functional.cross_entropy(logits, labels).item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return {'loss': loss_sum / count, 'accuracy': correct / count,
            'correct': correct, 'samples': count}


def save_curves(output, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(epochs, [row['train_loss'] for row in history], 'o-', label='Train')
    axes[0].plot(epochs, [row['val_clean_loss'] for row in history], 'o-', label='Clean val')
    axes[0].set_ylabel('Loss')
    axes[1].plot(epochs, [100 * row['train_accuracy'] for row in history], 'o-', label='Train')
    axes[1].plot(epochs, [100 * row['val_clean_accuracy'] for row in history], 'o-', label='Clean val')
    axes[1].set_ylabel('Accuracy (%)')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(output / 'curves.png', dpi=160)
    plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--method', choices=['clean'],
                        default='clean')
    result.add_argument('--epochs', type=int, default=12)
    result.add_argument('--batch-size', type=int, default=32)
    result.add_argument('--lr', type=float, default=0.0001)
    result.add_argument('--lr-milestones', nargs='*', type=int, default=[7, 10])
    result.add_argument('--lr-gamma', type=float, default=0.2)
    result.add_argument('--weight-decay', type=float, default=0.01)
    result.add_argument('--label-smoothing', type=float, default=0.0)
    result.add_argument('--dropout', type=float, default=0.0)
    result.add_argument('--freeze-backbone-epochs', type=int, default=0)
    result.add_argument('--clean-augmentations', action='store_true',
                        help='sample valid rotated/shifted JPEG views during training')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--device', default='auto')
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--dataset', type=Path,
                        default=DATA_DIR / 'mnist_clean_balanced.npz')
    result.add_argument('--weights', type=Path,
                        default=ROOT / 'checkpoints' / WEIGHT_NAME)
    result.add_argument('--output', type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        raise ValueError('epochs, batch-size, threads and lr must be positive')
    if args.weight_decay < 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError('weight-decay must be nonnegative; label-smoothing must be in [0, 1)')
    if not 0 <= args.dropout < 1 or args.freeze_backbone_epochs < 0:
        raise ValueError('dropout must be in [0, 1); freeze-backbone-epochs must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'{args.output} is not empty')
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(args.device)
    with np.load(args.dataset, allow_pickle=False) as data:
        key = 'train_aug_tokens' if args.clean_augmentations else 'train_tokens'
        if key not in data:
            raise KeyError(f'{key} is missing; rebuild the course dataset')
        train_tokens, train_labels = data[key], data['train_labels']
        val_tokens, val_labels = data['val_tokens'], data['val_labels']
    train_loader = DataLoader(ArrayDataset(
                                  train_tokens, train_labels,
                                  random_variant=args.clean_augmentations),
                              batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(ArrayDataset(val_tokens, val_labels),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.weights, num_classes=10).to(device)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = args.dropout
    backbone = [parameter for name, parameter in model.named_parameters()
                if not name.startswith('classifier.')]
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr * 10},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(train_samples=5000, val_samples=1000, test_samples=1000,
                  encoding=ENCODING, corrected_masks=True,
                  selection='clean validation accuracy')
    best_score, best_epoch, history = -1.0, 0, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        backbone_trainable = epoch > args.freeze_backbone_epochs
        for parameter in backbone:
            parameter.requires_grad_(backbone_trainable)
        model.train()
        loss_sum = correct = count = 0
        epoch_start = time.perf_counter()
        for step, (clean, labels) in enumerate(train_loader):
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.method == 'clean':
                logits = model(clean.to(device))
                loss = nn.functional.cross_entropy(
                    logits, labels, label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
        clean_val = evaluate(model, val_loader, device)
        selection = clean_val['accuracy']
        row = {
            'epoch': epoch,
            'train_loss': loss_sum / count,
            'train_accuracy': correct / count,
            'val_clean_loss': clean_val['loss'],
            'val_clean_accuracy': clean_val['accuracy'],
            'selection_score': selection,
            'lr': optimizer.param_groups[0]['lr'],
            'seconds': time.perf_counter() - epoch_start,
        }
        history.append(row)
        with (args.output / 'history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator='\n')
            writer.writeheader()
            writer.writerows(history)
        if selection > best_score:
            best_score, best_epoch = selection, epoch
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({'model': state, 'config': config, 'best_epoch': best_epoch,
                        'best_selection_score': best_score,
                        'best_clean_val_accuracy': clean_val['accuracy'],
                        },
                       args.output / 'best.pt')
        scheduler.step()
        print(f'[{args.method} epoch {epoch:02d}] train={100*row["train_accuracy"]:.2f}% '
              f'clean-val={100*clean_val["accuracy"]:.2f}% '
              f'time={row["seconds"]:.1f}s', flush=True)
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': config,
        'best_epoch': best_epoch,
        'best_selection_score': best_score,
        'elapsed_seconds': time.perf_counter() - start,
        'history': history,
        'pretrained_sha256': digest(args.weights),
        'dataset_sha256': digest(args.dataset),
        'environment': {
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'device': str(device),
            'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda'
                        else platform.processor(),
        },
    }
    (args.output / 'metrics.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    save_curves(args.output, history)
    print(f'[DONE] method={args.method} best_epoch={best_epoch} '
          f'elapsed={result["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
