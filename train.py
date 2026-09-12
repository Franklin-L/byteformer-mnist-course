"""Fine-tune the complete pretrained ByteFormer Tiny on MNIST JPEG bytes."""
import argparse
import csv
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from byteformer_model import build_model
from data_utils import ByteMNIST, ENCODING, split_indices
from prepare import ROOT, WEIGHT_NAME, digest


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Enable GPU on Kaggle, or use --device cpu.')
    return torch.device(name)


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    total_loss, correct, count, preds, labels = 0.0, 0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite evaluation loss')
        prediction = logits.argmax(1)
        total_loss += loss.item() * len(y)
        correct += (prediction == y).sum().item()
        count += len(y)
        preds.extend(prediction.cpu().tolist())
        labels.extend(y.cpu().tolist())
    return {'loss': total_loss / count, 'accuracy': correct / count, 'correct': correct, 'samples': count}, np.array(preds), np.array(labels)


def write_plots(out, history, dataset, preds, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 140})
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
        for key, name, color in [('train', 'Train', '#1967b3'), ('val', 'Validation', '#da7640')]:
            axes[0].plot([r['epoch'] for r in history], [r[f'{key}_loss'] for r in history], 'o-', label=name, color=color)
            axes[1].plot([r['epoch'] for r in history], [100*r[f'{key}_accuracy'] for r in history], 'o-', label=name, color=color)
        for ax in axes:
            ax.set_xlabel('Epoch'); ax.set_xticks([r['epoch'] for r in history]); ax.legend(); ax.grid(alpha=.15)
        axes[0].set_ylabel('Cross-entropy loss'); axes[1].set_ylabel('Accuracy (%)')
        fig.savefig(out / 'curves.png'); plt.close(fig)
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    np.savetxt(out / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cm, cmap='Blues'); fig.colorbar(im, ax=ax, label='Count')
    ax.set(xlabel='Predicted digit', ylabel='True digit', xticks=range(10), yticks=range(10), title=f'Official MNIST test (n={len(labels)})')
    for i in range(10):
        for j in range(10): ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=8, color='white' if cm[i,j] > cm.max()*.5 else '#153149')
    fig.savefig(out / 'confusion_matrix.png'); plt.close(fig)
    # A fixed first 12, plus up to four actual mistakes: selection is disclosed.
    selected = list(range(min(12, len(labels))))
    selected += [int(i) for i in np.flatnonzero(preds != labels) if i not in selected][:4]
    fig, axes = plt.subplots(4, 4, figsize=(7, 7), constrained_layout=True)
    for ax in axes.flat: ax.axis('off')
    for ax, i in zip(axes.flat, selected):
        ax.imshow(dataset.images[i], cmap='gray')
        ax.set_title(f'True {labels[i]} | Pred {preds[i]}', fontsize=10, color='#15805a' if preds[i]==labels[i] else '#c63c3c')
    fig.suptitle('First 12 test examples + up to 4 mistakes', fontsize=13)
    fig.savefig(out / 'predictions.png'); plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--train-samples', type=int, default=50000)
    p.add_argument('--val-samples', type=int, default=1000)
    p.add_argument('--test-samples', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--lr-milestones', type=int, nargs='*', default=[4, 6],
                   help='Completed epochs after which learning rates decay; empty disables decay')
    p.add_argument('--lr-gamma', type=float, default=0.2,
                   help='Multiply both backbone and head learning rates at each milestone')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', type=Path, default=Path('outputs/baseline'))
    p.add_argument('--weights', type=Path, default=ROOT / 'checkpoints' / WEIGHT_NAME)
    return p


def main():
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        raise ValueError('epochs, batch-size, threads and lr must be positive')
    if args.lr_milestones != sorted(set(args.lr_milestones)) or any(e < 1 for e in args.lr_milestones):
        raise ValueError('lr-milestones must be unique, increasing positive epoch numbers')
    if not 0 < args.lr_gamma <= 1:
        raise ValueError('lr-gamma must be greater than 0 and at most 1')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists() or (out / 'best.pt').exists():
        raise FileExistsError(f'{out} already contains a run. Use a different --output to preserve your comparison.')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(args.device)
    print(f'[ENV] Python {platform.python_version()} | PyTorch {torch.__version__} | device={device}', flush=True)
    if device.type == 'cuda': print(f'[GPU] {torch.cuda.get_device_name(device)}', flush=True)
    if not args.weights.exists(): raise FileNotFoundError('Run python prepare.py first.')
    tr_idx, va_idx, te_idx = split_indices(args.seed, args.train_samples, args.val_samples, args.test_samples)
    assert not set(tr_idx) & set(va_idx)
    np.savez_compressed(out / 'split_indices.npz', train=tr_idx, validation=va_idx, test=te_idx)
    start = time.perf_counter()
    train_ds, val_ds = ByteMNIST(tr_idx), ByteMNIST(va_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.weights, num_classes=10).to(device)
    print(f'[MODEL] Full pretrained Tiny; {sum(p.numel() for p in model.parameters()):,} trainable parameters; new 10-class head', flush=True)
    # Separate head rate helps the randomly initialized classifier start learning.
    backbone = [p for name, p in model.named_parameters() if not name.startswith('classifier.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.lr}, {'params': model.classifier.parameters(), 'lr': args.lr * 10}], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    initial, _, _ = score(model, val_loader, device)
    print(f'[BEFORE] validation accuracy={initial["accuracy"]*100:.2f}% (new untrained 10-class head)', flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(encoding=ENCODING, corrected_masks=True, head_lr_multiplier=10)
    history, best_acc, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = correct = count = 0; epoch_start = time.perf_counter()
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; run model validation and inspect input bytes.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item()*len(y); correct += (logits.argmax(1)==y).sum().item(); count += len(y)
            if step == 1 or step % 50 == 0:
                print(f'[TRAIN] epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss={loss.item():.4f}', flush=True)
        val, _, _ = score(model, val_loader, device)
        row = {'epoch': epoch, 'train_loss': loss_sum/count, 'train_accuracy': correct/count, 'val_loss': val['loss'], 'val_accuracy': val['accuracy'], 'lr': optimizer.param_groups[0]['lr'], 'head_lr': optimizer.param_groups[1]['lr'], 'seconds': time.perf_counter()-epoch_start}
        history.append(row)
        with (out/'history.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(row), lineterminator='\n'); writer.writeheader(); writer.writerows(history)
        if val['accuracy'] > best_acc:
            best_acc, best_epoch = val['accuracy'], epoch
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({'model': cpu_state, 'config': config, 'best_epoch': best_epoch, 'best_val_accuracy': best_acc}, out/'best.pt')
        scheduler.step()
        print(f'[EPOCH {epoch}] train_acc={100*row["train_accuracy"]:.2f}% val_acc={100*val["accuracy"]:.2f}% lr={row["lr"]:.2g} time={row["seconds"]:.1f}s', flush=True)
    checkpoint = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    test_ds = ByteMNIST(te_idx, train=False)
    test, preds, labels = score(model, DataLoader(test_ds,batch_size=args.batch_size),device)
    elapsed = time.perf_counter()-start
    result={'created_utc':datetime.now(timezone.utc).isoformat(), 'config':config, 'initial_validation':initial, 'best_epoch':best_epoch, 'best_validation_accuracy':best_acc, 'test':test, 'elapsed_seconds':elapsed, 'history':history, 'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':str(device),'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),'numpy':np.__version__}, 'pretrained_sha256':digest(args.weights), 'split_sha256':digest(out/'split_indices.npz')}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'test_predictions.npz', indices=te_idx, predictions=preds, labels=labels)
    write_plots(out,history,test_ds,preds,labels)
    print(f'[DONE] best epoch={best_epoch}; test accuracy={100*test["accuracy"]:.2f}% ({test["correct"]}/{test["samples"]}); elapsed={elapsed:.1f}s',flush=True)
    print(f'[SAVED] {out.resolve()} / best.pt, metrics.json, history.csv, curves.png, predictions.png, confusion_matrix.png',flush=True)


if __name__ == '__main__': main()
