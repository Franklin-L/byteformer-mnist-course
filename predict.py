"""Predict one digit with a trained course best.pt.

--index is an original official MNIST test index (0..9999), not a subset row.
Custom --image files should contain a white digit on a black background. They
are resized by the same input function used in training, never auto-inverted.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from data_utils import image_to_bytes, read_mnist
from evaluate import load_course_checkpoint, positive_int
from train import choose_device


def test_index(value):
    index = int(value)
    if not 0 <= index < 10000:
        raise argparse.ArgumentTypeError('official test index must be between 0 and 9999')
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trained course best.pt')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--index', type=test_index, help='Original official test index; default 0')
    source.add_argument('--image', type=Path, help='Your black-background, white-digit image')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=positive_int, default=4)
    parser.add_argument('--output', type=Path, help='Output directory; default checkpoint directory')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, _, _ = load_course_checkpoint(args.checkpoint, device)
    if args.image is not None:
        with Image.open(args.image) as opened:
            image = opened.copy()
        truth = None
        source_name = str(args.image)
        print('[INPUT] Custom images should be white digits on a black background. '
              'No automatic inversion is applied.', flush=True)
    else:
        index = args.index if args.index is not None else 0
        images, labels = read_mnist(train=False)
        image, truth = Image.fromarray(images[index]), int(labels[index])
        source_name = f'Official MNIST test index {index}'
    tokens, byte_length = image_to_bytes(image)
    x = torch.from_numpy(tokens.astype('int64')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Model returned non-finite logits.')
        probabilities = logits.softmax(dim=1)[0].cpu()
    prediction = probabilities.argmax().item()
    confidence = probabilities[prediction].item()
    true_text = str(truth) if truth is not None else 'unknown (custom image)'
    print(f'[PREDICT] source={source_name}', flush=True)
    print(f'[RESULT] true={true_text} | pred={prediction} | confidence={confidence:.4f} '
          f'({100*confidence:.2f}%) | JPEG bytes={byte_length}', flush=True)
    out = args.output if args.output is not None else args.checkpoint.parent
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    display = image.convert('L').resize((28, 28), Image.Resampling.BILINEAR)
    axes[0].imshow(display, cmap='gray', vmin=0, vmax=255)
    axes[0].axis('off')
    axes[0].set_title(f'True: {truth if truth is not None else "unknown"} | Prediction: {prediction}')
    colors = ['#176ba0' if i == prediction else '#bed7e6' for i in range(10)]
    axes[1].bar(range(10), probabilities.numpy(), color=colors)
    axes[1].set(xlabel='Digit', ylabel='Softmax probability', xticks=range(10),
                ylim=(0, 1), title=f'Confidence: {100*confidence:.2f}%')
    axes[1].spines[['top', 'right']].set_visible(False)
    fig.savefig(out / 'prediction_single.png', dpi=150)
    plt.close(fig)
    print(f'[SAVED] {(out / "prediction_single.png").resolve()}', flush=True)


if __name__ == '__main__':
    main()
