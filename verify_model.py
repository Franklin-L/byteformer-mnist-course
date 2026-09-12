"""Maintainer checks for the port; CoreNet is optional and not a course dependency.

python verify_model.py --checkpoint checkpoints/imagenet_jpeg_q100_k8_w128.pt
python verify_model.py --checkpoint ... --corenet-root /path/to/apple/corenet
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from byteformer_model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--corenet-root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('model_validation.json'))
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    report = {'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'pytorch': torch.__version__, 'device': 'cpu'}
    model = build_model(args.checkpoint).eval()
    report['loading'] = model.pretrained_report
    assert len(model.transformer) == 12
    tokens = torch.randint(0, 256, (2, 1100))
    original = tokens.clone()
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tokens)
        padded_logits = model(F.pad(tokens, (0, 436), value=-1))
    torch.testing.assert_close(tokens, original)
    torch.testing.assert_close(logits, padded_logits, rtol=2e-5, atol=2e-5)
    report['right_padding_max_abs_error'] = (logits - padded_logits).abs().max().item()
    report['two_forwards_seconds'] = time.perf_counter() - start
    # Test a batch containing different valid lengths, then each sample alone.
    mixed = F.pad(tokens[:1, :713], (0, 1536 - 713), value=-1)
    mixed = torch.cat((mixed, F.pad(tokens[1:], (0, 436), value=-1)))
    with torch.no_grad():
        mixed_logits = model(mixed)
        separate = torch.cat([model(tokens[:1, :713]), model(tokens[1:])])
    torch.testing.assert_close(mixed_logits, separate, rtol=2e-5, atol=2e-5)
    report['mixed_length_batch_max_abs_error'] = (mixed_logits-separate).abs().max().item()
    model.train()
    loss = F.cross_entropy(model(mixed), torch.tensor([3, 7]))
    loss.backward()
    for name in ('embeddings.weight', 'token_reduction_net.weight',
                 'transformer.0.pre_norm_mha.1.qkv_proj.weight',
                 'transformer.11.pre_norm_ffn.4.weight', 'classifier.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    report['backpropagation'] = 'finite nonzero gradients in embedding, conv, first/last layers, head'
    del model
    if args.corenet_root:
        sys.path.insert(0, str(args.corenet_root.resolve()))
        from corenet.options.opts import get_training_arguments
        from corenet.modeling.models.classification.byteformer import ByteFormer
        config = args.corenet_root / 'projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml'
        opts = get_training_arguments(args=['--common.config-file', str(config)])
        official = ByteFormer(opts).eval()
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        official.load_state_dict(state, strict=True)
        port = build_model(args.checkpoint, num_classes=1000,
                           corrected_masks=False, use_sdpa=False).eval()
        errors = {}
        for length in (1100, 1536):
            x = torch.randint(0, 256, (2, length))
            with torch.no_grad():
                a, b = official(x.clone()), port(x.clone())
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
            errors[str(length)] = (a - b).abs().max().item()
        report['official_compatibility_mode_max_abs_errors'] = errors
        report['official_parity_scope'] = ('corrected_masks=False/use_sdpa=False only; '
                                          'course mask fixes intentionally change upstream outputs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
