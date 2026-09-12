"""A dependency-light port of Apple CoreNet's complete ByteFormer Tiny.

Derived from Apple CoreNet, Copyright (C) 2024 Apple Inc. All Rights Reserved.
See THIRD_PARTY_NOTICES.md for the original license and exact source files.

The course default fixes upstream padding/shift-mask handling. Set
``corrected_masks=False`` to reproduce the upstream forward implementation;
this compatibility mode is for validation, not variable-length course data.
Input: integer [batch, bytes], values 0..255, right padding -1. Never normalize
bytes to 0..1 and never pass image patches. All 12 pretrained layers are kept.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _LayerNorm(nn.LayerNorm):
    """Use standard channel-last LN, or the exact upstream compatibility path."""

    def __init__(self, dim: int, corrected_masks: bool = True):
        super().__init__(dim, eps=1e-5)
        self.corrected_masks = corrected_masks

    def forward(self, x: Tensor) -> Tensor:
        if not self.corrected_masks and x.ndim > 2 and x.shape[1] == self.normalized_shape[0]:
            std, mean = torch.std_mean(x, dim=1, keepdim=True, unbiased=False)
            x = (x - mean) / (std + self.eps)
            shape = [1, self.normalized_shape[0]] + [1] * (x.ndim - 2)
            return torch.addcmul(self.bias.reshape(shape), x, self.weight.reshape(shape))
        return super().forward(x)


class _LearnablePositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, 50000, 192))
        nn.init.trunc_normal_(self.pos_embed, std=192**-0.5)

    def forward(self, length: int) -> Tensor:
        # The official checkpoint always takes a PREFIX, not interpolated short PE.
        return self.pos_embed[0, :, :length, :]


class _PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = _LearnablePositionalEmbedding()

    def forward(self, length: int) -> Tensor:
        return self.pos_embed(length)


def _pad(x: Tensor, mask: Tensor, window: int):
    extra = (-x.shape[1]) % window
    return F.pad(x, (0, 0, 0, extra)), F.pad(mask, (0, extra), value=float('-inf'))


class _MultiHeadAttention(nn.Module):
    def __init__(self, use_sdpa: bool):
        super().__init__()
        self.qkv_proj = nn.Linear(192, 576)
        self.out_proj = nn.Linear(192, 192)
        self.use_sdpa = use_sdpa

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv_proj(x).reshape(b, n, 3, 3, 64).transpose(1, 3).contiguous()
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.use_sdpa:
            attn_mask = None if mask is None else mask.unsqueeze(1).to(q.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            attn = torch.matmul(q * (64**-0.5), k.transpose(-1, -2))
            if mask is not None:
                attn = attn + mask.unsqueeze(1)
            attn = attn.float().softmax(dim=-1).to(attn.dtype)
            out = torch.matmul(attn, v)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, 192))


class _WindowedTransformerEncoder(nn.Module):
    def __init__(self, shift: int, corrected_masks: bool, use_sdpa: bool):
        super().__init__()
        self.window_size = 128
        self.window_shift = shift
        self.corrected_masks = corrected_masks
        self.pre_norm_mha = nn.Sequential(
            _LayerNorm(192, corrected_masks), _MultiHeadAttention(use_sdpa), nn.Dropout(0.0)
        )
        self.pre_norm_ffn = nn.Sequential(
            _LayerNorm(192, corrected_masks), nn.Linear(192, 768), nn.GELU(),
            nn.Dropout(0.0), nn.Linear(768, 192), nn.Dropout(0.0)
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
        b, n, c = x.shape
        w, shift = self.window_size, self.window_shift
        x, padding_mask = _pad(x, key_padding_mask, w)
        padded_n = x.shape[1]
        if shift:
            x = x.roll(-shift, dims=1)
            padding_mask = padding_mask.roll(-shift, dims=1)
        x = x.reshape(-1, w, c)
        total_mask = None
        if self.corrected_masks:
            shift_mask = x.new_zeros(padded_n // w, w, w)
            if shift:
                shift_mask[-1].fill_(float('-inf'))
                shift_mask[-1, :w-shift, :w-shift] = 0
                shift_mask[-1, w-shift:, w-shift:] = 0
            shift_mask = shift_mask.unsqueeze(0).expand(b, -1, -1, -1).reshape(-1, w, w)
            total_mask = padding_mask.reshape(-1, w).unsqueeze(1) + shift_mask
            # Entirely masked query rows are discarded later; avoid NaN softmax.
            fully_masked = torch.isneginf(total_mask).all(dim=-1, keepdim=True)
            total_mask = total_mask.masked_fill(fully_masked, 0)
        # Upstream currently computes but omits total_mask in its attention call.
        x = x + self.pre_norm_mha[2](self.pre_norm_mha[1](self.pre_norm_mha[0](x), total_mask))
        x = x + self.pre_norm_ffn(x)
        x = x.reshape(b, padded_n, c)
        if shift:
            x = x.roll(shift, dims=1)
        return x[:, :n]


class _TokenMerging(nn.Module):
    def __init__(self, corrected_masks: bool):
        super().__init__()
        self.reduction = nn.Linear(384, 192, bias=False)
        self.norm = _LayerNorm(192, corrected_masks)

    def forward(self, x: Tensor, mask: Tensor):
        x = x.masked_fill(torch.isneginf(mask).unsqueeze(-1), 0)
        x, mask = _pad(x, mask, 2)
        b, n, c = x.shape
        # Preserve Apple's interleaved channel/token order (not a plain reshape).
        x = x.unfold(1, 2, 2).reshape(b, n // 2, c * 2)
        x = self.norm(self.reduction(x))
        mask = mask.unfold(1, 2, 2).max(dim=-1).values
        return x, mask


class ByteFormerTiny(nn.Module):
    """12 layers, dim 192, 3 heads, FFN 768, k8/s4, windows 128/shift 64."""

    def __init__(self, num_classes: int = 10, corrected_masks: bool = True,
                 use_sdpa: bool = True):
        super().__init__()
        self.corrected_masks = corrected_masks
        self.embeddings = nn.Embedding(257, 192, padding_idx=256)
        self.conv_kernel_size = 8
        self.token_reduction_net = nn.Conv1d(192, 192, 8, stride=4, bias=False)
        self.max_num_tokens = 50000
        self.pos_embed = _PositionalEmbedding()
        self.emb_dropout = nn.Dropout(0.0)
        self.downsamplers = nn.ModuleDict({
            f'downsample_{i}': _TokenMerging(corrected_masks) for i in (0, 1, 3, 5, 7, 9)
        })
        self.transformer = nn.Sequential(*[
            _WindowedTransformerEncoder(0 if i % 2 == 0 else 64, corrected_masks, use_sdpa)
            for i in range(12)
        ])
        self.post_transformer_norm = _LayerNorm(192, corrected_masks)
        self.classifier = nn.Linear(192, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def get_backbone_inputs(self, tokens: Tensor):
        if tokens.ndim != 2 or tokens.shape[1] < 8:
            raise ValueError('Expected integer bytes with shape [B, L], L >= 8.')
        tokens = tokens.long()
        padding = tokens.eq(-1)
        # Do not mutate the caller's tensor (the upstream implementation does).
        x = self.embeddings(tokens.masked_fill(padding, 256))
        x = self.token_reduction_net(x.transpose(1, 2)).transpose(1, 2)
        if x.shape[1] > self.max_num_tokens:
            raise ValueError('Byte stream exceeds the pretrained position table.')
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        if self.corrected_masks:
            mask = mask.masked_fill(padding, float('-inf'))
            # Only full convolution windows count. This preserves results when
            # the same byte stream receives a longer amount of right padding.
            mask = mask.unfold(1, 8, 4).min(dim=-1).values
        else:
            # Exact upstream effect of mask[tokens == -1].fill_(-inf): no change.
            mask = mask.unfold(1, 8, 4).max(dim=-1).values
        x = self.emb_dropout(x + self.pos_embed(x.shape[1]))
        return x, mask

    def backbone_forward(self, x: Tensor, key_padding_mask: Tensor):
        for i, layer in enumerate(self.transformer):
            x = layer(x, key_padding_mask)
            name = f'downsample_{i}'
            if name in self.downsamplers:
                x, key_padding_mask = self.downsamplers[name](x, key_padding_mask)
        return self.post_transformer_norm(x), key_padding_mask

    def forward_features(self, tokens: Tensor) -> Tensor:
        x, mask = self.get_backbone_inputs(tokens)
        x, mask = self.backbone_forward(x, mask)
        x = x.masked_fill(torch.isneginf(mask).unsqueeze(-1), 0)
        count = mask.eq(0).sum(dim=1, keepdim=True).clamp_min(1)
        return x.sum(dim=1) / count

    def forward(self, tokens: Tensor) -> Tensor:
        return self.classifier(self.forward_features(tokens))


def build_model(checkpoint_path: Union[str, Path], num_classes: int = 10, *,
                corrected_masks: bool = True, use_sdpa: bool = True) -> ByteFormerTiny:
    """Load the Apple ImageNet checkpoint; replace only its 1000-class head.

    All backbone keys AND tensor shapes must match. Missing or unexpected keys
    raise an error; no silent partial loading, downloads, or random fallback.
    Use num_classes=1000 to preserve the ImageNet head for parity verification.
    A course checkpoint with an already matching head can also be loaded.
    """
    state = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must contain a PyTorch state dictionary.')
    if 'model' in state:
        state = state['model']
    elif 'state_dict' in state:
        state = state['state_dict']
    model = ByteFormerTiny(num_classes, corrected_masks, use_sdpa)
    expected = model.state_dict()
    head_keys = {'classifier.weight', 'classifier.bias'}
    if set(state) != set(expected):
        raise ValueError(f'Checkpoint keys mismatch: missing={sorted(set(expected)-set(state))}; '
                         f'unexpected={sorted(set(state)-set(expected))}')
    bad = {k: (tuple(state[k].shape), tuple(expected[k].shape))
           for k in expected if k not in head_keys and state[k].shape != expected[k].shape}
    if bad:
        raise ValueError(f'Backbone tensor shape mismatch: {bad}')
    old_classes = state['classifier.weight'].shape[0]
    if state['classifier.weight'].shape != (old_classes, 192) or state['classifier.bias'].shape != (old_classes,):
        raise ValueError('Invalid checkpoint classifier shape.')
    if old_classes != num_classes:
        if old_classes != 1000:
            raise ValueError(f'Only an ImageNet 1000-class head may be replaced, got {old_classes}.')
        state = dict(state)
        for key in head_keys:
            state[key] = expected[key]
    model.load_state_dict(state, strict=True)
    model.pretrained_report = {
        'backbone_tensors_loaded': len(expected) - 2,
        'classifier_replaced': old_classes != num_classes,
        'source_classes': old_classes,
        'target_classes': num_classes,
        'corrected_masks': corrected_masks,
        'parameters_total': sum(p.numel() for p in model.parameters()),
    }
    return model
