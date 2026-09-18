"""Deterministic JPEG-byte corruption used by the course extension experiment.

The protocol follows the segment corruption process used in the local
ByteAction manuscript.  A stream is split into segments of S bytes.  Each
segment is selected with probability P.  A selected segment uses bit flip with
probability p_f and byte loss otherwise; q controls the per-byte corruption
probability inside the selected segment.
"""

from dataclasses import asdict, dataclass
import io
import warnings

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CorruptionParams:
    segment_size: int
    segment_probability: float
    flip_probability: float
    byte_probability: float

    @property
    def expected_corruption_ratio(self):
        return self.segment_probability * self.byte_probability

    def to_dict(self):
        result = asdict(self)
        result['expected_corruption_ratio'] = self.expected_corruption_ratio
        return result


MEDIUM_SCENARIOS = {
    'Medium-Flip': CorruptionParams(64, 0.55, 1.0, 0.55),
    'Medium-Loss': CorruptionParams(64, 0.55, 0.0, 0.55),
}


def corrupt_bytes(data, params, seed):
    """Return corrupted bytes and exact operation counts for one stream."""
    rng = np.random.default_rng(seed)
    source = bytearray(data)
    flipped = 0
    deleted = set()
    selected_segments = 0
    size = params.segment_size
    # Include a final short segment. MNIST JPEG files are short enough that
    # discarding it would noticeably reduce the requested corruption strength.
    for start in range(0, len(source), size):
        end = min(start + size, len(source))
        if rng.random() >= params.segment_probability:
            continue
        selected_segments += 1
        use_flip = rng.random() < params.flip_probability
        for index in range(start, end):
            if rng.random() >= params.byte_probability:
                continue
            if use_flip:
                source[index] ^= 1 << int(rng.integers(0, 8))
                flipped += 1
            else:
                deleted.add(index)
    if deleted:
        source = bytearray(value for index, value in enumerate(source)
                           if index not in deleted)
    stats = {
        'input_bytes': len(data),
        'output_bytes': len(source),
        'selected_segments': selected_segments,
        'flipped_bytes': flipped,
        'deleted_bytes': len(deleted),
        'changed_fraction': (flipped + len(deleted)) / max(1, len(data)),
    }
    return bytes(source), stats


def random_training_params(rng, strength):
    """Sample weak or strong corruption for online robust training."""
    if strength == 'weak':
        low, high = 0.05, 0.25
    elif strength == 'strong':
        low, high = 0.25, 0.55
    else:
        raise ValueError("strength must be 'weak' or 'strong'")
    return CorruptionParams(
        segment_size=int(rng.choice([32, 64, 128, 256])),
        segment_probability=float(rng.uniform(low, high)),
        flip_probability=float(rng.choice([0.0, 0.5, 1.0])),
        byte_probability=float(rng.uniform(low, high)),
    )


def can_decode(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if max(image.size) > 4096:
                    return False
                image.load()
        return True
    except (OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        return False
