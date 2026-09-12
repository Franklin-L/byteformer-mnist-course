"""Download and verify the four original MNIST archives and Apple weights."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}
WEIGHT_NAME = 'imagenet_jpeg_q100_k8_w128.pt'
WEIGHT_URL = 'https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/' + WEIGHT_NAME


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(urls, path, expected, algorithm='md5'):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path, algorithm) == expected:
        print(f'[OK] {path.relative_to(ROOT)}', flush=True)
        return
    errors = []
    for url in urls:
        tmp = path.with_suffix(path.suffix + '.part')
        try:
            print(f'[DOWNLOAD] {url}', flush=True)
            with requests.get(url, stream=True, timeout=(20, 90)) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            if digest(tmp, algorithm) != expected:
                raise ValueError('Checksum mismatch: downloaded file is incomplete or changed')
            tmp.replace(path)
            print(f'[OK] {path.relative_to(ROOT)}', flush=True)
            return
        except (requests.RequestException, OSError, ValueError) as e:
            errors.append(f'{url}: {e}')
            tmp.unlink(missing_ok=True)
    raise RuntimeError('Download failed. Enable Internet, or use the teacher offline pack.\n' + '\n'.join(errors))


def prepare(local_weights=None):
    for name, md5 in FILES.items():
        download([
            'https://storage.googleapis.com/cvdf-datasets/mnist/' + name,
            'https://ossci-datasets.s3.amazonaws.com/mnist/' + name,
        ], ROOT / 'data/MNIST/raw' / name, md5)
    manifest = json.loads((ROOT / 'assets/resource_manifest.json').read_text())
    dest = ROOT / 'checkpoints' / WEIGHT_NAME
    if local_weights and Path(local_weights).resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if digest(local_weights) != manifest['weights']['sha256']:
            raise ValueError('Local weights checksum mismatch')
        shutil.copy2(local_weights, dest)
    download([WEIGHT_URL], dest, manifest['weights']['sha256'], 'sha256')
    print('[READY] MNIST (60000 train / 10000 test) and pretrained ByteFormer are ready.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-weights', help='Optional teacher-provided official checkpoint')
    args = parser.parse_args()
    prepare(args.local_weights)
