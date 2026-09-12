"""Build a portable class resource pack, including verified pretrained weights.

The recipient still needs an installed Python/PyTorch environment. This pack
does not contain third-party Python wheels or claim fully offline installation.
Run from anywhere after the course files have been added to Git.
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'ByteFormer_MNIST_课堂资源包_含数据与权重.zip')
    args = parser.parse_args()
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    required = [ROOT / 'checkpoints/imagenet_jpeg_q100_k8_w128.pt', ROOT / 'outputs/baseline/best.pt']
    for p in required:
        if not p.is_file(): raise FileNotFoundError(f'Missing {p}; run preparation and the baseline first.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'byteformer-mnist-course/'
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for name in files:
            if name and (ROOT / name).is_file(): archive.write(ROOT / name, prefix + name)
        archive.write(required[0], prefix + 'checkpoints/' + required[0].name)
        archive.write(required[1], prefix + 'teacher_checkpoint/best.pt')
        archive.writestr(prefix + 'teacher_checkpoint/README.txt',
            '教师参考模型：3轮，6000训练/1000验证，官方10000测试准确率88.72%。\n'
            '仅用于演示或恢复；学生应自己运行train.py完成微调。\n'
            '已有Python/PyTorch环境后，在课程根目录执行：\n'
            'python prepare.py\n'
            'python evaluate.py --checkpoint teacher_checkpoint/best.pt --output outputs/teacher_reference\n'
            '此资源包含数据和权重，不含Python安装包。安装依赖仍可能需要网络。\n')
    print(f'{args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)')


if __name__ == '__main__': main()
