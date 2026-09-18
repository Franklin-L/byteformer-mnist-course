"""Collect course corruption evaluations into a table, chart, and Markdown report."""

import argparse
import csv
import json
from pathlib import Path
import shutil


SCENARIOS = ['Clean', 'Medium-Flip', 'Medium-Loss']


def parse_run(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('run must use LABEL=EVALUATION_DIRECTORY')
    label, path = value.split('=', 1)
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=parse_run, required=True)
    parser.add_argument('--output', type=Path, default=Path('research/results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, directory in args.run:
        evaluation_path = directory / 'evaluation.json'
        report = json.loads(evaluation_path.read_text(encoding='utf-8'))
        results = report['results']
        row = {'method': label}
        for scenario in SCENARIOS:
            row[scenario] = results[scenario]['accuracy']
        corrupt_values = [row[name] for name in SCENARIOS[1:]]
        row['corrupted_mean'] = sum(corrupt_values) / len(corrupt_values)
        row['mean_drop_from_clean'] = row['Clean'] - row['corrupted_mean']
        row['checkpoint_sha256'] = report['checkpoint_sha256']
        row['best_epoch'] = report.get('best_epoch')
        rows.append(row)
        method_dir = args.output / label.lower().replace(' ', '_')
        method_dir.mkdir(parents=True, exist_ok=True)
        for filename in ['evaluation.json', 'accuracy.csv', 'accuracy.png']:
            source = directory / filename
            target = method_dir / filename
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        training_dir = Path(report['checkpoint']).parent
        for filename in ['metrics.json', 'history.csv', 'curves.png']:
            source = training_dir / filename
            if source.exists():
                shutil.copy2(source, method_dir / filename)

    baseline = rows[0]['corrupted_mean']
    for row in rows:
        row['corrupted_gain_vs_first'] = row['corrupted_mean'] - baseline
    fieldnames = list(rows[0])
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(
        json.dumps({'scenarios': SCENARIOS, 'methods': rows}, indent=2) + '\n',
        encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(SCENARIOS))
    width = 0.8 / len(rows)
    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for index, row in enumerate(rows):
        values = [100 * row[name] for name in SCENARIOS]
        axis.bar(x - 0.4 + width / 2 + index * width, values, width,
                 label=row['method'])
    axis.set_xticks(x, SCENARIOS)
    axis.set_ylim(0, 100)
    axis.set_ylabel('Top-1 accuracy (%)')
    axis.set_title('ByteFormer on clean and medium-corrupted MNIST bitstreams')
    axis.grid(axis='y', alpha=.2)
    axis.legend()
    fig.savefig(args.output / 'summary.png', dpi=180)
    plt.close(fig)

    lines = [
        '# 1/10 MNIST 码流损坏实验结果', '',
        '以下结果来自固定的 1,000 张平衡测试样本。两种受损测试集与干净测试集使用相同的样本索引。', '',
        '| 方法 | Clean | Medium-Flip | Medium-Loss | 受损平均 | 相对首行提升 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {100*row['Clean']:.1f}% | "
            f"{100*row['Medium-Flip']:.1f}% | {100*row['Medium-Loss']:.1f}% | "
            f"{100*row['corrupted_mean']:.1f}% | "
            f"{100*row['corrupted_gain_vs_first']:+.1f} 个百分点 |")
    strongest = max(rows, key=lambda item: item['corrupted_mean'])
    lines.extend([
        '',
        f"只用正常 JPEG 训练时，干净测试准确率为 {100*rows[0]['Clean']:.1f}%，"
        f"两种 Medium 损坏的平均准确率降到 {100*rows[0]['corrupted_mean']:.1f}%，"
        f"下降 {100*rows[0]['mean_drop_from_clean']:.1f} 个百分点。",
        '',
        f"{strongest['method']} 的受损平均准确率为 "
        f"{100*strongest['corrupted_mean']:.1f}%，相对纯干净训练提高 "
        f"{100*strongest['corrupted_gain_vs_first']:.1f} 个百分点；"
        f"干净测试准确率为 {100*strongest['Clean']:.1f}%。"
        '两种损坏中 byte loss 通常更困难，因为删除字节会改变后续序列位置。',
        '', '![准确率对比](results/summary.png)', '',
                  '每种方法的最终测试评估 JSON、逐场景 CSV 和准确率图保存在对应子目录。', ''])
    (args.output.parent / 'experiment_results.md').write_text(
        '\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
