# 演示运行记录

本页记录 MNIST 演示实验的配置与输出，供复现时核对。

## 数据与配置

- 训练：50,000 张；验证：1,000 张；其余 9,000 张验证池样本未使用。
- 轮数：8；batch size：32；随机种子：42。
- 主干学习率：0.0001；分类头学习率为主干的 10 倍。
- 第 4、6 轮结束后，学习率乘以 0.2。
- 最佳模型按验证准确率选择。

```bash
python train.py --epochs 8 --train-samples 50000 --val-samples 1000 --test-samples 1000 --batch-size 32 --lr 0.0001 --lr-milestones 4 6 --lr-gamma 0.2 --output outputs/course_baseline
python evaluate.py --checkpoint outputs/course_baseline/best.pt --test-samples 10000 --output outputs/course_full_test
```

首次运行在训练末尾评估了 1,000 张测试图，随后使用同一个最佳模型独立评估全部 10,000 张测试图。课程 Notebook 直接设置 `--test-samples 10000`。

## 结果

| 项目 | 记录 |
| --- | --- |
| 最佳验证准确率 | 97.30%，第 5 轮 |
| 完整测试准确率 | 97.07%，9,707 / 10,000 |
| 首次运行耗时 | 712.90 秒，包含当次 1,000 张测试 |
| 独立完整测试耗时 | 5.93 秒 |
| 训练与验证记录 | [metrics.json](../examples/course_baseline/metrics.json)、[history.csv](../examples/course_baseline/history.csv) |
| 完整测试记录 | [evaluation.json](../examples/course_full_test/evaluation.json) |

耗时不含依赖安装和网络下载。

## 运行环境

Linux、Python 3.9.25、PyTorch 2.3.0+cu121、NumPy 1.26.4、Pillow 11.3.0、matplotlib 3.9.4、requests 2.32.5；GPU 为 NVIDIA GeForce RTX 4090。依赖版本见 `requirements-reproduce.txt`。

CPU 流程检查使用 4 线程、1 轮、1,000 / 200 / 200 张训练 / 验证 / 测试图，测试准确率 20.5%，用时 65.70 秒。运行验证在本地完成，Kaggle 和 AutoDL 尚未登录实例运行。
