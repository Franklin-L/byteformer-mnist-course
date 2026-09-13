# 课程流程验证记录

本页记录固定 1/10 MNIST 数据上的完整流程验证。学生的训练轮数和 batch size 仍由自己设置，实际参数以各输出目录中的 `metrics.json` 为准。

## 数据

- 训练集：5,000 张，每类 500 张。
- 验证集：1,000 张，每类 100 张。
- 测试集：1,000 张，每类 100 张。
- 三类损坏测试与干净测试使用相同样本索引。

## 已验证命令

```bash
python prepare.py
python train_course_subset.py --method clean --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --clean-augmentations --output outputs/course_clean
python evaluate_course_corruption.py --checkpoint outputs/course_clean/best.pt --output outputs/course_clean_eval
python predict.py --checkpoint outputs/course_clean/best.pt --index 0
```

训练、验证、独立测试、单图预测、损坏增强和一致性训练流程均已完成验证。基础干净训练的测试准确率达到 95% 以上；损坏训练的详细对比见 [实验结果](../research/experiment_results.md)。

## 输出

每次训练保存 `best.pt`、`metrics.json`、`history.csv` 和 `curves.png`。评估保存 `evaluation.json`、`accuracy.csv`、`accuracy.png` 和预测数组。

运行时间与硬件、轮数和 batch size 有关，应以学生自己的记录为准。
