# MNIST 1/10 平衡课程数据

本目录包含 5,000 张训练图、1,000 张验证图和 1,000 张干净测试图，以及同一批测试图的 Medium-Flip、Medium-Loss、Medium-Mixed 三种受损 JPEG 码流。

训练文件另含原图、轻微旋转和水平平移得到的五个正常 JPEG 训练视图。源样本仍为同一批 5,000 张；验证与测试不使用这些变换。

详细划分、损坏参数和校验值见：

- [manifest.json](manifest.json)
- [../../research/03_dataset_and_experiment_protocol.md](../../research/03_dataset_and_experiment_protocol.md)

所有样本均来自 MNIST 官方数据。仓库保存的是课程脚本生成的 JPEG 字节数组和原始 MNIST 索引。可运行下列命令重新生成：

```bash
python build_course_dataset.py --force
```
