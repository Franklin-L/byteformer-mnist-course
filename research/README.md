# 字节域语义内容理解：课程前期资料

本目录保存课程课件制作前的背景材料、数据方案和实验记录。当前阶段先把技术路线和可复现实验做完整，PPT 后续再根据教师样例压缩整理。

## 材料索引

- [01_background_and_bitstreams.md](01_background_and_bitstreams.md)：多媒体通信、Gilbert–Elliott 信道、JPEG 与 H.264 码流结构。
- [02_byte_model_landscape.md](02_byte_model_landscape.md)：字节模型与码流语义理解工作的梳理，以及选择 ByteFormer 的理由。
- [03_dataset_and_experiment_protocol.md](03_dataset_and_experiment_protocol.md)：MNIST 1/10 平衡子集、Medium 损坏集和实验方案。
- `experiment_results.md`：训练和测试完成后写入实测结果。
- `figures/byteformer_model_arch.png`：ByteFormer 官方模型结构图。
- `figures/byteaction_overview.png`、`byteaction_framework.png`：ByteAction 的对比图和方法框架。
- `figures/vub_motivation.png`、`vub_framework.png`：视频像素流程与码流流程对比，以及 VUB 框架。

图中内容来自相应论文，课件使用时保留论文题名和引用。

## 本轮实验的核心问题

1. 只用干净 JPEG 码流微调 ByteFormer，能否完成 MNIST 分类？
2. 测试码流发生 bit flip 或 byte loss 后，分类准确率会下降多少？
3. 在训练中加入码流损坏增强，以及损坏前后预测与特征的一致性约束，能否提高鲁棒性？

数据采用固定索引、固定随机种子和固定损坏参数。训练集用于更新参数，验证集用于选择模型，测试集只在模型确定后用于最终评估。
