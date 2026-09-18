# 加分项：损坏码流理解参考资料

基础实验只要求在平衡 MNIST 干净数据上完成 ByteFormer 微调和测试。学生继续测试受损码流、分析不同损坏类型或尝试鲁棒训练，可作为加分项。

## 1. 加分项可以完成的内容

1. 使用课程提供的 Medium-Flip、Medium-Loss 测试集评估干净模型。
2. 比较两种损坏下的准确率，说明 bit flip 与 byte loss 的区别。
3. 在训练阶段加入随机码流损坏增强。
4. 与干净训练基线比较干净准确率、受损平均准确率和提升幅度。

## 2. 主要参考工作

| 工作 | 对象 | 可供学生参考的内容 | 资料 |
| --- | --- | --- | --- |
| CBSU-ALLM | 受损图像码流语义理解 | 图像码流受损后绕过传统解码流程进行语义分析 | *Corrupted bitstream semantic understanding by adaptive-modal large language models*，[DOI](https://doi.org/10.1016/j.patcog.2026.114151) |
| BRACE | 受损图像码流动作识别 | RBCS 四参数损坏模型 `(S, P, p_f, q)`；bit flip、byte loss 与不同强度设置 | *Bitstream Action Recognition is Byte Modeling*，[论文](https://arxiv.org/abs/2608.15695) |
| BSCV | 受损视频恢复 | 视频码流中的连续片段丢失、损坏位置与长度设置 | *Bitstream-Corrupted Video Recovery: A Novel Benchmark Dataset and Method*，[论文](https://arxiv.org/abs/2309.13890)，[代码](https://github.com/LIUTIGHE/BSCV-Dataset) |
| ByteAction | 受损图像码流动作识别 | 双损坏视图、码流模式增强 | *ByteAction: Byte-space Action Recognition Foundation Model*，[论文](https://arxiv.org/abs/2608.22760) |
| Cibic | 受损图像码流描述 | 标准解码失败时直接从 JPEG 字节生成语义描述 | *Cibic: Pixel-free foundation model for robust corrupted image bitstream captioning*，[DOI](https://doi.org/10.1016/j.patcog.2026.114238) |
| VUB | 视频码流内容理解 | 长视频码流分块建模和跨编码格式知识蒸馏 | *Learn to Understand Video from Bitstream Modeling and Distillation*，本地稿件 |

这些工作用于帮助学生理解加分实验的背景和可选方法，不属于基础实验必须复现的模型。

可供课件使用的框架图：

- [ByteAction 方法框架](figures/byteaction_framework.png)
- [ByteAction 方法对比图](figures/byteaction_overview.png)
- [视频码流理解流程对比](figures/vub_motivation.png)
- [VUB 方法框架](figures/vub_framework.png)

## 3. 与课程数据的对应关系

课程损坏测试集采用 BRACE/ByteAction 中的分段损坏思路：

- Medium-Flip：只执行 bit flip；
- Medium-Loss：只执行 byte loss；
- 固定 `S=64`、`P=0.55`、`q=0.55`，期望受损字节比例约为 30.25%。

课程实测中，只用干净码流训练的 ByteFormer 在干净测试集达到 95.6%，在两种 Medium 损坏上的平均准确率可用于观察模型鲁棒性。加入损坏增强后，再比较干净准确率和两种受损准确率。完整结果见 [experiment_results.md](experiment_results.md)。
