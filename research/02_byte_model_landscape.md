# 字节模型与码流语义理解工作梳理

## 1. 从字节序列建模到多媒体内容理解

字节模型把 `0–255` 的字节值作为基本符号，避免为每种文件格式单独设计像素、音频或文本输入接口。不同工作关注的目标并不完全相同：有的研究通用字节生成，有的研究长序列效率，有的直接处理图像或视频码流。

| 工作 | 主要输入与任务 | 结构特点 | 代码情况 | 与本课程的关系 |
| --- | --- | --- | --- | --- |
| ByteFormer | 图像、音频等文件字节；分类 | 字节嵌入、局部下采样、Transformer 编码器 | [Apple CoreNet](https://github.com/apple/corenet/tree/main/projects/byteformer) | 有公开代码和预训练权重，直接支持分类，本课程采用 |
| bGPT | 任意数字文件的字节；生成与多任务 | decoder-only Transformer，学习字节世界模型 | [公开仓库](https://github.com/sanderwood/bgpt) | 展示通用字节建模能力，模型规模和训练目标不适合零基础分类实验 |
| MEGABYTE | 长字节序列生成 | 全局块模型与局部字节模型分层建模 | [论文](https://arxiv.org/abs/2305.07185)，[社区实现](https://github.com/lucidrains/MEGABYTE-pytorch) | 说明长字节序列可通过多尺度结构降低计算量 |
| MambaByte | 无 token 的长字节序列建模 | 基于状态空间模型处理长上下文 | [公开仓库](https://github.com/jxiw/MambaByte) | 代表 Transformer 之外的长字节序列路线 |
| Byte Latent Transformer | 动态字节 patch 的语言建模 | 根据局部复杂度形成可变长度 patch | [公开仓库](https://github.com/facebookresearch/blt) | 代表动态分块和潜变量字节建模路线 |
| TransFace++ | JPEG 字节；人脸识别 | ByteFormer 类字节主干与人脸识别训练 | [论文](https://arxiv.org/abs/2308.10133) | 证明字节输入可以用于专门的视觉识别任务 |
| CBSU-ALLM | 受损图像码流；语义理解 | 自适应模态与大语言模型结合 | [论文 DOI](https://doi.org/10.1016/j.patcog.2026.114151)；未检索到同名公开代码仓库 | 图片码流受损后仍进行语义理解的背景工作 |
| Cibic | 受损 JPEG 字节；图像描述 | 字节编码器、对齐层、语言解码器 | [论文 DOI](https://doi.org/10.1016/j.patcog.2026.114238)；本地有稿件 | 从分类扩展到开放文本描述 |
| Bitstream Action Recognition is Byte Modeling | 受损图像码流；动作分类 | 从原始字节学习动作语义 | [论文](https://arxiv.org/abs/2608.15695)；未检索到同名公开代码仓库 | 本课程受损码流分类任务的直接背景 |
| ByteAction | 受损图像码流；动作分类 | ByteFormer 主干、码流增强、损坏一致性 | [论文](https://arxiv.org/abs/2608.22760)；本地有稿件 | 本课程 Medium 损坏参数和鲁棒训练思路的主要来源 |
| VUB | 视频原始码流；动作识别 | 分块并行建模、跨编码格式蒸馏 | 本地稿件，未作为公开课程代码发布 | 说明字节域语义理解可以从图像扩展到视频 |

开源状态以相应论文和项目页面为准。课程只要求学生运行 ByteFormer；其余模型用于建立研究背景，不要求安装。

## 2. ByteFormer 的输入和结构

ByteFormer 的输入是一维字节序列。文件中的每个字节先映射为向量，较短样本在末尾补齐，补齐位置由 padding mask 标记。随后模型通过卷积式局部下采样缩短序列，再送入 Transformer 编码器，最后将全局特征送到分类头。

```text
JPEG 文件字节
    ↓
字节值嵌入（0–255）与 padding mask
    ↓
局部卷积与序列下采样
    ↓
Transformer 编码器
    ↓
全局特征
    ↓
10 类分类头
```

本仓库使用 Apple 发布的 ByteFormer Tiny ImageNet JPEG 预训练参数，将最后的分类头改为 10 类，再用 MNIST JPEG 码流微调。

## 3. 本课程选择 ByteFormer 的理由

1. 输入与课程主题一致。模型直接读取 JPEG 文件字节，学生能看到“图像文件也是字节序列”。
2. 分类接口简单。更换分类头即可完成 MNIST 十分类，不需要语言模型和复杂解码过程。
3. 有公开实现和预训练权重。课程仓库可以给出从准备数据到训练、验证、测试的完整命令。
4. 模型规模适合教学。单张消费级 GPU 或 Kaggle GPU 能完成微调。
5. 易于扩展鲁棒性实验。同一模型可以直接读取 bit flip 或 byte loss 后的字节，用于比较干净训练、损坏增强和一致性训练。

## 4. 本课程与相关工作的衔接

课程主任务仍是 MNIST 码流图像分类。学生先理解 ByteFormer 的字节输入、训练集/验证集/测试集和分类准确率。扩展任务再引入受损码流：

- 参考 ByteAction 的分段损坏过程，构造 Medium-Flip、Medium-Loss 和 Medium-Mixed 测试集；
- 观察只在干净数据上训练的模型在受损码流上的性能下降；
- 加入在线损坏增强；
- 在两个损坏视图之间约束预测分布和全局特征的一致性。

这一顺序把通信中的码流错误、文件结构、字节模型和具体分类实验连接起来，同时保留学生可完成的代码修改与结果分析。
