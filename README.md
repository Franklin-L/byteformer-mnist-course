# 字节域语义内容理解：码流图像分类

本课程使用 ByteFormer 预训练模型完成 MNIST 码流图像分类。学生需要完成模型微调、训练/验证/测试、参数对比、单图预测和结果分析。课程另提供损坏码流测试集，供加分实验使用。

| 任课人员 | 姓名 | 邮箱 |
| --- | --- | --- |
| 教师 | 吴科君 | kjwu@hust.edu.cn |
| 助教 | 李方成 | lifangcheng2002@163.com |

课程材料：

- [课程 PPT](docs/ByteFormer_MNIST_零基础实验课.pptx) / [PDF 预览](docs/ByteFormer_MNIST_零基础实验课.pdf)
- [Kaggle 课程 Notebook](course_kaggle.ipynb)
- [Word 实验报告模板](docs/学生实验报告模板.docx) / [Markdown 模板](docs/student_report_template.md)
- [课程仓库](https://github.com/Franklin-L/byteformer-mnist-course)

## 任务与数据划分

MNIST 图像先编码为 JPEG，再读取为 0–255 的字节序列。模型使用公开的 ByteFormer Tiny 预训练权重，并将分类头调整为 10 类。

| 数据 | 数量 | 每类数量 | 用途 |
| --- | ---: | ---: | --- |
| 训练集 | 5,000 | 500 | 更新模型参数 |
| 验证集 | 1,000 | 100 | 每轮评估并选择最佳模型 |
| 测试集 | 1,000 | 100 | 模型确定后独立评估 |

训练文件还包含原图、±10° 旋转和 ±2 像素水平平移形成的有效 JPEG 视图。验证集和测试集不使用这些变换。固定数据位于 `data/course_1of10/`。

## Kaggle 免费 GPU

课前可参考：

- [知乎：Kaggle GPU 资源使用教程](https://zhuanlan.zhihu.com/p/18209757723)
- [CSDN：Kaggle 平台使用指导](https://blog.csdn.net/yyyyyybw/article/details/148336854)
- [Kaggle 官方 Notebook 文档](https://www.kaggle.com/docs/notebooks)

下载并导入 [course_kaggle.ipynb](course_kaggle.ipynb)，在设置中打开 GPU 和 Internet，然后从上到下运行单元格。看到 `GPU available: True` 后继续。

## 准备代码、数据和权重

Kaggle 单元格：

```python
!git clone https://github.com/Franklin-L/byteformer-mnist-course.git
%cd /kaggle/working/byteformer-mnist-course
!python -m pip install -r requirements.txt
!python prepare.py
```

AutoDL 或普通 Linux 终端去掉命令前的 `!`，并使用 `cd` 进入仓库目录。`prepare.py` 会检查 MNIST 文件并下载、校验 ByteFormer 预训练权重。

## 本地电脑安装环境

如果学生使用自己的 NVIDIA GPU 电脑，推荐 Python 3.9–3.12，在项目目录运行：

```bash
python -m pip install -r requirements-local.txt
python prepare.py
```

`requirements-local.txt` 会安装 CUDA 12.1 版 PyTorch 和课程依赖。电脑需要有可用的 NVIDIA 显卡驱动。Kaggle 和 AutoDL 的 PyTorch 镜像已经安装 PyTorch，仍使用较轻量的 `requirements.txt`。

## 基础训练

训练轮数和 batch size 由学生设置。batch size 可参考 32；显存不足时可减小。

```bash
read -p "请输入训练轮数：" EPOCHS
read -p "请输入 batch size（参考32）：" BATCH_SIZE
python train_course_subset.py \
  --method clean \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --clean-augmentations \
  --output outputs/course_clean
```

| 参数 | 说明 |
| --- | --- |
| `--epochs` | 训练轮数，自行设置 |
| `--batch-size` | 每批样本数，自行设置，参考 32 |
| `--lr` | 主干学习率，默认 `1e-4`；分类头使用 10 倍学习率 |
| `--clean-augmentations` | 从原图、轻微旋转和平移视图中随机取一个训练输入 |
| `--output` | 独立输出目录；目录已有文件时需改用新名称 |

训练完成后生成：

```text
outputs/course_clean/
├── best.pt
├── metrics.json
├── history.csv
└── curves.png
```

`best.pt` 根据验证集结果选择。`history.csv` 保存每轮训练与验证结果；`metrics.json` 保存参数、最佳轮次、运行时间和环境信息。

## 独立测试与单图预测

```bash
python evaluate_course_corruption.py \
  --checkpoint outputs/course_clean/best.pt \
  --output outputs/course_clean_eval

python predict.py \
  --checkpoint outputs/course_clean/best.pt \
  --index 0
```

评估脚本同时给出 `Clean`、`Medium-Flip`、`Medium-Loss` 和 `Medium-Mixed` 四类结果。基础任务报告 `Clean` 测试准确率；三类损坏结果用于加分项。

`predict.py --index` 使用官方 MNIST 测试索引，可填写 0–9999。也可以使用黑底白字的自制图片：

```bash
python predict.py --checkpoint outputs/course_clean/best.pt --image assets/example_digit.png
```

## 参数对比

自行选择第二组轮数和 batch size，建议一次只改变一个参数：

```bash
python train_course_subset.py \
  --method clean \
  --epochs "$COMPARISON_EPOCHS" \
  --batch-size "$COMPARISON_BATCH_SIZE" \
  --clean-augmentations \
  --output outputs/comparison
```

比较两次实验的验证准确率、最佳轮次、训练时间和学习曲线，并说明参数变化带来的收益与代价。

## 加分项：损坏码流分类

课程提供与干净测试集一一对应的损坏测试数据：

- `Medium-Flip`：选中一个字节，随机翻转其中一位，码流长度不变。
- `Medium-Loss`：删除选中的字节，后续字节前移。
- `Medium-Mixed`：对每个样本随机选择 bit flip 或 byte loss。

可以从以下方向完成加分实验：

1. 比较 Clean、Flip、Loss、Mixed，分析哪类损坏影响更大。
2. 在训练阶段加入随机 bit flip 和 byte loss 增强。
3. 对同一图像的干净码流与损坏码流加入预测一致性或特征对齐约束。

仓库已提供损坏增强训练入口：

```bash
python train_course_subset.py \
  --method augmentation \
  --epochs "$BONUS_EPOCHS" \
  --batch-size "$BONUS_BATCH_SIZE" \
  --clean-augmentations \
  --output outputs/course_corruption_aug

python evaluate_course_corruption.py \
  --checkpoint outputs/course_corruption_aug/best.pt \
  --output outputs/course_corruption_aug_eval
```

还可以将 `--method augmentation` 改为 `--method consistency`。背景与方法资料见 [加分项参考资料](research/04_bonus_corrupted_bitstream_references.md)，其中包括 CBSU-ALLM、BRACE、BSCV、ByteAction 等工作。

## AutoDL 备用路线

登录 [AutoDL](https://www.autodl.com/)，租用一张 GPU，选择 PyTorch 镜像并打开 JupyterLab Terminal。价格与可用资源以平台当前页面为准。参考 [AutoDL 官方快速开始](https://www.autodl.com/docs/quick_start/)。

```bash
cd /root/autodl-tmp
git clone https://github.com/Franklin-L/byteformer-mnist-course.git
cd byteformer-mnist-course
python -m pip install -r requirements.txt
python prepare.py
```

之后运行上面的基础训练、评估、预测和参数对比命令。实验结束后先下载 `outputs/` 中的结果，再回 AutoDL 控制台关闭实例。

## 常见问题

| 现象 | 处理方法 |
| --- | --- |
| `GPU available: False` | 检查 Kaggle 加速器设置，或改用 AutoDL |
| `No module named ...` | 在课程目录重新安装 `requirements.txt` |
| `CUDA out of memory` | 减小 batch size，如从 32 改为 16 或 8 |
| 输出目录已有文件 | 使用新的 `--output` 目录，保留已有结果 |
| 找不到 `best.pt` | 检查训练是否完成，以及 checkpoint 路径是否对应输出目录 |
| 下载权重失败 | 检查 Internet 后重新执行 `python prepare.py` |

## 提交内容

1. 实验报告，写明环境、轮数、batch size、命令和数据划分。
2. 基础训练与参数对比的 `metrics.json`、`history.csv` 和 `curves.png`。
3. 独立测试结果、单图预测或典型错例。
4. 对训练曲线、验证结果、测试结果和参数对比的分析。

Notebook 最后一格会把 JSON、CSV、PNG 和预测数组打包为 `byteformer_mnist_results.zip`。大型 checkpoint 不放入压缩包。

## 参考文献

- ByteFormer：[论文](https://arxiv.org/abs/2306.00238) / [Apple CoreNet](https://github.com/apple/corenet/tree/main/projects/byteformer)
- MEGABYTE：[论文](https://arxiv.org/abs/2305.07185)
- MambaByte：[论文](https://arxiv.org/abs/2401.13660)
- bGPT：[论文](https://arxiv.org/abs/2402.19155)
- mBLM：[论文](https://arxiv.org/abs/2502.14553)
- [研究背景与课件资料](research/README.md)
