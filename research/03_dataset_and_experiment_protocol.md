# MNIST 1/10 平衡数据与受损码流实验方案

## 1. 数据划分

课程实验使用 MNIST 官方数据的固定平衡子集，总计 7,000 张，约为完整 70,000 张数据的十分之一。

| 划分 | 每类数量 | 10 类合计 | 用途 |
| --- | ---: | ---: | --- |
| 训练集 | 500 | 5,000 | 更新模型参数 |
| 验证集 | 100 | 1,000 | 每轮评估并选择最佳模型 |
| 干净测试集 | 100 | 1,000 | 模型确定后的最终评估 |

训练集和验证集来自 MNIST 官方 60,000 张训练部分，二者索引不重叠。测试集来自官方 10,000 张测试部分。每个数字 `0–9` 数量完全相同，固定随机种子为 `20260913`。

每张 28×28 灰度图先转为 RGB，再编码为 JPEG quality 100、4:4:4。JPEG 字节序列补齐到 2,048 个位置，补齐值为 `-1`。

训练文件还保存同一批 5,000 张图的五个正常 JPEG 视图：原图、旋转 `-10°/+10°`、水平平移 `-2/+2` 像素。它们不包含 bit flip 或 byte loss，用于小数据训练时抑制过拟合；验证和测试始终使用未变换样本。

数据文件：

- `data/course_1of10/mnist_clean_balanced.npz`：训练、验证和干净测试；
- `data/course_1of10/mnist_test_corrupted_medium.npz`：同一批 1,000 张测试样本的两种受损版本；
- `data/course_1of10/manifest.json`：数据数量、损坏统计、文件大小和 SHA-256。

重新生成：

```bash
python build_course_dataset.py --force
```

## 2. Medium 码流损坏

损坏任务参考 *Bitstream Action Recognition is Byte Modeling*；具体四元组和 Medium 参数依据本地 ByteAction 稿件。损坏过程使用 `(S, P, p_f, q)`：

- `S`：把 JPEG 字节序列切分成长度为 `S` 的不重叠段；
- `P`：每段被选中的概率；
- `p_f`：选中段采用 bit flip 的概率，未采用 bit flip 时执行 byte loss；
- `q`：选中段内每个字节发生操作的概率。

bit flip 对被选中的字节随机翻转一个 bit，保持码流长度；byte loss 删除被选中的字节，使后续字节前移。固定测试参数如下：

| 场景 | S | P | p_f | q | 期望受损比例 `P×q` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Medium-Flip | 64 | 0.55 | 1.0 | 0.55 | 30.25% |
| Medium-Loss | 64 | 0.55 | 0.0 | 0.55 | 30.25% |

当前 1,000 张测试样本的实测统计：

| 场景 | 平均受损字节比例 | Pillow 完整解码成功 |
| --- | ---: | ---: |
| Medium-Flip | 30.208% | 2/1,000（0.2%） |
| Medium-Loss | 30.255% | 4/1,000（0.4%） |

这里的解码率只用于说明标准像素流程受到破坏。ByteFormer 测试直接读取损坏后的字节，不需要 Pillow 成功恢复图像。

## 3. 对比方法

### A. Clean fine-tuning

只用干净训练码流微调 ByteFormer。最佳模型按干净验证集准确率选择。这是主任务基线，也是观察受损测试性能下降的参照。

### B. Corruption augmentation

训练时在线产生 weak 或 strong 损坏视图，并保留一部分干净输入。损坏类型在 bit flip 和 byte loss 之间采样。最佳模型按干净验证、Medium-Flip 验证和 Medium-Loss 验证的平均值选择。

## 4. 训练与评估命令

基础实验使用下面的干净训练命令。损坏增强属于课程加分项，当前课程仓库不提供可直接完成训练的命令，学生应按照 [`examples/bonus_augmentation_template.py`](../examples/bonus_augmentation_template.py) 中的伪代码自行实现。

训练轮数属于实验设置，可根据验证曲线继续调整，不写成课程固定要求。

```bash
python train_course_subset.py --method clean \
  --epochs 12 --batch-size 32 --clean-augmentations \
  --output outputs/subset_clean
```

基础模型只在干净测试集上评估：

```bash
python evaluate.py \
  --checkpoint outputs/subset_clean/best.pt \
  --output outputs/subset_clean_eval
```

完成加分项自己的训练脚本后，再使用两种固定 Medium 测试集评估：

```bash
python evaluate_course_corruption.py \
  --checkpoint outputs/subset_clean/best.pt \
  --output outputs/subset_clean_eval
```

其他方法只需替换 checkpoint 与输出目录。

## 5. 记录指标

每种方法报告四个互不混用的测试结果：

- Clean test accuracy；
- Medium-Flip accuracy；
- Medium-Loss accuracy。

同时记录：

- 干净准确率下降：`Clean - Corrupted`；
- 三种受损测试的平均准确率；
- 鲁棒方法相对 clean fine-tuning 的受损平均准确率提升；
- 最佳验证轮次、训练时间、数据与 checkpoint 的 SHA-256。

测试集只用于最终比较，不根据测试准确率反复选择训练参数。

## 6. 数据校验值

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `mnist_clean_balanced.npz` | 27,982,939 bytes | `526a1d75f69276de5146a918cdfc3b5208f881ca280b290a88d9a8271e97aeaa` |
| `mnist_test_corrupted_medium.npz` | 3,047,128 bytes | `38b4e1e797ee1888139253a87c771958adfd8745209e9f6fd061bef21f4399ef` |

这些值与 `manifest.json` 一致时，说明学生与教师使用的是同一份课程数据。
