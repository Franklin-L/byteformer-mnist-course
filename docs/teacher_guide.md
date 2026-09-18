# 教师指南：码流图像分类

教师：吴科君　　助教：李方成

## 实验目标

基于 ByteFormer 预训练模型完成 MNIST 分类，理解训练、验证与测试的分工，学习参数设置、学习曲线分析和错例分析。

模型采用 ImageNet JPEG 预训练 ByteFormer Tiny，保留 12 层 Transformer，将分类头调整为 10 类。MNIST 图像由脚本编码为 JPEG 文件字节后输入模型。

## 课前准备

1. 检查学生是否能够访问课程仓库、下载 Notebook 并开启 GPU。课件中给出 Kaggle 使用教程；AutoDL 作为备用平台。
2. 在授课环境运行 `python prepare.py`，准备数据与预训练权重。
3. 设置训练轮数与 batch size，完成一次基线训练，检查曲线、指标和预测图。
4. 更改一个参数完成对比实验，结果保存到 `outputs/comparison`。
5. 准备课程资源包、演示模型和报告模板。每位学生使用独立的课程目录与输出目录。

## 实验命令

学生自行设置训练轮数和 batch size，batch size 可参考 32。Kaggle 使用 Notebook 输入参数；下列命令用于终端。

### 基线命令

训练轮数和 batch size 均由学生自行设置，不固定数值。Notebook 使用 `input()` 接收学生输入；以下为 AutoDL/Linux 终端命令：

```bash
python prepare.py
read -p "请输入训练轮数：" EPOCHS
read -p "请输入 batch size（参考32）：" BATCH_SIZE
python train_course_subset.py --method clean --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --clean-augmentations --output outputs/course_clean
python evaluate_course_corruption.py --checkpoint outputs/course_clean/best.pt --output outputs/course_clean_eval
python predict.py --checkpoint outputs/course_clean/best.pt --index 0
```

### 自选对比命令

```bash
read -p "请输入对比实验的训练轮数：" COMPARISON_EPOCHS
read -p "请输入对比实验的 batch size：" COMPARISON_BATCH_SIZE
python train_course_subset.py --method clean --epochs "$COMPARISON_EPOCHS" --batch-size "$COMPARISON_BATCH_SIZE" --clean-augmentations --output outputs/comparison
```

## 课堂安排

可根据学生进度和设备运行情况调整时间。

| 阶段 | 建议时间 | 教师引导 | 学生操作 |
| --- | --- | --- | --- |
| 任务和结果预览 | 10 分钟 | 展示输入、预测、课程交付物 | 判断这是一个 10 类分类任务 |
| 环境与数据检查 | 15 分钟 | 演示终端、目录、GPU 检查 | 安装环境，运行 `prepare.py` |
| 运行基线 | 20 分钟 | 解释 epoch、batch、loss、accuracy | 按自选轮数训练，记录日志 |
| 查看输出 | 15 分钟 | 区分训练、验证、测试 | 打开曲线，读取指标，单图预测 |
| 修改一个参数 | 20 分钟 | 说明公平比较与验证集作用 | 自选另一设置，填写对照表 |
| 错例与报告 | 10 分钟 | 讨论容易混淆的数字 | 找错例，完成反思和提交 |

训练等待期间可讲解数据划分、参数含义和结果文件。环境安装与数据下载可提前安排为课前任务。

## 教学要点

- ByteFormer 输入为文件字节序列；MNIST 图像转换由脚本完成。
- 使用固定的平衡划分：5,000 张训练图、1,000 张验证图和 1,000 张测试图，每类数量相同。
- 训练集更新参数，验证集选择 `best.pt`，模型确定后在测试集上评估。
- 对比实验从同一预训练权重开始，保持数据划分、随机种子和学习率策略一致，建议每次调整一个参数。
- 结合训练损失与验证准确率讨论参数效果；错例分析应描述图像特征、预测结果和可能原因。

## 报告与加分

完成 MNIST 规定任务并提交实验报告，即达标及格。报告包含环境、参数、曲线、评估结果、一次参数对比和错例分析。

加分项使用课程提供的损坏测试集。学生比较 bit flip 和 byte loss 的影响，并可尝试损坏增强、预测一致性或特征对齐，提交三类测试结果与分析。

## 课程材料

- 19 页课件由教师单独保存和提交；仓库提供对应的课程代码与实验材料。
- [课程 Notebook](../course_kaggle.ipynb) / [操作指南](../README.md)
- [实验报告模板](学生实验报告模板.docx)
- [演示运行记录](run_record.md)

Notebook 最后生成 `/kaggle/working/byteformer_mnist_results.zip`。学生下载结果包并填写报告；如需后续预测，另存 `best.pt`。AutoDL 的启动、文件下载和关机操作见 README。
