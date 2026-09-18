# 加分项说明：损坏码流增强

基础任务只训练干净码流，并在干净测试集上完成分类。损坏码流增强是可选加分项，完成后再对三类测试集进行评估：

- **Clean**：干净测试码流；
- **Medium-Flip**：随机翻转一个字节中的 1 个 bit，码流长度不变；
- **Medium-Loss**：删除一个字节，后续字节前移。

## 1. 先看伪代码模板

打开项目中的 [`examples/bonus_augmentation_template.py`](../examples/bonus_augmentation_template.py)。模板只给出数据流、训练循环和保存结果的位置，关键的损坏生成逻辑留给学生完成。需要补上的部分包括：

1. 找到每条码流的有效字节范围；
2. 设计 bit flip、byte loss，或两者的随机组合；
3. 控制损坏位置和强度，并保持 batch 的张量形状；
4. 用增强后的训练样本训练模型；
5. 保持验证集干净，根据干净验证准确率保存 `best.pt`；
6. 保存训练曲线和实验参数，便于写入报告。

验证集不要随机损坏。这样才能比较不同增强设计在同一验证集上的表现。

## 2. 两类损坏测试数据

两类数据保存在同一个文件中，避免学生下载两份重复数据：

- [数据说明与字段](../data/course_1of10/README.md)
- [GitHub 中的 NPZ 文件](../data/course_1of10/mnist_test_corrupted_medium.npz)
- [直接下载 NPZ](https://raw.githubusercontent.com/Franklin-L/byteformer-mnist-course/main/data/course_1of10/mnist_test_corrupted_medium.npz)

文件中包含两个数组：

```python
import numpy as np

with np.load("data/course_1of10/mnist_test_corrupted_medium.npz") as data:
    medium_flip_tokens = data["medium_flip_tokens"]
    medium_loss_tokens = data["medium_loss_tokens"]
    labels = data["labels"]
```

两个数组与同一批测试标签一一对应，测试样本数量和顺序保持一致。

## 3. 评估自己的加分模型

完成模板中的训练流程并保存模型后，运行：

```bash
python evaluate_course_corruption.py \
  --checkpoint <你的加分模型目录>/best.pt \
  --output <你的加分模型目录>_eval
```

评估结果保存在输出目录中：

```text
evaluation.json   # Clean、Medium-Flip、Medium-Loss 的准确率
accuracy.csv      # 三类准确率表
accuracy.png      # 三类准确率柱状图
```

报告中记录增强方法、训练/验证划分、实际参数，以及三类测试准确率。加分项只需提交自己训练得到的 `best.pt` 和上述评估结果；基础任务仍按 README 中的基础提交清单完成。
