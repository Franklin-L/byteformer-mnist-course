"""损坏码流增强训练模板（加分项）。

本文件只提供训练流程和接口提示，不是可以直接得到结果的完整答案。
请根据注释完成 TODO，再保存自己的 best.pt，并使用
evaluate_course_corruption.py 评估 Clean、Medium-Flip 和 Medium-Loss。
"""

from pathlib import Path


# TODO 1：读取 data/course_1of10/mnist_clean_balanced.npz。
# 需要使用：train_tokens、train_aug_tokens、train_labels、val_tokens、val_labels。
# 训练集可以加入你设计的损坏增强；验证集保持干净，不要加入随机损坏。


def augment_byte_batch(clean_tokens, rng):
    """根据自己的设计，把一个干净 batch 转成增强后的字节 batch。

    输入：
        clean_tokens：形状为 [batch, sequence_length] 的补齐字节序列。
        有效字节为 0–255，补齐位置为 -1。

    需要自己完成：
        1. 对每个样本找到有效字节长度；
        2. 自行决定使用 bit flip、byte loss，或两者按概率混合；
        3. 控制损坏位置和损坏强度；
        4. 保持输出形状不变，重新补齐或截断；
        5. 返回可以输入模型的整数张量。
    """
    # TODO: 编写你的码流增强逻辑。
    raise NotImplementedError("请完成 augment_byte_batch")


def train_one_epoch(model, loader, optimizer, device, rng):
    """训练一轮的伪代码。"""
    # TODO：
    # 1. model.train()
    # 2. 遍历 clean_tokens, labels
    # 3. augmented_tokens = augment_byte_batch(clean_tokens, rng)
    # 4. logits = model(augmented_tokens.to(device))
    # 5. loss = cross_entropy(logits, labels.to(device))
    # 6. 清空梯度、反向传播、更新参数
    # 7. 记录本轮平均 loss 和 accuracy
    raise NotImplementedError("请完成 train_one_epoch")


def validate_clean(model, loader, device):
    """验证伪代码：验证集保持干净，只用于选择 best.pt。"""
    # TODO：不要在这里随机生成损坏码流。
    # 计算 clean validation loss 和 accuracy，并返回结果。
    raise NotImplementedError("请完成 validate_clean")


def main():
    # TODO 1：解析 epochs、batch size、learning rate 和 output 目录。
    # TODO 2：加载 ByteFormer 预训练权重和课程数据。
    # TODO 3：创建 optimizer、DataLoader 和随机数生成器。
    # TODO 4：逐轮训练，并根据 clean validation accuracy 保存 best.pt。
    # TODO 5：保存 history.csv、metrics.json 和 curves.png。
    # TODO 6：训练结束后运行：
    # python evaluate_course_corruption.py \
    #   --checkpoint <你的输出目录>/best.pt \
    #   --output <你的输出目录>_eval
    raise NotImplementedError("请根据上面的伪代码完成加分训练流程")


if __name__ == "__main__":
    main()
