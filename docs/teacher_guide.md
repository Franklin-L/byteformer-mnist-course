# 教师指南：ByteFormer 微调 MNIST 入门课

## 课程定位

面向没有深度学习实践基础的学生。目标是让学生亲手完成一次真实预训练模型的微调，知道训练、验证和测试如何分工，能读懂最基本的学习曲线，并完成一次自选训练参数的对比。

教学主线为 Kaggle 免费 GPU；AutoDL 租 GPU 作为备用，CPU 路线仅用于流程验证。Kaggle/AutoDL 的入口与下载步骤见 README。请在授课前用学生实际使用的账号、网络和设备完整运行一遍，记录真实耗时。不要把教师机器上的速度承诺为学生环境的固定速度。

本项目采用官方 ImageNet JPEG 预训练 ByteFormer Tiny 权重，教学代码保留 12 层 Transformer，替换为 10 类分类头，进行全参数微调。输入是由 MNIST RGB 图像编码得到的 JPEG 字节。课程实现修正了官方 padding mask 处理问题；请向学生明确这是一份教学适配，以及它与官方实现的差异。不要称其为论文 ImageNet 性能的复现。

## 课前准备

PPT 第 6 页提供知乎、CSDN 和 Kaggle 官方文档链接，GPU 开通步骤由学生课前参考教程完成，课堂从课程 Notebook 开始。中文教程已通过搜索检索到；因站点限制自动读取，本次未逐图核验，请课前用学生浏览器确认能否打开。教程中的平台额度与界面信息以 Kaggle 实际页面为准。

1. 确认课程仓库已经实际发布且普通学生账号可读取；如果仓库尚未发布，使用完整课程压缩包。README 与 PPT 中的仓库入口必须与真实状态一致。
2. 在干净环境中安装 `requirements.txt`，运行 `python prepare.py`，确保数据与官方权重通过校验。
3. 自行设置轮数，运行完整基线命令，保存终端日志、环境版本、代码提交号和全部结果文件。
4. 使用相同环境和另一自选轮数作对比，另存到 `outputs/comparison`。检查最佳模型由验证集选择，测试集只用于最终评估。
5. 保存演示运行的设备、样本数、epoch 数和实验日期，供复现核对；PPT 引导学生查看自己的曲线和指标，不展示教师准确率。
6. 预备可供课堂恢复的预训练文件和一份教师已完成的 checkpoint。学生使用教师 checkpoint 时必须标注来源，不能将教师训练记录写成自己的记录。
7. 检查学生账号的 GPU 与磁盘配额。多人同时训练时，按 GPU 容量分组或安排批次。

建议每位学生拥有自己的课程目录和虚拟环境，公共数据可由教师提前准备。不要让学生向同一 `outputs/baseline` 目录写文件。

### 基线命令

训练轮数和 batch size 均由学生自行设置，不固定数值。Notebook 使用 `input()` 接收学生输入；以下为 AutoDL/Linux 终端命令：

```bash
python prepare.py
read -p "请输入训练轮数：" EPOCHS
read -p "请输入 batch size（参考32）：" BATCH_SIZE
python train.py --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --output outputs/baseline
python evaluate.py --checkpoint outputs/baseline/best.pt --output outputs/baseline
python predict.py --checkpoint outputs/baseline/best.pt --index 0
```

### 自选对比命令

```bash
read -p "请输入对比实验的训练轮数：" COMPARISON_EPOCHS
read -p "请输入对比实验的 batch size：" COMPARISON_BATCH_SIZE
python train.py --epochs "$COMPARISON_EPOCHS" --batch-size "$COMPARISON_BATCH_SIZE" --output outputs/comparison
```

batch size可以给出32作为参考，轮数可参照教师实测记录，但均不是固定要求。根据训练与验证曲线讨论设置是否合适；不把教师的实测轮数当作学生必须遵守的任务要求。

### CPU 备用命令

```bash
python train.py --epochs 1 --train-samples 1000 --val-samples 200 --test-samples 200 --batch-size 16 --lr 0.0001 --device cpu --output outputs/cpu_smoke
```

CPU 路线没有准确率门槛；它仅证明程序能完成一轮训练、验证和测试。若 GPU 条件不足，请明确降低课堂完成范围，并单独安排完整对比；不要将冒烟测试等同于课程完整实验。

课堂要求学生完成流程、记录结果并解释参数变化，不把演示准确率作为成绩标准。学生可能取得更高或更低的结果；根据训练与验证表现讨论改进，测试集用于最终评估。

## 建议课堂流程

下表为教学活动的时间分配建议，不是训练速度保证。根据授课设备实测用时调整。

| 阶段 | 建议时间 | 教师引导 | 学生操作 |
| --- | --- | --- | --- |
| 任务和结果预览 | 10 分钟 | 展示输入、预测、课程交付物 | 判断这是一个 10 类分类任务 |
| 环境与数据检查 | 15 分钟 | 演示终端、目录、GPU 检查 | 安装环境，运行 `prepare.py` |
| 运行基线 | 20 分钟 | 解释 epoch、batch、loss、accuracy | 按自选轮数训练，记录日志 |
| 查看输出 | 15 分钟 | 区分训练、验证、测试 | 打开曲线，读取指标，单图预测 |
| 修改一个参数 | 20 分钟 | 说明公平比较与验证集作用 | 自选另一设置，填写对照表 |
| 错例与报告 | 10 分钟 | 讨论容易混淆的数字 | 找错例，完成反思和提交 |

训练等待期间可以讲解数据划分、查看教师预生成图、练习阅读命令参数。若实际训练时间超过课堂时间，请将依赖安装与数据下载安排为课前任务，并允许训练在课后继续。

## 平台操作与笔记本维护

16 页课件为 `docs/ByteFormer_MNIST_零基础实验课.pptx`，另有同名 PDF。课件保留课堂主线，完整操作细节在 README 和 `course_kaggle.ipynb`。学生实操入口为仓库中的 `course_kaggle.ipynb`。学生导入后开启 GPU 与 Internet，代码克隆到 `/kaggle/working/`，依次运行。轻量 `requirements.txt` 保留预装 PyTorch。不要让学生把 `requirements-reproduce.txt` 安装进托管环境。

Notebook 最后生成 `/kaggle/working/byteformer_mnist_results.zip`；提醒学生从文件/输出面板下载，报告另行填写。模型文件不包含在默认作业压缩包中，需要以后推理的学生应另存 `best.pt`。AutoDL 的 JupyterLab、终端、GitHub ZIP 备用获取与关闭实例步骤已写入 README，可直接用于课堂演示。

## 教学提示

- 第一次讲命令时，指出“当前目录”“一个空格分隔一个参数”“按回车运行”“终端报错需要保留”四件事即可。
- 先让学生看到数字图和预测，再解释字节输入。无需推导注意力公式即可完成本课。
- “模型曾在 ImageNet 上学习”不意味着数字任务无需训练；MNIST 与 ImageNet 的图像内容不同，分类头也不同。
- 说明 MNIST 训练、验证、测试来自不同样本；正式设置为 50,000 训练、1,000 验证、10,000 测试，另有 9,000 验证池样本未用。复现实验需要记录实际样本数。
- 强调 `best.pt` 按验证集选择。即使最后一轮训练损失更低，最佳模型也可能来自更早的轮次。
- 对比实验每次从同一官方预训练权重开始，保持数据划分、随机种子和学习率策略相同。建议轮数与batch size一次只改一项；两项都变时说明比较的局限。不要误以为是接着上次的最佳模型继续训练。
- 学生自行设置训练轮数和batch size，记录选择依据。学生可以报告两组最终测试结果，但不应再根据测试成绩反复改参数。
- 学生发现错例时，把“观察到的笔画特征”和“对错误原因的推测”分开。一次错例不足以证明模型内部学到了什么。

## 达标与拓展加分

完成 MNIST 规定任务并提交自己的实验报告，即达标及格。检查训练、验证与测试流程、参数记录、曲线解读、一次参数对比和错例分析。

PPT 增加一页可选拓展：完成 CIFAR-10 微调可加分，进一步完成 Stanford40 动作识别微调分类可争取更高分。学生附代码、数据划分、实际结果和简要分析，不设统一准确率门槛或固定加分值。本次只补充任务说明，未开展这两个数据集的实验；现有 MNIST 脚本需要由学生自行适配。

## 教师复现记录与结果来源

课堂使用 50,000 张训练、1,000 张验证、10,000 张独立测试，轮数由学生设置。以下配置和时间仅记录教师实际完成的运行，不是固定课程要求。官方 60,000 张训练图按种子 42 固定划分为 50,000 张训练池和 10,000 张验证池；只使用验证池前 1,000 张，另外 9,000 张保留未用。

教师首次正式训练的命令与课堂默认测试数量有一处区别：先保存了 1,000 张测试的训练输出，随后对同一个、已经由验证集选定的模型独立评估全部 10,000 张测试图。测试数量不会改变训练过程；保留原始训练配置，不把原日志的 `test_samples: 1000` 改写为 10,000。完整测试准确率应读取 `examples/course_full_test/evaluation.json`，训练与验证曲线读取 `examples/course_baseline/`。

教师首次训练与完整测试的准确命令：

```bash
python train.py --epochs 8 --train-samples 50000 --val-samples 1000 --test-samples 1000 --batch-size 32 --lr 0.0001 --lr-milestones 4 6 --lr-gamma 0.2 --output outputs/course_baseline
python evaluate.py --checkpoint outputs/course_baseline/best.pt --test-samples 10000 --output outputs/course_full_test
```

上面的 1,000 张测试是教师首次运行记录，**不是学生正式默认**。为降低课堂操作复杂度，学生在一次训练命令中直接使用 `--test-samples 10000`，训练结束后自动完整测试。学生对比轮数自行确定，不用教师记录代替自己的运行。

| 记录项 | 实际内容 |
| --- | --- |
| 日期与代码提交号 | 2026-09-12；发布提交号见 GitHub，原始指标见结果记录 |
| 操作系统 / Python / PyTorch | Linux / Python 3.9.25 / PyTorch 2.3.0+cu121 |
| GPU / CPU | NVIDIA GeForce RTX 4090；CPU 快速路线使用 4 线程 |
| 官方预训练来源与文件校验 | Apple 官方 JPEG 预训练 Tiny，SHA256 见资源清单和指标记录 |
| 正式最佳验证准确率 / 轮次 | 97.30%，第 5 轮（验证样本 1,000） |
| 正式完整测试准确率 | 97.07%（9,707/10,000），达到 95% 以上目标 |
| 教师首次 8 轮运行耗时 | 712.90 秒（约 11.88 分钟），包含当次 1,000 张测试，不含安装下载 |
| 独立完整测试耗时 | 5.93 秒；对已固定的模型评估全部 10,000 张 |
| 学生对比实验 | 轮数自行设置，报告各自的实际结果 |
| 正式训练与验证记录 | `examples/course_baseline/metrics.json`、`history.csv` |
| 正式完整测试记录 | `examples/course_full_test/evaluation.json`，分母 10,000 |
| CPU 快速路线是否实际完成 | 已完成；1 轮 1,000/200/200，测试 20.5%，65.70 秒，仅验证流程 |
| Kaggle / AutoDL 是否实机验证 | 未登录实机验证；网页步骤依据官方文档核验 |
| 官方 padding mask 差异与修复说明 | 课程实现修正 padding mask；具体代码与模型移植验证记录随仓库提供 |

早期 6,000 张、3 轮及 5 轮实验保存在历史示例中，用于记录快速体验过程。其 88.72% 全测试结果不能作为正式课程基线；也不能混入新的 8 轮结果表。

## 课程材料与发布

建议 GitHub 保存源代码、中文指南、课程笔记本、少量结果示例及其运行说明。MNIST 的 4 个原始压缩文件（约 11.6 MB）随仓库提供，脚本会校验；官方预训练权重由脚本下载，checkpoint 由训练生成，不放入普通 Git 提交。保留原项目所要求的许可与归属信息。

学生下载入口、PPT 二维码和 Kaggle 笔记本入口只能指向已经可访问的实际仓库。发布前用未登录状态检查一次。发布材料应清理个人路径与账号信息；使用教师 PPT 模板及其校徽等素材时保留原有权利说明。
