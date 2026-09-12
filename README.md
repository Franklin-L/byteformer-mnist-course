# 码流图像分类

本实验基于 ByteFormer 预训练模型，在 MNIST 数据集上完成码流图像分类。实验内容包括环境配置、模型微调、参数对比和结果分析。

课程后续将加入受损码流分类。备课阶段已经整理多媒体通信、Gilbert–Elliott 信道、JPEG/H.264 码流结构、字节模型和 1/10 MNIST 受损测试方案，见 [research/README.md](research/README.md)。这一部分当前作为实验与课件素材保存，尚未并入课程 PPT。

| 任课人员 | 姓名 | 邮箱 |
| --- | --- | --- |
| 教师 | 吴科君 | kjwu@hust.edu.cn |
| 助教 | 李方成 | lifangcheng2002@163.com |

16 页课程课件：[PowerPoint](docs/ByteFormer_MNIST_零基础实验课.pptx) / [PDF](docs/ByteFormer_MNIST_零基础实验课.pdf)。课程仓库：[Franklin-L/byteformer-mnist-course](https://github.com/Franklin-L/byteformer-mnist-course)。推荐使用 **Kaggle 免费 GPU + 本仓库笔记本**，无需在自己的电脑上配置深度学习环境。

实验采用 Apple 发布的 ImageNet 预训练 ByteFormer Tiny，保留 12 层 Transformer，将分类头调整为 10 类，并使用 MNIST 更新模型参数。

## 实验要求

| 实验任务 | 提交内容 |
| --- | --- |
| 下载数据与预训练权重，完成一次基线训练 | 下载成功记录、训练日志和 `metrics.json` |
| 读懂训练损失与验证准确率曲线 | `curves.png`，用自己的话解释两个指标 |
| 自行设置训练轮数和 batch size，并作参数对比 | 两个独立输出目录与验证集结果比较 |
| 查看一个识别错误的数字并提出解释 | 数字图像、真实标签、预测标签与简短分析 |
| 提交实验报告 | 使用 [Word 报告模板](docs/学生实验报告模板.docx)（也提供 [Markdown 版](docs/student_report_template.md)） |

**微调**：从已训练的参数继续学习新任务。**MNIST**：手写数字图像数据集。**epoch（轮）**：把本次选定的训练样本学习一遍。**batch（批次）**：每次一起送入模型的一小组样本。**loss（损失）**：训练时使用的误差，通常希望它下降。**accuracy（准确率）**：预测正确的比例，通常希望它上升。

ByteFormer 读取文件字节序列。本实验通过脚本将 MNIST 图像转换为 RGB 格式并编码为 JPEG，再将 JPEG 字节输入模型。

## 1. 在 Kaggle 打开课程笔记本（推荐）

课前可先参考以下教程完成 Kaggle 平台准备：

- [知乎：Kaggle GPU资源使用教程——针对超级小白](https://zhuanlan.zhihu.com/p/18209757723)
- [CSDN：科研小白扫盲：Kaggle平台使用指导指南](https://blog.csdn.net/yyyyyybw/article/details/148336854)
- [Kaggle 官方 Notebook 文档](https://www.kaggle.com/docs/notebooks)

准备好 GPU 后，导入本课程 Notebook。GPU 验证要求和可用额度以账号页面为准。

1. 注册并登录 [Kaggle](https://www.kaggle.com/)，进入 [Code](https://www.kaggle.com/code) 页面，新建 Notebook。
2. 从本仓库下载 [`course_kaggle.ipynb`](course_kaggle.ipynb)，在 Kaggle 的 **File → Import Notebook** 中导入该文件。界面名称可能调整，寻找“导入笔记本 / Import Notebook”即可。
3. 在笔记本设置中，把 **Accelerator** 设为可用的 **GPU**，并打开 **Internet**。如果提示账号验证，按 Kaggle 的页面提示完成。GPU 是否可用及使用额度以账号页面为准。
4. 从上到下逐个运行单元格：点击左侧运行按钮，等待当前单元格结束后继续。第一次不要直接运行全部单元格。
5. 环境检查单元格显示 `GPU available: True` 即可继续。若为 `False`，先确认设置；无法获取 GPU 时使用下方 AutoDL 备用路线。

## 2. 获取代码并检查环境

课程 Notebook 已包含以下命令，运行对应单元格即可。Kaggle 的 Python 单元格中，shell 命令前加 `!`，切换目录使用 `%cd`：

```python
!git clone https://github.com/Franklin-L/byteformer-mnist-course.git /kaggle/working/byteformer-mnist-course
%cd /kaggle/working/byteformer-mnist-course
!python -m pip install -r requirements.txt
```

`requirements.txt` 仅安装轻量依赖，保留 Kaggle 已有的 PyTorch，不需要重新安装 CUDA。代码必须放在可写的 `/kaggle/working/` 下；`/kaggle/input/` 中的数据通常只读。

检查 Python 和 GPU：

```python
import sys, torch
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

**成功判断：** 当前目录能看到 `prepare.py`、`train.py`、`evaluate.py`、`predict.py` 和 `requirements.txt`，并且 GPU 检查通过。笔记本第一次运行会获取代码；重跑时会复用已有课程目录。

**后续第 3–7 步展示的是终端命令。** 在 Kaggle 的 Python 单元格中，前面加 `!`；课程笔记本已经处理好，无需手工改写。在 AutoDL 终端中不加 `!`。

## 3. 准备 MNIST 和官方预训练权重

```bash
python prepare.py
```

仓库附带 MNIST 的 4 个原始压缩文件（合计约 11.6 MB）。脚本检查这些文件，并下载约 61 MB 的官方预训练权重；若数据缺失，也会自动补齐。已有且通过校验的文件会复用。无需自己找数据集、解压 MNIST 或修改路径。

**成功判断：** 命令正常结束，没有下载失败或校验失败的错误。数据放在 `data/` 下，预训练文件放在 `checkpoints/` 下。

预训练权重由 `prepare.py` 自动下载。网络下载困难时，可使用课程资源包，其中包含代码、MNIST、预训练权重和演示模型 `teacher_checkpoint/best.pt`。使用资源包仍需安装 Python/PyTorch；训练结果默认保存到 `outputs/baseline`。

## 4. 完成基线微调

训练轮数和 batch size 均由学生设置，不作统一规定。Kaggle 在课程 Notebook 中输入自己选择的正整数；AutoDL/Linux 终端使用下面的命令：

```bash
read -p "请输入训练轮数：" EPOCHS
read -p "请输入 batch size（参考32）：" BATCH_SIZE
python train.py --epochs "$EPOCHS" --train-samples 50000 --val-samples 1000 --test-samples 10000 --batch-size "$BATCH_SIZE" --lr 0.0001 --lr-milestones 4 6 --lr-gamma 0.2 --output outputs/baseline
```

`--epochs` 和 `--batch-size` 填写正整数，batch size 可参考 `32`。根据训练与验证曲线调整设置，测试集用于最终评估。Kaggle 使用 Notebook 中的 `input()` 输入参数；以下 `read` 命令用于终端。

| 参数 | 本次设置 | 含义 |
| --- | --- | --- |
| `--epochs` | 自行设置的正整数 | 本次计划学习训练集的遍数；不固定 |
| `--train-samples` | `50000` | 使用 50,000 个训练样本 |
| `--val-samples` | `1000` | 使用 1,000 个验证样本 |
| `--test-samples` | `10000` | 最后评估全部 10,000 个官方测试样本 |
| `--batch-size` | 自行设置，参考 `32` | 每批样本数；可结合显存与训练情况调整 |
| `--lr` | `0.0001` | 主干学习率；新分类头使用其 10 倍，即 `0.001` |
| `--lr-milestones` | `4 6` | 第 4、6 轮结束后降低学习率 |
| `--lr-gamma` | `0.2` | 每次降低时乘以 0.2 |
| `--output` | `outputs/baseline` | 本次结果保存目录 |

正式课程使用 **50,000 张训练图 + 1,000 张验证图 + 10,000 张测试图**。默认随机种子为 `42`：官方 60,000 张训练图先固定分为 50,000 张训练池和 10,000 张验证池，本实验使用整个训练池，以及验证池的前 1,000 张；**验证池其余 9,000 张保留未用**。训练与验证没有重叠。官方 10,000 张测试图单独保留，在训练完成后全部评估；测试数据不用于更新参数或挑选最佳模型。

默认学习率在第 4、6 轮结束后分别乘以 `0.2`，分类头学习率为主干的 10 倍。可通过 `--lr-milestones` 和 `--lr-gamma` 调整。

- **训练集：** 用来更新模型参数。
- **验证集：** 每轮结束后检查，用来选出最佳模型。
- **测试集：** 训练结束后评估选出的模型，不参与梯度更新或最佳轮次选择。

运行期间不要关闭终端。每轮日志会显示进度和指标；实际用时取决于 GPU、CPU 和环境，请以自己的运行记录为准。

**成功判断：** 命令正常结束，并生成以下文件：

```text
outputs/baseline/
├── best.pt                 # 按验证结果选出的模型
├── metrics.json            # 本次实验设置与最终指标
├── history.csv             # 各轮训练和验证记录
├── curves.png              # 损失、准确率等学习曲线
├── predictions.png         # 数字图像与预测示例
└── confusion_matrix.png    # 各数字之间的混淆情况
```

在全部 10,000 张测试图上评估所选模型，将准确率及训练、验证曲线分析写入报告。

### CPU 快速路线

如果 GPU 不可用，先运行：

```bash
python train.py --epochs 1 --train-samples 1000 --val-samples 200 --test-samples 200 --batch-size 16 --lr 0.0001 --device cpu --output outputs/cpu_smoke
```

CPU 命令使用较少样本检查数据准备、训练与输出流程。运行前需要安装 PyTorch；完整实验建议使用 GPU。

## 5. 读取结果，单独评估与预测

用文件浏览器打开 `outputs/baseline/curves.png`。观察损失是否总体下降、验证准确率是否提高；个别轮次波动并不一定意味着程序错误。

笔记本会直接展示指标，也可以用文本编辑器打开 `outputs/baseline/metrics.json` 和 `history.csv`。JSON 中 `best_validation_accuracy` 是最佳验证准确率，`test.accuracy` 是最终测试准确率，数值使用 0–1；乘以 100 才是百分比。把自己的实际结果写入报告。`history.csv` 也可用电子表格软件打开。**曲线上的验证结果与最终测试结果来自不同数据，不应混写。**

如果想重新评估已经训练好的模型，无需再次训练：

```bash
python evaluate.py --checkpoint outputs/baseline/best.pt --output outputs/baseline
```

复评结果另存为 `evaluation.json`，原训练的 `metrics.json` 不会被覆盖。请保持原来的测试设置，使用脚本保存的配置完成复评；不要反复根据测试结果挑选学习率或训练轮数。

预测官方测试集的第 0 张图：

```bash
python predict.py --checkpoint outputs/baseline/best.pt --index 0
```

结果图保存为 `outputs/baseline/prediction_single.png`。将 `--index 0` 改为 `0–9999` 中的其他索引，可查看对应测试样本。

可选：先预测仓库附带的示例图片；之后将路径替换为自己上传的图片：

```bash
python predict.py --checkpoint outputs/baseline/best.pt --image assets/example_digit.png
```

自制图片采用黑底白字、单个居中的手写数字。可比较自制图片与 MNIST 样本的笔画、背景和方向，分析预测差异。

## 6. 自选参数，观察验证结果

自行设置另一组训练轮数和 batch size；不规定两组的具体数值。建议一次只改变其中一项，另一项沿用基线，便于判断效果。Kaggle 在对比单元格输入自己的设置；AutoDL/Linux 终端运行：

```bash
read -p "请输入对比实验的训练轮数：" COMPARISON_EPOCHS
read -p "请输入对比实验的 batch size：" COMPARISON_BATCH_SIZE
python train.py --epochs "$COMPARISON_EPOCHS" --batch-size "$COMPARISON_BATCH_SIZE" --output outputs/comparison
```

两次都从同一官方预训练权重开始，保持数据划分、随机种子和学习率策略相同；若两项参数都改变，请说明比较的局限。保留 `outputs/baseline`，另存 `outputs/comparison`；重跑使用新的输出目录，并让评估与预测的 checkpoint 路径对应。

比较两次实验的训练损失、最佳验证准确率和耗时，说明参数调整的影响，并记录各自的最终测试结果。

## 7. 找一个错例并提交报告

预测图包含前 12 个测试样本及最多 4 个错例，整体准确率记录在 `metrics.json` 中。Notebook 会从 `test_predictions.npz` 读取错例的测试索引，可用于再次预测和保存图片。选择一个错例，分析图像特征与预测差异；若没有错例，可分析一个容易混淆的样本，并说明样本来源。

在 [Word 报告模板](docs/学生实验报告模板.docx)（也提供 [Markdown 版](docs/student_report_template.md)）中完成记录，提交以下材料：

1. 完整实验报告。
2. 基线和自选对比实验各自的 `metrics.json`、`history.csv` 和 `curves.png`。
3. 基线的 `predictions.png`、`confusion_matrix.png`，以及错例图或终端记录。
4. 简短说明：自己运行了哪些命令、修改了什么、遇到什么问题。

笔记本最后一格会生成 `/kaggle/working/byteformer_mnist_results.zip`。在 Kaggle 文件/输出面板找到这个文件，点击下载；如需保存整个会话，按页面提供的保存版本操作执行。压缩包包含两组实验的小型结果文件，不含大型模型。会话结束前请确认文件已经下载到自己的电脑。

默认不需要提交大型模型文件、下载数据、`.venv/` 或整份 Python 环境。若要以后继续做预测，请另外下载 `best.pt`。

## 常见问题

| 现象 | 操作 |
| --- | --- |
| `python: command not found` | Linux 先用 `python3` 创建并激活 `.venv`；Windows 使用 `py`。 |
| `No module named torch` | 请使用 Kaggle GPU 环境或 AutoDL 的 PyTorch 镜像。轻量 requirements 不安装 torch；本地 CPU 安装可参考下方说明。 |
| `No module named ...`（其他依赖） | 确认使用正确环境，在课程目录运行 `python -m pip install -r requirements.txt`。 |
| `can't open file 'train.py'` | 当前目录不对。先进入能看到 `train.py` 的课程目录。 |
| `GPU available: False` | Kaggle 设置中开启 GPU；无可用额度时切换 AutoDL，或先使用 CPU 快速路线。 |
| 下载超时或校验失败 | 检查网络后重新执行 `python prepare.py`；持续失败时使用教师预下载的材料。不要忽略校验错误。 |
| `CUDA out of memory` | 降低自己设置的 batch size，例如从 32 降到 16 或 8，并记录实际值；若比较轮数的影响，则保持 batch size 相同。 |
| 训练很慢 | 检查是否在 CPU 上运行；先验证 CPU 快速路线，完整实验使用 Kaggle 或 AutoDL GPU。 |
| 找不到 `best.pt` | 检查训练是否正常完成，以及 `--checkpoint` 是否与训练时的 `--output` 对应。 |
| 准确率比同学低 | 比较样本数、轮数、batch size 和输入设置，结合训练与验证曲线分析差异。 |
| `already contains a run` | 已有结果受保护。改用新 `--output` 目录，并同步修改评估/预测的 checkpoint 路径。 |
| 重启 Kaggle 会话后文件丢失 | 从保存的输出或已下载结果恢复；重新训练要使用原设置并如实记录。 |
| Kaggle 提示只读目录 | 把代码放到 `/kaggle/working/`，不要在 `/kaggle/input/` 直接训练。 |

如果确实要在自己电脑的独立 Python 环境中做 CPU 验证，可以先执行以下安装命令；**Kaggle/AutoDL 已有 PyTorch 的环境不要运行此安装命令**：

```bash
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

本地建议 Python 3.10/3.11。如果安装器提示没有匹配版本，优先使用课程推荐的托管 PyTorch 环境。

## AutoDL 备用路线：租 GPU 并打开终端

1. 登录 [AutoDL 官网](https://www.autodl.com/)，进入控制台，创建 GPU 实例。选择平台提供的 **PyTorch 镜像**；建议 Python 3.10/3.11、PyTorch 2.3 或更新兼容版本，单 GPU 显存至少 4 GB。价格、余额、可用资源与计费方式以创建页面为准，不需要为本课购买最高规格。
2. 启动实例，在实例页面打开 **JupyterLab**。在 Launcher 中点击 **Terminal / 终端**；如果没有 Launcher，可通过 File → New → Terminal 打开。参考 [AutoDL 官方 JupyterLab 指南](https://www.autodl.com/docs/jupyterlab/)。
3. 在终端逐行执行以下命令。这里不加 Kaggle 单元格使用的 `!`，也不使用 `%cd`：

```bash
mkdir -p /root/autodl-tmp
cd /root/autodl-tmp
git clone https://github.com/Franklin-L/byteformer-mnist-course.git
cd byteformer-mnist-course
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('GPU available:', torch.cuda.is_available())"
python prepare.py
read -p "请输入训练轮数：" EPOCHS
read -p "请输入 batch size（参考32）：" BATCH_SIZE
python train.py --epochs "$EPOCHS" --train-samples 50000 --val-samples 1000 --test-samples 10000 --batch-size "$BATCH_SIZE" --lr 0.0001 --lr-milestones 4 6 --lr-gamma 0.2 --output outputs/baseline
```

4. 按第 5–7 步继续评估、预测和自选参数对比。JupyterLab 左侧文件浏览器可以打开生成的图片。
5. 下载作业结果：运行下面的打包命令，然后在左侧文件浏览器中找到课程目录里的 `student_results.zip`，右键下载。打包不包含大型 checkpoint。

```bash
python -c "from pathlib import Path; from zipfile import ZipFile, ZIP_DEFLATED; files=[p for p in Path('outputs').rglob('*') if p.is_file() and p.suffix in {'.json','.csv','.png','.npz'}]; z=ZipFile('student_results.zip','w',ZIP_DEFLATED); [z.write(p,p.as_posix()) for p in files]; z.close(); print('student_results.zip')"
```

6. 确认文件已下载，再回 AutoDL 控制台关闭实例。**只关浏览器不等于关闭服务器**；关机后的存储费用等规则请查看平台当前计费说明。

如果 GitHub 克隆失败，可以在自己电脑的仓库页面点 **Code → Download ZIP**，将 ZIP 上传到 JupyterLab 的 `/root/autodl-tmp/`，再在终端执行：

```bash
cd /root/autodl-tmp
python -m zipfile -e byteformer-mnist-course-main.zip .
cd byteformer-mnist-course-main
python -m pip install -r requirements.txt
python prepare.py
```

压缩包名称以实际下载文件为准。进入解压后的 `byteformer-mnist-course-main` 文件夹，不要误进入 ZIP 所在目录直接训练。

需要 SSH 时，从实例页面复制**自己的 SSH 登录命令**，在本机 PowerShell/终端执行，再输入平台显示的密码；输入密码时终端不会显示字符。连接后执行与上面相同的 `cd` 和训练命令。初学者优先使用网页 JupyterLab；SSH 长任务需按 [官方 SSH 指南](https://www.autodl.com/docs/ssh/)使用终端会话保活。

## 拓展任务

完成 MNIST 规定任务并提交实验报告，即达标及格。额外完成 CIFAR-10 微调可加分，Stanford40 动作识别分类为进阶加分任务。拓展任务需自行适配数据读取和分类头，提交代码、数据划分、实验结果与分析。

## 配套材料与参考文献

- [实验报告 Word 模板](docs/学生实验报告模板.docx) / [Markdown 模板](docs/student_report_template.md)
- [教师指南](docs/teacher_guide.md) / [演示运行记录](docs/run_record.md)
- ByteFormer 论文：[Bytes Are All You Need](https://arxiv.org/abs/2306.00238)
- 原始代码与预训练模型：[Apple CoreNet](https://github.com/apple/corenet/tree/main/projects/byteformer)
- MNIST 数据：[CVDF 镜像](https://github.com/cvdfoundation/mnist)
- [代码、模型与素材来源说明](THIRD_PARTY_NOTICES.md)
