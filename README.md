# ByteFormer 微调 MNIST：零基础课程实验

用一个已经学过图像知识的 **ByteFormer** 模型，识别手写数字 `0–9`。你将完成下载材料、运行训练、查看结果、修改一个参数和分析一个错例。无需从零编写神经网络。

15 页课程课件：[PowerPoint](docs/ByteFormer_MNIST_零基础实验课.pptx) / [PDF](docs/ByteFormer_MNIST_零基础实验课.pdf)。课程仓库：[Franklin-L/byteformer-mnist-course](https://github.com/Franklin-L/byteformer-mnist-course)。推荐使用 **Kaggle 免费 GPU + 本仓库笔记本**，无需在自己的电脑上配置深度学习环境。

本课程使用 Apple 官方发布的 ImageNet 预训练 ByteFormer Tiny 权重，替换为 10 类分类头，再对 MNIST 进行**全参数微调**。我们提供适合教学的精简 PyTorch 实现，保留 12 层 Transformer 和预训练参数结构。实现对官方代码中的 padding mask 处理问题做了修正，具体差异见代码和教师复现记录。

## 你需要完成什么

| 必做任务 | 完成证据 |
| --- | --- |
| 下载数据与预训练权重，完成一次基线训练 | 下载成功记录、训练日志和 `metrics.json` |
| 读懂训练损失与验证准确率曲线 | `curves.png`，用自己的话解释两个指标 |
| 只把训练轮数从 8 改为 10，再运行一次 | 两个独立输出目录与验证集结果比较 |
| 查看一个识别错误的数字并提出解释 | 数字图像、真实标签、预测标签与简短分析 |
| 提交实验报告 | 使用 [Word 报告模板](docs/学生实验报告模板.docx)（也提供 [Markdown 版](docs/student_report_template.md)） |

**微调**：从已训练的参数继续学习新任务。**MNIST**：手写数字图像数据集。**epoch（轮）**：把本次选定的训练样本学习一遍。**batch（批次）**：每次一起送入模型的一小组样本。**loss（损失）**：训练时使用的误差，通常希望它下降。**accuracy（准确率）**：预测正确的比例，通常希望它上升。

ByteFormer 读取的是**文件字节序列**。本实验把 MNIST 灰度图转成 RGB 图像，以 JPEG 格式编码，再把 JPEG 字节交给模型；学生不需要手工转换图片。它与直接把像素小块输入普通视觉 Transformer 的处理流程不同。

## 1. 在 Kaggle 打开课程笔记本（推荐）

1. 注册并登录 [Kaggle](https://www.kaggle.com/)，进入 [Code](https://www.kaggle.com/code) 页面，新建 Notebook。
2. 从本仓库下载 [`course_kaggle.ipynb`](course_kaggle.ipynb)，在 Kaggle 的 **File → Import Notebook** 中导入该文件。界面名称可能调整，寻找“导入笔记本 / Import Notebook”即可。
3. 在笔记本设置中，把 **Accelerator** 设为可用的 **GPU**，并打开 **Internet**。如果提示账号验证，按 Kaggle 的页面提示完成。GPU 是否可用及使用额度以账号页面为准。
4. 从上到下逐个运行单元格：点击左侧运行按钮，等待当前单元格结束后继续。第一次不要直接运行全部单元格。
5. 环境检查单元格显示 `GPU available: True` 即可继续。若为 `False`，先确认设置；无法获取 GPU 时使用下方 AutoDL 备用路线。

Kaggle 操作参考：[官方 Notebook 文档](https://www.kaggle.com/docs/notebooks)。本项目已在本地 GPU 与 CPU 完成实测；Kaggle 和 AutoDL 的网页步骤依据官方文档整理，尚未用本课程账号登录实机验证。

## 2. 获取代码并检查环境

**使用课程笔记本时，这些步骤已经写好，直接运行对应单元格。** 以下保留命令，方便你看懂实际做了什么。Kaggle 的 Python 单元格中，shell 命令前需要 `!`，切换目录使用 `%cd`：

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

官方预训练权重不放入普通 Git 提交，由脚本从官方来源下载。网络受限时，可使用教师准备的课堂资源包，内含代码、MNIST、官方预训练权重和 `teacher_checkpoint/best.pt`。资源包不含 Python/PyTorch 安装包，仍需已有 PyTorch 环境；学生自己的默认训练输出为 `outputs/baseline`，与教师模型分开保存。MNIST 原始文件的来源、校验和许可说明见仓库材料。

## 4. 完成基线微调

在已开启 GPU 的环境中运行（Kaggle 单元格加 `!`）：

```bash
python train.py --epochs 8 --train-samples 50000 --val-samples 1000 --test-samples 10000 --batch-size 32 --lr 0.0001 --lr-milestones 4 6 --lr-gamma 0.2 --output outputs/baseline
```

上述参数也是默认设置，因此直接运行 `python train.py` 也可完成基线。第一次建议使用完整命令，让参数含义更清楚。

| 参数 | 本次设置 | 含义 |
| --- | --- | --- |
| `--epochs` | `8` | 训练 8 轮 |
| `--train-samples` | `50000` | 使用 50,000 个训练样本 |
| `--val-samples` | `1000` | 使用 1,000 个验证样本 |
| `--test-samples` | `10000` | 最后评估全部 10,000 个官方测试样本 |
| `--batch-size` | `32` | 每批 32 个样本 |
| `--lr` | `0.0001` | 主干学习率；新分类头使用其 10 倍，即 `0.001` |
| `--lr-milestones` | `4 6` | 第 4、6 轮结束后降低学习率 |
| `--lr-gamma` | `0.2` | 每次降低时乘以 0.2 |
| `--output` | `outputs/baseline` | 本次结果保存目录 |

正式课程使用 **50,000 张训练图 + 1,000 张验证图 + 10,000 张测试图**。默认随机种子为 `42`：官方 60,000 张训练图先固定分为 50,000 张训练池和 10,000 张验证池，本实验使用整个训练池，以及验证池的前 1,000 张；**验证池其余 9,000 张保留未用**。训练与验证没有重叠。官方 10,000 张测试图单独保留，在训练完成后全部评估；测试数据不用于更新参数或挑选最佳模型。

不要把“验证池有 10,000 张”写成“本实验使用 10,000 张验证图”。我们实际验证样本数是 **1,000**，也没有使用官方全部 60,000 张训练图更新参数。

学习率会自动降低：第 1–4 轮主干为 `0.0001`，第 5–6 轮为 `0.00002`，第 7 轮起为 `0.000004`；分类头始终使用主干的 10 倍。可以理解为先学习，再用较小步幅调整，学生无需手工干预。

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

正式课程的参考目标是全部 10,000 张官方测试图准确率达到 **95% 以上**。请保留自己的真实结果；若低于目标，先检查样本数、训练轮数、学习率衰减和权重来源，再向教师说明。不要把教师演示结果或截图中的数值抄成自己的实验数据。

### CPU 快速路线

如果 GPU 不可用，先运行：

```bash
python train.py --epochs 1 --train-samples 1000 --val-samples 200 --test-samples 200 --batch-size 16 --lr 0.0001 --device cpu --output outputs/cpu_smoke
```

这条路线仍需已经安装 PyTorch，可在 Kaggle/AutoDL 的 PyTorch 环境里使用 `--device cpu` 执行。它用于验证下载、训练和输出流程，**不保证准确率，也不等同于完整基线实验**。ByteFormer 仍然是一个需要计算资源的 Transformer 模型，小样本不代表 CPU 训练一定很快。报告中请明确写出你实际使用的路线和参数。

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

结果图保存为 `outputs/baseline/prediction_single.png`。把 `0` 改成 `0–9999` 中的其他索引，可查看另一个测试样本。这只是查看个例，不能用一张图代表整体性能。

可选：先预测仓库附带的示例图片；之后将路径替换为自己上传的图片：

```bash
python predict.py --checkpoint outputs/baseline/best.pt --image assets/example_digit.png
```

为了接近 MNIST 的输入形式，请使用**黑底白字、一个居中的手写数字**。手机拍照中的背景、光照、方向和书写风格可能与 MNIST 差别较大，出现错误并不意味着训练脚本运行失败。自制图片是额外体验，不替代规定的测试集评估。

## 6. 只改一个参数：8 轮变为 10 轮

保持其他参数相同，把 `--epochs 8` 改为 `--epochs 10`，并使用新输出目录：

```bash
python train.py --epochs 10 --train-samples 50000 --val-samples 1000 --test-samples 10000 --batch-size 32 --lr 0.0001 --lr-milestones 4 6 --lr-gamma 0.2 --output outputs/epochs10
```

如果输出目录已有实验结果，脚本会报错以保护结果；需要重新运行时，请改用新目录，例如 `outputs/baseline_retry`，并让评估与预测的 checkpoint 路径对应新目录。

这次实验应再次从官方预训练参数出发。它不是在 `baseline/best.pt` 上接着训练。保留默认随机种子 `42`、相同的数据与学习率衰减设置，使对比更公平。两组都在第 4、6 轮后乘以 0.2，不要同时修改这些参数。

用两次实验的 `history.csv` 比较**验证准确率**和训练损失，记录最佳验证轮次。不要预设“10 轮一定更好”。如果没有改善，说明观察到的事实，并讨论可能原因。两组训练轮数应在查看测试结果之前约定；测试结果用于最终报告，不用于继续试参。

CPU 同学请按教师安排使用 GPU 完成这部分；如课堂只允许 CPU 冒烟验证，应在报告中注明“完整对比实验未完成”，不要填写虚构结果。

## 7. 找一个错例并提交报告

预测示例图展示前 12 个测试样本，并额外加入最多 4 个实际错例；这些额外错例是为了讲解错误而选择的，不能用图中正确比例当作整体准确率。笔记本还会从 `test_predictions.npz` 读出错例的官方测试索引，供你再次预测并保存证据。打开预测图或混淆矩阵，寻找一个被识别错的数字。如果本次测试没有错例，报告真实情况，并分析一个自制图片的错误或一个容易混淆的样本，明确说明它的来源。

在 [Word 报告模板](docs/学生实验报告模板.docx)（也提供 [Markdown 版](docs/student_report_template.md)）中完成记录，提交以下材料：

1. 完整实验报告。
2. 基线和 10 轮实验各自的 `metrics.json`、`history.csv` 和 `curves.png`。
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
| `CUDA out of memory` | 把 `--batch-size 32` 改为 `--batch-size 16`，必要时改为 `8`；在报告中记录变更，两组实验保持相同批次大小。 |
| 训练很慢 | 检查是否在 CPU 上运行；先验证 CPU 快速路线，完整实验使用 Kaggle 或 AutoDL GPU。 |
| 找不到 `best.pt` | 检查训练是否正常完成，以及 `--checkpoint` 是否与训练时的 `--output` 对应。 |
| 准确率比同学低 | 先对比样本数、轮数、种子和输入设置。保留真实结果，再解释差异；不要通过反复测试来选择参数。 |
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
python train.py --epochs 8 --train-samples 50000 --val-samples 1000 --test-samples 10000 --batch-size 32 --lr 0.0001 --lr-milestones 4 6 --lr-gamma 0.2 --output outputs/baseline
```

4. 按第 5–7 步继续评估、预测和 10 轮对比。JupyterLab 左侧文件浏览器可以打开生成的图片。
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

## 教师实测参考：不要抄作自己的结果

正式课程已达到 95% 以上的参考目标：在全部 **10,000 张官方测试图上，准确率为 97.07%（9,707/10,000）**。训练使用 50,000 张，实际验证使用 1,000 张。

| 正式方案 | 最佳验证准确率 | 完整 10,000 张测试准确率 | 验证选出的最佳轮次 |
| --- | --- | --- | --- |
| 50,000 张训练、8 轮、第 4/6 轮后学习率乘 0.2 | 97.30% | 97.07% | 第 5 轮 |

教师首次 8 轮运行用时 **712.90 秒（约 11.88 分钟）**，该次原始配置末尾测试了 1,000 张；锁定模型后，另行完整评估 10,000 张用时 **5.93 秒**。学生命令已直接设置 `--test-samples 10000`，一次完成完整测试。不要把教师原始 `metrics.json` 中的 1,000 张结果当作完整测试成绩。

训练与验证记录见 [`examples/course_baseline/`](examples/course_baseline/)，正式完整测试证据见 [`examples/course_full_test/evaluation.json`](examples/course_full_test/evaluation.json)。8→10 轮是学生实操，没有填入未执行的教师 10 轮成绩。旧的 6,000 张、3 轮实验只保留在历史记录中。

软件参考环境为 Python 3.9.25、PyTorch 2.3.0+cu121、NumPy 1.26.4、Pillow 11.3.0、matplotlib 3.9.4、requests 2.32.5，GPU 为 NVIDIA GeForce RTX 4090。记录中的用时不含依赖安装和网络下载，不是 Kaggle 或 AutoDL 的速度承诺。环境差异可能影响复现结果。

`requirements.txt` 用于已有 PyTorch 的 Kaggle/AutoDL 环境；`requirements-reproduce.txt` 记录教师实测依赖，仅供需要复现该软件组合的教师参考，**学生不需要在 Kaggle 中安装后者**。

## 教师与来源说明

- 学生报告：[Word 模板](docs/学生实验报告模板.docx) / [Markdown 模板](docs/student_report_template.md)。备课与评分建议：[教师指南](docs/teacher_guide.md)。
- 原始方法与预训练模型：Apple 的 [CoreNet 项目](https://github.com/apple/corenet)；权重来源、文件校验与实现差异以课程脚本和复现记录为准。
- MNIST 原始压缩文件随课程提供；来源、校验与许可说明见仓库材料。

本课程是教学适配，并非 Apple 官方课程。精简实现、MNIST 输入转换、训练/验证划分和 padding mask 修正可能使结果与官方完整 ImageNet 配置不同，不能据此声称复现了论文中的 ImageNet 性能。
