# MNIST 原始数据

本仓库随附四个未修改的 MNIST IDX gzip 文件，共 11,594,722 字节，方便课堂免去数据下载。
原作者：Yann LeCun、Corinna Cortes、Christopher J. C. Burges。

- 原始项目页：http://yann.lecun.com/exdb/mnist/
- CVDF 镜像说明：https://github.com/cvdfoundation/mnist
- 本次下载使用 PyTorch/torchvision 提供的镜像：https://ossci-datasets.s3.amazonaws.com/mnist/
- MD5 值来自 PyTorch 官方 MNIST 实现：https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
- 本地每个文件的 MD5、大小记录在 `assets/resource_manifest.json`；`python prepare.py` 每次都会检查。

CVDF 镜像声明已获得 Yann LeCun 同意托管 MNIST。数据保持原样并保留原作者权利；本仓库代码许可不对 MNIST 重新授权。
本课程 JPEG 字节缓存仅在学生运行时生成，不作为另一套训练/测试数据发布。

官方训练集共 60,000 张，测试集共 10,000 张。课程从官方训练集中独立划分训练和验证子集；不会用测试集训练或选择最佳轮次。
