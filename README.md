# 目录

<!-- TOC -->

- [目录](#目录)
- [DPRNN介绍](#DPRNN介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [数据预处理过程](#数据预处理过程)
        - [数据预处理](#数据预处理)
    - [训练过程](#训练过程)
        - [训练](#训练)  
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# DPRNN介绍

双路径循环语音分离神经网络（Dual-Path RNN）由三个处理阶段组成, 编码器、分离和解码器。首先，编码器模块用于将混合波形的短段转换为它们在中间特征空间中的对应表示。然后，该表示用于在每个时间步估计每个源的乘法函数（掩码）。最后利用解码器模块对屏蔽编码器特征进行变换，重构源波形。 DPRNN被广泛的应用在语音分离等任务上，取得了显著的效果。

[论文](https://arxiv.org/pdf/1910.06379.pdf): DUAL-PATH RNN: EFFICIENT LONG SEQUENCE MODELING FOR
TIME-DOMAIN SINGLE-CHANNEL SPEECH SEPARATION

# 模型架构

模型包括  
encoder：编码器，类似fft，提取语音特征。
decoder：解码器，类似ifft，得到语音波形。
separation：类似得到mask，通过mix*单个语音的mask，得到单个语音的一个语谱图。通过decoder解码器还原出语音波形。

# 数据集

使用的数据集为: [librimix](<https://catalog.ldc.upenn.edu/docs/LDC93S1/TIMIT.html>)，LibriMix 是一个开源数据集，用于在嘈杂环境中进行源代码分离。
要生成 LibriMix，请参照开源项目：https://github.com/JorisCos/LibriMix

# 环境要求

- 硬件（ASCEND）
    - ASCEND处理器
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 通过下面网址可以获得更多信息:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 依赖
    - 见requirements.txt文件，使用方法如下:

```python
pip install -r requirements.txt
 ```

# 脚本说明

## 脚本及样例代码

```path
dprnn
├─ requirements.txt                   # requirements
├─ README.md                          # descriptions
├── scripts
  ├─ run_distribute_train.sh          # launch ascend training(8 pcs)
  ├─ run_stranalone_train.sh          # launch ascend training(1 pcs)
  ├─ run_eval.sh                      # launch ascend eval
  ├─ run_infer_310.sh                 # launch infer 310
├─ train.py                           # train script
├─ train_wrapper.py                   # global clip norm Settings
├─ evaluate.py                        # eval
├─ preprocess.py                      # preprocess json
├─ data.py                            # postprocess data
├─ export.py                          # export mindir script
├─ network_define.py                  # define network
├─ model.py                           # dprnn model
├─ loss.py                            # loss function
├─ preprocess_310.py                  # preprocess of 310
└─ postprocess.py                     # postprocess of 310
```

## 脚本参数

数据预处理、训练、评估的相关参数在`train.py`等文件

```text
数据预处理相关参数
in-dir                    预处理前加载原始数据集目录
out-dir                   预处理后的json文件的目录
sample-rate               采样率
train_name                预处理后的训练MindRecord文件的名称
test_name                 预处理后的测试MindRecord文件的名称  
```

```text
训练和模型相关参数
train_dir                  训练集
valid_dir                  测试集
sample_rate                采样率
segment                    取得音频的长度
cv_maxlen                  最大音频长度
in_channels                输入通道数(过滤器数量/掩蔽网络的输入维度)
out_channels               隐藏状态的特征个数
hidden_channels            RNN层状态中的神经元数量
bn_channels                瓶颈层后的通道数
kernel_size                编码器和解码器内核大小
rnn_type                   使用的RNN类型
norm                       要使用的规范化类型
num_layers                 Dual-Path-Block的数量
K                          重叠窗口大小
num_spks                   说话者数量
lr                         学习率
l2                         权重衰减
momentum                   动量
```

```text
评估相关参数
model_path                 ckpt文件路径
cal_sdr                    是否计算SDR
data-dir                   测试集路径
batch_size                 测试集batch大小
```

```text
配置相关参数
device_target                硬件，只支持ASCEND
device_id                    设备号
```

# 数据预处理过程

## 数据预处理

数据预处理运行示例:

```text
python preprocess.py
```

数据预处理过程需要三分钟时间

# 训练过程

## 训练

- ### 单卡训练

运行示例:

```text
python train.py
参数:
train_dir                  训练集
valid_dir                  测试集
sample_rate                采样率
segment                    取得音频的长度
cv_maxlen                  最大音频长度
in_channels                输入通道数(过滤器数量/掩蔽网络的输入维度)
out_channels               隐藏状态的特征个数
hidden_channels            RNN层状态中的神经元数量
bn_channels                瓶颈层后的通道数
kernel_size                编码器和解码器内核大小
rnn_type                   使用的RNN类型
norm                       要使用的规范化类型
num_layers                 Dual-Path-Block的数量
K                          重叠窗口大小
num_spks                   说话者数量
lr                         学习率
l2                         权重衰减
momentum                   动量
```

或者可以运行脚本:

```bash
bash run_standalone_train.sh [DEVICE_ID] [DATA_DIR]
```

上述命令将在后台运行，可以通过train.log查看结果  
每个epoch将运行12小时左右

- ### 分布式训练

分布式训练脚本如下

```bash
bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [DATA_DIR]
```

# 评估过程

## 评估

运行示例:

```text
python eval.py
参数:
model_path                 ckpt文件
data-dir                   测试集路径
batch_size                 测试集batch大小
```

或者可以运行脚本:

```bash
bash run_eval.sh [DEVICE_ID] [CKPT_PATH] [DATA_DIR]
```

上述命令在后台运行，可以通过eval.log查看结果,测试结果如下

# 导出mindir模型

## 导出

```bash
python export.py
```

# 推理过程

## 推理

### 用法

```bash
bash scripts/run_infer_310.sh [MINDIR_PATH] [TEST_PATH] [NEED_PREPROCESS]
```

### 结果

```text
Average SISNR improvement: 12.77
```

# 模型描述

## 性能

### 训练性能

| 参数                 | DPRNN                                                      |
| -------------------------- | ---------------------------------------------------------------|
| 资源                   | Ascend910             |
| 上传日期              | 2022-9-2                                    |
| MindSpore版本           | 1.6.1                                                          |
| 数据集                    | Librimix                                                 |
| 训练参数       | 8p, epoch = 100, batch_size = 4   |
| 优化器                  | Adam                                                           |
| 损失函数              | SI-SNR                                |
| 输出                    | SI-SNR(12.77)                                                    |
| 损失值                       | -16.97                                                       |
| 运行速度                      | 8p 14099.233 ms/step                                   |
| 训练总时间       | 8p:约173h;                                  |                                           |

# 随机情况说明

随机性主要来自下面两点:

- 参数初始化
- 轮换数据集

# ModelZoo主页

 [ModelZoo主页](https://gitee.com/mindspore/models).
