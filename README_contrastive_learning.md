# HRRP对比学习训练指南

本项目实现了基于对比学习的HRRP（高分辨率距离像）目标识别系统，结合了您提供的三个核心组件：

1. **CompositeTripletLoss**: 复合三元组损失函数
2. **MyCustomModel**: 基于角度信息的HRRP特征提取模型
3. **训练框架**: 完整的训练和评估流程

## 系统架构

### 核心组件

#### 1. 对比学习模型 (`model/contrastive_hrrp_model.py`)

- **ContrastiveHRRPModel**: 主要的对比学习模型
  - HRRP特征提取器（1D CNN）
  - 角度信息编码器（MLP）
  - 旋转融合层
  - Transformer编码器
  - 投影头（用于对比学习）

- **ClassificationHRRPModel**: 分类模型
  - 基于预训练的对比学习模型
  - 冻结编码器参数
  - 只训练分类头

#### 2. 复合三元组损失 (`model/tripletloss.py`)

- **L_class**: 类间分离损失
- **L_angular**: 类内角度排序损失
- **λ平衡**: 两个损失的权重平衡

#### 3. 训练脚本

- `train_contrastive.py`: 对比学习预训练
- `finetune_classification.py`: 分类微调
- `train_full_pipeline.py`: 完整训练流程

## 使用方法

### 1. 环境准备

确保安装了所需的依赖：

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn tqdm
```

### 2. 数据准备

确保您的数据格式符合`HrrpDataset`的要求：

```python
{
    "hrrp": torch.Tensor,      # HRRP信号数据
    "azimuth": torch.Tensor,   # 方位角
    "elevation": torch.Tensor, # 俯仰角
    "class": torch.Tensor,     # 类别标签
}
```

### 3. 训练流程

#### 方式一：完整流程训练（推荐）

```bash
# 完整的两阶段训练
python train_full_pipeline.py

# 只进行对比学习预训练
python train_full_pipeline.py --skip_finetuning

# 只进行分类微调（需要已有预训练模型）
python train_full_pipeline.py --skip_contrastive
```

#### 方式二：分阶段训练

```bash
# 第一阶段：对比学习预训练
python train_contrastive.py

# 第二阶段：分类微调
python finetune_classification.py --pretrained_path ckpt/contrastive_best.pth
```

### 4. 配置参数

在`config/config.py`中可以调整以下参数：

#### 模型配置

```python
@dataclass
class ModelConfig:
    sequence_length: int = 256    # HRRP序列长度
    d_model: int = 512           # 特征维度
    num_heads: int = 8           # 注意力头数
    num_layers: int = 6          # Transformer层数
    num_classes: int = 10        # 类别数量
```

#### 训练配置

```python
@dataclass
class TrainingConfig:
    batch_size: int = 32         # 批次大小
    learning_rate: float = 1e-4  # 学习率
    num_epochs: int = 100        # 训练轮数
    weight_decay: float = 0.01   # 权重衰减
```

#### 三元组损失配置

```python
# 在训练脚本中调整
criterion = CompositeTripletLoss(
    margin_class=1.0,      # 类间分离边距
    margin_angular=0.2,    # 类内角度排序边距
    lambda_angular=0.5,    # 角度损失权重
)
```

## 训练策略

### 1. 对比学习预训练

**目标**: 学习具有良好表示能力的特征空间

**损失函数**: `CompositeTripletLoss`

- 类间分离：确保不同类别的样本在特征空间中分离
- 类内角度排序：根据角度信息对同类样本进行排序

**关键特点**:

- 使用难例挖掘（Hard Mining）
- 结合角度信息的周期性距离计算
- 平衡类间和类内损失

### 2. 分类微调

**目标**: 在预训练特征基础上进行下游分类任务

**策略**:

- 冻结编码器参数
- 只训练分类头
- 使用较小的学习率（原始学习率的0.1倍）

## 模型优势

### 1. 角度感知

- 利用方位角和俯仰角信息
- 通过旋转融合层增强特征表示
- 考虑角度信息的周期性

### 2. 对比学习

- 学习判别性特征表示
- 减少对标注数据的依赖
- 提高模型的泛化能力

### 3. 复合损失

- 同时优化类间分离和类内排序
- 平衡不同损失项的权重
- 使用难例挖掘提高训练效率

## 监控和评估

### 1. 训练监控

- 训练损失和验证损失
- 学习率变化
- 早停机制

### 2. 模型保存

- 自动保存最佳模型
- 保存训练历史
- 支持断点续训

### 3. 日志记录

- 详细的训练日志
- 损失曲线可视化
- 模型性能指标

## 常见问题

### 1. 内存不足

```bash
# 减小批次大小
--batch_size 16

# 减少工作进程数
--num_workers 2
```

### 2. 训练不收敛

- 检查学习率设置
- 调整损失函数权重
- 增加数据增强

### 3. 过拟合

- 增加权重衰减
- 使用更多数据增强
- 减少模型复杂度

## 扩展功能

### 1. 自定义损失函数

可以修改`CompositeTripletLoss`的参数：

```python
# 调整边距
margin_class = 0.8
margin_angular = 0.1

# 调整权重
lambda_angular = 0.3
```

### 2. 数据增强

在`HrrpDataset`中添加更多数据增强策略：

```python
# 添加噪声
noise_std = 0.02

# 随机缩放
scale_std = 0.15

# 时间偏移
time_shift = True
```

### 3. 模型架构

可以修改`ContrastiveHRRPModel`的架构：

```python
# 调整投影头
projection_hidden_dim = 512
projection_output_dim = 256

# 调整Transformer
num_encoder_layers = 8
nhead = 16
```

## 性能优化建议

1. **数据预处理**: 确保数据标准化和增强
2. **批次大小**: 根据GPU内存调整
3. **学习率调度**: 使用余弦退火调度
4. **早停**: 设置合适的早停参数
5. **混合精度**: 对于大模型考虑使用FP16

## 引用

如果您使用了这个对比学习框架，请引用相关论文：

```bibtex
@article{hrrp_contrastive_2024,
  title={Contrastive Learning for HRRP Target Recognition with Angular Information},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```
