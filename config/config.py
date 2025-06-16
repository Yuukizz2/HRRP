from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ModelConfig:
    # 数据参数
    input_hrrp_dim: int = 1
    sequence_length: int = 256
    num_classes: int = 10

    # 模型架构参数
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # 训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    save_dir: str = "ckpt"
    log_dir: str = "logs"

    # 其他
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    save_path: str = "hrrp_transformer_best.pth"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    save_dir: str = "ckpt"
    log_dir: str = "logs"


@dataclass
class DataConfig:
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    num_workers: int = 4
    pin_memory: bool = True
