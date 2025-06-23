from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ModelConfig:
    sequence_length: int = 512
    num_classes: int = 8

    # 模型架构参数
    num_layers: int = 4
    d_model: int = 1024
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # 其他
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path: str = "hrrp_transformer_best.pth"


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1
    num_epochs: int = 50
    warmup_steps: int = 1000
    weight_decay: float = 0.001
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 20
    save_dir: str = "ckpt"
    log_dir: str = "logs"


@dataclass
class DataConfig:
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    num_workers: int = 8
    pin_memory: bool = True
