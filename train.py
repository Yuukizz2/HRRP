import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse
from config.config import ModelConfig, TrainingConfig, DataConfig
from model.hrrp_model import HrrpTransformerClassifier
from model.tripletloss import CompositeTripletLoss
from dataset.hrrp_dataset import HrrpDataset
from utils.trainer import Trainer


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建配置
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()

    # 设置日志
    logger = setup_logging(Path(training_config.log_dir))
    logger.info("Starting training...")

    # 创建数据集
    train_dataset = HrrpDataset(
        data_path=data_config.train_data_path,
        transform={"normalize": True, "augment": True},
    )
    val_dataset = HrrpDataset(
        data_path=data_config.val_data_path,
        transform={"normalize": True, "augment": False},
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )

    # 创建模型
    model = HrrpTransformerClassifier(
        input_hrrp_dim=model_config.input_hrrp_dim,
        sequence_length=model_config.sequence_length,
        num_classes=model_config.num_classes,
        num_layers=model_config.num_layers,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        dropout=model_config.dropout,
    )

    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.num_epochs
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_dir=training_config.save_dir,
        log_dir=training_config.log_dir,
        early_stopping_patience=training_config.early_stopping_patience,
    )

    # 开始训练
    trainer.train(training_config.num_epochs)

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRRP Transformer model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
