import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse
from config.config import ModelConfig, TrainingConfig, DataConfig
from model.contrastive_hrrp_model import ContrastiveHRRPModel, ClassificationHRRPModel
from dataset.hrrp_dataset import HrrpDataset
from utils.trainer import Trainer


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "classification_finetune.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_pretrained_model(checkpoint_path: str, model_config: ModelConfig):
    """加载预训练的对比学习模型"""
    # 创建对比学习模型
    contrastive_model = ContrastiveHRRPModel(
        hrrp_len=model_config.sequence_length,
        feature_dim=model_config.d_model,
        nhead=model_config.num_heads,
        num_encoder_layers=model_config.num_layers,
        dim_feedforward=model_config.d_ff,
        projection_hidden_dim=256,
        projection_output_dim=128,
    )

    # 加载预训练权重
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    contrastive_model.load_state_dict(checkpoint["model_state_dict"])

    return contrastive_model


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
    logger.info("开始分类微调训练...")

    # 加载预训练的对比学习模型
    logger.info(f"加载预训练模型: {args.pretrained_path}")
    pretrained_model = load_pretrained_model(args.pretrained_path, model_config)

    # 创建分类模型
    classification_model = ClassificationHRRPModel(
        contrastive_model=pretrained_model,
        num_classes=model_config.num_classes,
    )

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

    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 只优化分类头的参数
    optimizer = torch.optim.AdamW(
        classification_model.classifier.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.num_epochs
    )

    # 创建训练器
    trainer = Trainer(
        model=classification_model,
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

    logger.info("分类微调训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分类微调训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="ckpt/contrastive_best.pth",
        help="预训练模型路径",
    )
    args = parser.parse_args()
    main(args)
