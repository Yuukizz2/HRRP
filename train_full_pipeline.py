#!/usr/bin/env python3
"""
完整的HRRP对比学习训练流程

这个脚本实现了两阶段训练：
1. 对比学习预训练：使用CompositeTripletLoss训练特征表示
2. 分类微调：在预训练模型基础上进行下游分类任务
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse
import time
from config.config import ModelConfig, TrainingConfig, DataConfig
from model.contrastive_hrrp_model import ContrastiveHRRPModel, ClassificationHRRPModel
from model.tripletloss import CompositeTripletLoss
from dataset.hrrp_dataset import HrrpDataset
from train_contrastive import ContrastiveTrainer
from utils.trainer import Trainer


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "full_pipeline_training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def stage1_contrastive_pretraining(
    model_config, training_config, data_config, logger, device
):
    """第一阶段：对比学习预训练"""
    logger.info("=" * 50)
    logger.info("第一阶段：对比学习预训练")
    logger.info("=" * 50)

    start_time = time.time()

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

    # 创建复合三元组损失函数
    criterion = CompositeTripletLoss(
        margin_class=1.0,  # 类间分离边距
        margin_angular=0.2,  # 类内角度排序边距
        lambda_angular=0.5,  # 角度损失权重
    )

    # 创建优化器
    optimizer = torch.optim.AdamW(
        contrastive_model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.num_epochs
    )

    # 创建对比学习训练器
    trainer = ContrastiveTrainer(
        model=contrastive_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=training_config.save_dir,
        log_dir=training_config.log_dir,
        early_stopping_patience=training_config.early_stopping_patience,
    )

    # 开始训练
    trainer.train(training_config.num_epochs)

    # 保存最终模型
    final_checkpoint = {
        "model_state_dict": contrastive_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": trainer.best_val_loss,
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "stage": "contrastive_pretraining",
    }
    torch.save(
        final_checkpoint, Path(training_config.save_dir) / "contrastive_final.pth"
    )

    elapsed_time = time.time() - start_time
    logger.info(f"对比学习预训练完成，耗时: {elapsed_time/3600:.2f}小时")

    return contrastive_model


def stage2_classification_finetuning(
    pretrained_model, model_config, training_config, data_config, logger, device
):
    """第二阶段：分类微调"""
    logger.info("=" * 50)
    logger.info("第二阶段：分类微调")
    logger.info("=" * 50)

    start_time = time.time()

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
        lr=training_config.learning_rate * 0.1,  # 微调时使用较小的学习率
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
        device=device,
        save_dir=training_config.save_dir,
        log_dir=training_config.log_dir,
        early_stopping_patience=training_config.early_stopping_patience,
    )

    # 开始训练
    trainer.train(training_config.num_epochs)

    # 保存最终模型
    final_checkpoint = {
        "model_state_dict": classification_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": trainer.best_val_loss,
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "stage": "classification_finetuning",
    }
    torch.save(
        final_checkpoint, Path(training_config.save_dir) / "classification_final.pth"
    )

    elapsed_time = time.time() - start_time
    logger.info(f"分类微调完成，耗时: {elapsed_time/3600:.2f}小时")

    return classification_model


def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建配置
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置日志
    logger = setup_logging(Path(training_config.log_dir))
    logger.info("开始完整训练流程...")
    logger.info(f"使用设备: {device}")
    logger.info(f"模型配置: {model_config}")
    logger.info(f"训练配置: {training_config}")

    # 第一阶段：对比学习预训练
    if not args.skip_contrastive:
        pretrained_model = stage1_contrastive_pretraining(
            model_config, training_config, data_config, logger, device
        )
    else:
        logger.info("跳过对比学习预训练阶段")
        # 加载已有的预训练模型
        checkpoint_path = Path(training_config.save_dir) / "contrastive_final.pth"
        if checkpoint_path.exists():
            logger.info(f"加载已有预训练模型: {checkpoint_path}")
            pretrained_model = ContrastiveHRRPModel(
                hrrp_len=model_config.sequence_length,
                feature_dim=model_config.d_model,
                nhead=model_config.num_heads,
                num_encoder_layers=model_config.num_layers,
                dim_feedforward=model_config.d_ff,
                projection_hidden_dim=256,
                projection_output_dim=128,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            pretrained_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise FileNotFoundError(f"未找到预训练模型: {checkpoint_path}")

    # 第二阶段：分类微调
    if not args.skip_finetuning:
        final_model = stage2_classification_finetuning(
            pretrained_model, model_config, training_config, data_config, logger, device
        )
    else:
        logger.info("跳过分类微调阶段")

    logger.info("=" * 50)
    logger.info("完整训练流程完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完整HRRP对比学习训练流程")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--skip_contrastive", action="store_true", help="跳过对比学习预训练阶段"
    )
    parser.add_argument(
        "--skip_finetuning", action="store_true", help="跳过分类微调阶段"
    )
    args = parser.parse_args()
    main(args)
