import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse
import numpy as np
from config.config import ModelConfig, TrainingConfig, DataConfig
from model.contrastive_hrrp_model import ContrastiveHRRPModel, ClassificationHRRPModel
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
            logging.FileHandler(log_dir / "contrastive_training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


class ContrastiveTrainer:
    """对比学习训练器"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir,
        log_dir,
        early_stopping_patience=10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.early_stopping_patience = early_stopping_patience

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 训练状态
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # 获取数据
            hrrp_data = batch["hrrp"].to(self.device)
            angles_data = torch.stack([batch["azimuth"], batch["elevation"]], dim=1).to(
                self.device
            )
            labels = batch["class"].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            # 获取投影后的特征
            projected_features = self.model(hrrp_data, angles_data)

            # 计算角度信息（用于三元组损失）
            # 这里我们使用方位角作为角度信息
            angles_rad = torch.deg2rad(angles_data[:, 0])  # 使用方位角

            # 计算对比学习损失
            loss = self.criterion(projected_features, labels, angles_rad)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # 获取数据
                hrrp_data = batch["hrrp"].to(self.device)
                angles_data = torch.stack(
                    [batch["azimuth"], batch["elevation"]], dim=1
                ).to(self.device)
                labels = batch["class"].to(self.device)

                # 前向传播
                projected_features = self.model(hrrp_data, angles_data)

                # 计算角度信息
                angles_rad = torch.deg2rad(angles_data[:, 0])

                # 计算损失
                loss = self.criterion(projected_features, labels, angles_rad)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        # 保存最新检查点
        torch.save(checkpoint, self.save_dir / "contrastive_latest.pth")

        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.save_dir / "contrastive_best.pth")
            self.logger.info(f"保存最佳模型，验证损失: {self.best_val_loss:.4f}")

    def train(self, num_epochs):
        """训练模型"""
        self.logger.info("开始对比学习训练...")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            # 记录日志
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 保存检查点
            self.save_checkpoint(is_best)

            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                break

        self.logger.info("对比学习训练完成！")


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
    logger.info("开始对比学习训练...")

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
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_dir=training_config.save_dir,
        log_dir=training_config.log_dir,
        early_stopping_patience=training_config.early_stopping_patience,
    )

    # 开始训练
    trainer.train(training_config.num_epochs)

    logger.info("对比学习训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比学习训练HRRP模型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    main(args)
