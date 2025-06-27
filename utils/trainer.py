import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Tuple
import os
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt


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
        early_stopping_patience=15,
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

    def normalize_hrrp(self, hrrp_data):
        """标准化HRRP数据"""
        # 对每个样本进行标准化
        mean = hrrp_data.mean(dim=-1, keepdim=True)
        std = hrrp_data.std(dim=-1, keepdim=True) + 1e-8
        normalized = (hrrp_data - mean) / std
        return normalized

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 获取数据
            hrrp_data = batch["hrrp"].to(self.device)
            angles_data = torch.stack([batch["azimuth"], batch["elevation"]], dim=1).to(
                self.device
            )
            labels = batch["class"].to(self.device)

            # 数据预处理：标准化HRRP
            hrrp_data = self.normalize_hrrp(hrrp_data)

            # 前向传播
            self.optimizer.zero_grad()

            # 获取投影后的特征（用于对比学习）
            projected_features = self.model(hrrp_data, angles_data)

            # 计算角度信息（用于三元组损失）
            angles_rad = torch.deg2rad(angles_data[:, 0])

            # 计算对比学习损失
            contrastive_loss = self.criterion(projected_features, labels, angles_rad)

            # 反向传播
            contrastive_loss.backward()

            self.optimizer.step()

            total_loss += contrastive_loss.item()
            num_batches += 1

            # 每100个batch打印一次进度
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {contrastive_loss.item():.4f}"
                )

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

                # 数据预处理：标准化HRRP
                hrrp_data = self.normalize_hrrp(hrrp_data)

                # 前向传播
                projected_features = self.model(hrrp_data, angles_data)

                # 计算角度信息
                angles_rad = torch.deg2rad(angles_data[:, 0])

                # 计算对比学习损失
                contrastive_loss = self.criterion(
                    projected_features, labels, angles_rad
                )

                total_loss += contrastive_loss.item()
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
        torch.save(checkpoint, self.save_dir / "latest_checkpoint.pth")

        # 如果是最佳模型，保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pth")
            self.logger.info(f"保存最佳模型，验证损失: {self.best_val_loss:.4f}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        # 损失曲线
        ax.plot(self.train_losses, label="Train Loss", color="blue")
        ax.plot(self.val_losses, label="Val Loss", color="red")
        ax.set_title("Training and Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.log_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    def train(self, num_epochs):
        """训练模型"""
        self.logger.info("开始改进的对比学习训练...")

        for epoch in range(num_epochs):
            self.epoch = epoch + 1

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss = self.validate_epoch()

            # 学习率调度
            if self.scheduler:
                self.scheduler.step(val_loss)

            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 记录训练信息
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {self.epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"LR: {current_lr:.6f}"
            )

            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(is_best=False)

            # # 早停
            # if self.patience_counter >= self.early_stopping_patience:
            #     self.logger.info(f"早停触发，在epoch {self.epoch}停止训练")
            #     break

            # 每10个epoch绘制一次训练曲线
            if self.epoch % 10 == 0:
                self.plot_training_curves()

        self.logger.info(f"训练完成！最佳验证损失: {self.best_val_loss:.4f}")
        self.plot_training_curves()
