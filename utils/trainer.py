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


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        save_dir: str = "ckpt",
        log_dir: str = "logs",
        early_stopping_patience: int = 10,
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
        self.setup_logging()

        # 训练状态
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []

    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / "training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(self.train_loader)

    def calculate_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """计算各种评测指标

        Args:
            outputs: 模型输出，形状为 [batch_size, num_classes]
            targets: 真实标签，形状为 [batch_size]

        Returns:
            包含各种指标的字典
        """
        # 将输出转换为预测类别
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()

        # 计算各项指标
        metrics = {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions, average="weighted"),
            "recall": recall_score(targets, predictions, average="weighted"),
            "f1": f1_score(targets, predictions, average="weighted"),
        }

        # 计算混淆矩阵
        cm = confusion_matrix(targets, predictions)

        # 计算ROC曲线和AUC（仅适用于二分类）
        if outputs.shape[1] == 2:
            fpr, tpr, _ = roc_curve(targets, outputs[:, 1].cpu().numpy())
            metrics["auc"] = auc(fpr, tpr)

            # 绘制ROC曲线
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["auc"]:.2f})')
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend(loc="lower right")
            plt.savefig(self.log_dir / f"roc_curve_epoch_{self.epoch}.png")
            plt.close()

        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(self.log_dir / f"confusion_matrix_epoch_{self.epoch}.png")
        plt.close()

        return metrics

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """验证模型并返回损失和指标"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                all_outputs.append(output)
                all_targets.append(target)

        # 合并所有批次的结果
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算指标
        metrics = self.calculate_metrics(all_outputs, all_targets)

        return total_loss / len(self.val_loader), metrics

    def save_checkpoint(self, is_best: bool = False):
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

        # 如果是最佳模型，额外保存一份
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pth")

    def train(self, num_epochs: int):
        """训练模型"""
        for epoch in range(num_epochs):
            self.epoch = epoch

            # 训练和验证
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 记录日志
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Accuracy: {metrics['accuracy']:.4f} - "
                f"Precision: {metrics['precision']:.4f} - "
                f"Recall: {metrics['recall']:.4f} - "
                f"F1 Score: {metrics['f1']:.4f}"
            )

            if "auc" in metrics:
                self.logger.info(f"AUC: {metrics['auc']:.4f}")

            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(is_best)

            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
