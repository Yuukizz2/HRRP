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
        num_classes,
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
        self.num_classes = num_classes

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 在CLS token上添加分类层，使用更好的初始化
        self.cls_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        ).to(device)
        
        # 使用标签平滑的交叉熵损失
        self.classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 训练状态
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def compute_accuracy(self, cls_features, labels):
        """使用CLS token计算分类准确率"""
        with torch.no_grad():
            logits = self.cls_classifier(cls_features)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
        return accuracy

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
        self.cls_classifier.train()
        total_loss = 0.0
        total_accuracy = 0.0
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
            
            # 获取CLS token特征（用于分类）
            cls_features = self.model.forward_encoder(hrrp_data, angles_data)

            # 计算角度信息（用于三元组损失）
            angles_rad = torch.deg2rad(angles_data[:, 0])

            # 计算对比学习损失
            contrastive_loss = self.criterion(projected_features, labels, angles_rad)
            
            # 计算分类损失（使用CLS token）
            logits = self.cls_classifier(cls_features)
            classification_loss = self.classification_criterion(logits, labels)
            
            # 动态调整损失权重：随着训练进行，逐渐增加分类损失的权重
            classification_weight = min(0.5, 0.1 + self.epoch * 0.02)
            total_batch_loss = contrastive_loss + classification_weight * classification_loss

            # 反向传播
            total_batch_loss.backward()
            
            # # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            # torch.nn.utils.clip_grad_norm_(self.cls_classifier.parameters(), max_norm=0.5)
            
            self.optimizer.step()

            # 计算准确率
            accuracy = self.compute_accuracy(cls_features, labels)

            total_loss += total_batch_loss.item()
            total_accuracy += accuracy
            num_batches += 1

            # 每100个batch打印一次进度
            if batch_idx % 100 == 0:
                self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {total_batch_loss.item():.4f}, "
                               f"Acc: {accuracy:.4f}")

        return total_loss / num_batches, total_accuracy / num_batches

    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        self.cls_classifier.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []

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
                cls_features = self.model.forward_encoder(hrrp_data, angles_data)

                # 计算角度信息
                angles_rad = torch.deg2rad(angles_data[:, 0])

                # 计算对比学习损失
                contrastive_loss = self.criterion(projected_features, labels, angles_rad)
                
                # 计算分类损失（使用CLS token）
                logits = self.cls_classifier(cls_features)
                classification_loss = self.classification_criterion(logits, labels)
                
                # 总损失
                classification_weight = min(0.5, 0.1 + self.epoch * 0.02)
                total_batch_loss = contrastive_loss + classification_weight * classification_loss

                # 计算准确率
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean().item()

                # 收集预测结果用于详细分析
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_loss += total_batch_loss.item()
                total_accuracy += accuracy
                num_batches += 1

        # 计算整体准确率
        overall_accuracy = sum(all_predictions[i] == all_labels[i] for i in range(len(all_predictions))) / len(all_predictions)
        
        # 生成详细的分类报告
        self._log_classification_report(all_predictions, all_labels)
        
        return total_loss / num_batches, overall_accuracy

    def _log_classification_report(self, predictions, labels):
        """记录详细的分类报告"""
        # 计算每个类别的指标
        class_report = {}
        for class_id in range(self.num_classes):
            class_predictions = [1 if p == class_id else 0 for p in predictions]
            class_labels = [1 if l == class_id else 0 for l in labels]
            
            # 计算该类别的指标
            tp = sum(1 for p, l in zip(class_predictions, class_labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(class_predictions, class_labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(class_predictions, class_labels) if p == 0 and l == 1)
            tn = sum(1 for p, l in zip(class_predictions, class_labels) if p == 0 and l == 0)
            
            # 计算精确率、召回率和F1分数
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_report[class_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(class_labels)
            }
        
        # 记录分类报告
        self.logger.info(f"Epoch {self.epoch} 分类报告:")
        for class_id, metrics in class_report.items():
            self.logger.info(f"  Class_{class_id}: Precision={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
                           f"Support={metrics['support']}")

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "cls_classifier_state_dict": self.cls_classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }

        # 保存最新检查点
        torch.save(checkpoint, self.save_dir / "latest_checkpoint.pth")

        # 如果是最佳模型，保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pth")
            self.logger.info(f"保存最佳模型，验证损失: {self.best_val_loss:.4f}, 验证准确率: {self.best_val_accuracy:.4f}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, num_epochs):
        """训练模型"""
        self.logger.info("开始改进的对比学习训练...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # 记录损失和准确率
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # 记录训练信息
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {self.epoch}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                           f"LR: {current_lr:.6f}")
            
            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
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
        
        self.logger.info(f"训练完成！最佳验证准确率: {self.best_val_accuracy:.4f}")
        self.plot_training_curves()