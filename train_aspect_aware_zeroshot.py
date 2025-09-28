import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset.advanced_dataset_splitter import AdvancedDatasetSplitter
from model.aspect_aware_hrrp_model import AspectAwareHRRPModel, AspectAwareClassifier
from model.aspect_alignment_loss import AspectAlignmentLoss, HierarchicalAspectLoss
from data.embedding_tensor import build_label_embeddings_tensor
import matplotlib.pyplot as plt
import random
import logging
import time
from pathlib import Path
import argparse
from typing import Optional, List, Dict, Tuple, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from dataclasses import dataclass
from tqdm import tqdm
import warnings
from contextlib import contextmanager

# 忽略警告
warnings.filterwarnings("ignore")

# 导入新的数据加载器


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 数据相关
    embedding_dir: str = "data/embedding/Qwen_QwQ-32B"
    balanced_data_path: str = "data/zeroshot/balanced_data.pt"
    
    # 模型相关
    hrrp_len: int = 512
    feature_dim: int = 512
    num_aspects: int = 11
    nhead: int = 8
    num_layers: int = 2
    label_embedding_dim: int = 768
    num_labels: int = 8
    
    # 训练相关
    num_epochs: int = 50
    lr: float = 0.0005
    batch_size: int = 32
    weight_decay: float = 0.1
    gradient_clip_norm: float = 0.5
    early_stopping_patience: int = 10
    min_delta: float = 0.001
    
    # 损失相关
    loss_type: str = 'aspect_alignment'
    align_loss_weight: float = 1
    calibration_factor: float = 0.5
    
    # 系统相关
    save_dir: str = "ckpt"
    experiment_name: Optional[str] = None
    ckpt_path: Optional[str] = None
    num_workers: int = 0  # 设置为0避免多进程问题
    device: Optional[str] = None
    
    # 零样本相关
    seen_classes: List[str] = None
    unseen_classes: List[str] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Aspect名称
    aspect_names: List[str] = None
    
    def __post_init__(self):
        if self.aspect_names is None:
            self.aspect_names = [
                "flightCharacteristics", "microDopplerSignature", "basicInformation",
                "thermalSignature", "roleAndMission", "geometricCharacteristics",
                "materialsAndCoatings", "aircraftModel", "radarCrossSection",
                "engineAndPerformance", "global"
            ]
        
        if self.seen_classes is None:
            self.seen_classes = ["A321", "A330", "ARJ21", "BY737", "BY787", "KJ600"]
        
        if self.unseen_classes is None:
            self.unseen_classes = ["Z8", "Y20"]
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

class Logger:
    """改进的日志记录器"""
    
    def __init__(self, save_dir: str, experiment_name: str):
        self.experiment_name = experiment_name
        
        self.log_dir = Path(save_dir) / self.experiment_name / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / "training.log"
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录器"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        self.logger = logger
        self.logger.info(f"日志文件保存在: {self.log_file}")
        self.logger.info(f"实验名称: {self.experiment_name}")
    
    def info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)

class EarlyStopping:
    """改进的早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """保存最佳权重"""
        self.best_weights = model.state_dict().copy()
    
    def get_best_epoch(self) -> int:
        """获取最佳epoch"""
        return self.best_epoch

@contextmanager
def timer(description: str, logger: Optional[Logger] = None):
    """计时上下文管理器"""
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if logger:
        logger.info(f"{description} 耗时: {elapsed_time:.2f}秒")
    else:
        print(f"{description} 耗时: {elapsed_time:.2f}秒")

def validate_model(
    classifier: AspectAwareClassifier, 
    hrrp_model: AspectAwareHRRPModel, 
    val_loader: DataLoader, 
    device: torch.device, 
    criterion: nn.Module, 
    aspect_loss_fn: nn.Module,
    config: TrainingConfig
) -> Tuple[float, float]:
    """验证模型性能"""
    classifier.eval()
    hrrp_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中", leave=False):
            hrrp_data, class_names = batch
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            
            # 将字符串标签转换为数字索引
            labels = torch.tensor([config.label_to_aircraft[cls] for cls in class_names]).to(device, non_blocking=True)
            
            # 前向传播
            hrrp_features, aspect_features, aspect_weights = hrrp_model(hrrp_data)
            logits, _, _ = classifier(hrrp_features, aspect_features, aspect_weights)
            
            # 计算损失 - 仅使用交叉熵损失
            total_loss_batch = criterion(logits, labels)
            
            total_loss += total_loss_batch.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def test_zeroshot_performance(
    classifier: AspectAwareClassifier, 
    hrrp_model: AspectAwareHRRPModel, 
    test_loader: DataLoader, 
    device: torch.device, 
    seen_classes: List[int], 
    unseen_classes: List[int],
    config: TrainingConfig,
    logger: Logger
) -> Tuple[float, float, float, Dict]:
    """测试零样本性能"""
    classifier.eval()
    hrrp_model.eval()
    all_preds = []
    all_labels = []
    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0
    
    # 用于保存unseen类的softmax概率
    unseen_softmax_probs = []
    unseen_sample_info = []

    with torch.no_grad():
        for batch_idx, (hrrp_data, class_names) in enumerate(tqdm(test_loader, desc="测试中")):
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            
            # 将字符串标签转换为数字索引
            labels = torch.tensor([config.label_to_aircraft[cls] for cls in class_names]).to(device, non_blocking=True)
            
            hrrp_features, aspect_features, aspect_weights = hrrp_model(hrrp_data)
            logits, _, _ = classifier(hrrp_features, aspect_features, aspect_weights)
            
            # 计算softmax概率
            softmax_probs = F.softmax(logits, dim=1)
            
            # 校准因子
            penalty_mask = torch.ones_like(logits)
            penalty_mask[:, seen_classes] = config.calibration_factor
            calibrated_logits = logits * penalty_mask
            preds = calibrated_logits.argmax(dim=1)

            for i, (pred, label) in enumerate(zip(preds.cpu().numpy(), labels.cpu().numpy())):
                all_preds.append(pred)
                all_labels.append(label)
                
                # 保存unseen类的softmax概率
                if label in unseen_classes:
                    sample_idx = batch_idx * test_loader.batch_size + i
                    sample_probs = softmax_probs[i].cpu().numpy()
                    
                    # 保存样本信息
                    sample_info = {
                        'sample_idx': sample_idx,
                        'true_label': int(label),
                        'predicted_label': int(pred),
                        'is_correct': int(pred == label),
                        'softmax_probs': sample_probs.tolist(),
                        'max_prob': float(sample_probs.max()),
                        'unseen_class_prob': float(sample_probs[label]),  # 真实unseen类的概率
                        'top3_probs': [(int(j), float(sample_probs[j])) for j in sample_probs.argsort()[-3:][::-1]]
                    }
                    unseen_sample_info.append(sample_info)
                    unseen_softmax_probs.append(sample_probs)
                    
                    logger.info(f"[UNSEEN] Sample {sample_idx}: pred={pred}, true={label}, "
                              f"unseen_prob={sample_probs[label]:.4f}, max_prob={sample_probs.max():.4f}")
                else:
                    logger.info(f"Sample {batch_idx * test_loader.batch_size + i}: pred={pred}, true={label}")
                    
                if label in seen_classes:
                    seen_total += 1
                    if pred == label:
                        seen_correct += 1
                elif label in unseen_classes:
                    unseen_total += 1
                    if pred == label:
                        unseen_correct += 1

    total_acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    seen_acc = seen_correct / seen_total if seen_total > 0 else 0
    unseen_acc = unseen_correct / unseen_total if unseen_total > 0 else 0
    print(f"unseen_total: {unseen_total}")
    
    # 整理unseen类的分析结果
    unseen_analysis = {
        'unseen_samples': unseen_sample_info,
        'unseen_softmax_probs': np.array(unseen_softmax_probs).tolist() if unseen_softmax_probs else [],
        'unseen_class_avg_prob': float(np.mean([info['unseen_class_prob'] for info in unseen_sample_info])) if unseen_sample_info else 0.0,
        'unseen_class_max_prob': float(np.max([info['unseen_class_prob'] for info in unseen_sample_info])) if unseen_sample_info else 0.0,
        'unseen_class_min_prob': float(np.min([info['unseen_class_prob'] for info in unseen_sample_info])) if unseen_sample_info else 0.0,
        'unseen_correct_predictions': sum([info['is_correct'] for info in unseen_sample_info]),
        'unseen_total_predictions': len(unseen_sample_info)
    }
    
    return total_acc, seen_acc, unseen_acc, unseen_analysis

def save_checkpoint(
    model_dict: Dict[str, torch.Tensor], 
    optimizer: optim.Optimizer, 
    epoch: int, 
    loss: float, 
    save_path: Path,
    logger: Logger
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    torch.save(checkpoint, save_path)
    logger.info(f"检查点已保存到: {save_path}")

def load_checkpoint(
    checkpoint_path: Path, 
    classifier: AspectAwareClassifier, 
    hrrp_model: AspectAwareHRRPModel, 
    optimizer: optim.Optimizer, 
    device: torch.device,
    logger: Logger
) -> Tuple[int, float]:
    """加载检查点"""
    if checkpoint_path.exists():
        try:
            logger.info(f"加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            classifier.load_state_dict(checkpoint['model_state_dict']['classifier'])
            hrrp_model.load_state_dict(checkpoint['model_state_dict']['hrrp_model'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            return checkpoint['epoch'], checkpoint['loss']
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return 0, float('inf')
    return 0, float('inf')

def plot_unseen_softmax_distribution(
    unseen_softmax_probs: List[List[float]], 
    unseen_samples: List[Dict], 
    save_path: Path,
    label_to_aircraft: Dict[int, str]
):
    """绘制unseen类softmax概率分布图"""
    try:
        if not unseen_softmax_probs:
            return
            
        unseen_probs = np.array(unseen_softmax_probs)
        num_classes = unseen_probs.shape[1]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Unseen类概率分布直方图
        unseen_class_probs = [sample['unseen_class_prob'] for sample in unseen_samples]
        ax1.hist(unseen_class_probs, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax1.set_title('Unseen Class Probability Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Unseen Class Probability', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(unseen_class_probs), color='blue', linestyle='--', 
                   label=f'Mean: {np.mean(unseen_class_probs):.3f}')
        ax1.legend()
        
        # 2. 所有类别的平均概率
        class_avg_probs = np.mean(unseen_probs, axis=0)
        class_names = [label_to_aircraft.get(i, f'Class_{i}') for i in range(num_classes)]
        colors = ['red' if i in [7] else 'blue' for i in range(num_classes)]  # unseen类标红
        
        bars = ax2.bar(range(num_classes), class_avg_probs, color=colors, alpha=0.7)
        ax2.set_title('Average Probability for All Classes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Average Probability', fontsize=12)
        ax2.set_xticks(range(num_classes))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 正确vs错误预测的概率分布
        correct_probs = [sample['unseen_class_prob'] for sample in unseen_samples if sample['is_correct']]
        incorrect_probs = [sample['unseen_class_prob'] for sample in unseen_samples if not sample['is_correct']]
        
        if correct_probs:
            ax3.hist(correct_probs, bins=15, alpha=0.7, color='green', label='Correct Predictions', edgecolor='black')
        if incorrect_probs:
            ax3.hist(incorrect_probs, bins=15, alpha=0.7, color='orange', label='Incorrect Predictions', edgecolor='black')
        
        ax3.set_title('Unseen Class Probability: Correct vs Incorrect', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Unseen Class Probability', fontsize=12)
        ax3.set_ylabel('Number of Samples', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Top-3预测概率分布
        top1_probs = [sample['top3_probs'][0][1] for sample in unseen_samples]
        top2_probs = [sample['top3_probs'][1][1] for sample in unseen_samples]
        top3_probs = [sample['top3_probs'][2][1] for sample in unseen_samples]
        
        ax4.hist(top1_probs, bins=15, alpha=0.7, color='red', label='Top-1', edgecolor='black')
        ax4.hist(top2_probs, bins=15, alpha=0.7, color='blue', label='Top-2', edgecolor='black')
        ax4.hist(top3_probs, bins=15, alpha=0.7, color='green', label='Top-3', edgecolor='black')
        
        ax4.set_title('Top-3 Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Probability', fontsize=12)
        ax4.set_ylabel('Number of Samples', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'unseen_softmax_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Unseen class softmax analysis plot saved to: {save_path / 'unseen_softmax_analysis.png'}")
        
    except Exception as e:
        print(f"Failed to plot unseen class softmax distribution: {e}")

def plot_training_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    val_accuracies: List[float], 
    loss_components_history: List[Dict[str, float]], 
    save_path: Path,
    logger: Logger
):
    """绘制训练曲线"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 总损失曲线
        ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        ax1.plot(val_losses, label='Val Loss', color='red', linewidth=2)
        ax1.set_title('Train & Val Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(val_accuracies, label='Val Accuracy', color='green', linewidth=2)
        ax2.set_title('Val Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"训练曲线已保存到: {save_path / 'training_curves.png'}")
    except Exception as e:
        logger.error(f"绘制训练曲线失败: {e}")

def train_aspect_aware_model(config: TrainingConfig, experiment_name: str) -> Tuple[AspectAwareClassifier, AspectAwareHRRPModel, Dict[int, str]]:
    """训练Aspect-Aware HRRP模型"""
    
    device = torch.device(config.device)
    
    save_path = Path(config.save_dir) / experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = Logger(config.save_dir, experiment_name)
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"保存目录: {save_path}")
    logger.info(f"使用设备: {device}")
    
    # 构建标签嵌入张量
    with timer("构建标签嵌入张量", logger):
        label_embeddings_tensor = build_label_embeddings_tensor(
            config.embedding_dir, 
            config.label_jsons, 
            config.aspect_names,
            config.label_to_aircraft
        )
    
    # 创建模型
    logger.info("初始化Aspect-Aware模型...")
    hrrp_model = AspectAwareHRRPModel(
        hrrp_len=config.hrrp_len,
        feature_dim=config.feature_dim,
        num_aspects=config.num_aspects,
        nhead=config.nhead,
        num_layers=config.num_layers
    ).float().to(device)
    
    classifier = AspectAwareClassifier(
        hrrp_dim=config.feature_dim,
        num_labels=config.num_labels,
        label_aspects=config.num_aspects,
        label_embedding_dim=config.label_embedding_dim,
        label_embeddings_tensor=label_embeddings_tensor,
        trainable_labels=True
    ).float().to(device)
    
    # 加载预训练模型
    if config.ckpt_path and os.path.exists(config.ckpt_path):
        try:
            logger.info(f"加载预训练HRRP模型: {config.ckpt_path}")
            checkpoint = torch.load(config.ckpt_path, map_location=device)
            hrrp_model.load_state_dict(checkpoint["model_state_dict"])
            hrrp_model.eval()
        except Exception as e:
            logger.error(f"加载预训练模型失败: {e}")
    else:
        logger.info("未找到预训练模型，将使用随机初始化的HRRP模型")
    
    # 初始化损失函数
    if config.loss_type == 'aspect_alignment':
        aspect_loss_fn = AspectAlignmentLoss(config.label_embedding_dim)
    elif config.loss_type == 'hierarchical':
        aspect_loss_fn = HierarchicalAspectLoss(config.label_embedding_dim)
    else:
        raise ValueError(f"Unknown loss_type: {config.loss_type}")
    
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.AdamW([
        {'params': classifier.parameters(), 'lr': config.lr},
        {'params': hrrp_model.parameters(), 'lr': config.lr * 0.1}
    ], weight_decay=config.weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.min_delta)
    
    # 使用新的数据加载方式
    logger.info("使用新的数据加载方式...")
    splitter = AdvancedDatasetSplitter(config.balanced_data_path)
    
    # 进行数据划分
    splits = splitter.split_data(
        seen_classes=config.seen_classes,
        unseen_classes=config.unseen_classes,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=42
    )
    
    # 创建DataLoader
    dataloaders = splitter.create_dataloader_splits(splits, batch_size=config.batch_size)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    logger.info(f"训练集样本数: {len(train_loader.dataset)}")
    logger.info(f"验证集样本数: {len(val_loader.dataset)}")
    logger.info(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_accuracies = []
    loss_components_history = []
    best_val_loss = float('inf')
    
    logger.info("=" * 50)
    logger.info("Aspect-Aware训练配置:")
    logger.info(f"  训练轮数: {config.num_epochs}")
    logger.info(f"  学习率: {config.lr}")
    logger.info(f"  批次大小: {config.batch_size}")
    logger.info(f"  早停耐心值: {config.early_stopping_patience}")
    logger.info(f"  Seen类别: {config.seen_classes}")
    logger.info(f"  Unseen类别: {config.unseen_classes}")
    logger.info("=" * 50)
    
    logger.info(f"开始训练，从第 0 轮开始...")
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        classifier.train()
        hrrp_model.train()
        epoch_loss = 0.0
        epoch_loss_components = {}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, (hrrp_data, class_names) in enumerate(progress_bar):
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            
            # 将字符串标签转换为数字索引
            labels = torch.tensor([config.label_to_aircraft[cls] for cls in class_names]).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            hrrp_features, aspect_features, aspect_weights = hrrp_model(hrrp_data)
            logits, _, _ = classifier(hrrp_features, aspect_features, aspect_weights)
            
            # 计算损失 - 仅使用交叉熵损失
            total_loss = criterion(logits, labels)
            ce_loss = total_loss
            align_loss = torch.tensor(0.0, device=device)  # 占位符，保持兼容性
            loss_components = {'ce_loss': ce_loss.detach().cpu().item(), 'align_loss': 0.0}
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=config.gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(hrrp_model.parameters(), max_norm=config.gradient_clip_norm)
            
            optimizer.step()
            
            # 清理缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            epoch_loss += total_loss.item()
            
            # 累积损失组件
            for key, value in loss_components.items():
                if key not in epoch_loss_components:
                    epoch_loss_components[key] = []
                epoch_loss_components[key].append(value)
            
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}'
            })
            
            # 调试信息：每100个batch打印一次损失
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: CE={ce_loss.item():.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 计算平均损失组件
        avg_loss_components = {key: np.mean(values) for key, values in epoch_loss_components.items()}
        loss_components_history.append(avg_loss_components)
        
        # 验证阶段
        val_loss, val_acc = validate_model(classifier, hrrp_model, val_loader, device, criterion, aspect_loss_fn, config)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        logger.info(f'Epoch {epoch+1}/{config.num_epochs}, '
                   f'Train Loss: {avg_train_loss:.4f}, '
                   f'Val Loss: {val_loss:.4f}, '
                   f'Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_path / 'best_model.pth'
            torch.save({
                'hrrp_model_state_dict': hrrp_model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'label_to_aircraft': config.label_to_aircraft,
                'seen_classes': config.seen_classes,
                'unseen_classes': config.unseen_classes,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'experiment_name': experiment_name,
                'config': config
            }, best_model_path)
            logger.info(f"新的最佳模型已保存: {best_model_path}")
        
        # 保存最新检查点
        checkpoint_path = save_path / "latest_checkpoint.pth"
        save_checkpoint(
            {
                'classifier': classifier.state_dict(),
                'hrrp_model': hrrp_model.state_dict()
            },
            optimizer, epoch, val_loss, checkpoint_path, logger
        )
        
        # 早停检查
        if early_stopping(val_loss, classifier, epoch):
            logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
            logger.info(f"最佳模型在第 {early_stopping.get_best_epoch() + 1} 轮")
            break
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_accuracies, loss_components_history, save_path, logger)
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'loss_components_history': loss_components_history,
        'best_val_loss': best_val_loss,
        'experiment_name': experiment_name,
        'total_epochs': len(train_losses),
        'config': config.__dict__
    }
    
    history_path = save_path / "training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 50)
    logger.info("Aspect-Aware训练完成！")
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"最终验证准确率: {val_accuracies[-1]:.4f}")
    logger.info(f"总训练轮数: {len(train_losses)}")
    logger.info(f"模型保存路径: {save_path}")
    logger.info("=" * 50)
    
    return classifier, hrrp_model, config.label_to_aircraft, test_loader

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练Aspect-Aware HRRP分类器（零样本版本）')
    parser.add_argument('--embedding_dir', type=str, default="data/embedding/Qwen_QwQ-32B",
                        help='嵌入文件目录')
    parser.add_argument('--balanced_data_path', type=str, default="data/zeroshot/balanced_data.pt",
                        help='平衡数据路径')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--save_dir', type=str, default="ckpt",
                        help='模型保存目录')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--experiment_name', type=str, default="aspect_aware_zeroshot",
                        help='实验名称')
    parser.add_argument('--loss_type', type=str, default='aspect_alignment', 
                        choices=['aspect_alignment', 'hierarchical'],
                        help='损失函数类型')
    parser.add_argument('--align_loss_weight', type=float, default=1.0,
                        help='对齐损失权重')
    parser.add_argument('--calibration_factor', type=float, default=0.5,
                        help='校准因子')
    parser.add_argument('--seen_classes', nargs='+', 
                        default=["A321", "A330", "ARJ21", "BY737", "BY787", "KJ600"],
                        help='seen类别列表')
    parser.add_argument('--unseen_classes', nargs='+', 
                        default=["Z8", "Y20"],
                        help='unseen类别列表')
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(
        embedding_dir=args.embedding_dir,
        balanced_data_path=args.balanced_data_path,
        ckpt_path=args.ckpt_path,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping_patience,
        experiment_name=args.experiment_name,
        loss_type=args.loss_type,
        align_loss_weight=args.align_loss_weight,
        calibration_factor=args.calibration_factor,
        seen_classes=args.seen_classes,
        unseen_classes=args.unseen_classes
    )
    
    # 设置标签映射
    config.label_to_aircraft = {
        0: "A321", 1: "A330", 2: "ARJ21", 3: "BY737",
        4: "BY787", 5: "KJ600", 6: "Y20", 7: "Z8"
    }
    
    # 创建反向映射：从字符串到数字
    config.aircraft_to_label = {v: k for k, v in config.label_to_aircraft.items()}
    
    config.label_jsons = [
        "A321_hierarchical_embedding.json",
        "A330_hierarchical_embedding.json", 
        "BY787_hierarchical_embedding.json",
        "BY737_hierarchical_embedding.json",
        "Z8_hierarchical_embedding.json",
        "ARJ21_hierarchical_embedding.json",
        "KJ600_hierarchical_embedding.json",
        "Y20_hierarchical_embedding.json"
    ]
    
    # 创建统一的实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.experiment_name is None:
        experiment_name = f"aspect_aware_zeroshot_{timestamp}"
    else:
        experiment_name = f"{config.experiment_name}_{timestamp}"
    
    # 训练模型
    print("开始Aspect-Aware零样本训练...")
    
    try:
        classifier, hrrp_model, label_to_aircraft, test_loader = train_aspect_aware_model(config, experiment_name)
        print("Aspect-Aware训练完成！")
        
        # 测试零样本性能
        print("开始在测试集上对所有类别进行评估...")
        
        # 获取类别索引映射
        seen_class_indices = [config.label_to_aircraft.index(cls) for cls in config.seen_classes]
        unseen_class_indices = [config.label_to_aircraft.index(cls) for cls in config.unseen_classes]
        
        # 使用相同的实验名称创建Logger用于测试
        temp_logger = Logger(config.save_dir, experiment_name)
        
        total_acc, seen_acc, unseen_acc, unseen_analysis = test_zeroshot_performance(
            classifier, hrrp_model, test_loader, torch.device(config.device), 
            seen_class_indices, unseen_class_indices, config, temp_logger
        )
        
        # 保存测试结果到实验文件夹
        test_results = {
            'total_accuracy': total_acc,
            'seen_accuracy': seen_acc,
            'unseen_accuracy': unseen_acc,
            'test_samples': len(test_loader.dataset),
            'seen_classes': config.seen_classes,
            'unseen_classes': config.unseen_classes,
            'experiment_name': experiment_name,
            'unseen_analysis': unseen_analysis
        }
        
        test_results_path = Path(config.save_dir) / experiment_name / "test_results.json"
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print("测试集全类别评估完成！")
        print(f"测试集样本数: {len(test_loader.dataset)}")
        print(f"测试集全类别准确率: {total_acc:.4f}")
        print(f"测试集已知类别准确率: {seen_acc:.4f}")
        print(f"测试集未知类别准确率: {unseen_acc:.4f}")
        
        # 打印unseen类的详细分析
        if unseen_analysis['unseen_total_predictions'] > 0:
            print(f"\n=== Unseen类详细分析 ===")
            print(f"Unseen类样本数: {unseen_analysis['unseen_total_predictions']}")
            print(f"Unseen类正确预测数: {unseen_analysis['unseen_correct_predictions']}")
            print(f"Unseen类平均概率: {unseen_analysis['unseen_class_avg_prob']:.4f}")
            print(f"Unseen类最大概率: {unseen_analysis['unseen_class_max_prob']:.4f}")
            print(f"Unseen类最小概率: {unseen_analysis['unseen_class_min_prob']:.4f}")
            
            # 保存unseen类softmax概率分布图
            if unseen_analysis['unseen_softmax_probs']:
                plot_unseen_softmax_distribution(
                    unseen_analysis['unseen_softmax_probs'], 
                    unseen_analysis['unseen_samples'],
                    Path(config.save_dir) / experiment_name,
                    config.label_to_aircraft
                )
        
        print(f"测试结果已保存到: {test_results_path}")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main() 