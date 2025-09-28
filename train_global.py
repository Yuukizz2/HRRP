import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset.advanced_dataset_splitter import AdvancedDatasetSplitter
from model.atten_classify import AttentionClassifier
from model.hrrp_model import HRRPModel
from model.tripletloss import TripletLossWithMining, FocalLoss
from data.embedding_tensor import build_label_embeddings_tensor
import matplotlib.pyplot as plt
import time
from pathlib import Path
import argparse
from typing import Optional, List, Dict, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import contextmanager
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.plot import plot_training_curves, plot_confusion_matrix

@dataclass
class TrainingConfig:
    """训练配置类"""
    # 数据相关
    embedding_dir: str = "data/embedding/Qwen_QwQ-32B"
    balanced_data_path: str = "data/zeroshot/balanced_data_enhanced.pt"
    
    # 模型相关
    hrrp_len: int = 512
    feature_dim: int = 512
    num_aspects: int = 1
    nhead: int = 8
    num_layers: int = 3
    label_embedding_dim: int = 768
    num_labels: int = 8
    
    # 训练相关
    num_epochs: int = 50
    lr: float = 0.0005
    batch_size: int = 64
    weight_decay: float = 0.005
    gradient_clip_norm: float = 0.5
    early_stopping_patience: int = 10
    min_delta: float = 0.001
    
    calibration_factor: float = 1.25
    
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
                "global"
            ]
        
        if self.seen_classes is None:
            self.seen_classes = ["A321", "A330", "ARJ21", "BY737", "KJ600", "Z8"]
        
        if self.unseen_classes is None:
            self.unseen_classes = ["BY787", "Y20"]
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"



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
    classifier: AttentionClassifier, 
    hrrp_model: HRRPModel, 
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
        
        
def validate(
    classifier: AttentionClassifier, 
    hrrp_model: HRRPModel, 
    val_loader: DataLoader, 
    device: torch.device, 
    criterion: nn.Module, 
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
            hrrp_data, class_names,angles = batch
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            angles = angles.float().to(device, non_blocking=True)
            
            # 将字符串标签转换为数字索引
            aircraft_to_label = {v: k for k, v in config.label_to_aircraft.items()}
            labels = torch.tensor([aircraft_to_label[cls] for cls in class_names]).to(device, non_blocking=True)
            
            # 前向传播
            hrrp_features = hrrp_model(hrrp_data,angles)
            logits, _, _ = classifier(hrrp_features)
            
            # 计算损失 - 仅使用交叉熵损失
            total_loss_batch = criterion(logits, labels)
            
            total_loss += total_loss_batch.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def test(
    classifier: AttentionClassifier, 
    hrrp_model: HRRPModel, 
    test_loader: DataLoader, 
    device: torch.device, 
    seen_classes: List[int], 
    unseen_classes: List[int],
    config: TrainingConfig,
    logger: Logger,
    save_path: Path = None
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
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试中")):
            hrrp_data, class_names,angles = batch
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            angles = angles.float().to(device, non_blocking=True)
            
            # 将字符串标签转换为数字索引
            aircraft_to_label = {v: k for k, v in config.label_to_aircraft.items()}
            labels = torch.tensor([aircraft_to_label[cls] for cls in class_names]).to(device, non_blocking=True)

            hrrp_features = hrrp_model(hrrp_data,angles)
            logits, _, _ = classifier(hrrp_features)
            
            # 校准因子
            penalty = torch.zeros(logits.shape[1], device=logits.device)
            gamma = config.calibration_factor 
            penalty[seen_classes] = gamma
            calibrated_logits = logits - penalty
            T = 0.5 
            temperature_scaled_logits = calibrated_logits / T
            
            preds = temperature_scaled_logits.argmax(dim=1)
            calibrated_softmax_probs = F.softmax(temperature_scaled_logits, dim=1)

            for i, (pred, label) in enumerate(zip(preds.cpu().numpy(), labels.cpu().numpy())):
                all_preds.append(pred)
                all_labels.append(label)
                sample_idx = batch_idx * test_loader.batch_size + i
                sample_probs = calibrated_softmax_probs[i].cpu().numpy()
                # 保存unseen类的softmax概率
                if label in unseen_classes:
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
                    
                    logger.info(f"[UNSEEN] Sample {sample_idx}: pred={pred}({config.label_to_aircraft[pred]}), true={label}({config.label_to_aircraft[label]}), "
                              f"unseen_prob={sample_probs[label]:.4f}, max_prob={sample_probs.max():.4f}, sample_probs={sample_probs}")
                else:
                    logger.info(f"Sample {sample_idx}: pred={pred}({config.label_to_aircraft[pred]}), true={label}({config.label_to_aircraft[label]}), "
                                f"seen_prob={sample_probs[label]:.4f}, max_prob={sample_probs.max():.4f}, sample_probs={sample_probs}")
                    
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
    
    # 绘制混淆矩阵和分类报告
    if save_path is not None:
        logger.info("Generating evaluation plots...")
        
        # 准备类别名称
        class_names = [config.label_to_aircraft[i] for i in range(config.num_labels)]
        
        # 绘制混淆矩阵
        logger.info("Plotting confusion matrix...")
        confusion_metrics = plot_confusion_matrix(
            y_true=all_labels,
            y_pred=all_preds,
            class_names=class_names,
            save_path=save_path,
            logger=logger,
            normalize=True,
            title="Zero-Shot Classification Confusion Matrix"
        )
        
        # 将混淆矩阵指标添加到unseen_analysis中
        if confusion_metrics:
            unseen_analysis['confusion_matrix_metrics'] = confusion_metrics
            logger.info(f"Confusion matrix metrics: {confusion_metrics}")
    
    return total_acc, seen_acc, unseen_acc, unseen_analysis        

        
def train(config: TrainingConfig, experiment_name: str) -> Tuple[AttentionClassifier, HRRPModel, Dict[int, str], DataLoader]:
    
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
    logger.info("初始化模型...")
    hrrp_model = HRRPModel(
        hrrp_len=config.hrrp_len,
        feature_dim=config.feature_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_layers
    ).float().to(device)
    
    classifier = AttentionClassifier(
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
    
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    triplet_loss = TripletLossWithMining(margin=1.0, mining_strategy='semi_hard')
    
    # 优化器
    optimizer = optim.AdamW([
        {'params': classifier.parameters(), 'lr': config.lr},
        {'params': hrrp_model.parameters(), 'lr': config.lr}
    ], weight_decay=config.weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.min_delta)
    
    # 使用新的数据加载方式
    logger.info("使用新的数据加载方式...")
    splitter = AdvancedDatasetSplitter(config.balanced_data_path)
    
    splits = splitter.get_or_create_splits(
        seen_classes=config.seen_classes,
        unseen_classes=config.unseen_classes,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=42,
        fixed_split_dir="data/splits",
        experiment_name="zeroshot"
    )
    # splits = splitter.load_splits("data/splits/zeroshot")
    
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
    best_val_loss = float('inf')
    
    logger.info("=" * 50)
    logger.info("训练配置:")
    logger.info(f"  训练轮数: {config.num_epochs}")
    logger.info(f"  学习率: {config.lr}")
    logger.info(f"  批次大小: {config.batch_size}")
    logger.info(f"  校准因子: {config.calibration_factor}")
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
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, (hrrp_data, class_names, angles) in enumerate(progress_bar):
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            angles = angles.float().to(device, non_blocking=True)
            
            # 将字符串标签转换为数字索引
            aircraft_to_label = {v: k for k, v in config.label_to_aircraft.items()}
            labels = torch.tensor([aircraft_to_label[cls] for cls in class_names]).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            hrrp_features = hrrp_model(hrrp_data,angles)
            logits, label_project, hrrp_project = classifier(hrrp_features)
            
            anchor = hrrp_project
            all_class_semantics = label_project.to(device)
            positive = all_class_semantics[labels].to(device)
            
            # 计算损失 - 仅使用交叉熵损失
            triplet_loss_value, _ = triplet_loss(anchor, positive, all_class_semantics, labels)
            ce_loss = criterion(logits, labels)
            total_loss = ce_loss + triplet_loss_value
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=config.gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(hrrp_model.parameters(), max_norm=config.gradient_clip_norm)
            
            optimizer.step()
            
            # 清理缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Triplet': f'{triplet_loss_value.item():.4f}'
            })
            
            # 调试信息：每100个batch打印一次损失
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: CE={ce_loss.item():.4f} Triplet={triplet_loss_value.item():.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
    
        
        # 验证阶段
        val_loss, val_acc = validate(classifier, hrrp_model, val_loader, device, criterion, config)
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
    plot_training_curves(train_losses, val_losses, val_accuracies, save_path, logger)
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'experiment_name': experiment_name,
        'total_epochs': len(train_losses),
        'config': config.__dict__
    }
    
    history_path = save_path / "training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 50)
    logger.info("训练完成！")
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"最终验证准确率: {val_accuracies[-1]:.4f}")
    logger.info(f"总训练轮数: {len(train_losses)}")
    logger.info(f"模型保存路径: {save_path}")
    logger.info("=" * 50)
    
    return classifier, hrrp_model, config.label_to_aircraft, test_loader



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练HRRP分类器')
    parser.add_argument('--embedding_dir', type=str, default="data/embedding/Qwen_QwQ-32B",
                        help='嵌入文件目录')
    parser.add_argument('--balanced_data_path', type=str, default="data/datav1/hrrp_dataset_filtered.pt",
                        help='数据路径')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--save_dir', type=str, default="ckpt",
                        help='模型保存目录')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--experiment_name', type=str, default="zeroshot_global",
                        help='实验名称')
    parser.add_argument('--calibration_factor', type=float, default=1.25,
                        help='校准因子')
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
        calibration_factor=args.calibration_factor
    )
    config.label_to_aircraft = {
        0: "A321", 1: "A330", 2: "BY787", 3: "BY737",
        4: "Z8", 5: "ARJ21", 6: "KJ600", 7: "Y20"
    }
    
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
        experiment_name = f"zeroshot_{timestamp}"
    else:
        experiment_name = f"{config.experiment_name}_{timestamp}"
    
    # 训练模型
    print("开始训练...")
    
    try:
        classifier, hrrp_model, label_to_aircraft, test_loader = train(config, experiment_name)
        print("训练完成！")
        
        # 测试零样本性能
        print("开始在测试集上对所有类别进行评估...")

        # 使用config中的seen_classes和unseen_classes来定义训练和测试标签
        aircraft_to_label = {v: k for k, v in config.label_to_aircraft.items()}
        
        # 将字符串类别名转换为数字标签
        train_labels = [aircraft_to_label[cls] for cls in config.seen_classes if cls in aircraft_to_label]
        test_labels = [aircraft_to_label[cls] for cls in config.unseen_classes if cls in aircraft_to_label]
        
        print(f"训练类别 (seen_classes): {config.seen_classes} -> {train_labels}")
        print(f"测试类别 (unseen_classes): {config.unseen_classes} -> {test_labels}")
        
        # 使用相同的实验名称创建Logger用于测试
        temp_logger = Logger(config.save_dir, experiment_name)
        
        # 获取实验保存路径
        experiment_save_path = Path(config.save_dir) / experiment_name
        
        total_acc, seen_acc, unseen_acc, unseen_analysis = test(
            classifier, hrrp_model, test_loader, torch.device(config.device), 
            train_labels, test_labels, config, temp_logger, experiment_save_path
        )
        
        # 保存测试结果到实验文件夹
        test_results = {
            'total_accuracy': total_acc,
            'seen_accuracy': seen_acc,
            'unseen_accuracy': unseen_acc,
            'seen_classes': train_labels,
            'unseen_classes': test_labels,
            'experiment_name': experiment_name,
            'unseen_analysis': unseen_analysis,
            'test_timestamp': datetime.now().isoformat()
        }
        
        test_results_path = experiment_save_path / "test_results.json"
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
            
        
        print(f"测试结果已保存到: {test_results_path}")
        print(f"评估图表已保存到: {experiment_save_path}")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main() 