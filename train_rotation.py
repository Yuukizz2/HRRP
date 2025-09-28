import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset.advanced_dataset_splitter import AdvancedDatasetSplitter
from model.rotation_model import RotationModel, AngleNormalizationLoss
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
from utils.plot import plot_training_curves

@dataclass
class RotationTrainingConfig:
    """旋转模型训练配置类"""
    # 数据相关
    balanced_data_path: str = "data/zeroshot/balanced_data_enhanced.pt"
    
    # 模型相关
    hrrp_len: int = 512
    feature_dim: int = 512
    
    # 训练相关
    num_epochs: int = 100
    lr: float = 0.001
    batch_size: int = 32
    weight_decay: float = 0.001
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    min_delta: float = 0.001
    
    # 系统相关
    save_dir: str = "ckpt"
    experiment_name: Optional[str] = None
    num_workers: int = 0
    device: Optional[str] = None
    
    # 数据分割相关
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    def __post_init__(self):
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
    model: RotationModel, 
    optimizer: optim.Optimizer, 
    epoch: int, 
    loss: float, 
    save_path: Path,
    logger: Logger
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    torch.save(checkpoint, save_path)
    logger.info(f"检查点已保存到: {save_path}")


def test(
    model: RotationModel,
    test_loader: DataLoader,
    device: torch.device,
    criterion: AngleNormalizationLoss,
    config: RotationTrainingConfig
) -> Dict:
    """测试模型性能"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # 存储结果用于分析
    results = {
        'similarity_analysis': {},
        'angle_analysis': {}
    }
    
    # 按类别收集特征和角度
    class_features_before = {}   # {class_name: [features]}
    class_features_after = {}    # {class_name: [features]}
    class_angles = {}            # {class_name: [angles]}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中", leave=False):
            hrrp_data, class_names, angles = batch
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            angles = angles.float().to(device, non_blocking=True)
            
            # 创建目标ID
            target_ids = torch.tensor([hash(cls) % 1000 for cls in class_names]).to(device)
            
            # 前向传播
            hrrp_features, rotated_features = model(hrrp_data, angles)
            
            # 计算损失
            loss = criterion(rotated_features, target_ids, angles)
            total_loss += loss.item() * hrrp_data.size(0)
            total_samples += hrrp_data.size(0)
            
            
            before_features = hrrp_features  # (B, S, D)
            before_global = before_features.mean(dim=1)  # (B, D)
            
            # 旋转后特征（使用全局平均池化）
            after_global = rotated_features.mean(dim=1)  # (B, D)
            
            # 按类别收集数据
            for i in range(hrrp_data.size(0)):
                class_name = str(class_names[i])
                angle = angles[i].cpu().numpy()
                
                if class_name not in class_features_before:
                    class_features_before[class_name] = []
                    class_features_after[class_name] = []
                    class_angles[class_name] = []
                
                class_features_before[class_name].append(before_global[i].cpu().numpy())
                class_features_after[class_name].append(after_global[i].cpu().numpy())
                class_angles[class_name].append(angle)
    
    avg_loss = total_loss / total_samples
    
    # 计算角度相关性（使用rotation_check.py中的函数）
    def calculate_angle_feature_correlation(angles, features):
        """计算角度与特征的相关性"""
        if len(angles) < 3:
            return np.array([0.0, 0.0])
        
        if angles.ndim > 2:
            angles = angles.reshape(-1, 2)
            
        # 计算每个特征维度与角度的相关性
        correlations = []
        for angle_idx in range(2):  # 方位角和俯仰角
            angle_values = angles[:, angle_idx].flatten()
            feature_correlations = []
            
            for feat_idx in range(features.shape[1]):
                try:
                    corr = np.corrcoef(angle_values, features[:, feat_idx])[0, 1]
                    if not np.isnan(corr):
                        feature_correlations.append(abs(corr))  # 取绝对值
                except ValueError:
                    # 如果corrcoef失败，跳过这个特征
                    continue
            
            if feature_correlations:
                correlations.append(np.mean(feature_correlations))
            else:
                correlations.append(0.0)
        
        return np.array(correlations)
    
    def calculate_intra_class_similarity(features):
        """计算类内特征相似性"""
        if len(features) < 2:
            return np.array([1.0])
        
        # 计算余弦相似性矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(features)
        
        # 提取上三角矩阵（排除对角线）
        n = len(features)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i, j])
        
        return np.array(similarities)
    
    # 分析每个类别的特征相似性和角度相关性
    for class_name in class_features_before.keys():
        if len(class_features_before[class_name]) < 2:
            continue
            
        before_features = np.array(class_features_before[class_name])
        after_features = np.array(class_features_after[class_name])
        angles = np.array(class_angles[class_name])
        if angles.ndim > 2:
            angles = angles.reshape(-1, 2)
        
        # 计算类内相似性
        before_similarity = calculate_intra_class_similarity(before_features)
        after_similarity = calculate_intra_class_similarity(after_features)
        
        # 计算角度与特征的相关性
        before_correlation = calculate_angle_feature_correlation(angles, before_features)
        after_correlation = calculate_angle_feature_correlation(angles, after_features)
        
        results['similarity_analysis'][class_name] = {
            'before_rotation_similarity': {
                'mean': float(np.mean(before_similarity)),
                'std': float(np.std(before_similarity)),
                'min': float(np.min(before_similarity)),
                'max': float(np.max(before_similarity))
            },
            'after_rotation_similarity': {
                'mean': float(np.mean(after_similarity)),
                'std': float(np.std(after_similarity)),
                'min': float(np.min(after_similarity)),
                'max': float(np.max(after_similarity))
            },
            'improvement': float(np.mean(after_similarity) - np.mean(before_similarity))
        }
        
        results['angle_analysis'][class_name] = {
            'before_rotation_correlation': {
                'azimuth': float(before_correlation[0]),
                'elevation': float(before_correlation[1])
            },
            'after_rotation_correlation': {
                'azimuth': float(after_correlation[0]),
                'elevation': float(after_correlation[1])
            },
            'correlation_improvement': {
                'azimuth': float(before_correlation[0] - after_correlation[0]),  # 降低为改进
                'elevation': float(before_correlation[1] - after_correlation[1])
            }
        }
    
    # 计算总体统计
    all_improvements = [results['similarity_analysis'][c]['improvement'] 
                       for c in results['similarity_analysis'].keys()]
    
    # 计算角度相关性改进
    angle_improvements = []
    for class_name in results['angle_analysis'].keys():
        az_improvement = results['angle_analysis'][class_name]['correlation_improvement']['azimuth']
        el_improvement = results['angle_analysis'][class_name]['correlation_improvement']['elevation']
        angle_improvements.extend([az_improvement, el_improvement])
    
    # 生成测试报告
    test_report = {
        'test_loss': avg_loss,
        'total_samples': total_samples,
        'summary': {
            'total_classes': len(results['similarity_analysis']),
            'similarity_improvement': {
                'mean': float(np.mean(all_improvements)) if all_improvements else 0.0,
                'std': float(np.std(all_improvements)) if all_improvements else 0.0,
                'positive_improvements': int(np.sum(np.array(all_improvements) > 0)) if all_improvements else 0,
                'improvement_rate': float(np.sum(np.array(all_improvements) > 0) / len(all_improvements)) if all_improvements else 0.0
            },
            'angle_correlation_improvement': {
                'mean': float(np.mean(angle_improvements)) if angle_improvements else 0.0,
                'std': float(np.std(angle_improvements)) if angle_improvements else 0.0,
                'positive_improvements': int(np.sum(np.array(angle_improvements) > 0)) if angle_improvements else 0,
                'improvement_rate': float(np.sum(np.array(angle_improvements) > 0) / len(angle_improvements)) if angle_improvements else 0.0
            }
        },
        'detailed_results': results
    }
    
    return test_report


def validate(
    model: RotationModel, 
    val_loader: DataLoader, 
    device: torch.device, 
    criterion: AngleNormalizationLoss,
    config: RotationTrainingConfig
) -> float:
    """验证模型性能"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中", leave=False):
            hrrp_data, class_names, angles = batch
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            angles = angles.float().to(device, non_blocking=True)
            
            # 前向传播
            hrrp_features, rotated_features = model(hrrp_data, angles)
            
            # 创建目标ID（同一类别在不同角度下的样本应该有相同的ID）
            target_ids = torch.tensor([hash(cls) % 1000 for cls in class_names]).to(device)
            
            # 计算损失
            loss = criterion(rotated_features, target_ids, angles)
            
            total_loss += loss.item() * hrrp_data.size(0)
            total_samples += hrrp_data.size(0)
            

    avg_loss = total_loss / total_samples
    
    return avg_loss

def train_rotation_model(config: RotationTrainingConfig, experiment_name: str) -> RotationModel:
    """训练旋转模型"""
    
    device = torch.device(config.device)
    
    save_path = Path(config.save_dir) / experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = Logger(config.save_dir, experiment_name)
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"保存目录: {save_path}")
    logger.info(f"使用设备: {device}")
    
    # 创建模型和损失函数
    logger.info("初始化模型...")
    model = RotationModel(
        hrrp_len=config.hrrp_len,
        feature_dim=config.feature_dim
    ).float().to(device)
    
    criterion = AngleNormalizationLoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.min_delta)
    
    # 加载数据
    logger.info("加载数据...")
    splitter = AdvancedDatasetSplitter(config.balanced_data_path)
    
    # 获取所有类别用于训练旋转模型
    all_classes = ["A321", "A330", "ARJ21", "BY737", "BY787", "KJ600", "Z8", "Y20"]
    
    splits = splitter.get_or_create_splits(
        seen_classes=all_classes,  # 使用所有类别训练旋转模型
        unseen_classes=[],  # 不需要unseen类别
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=42,
        fixed_split_dir="data/splits",
        experiment_name="rotation_training"
    )
    
    # 创建DataLoader
    dataloaders = splitter.create_dataloader_splits(splits, batch_size=config.batch_size)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    logger.info(f"训练集样本数: {len(train_loader.dataset)}")
    logger.info(f"验证集样本数: {len(val_loader.dataset)}")
    
    # 训练历史
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    logger.info("=" * 50)
    logger.info("训练配置:")
    logger.info(f"  训练轮数: {config.num_epochs}")
    logger.info(f"  学习率: {config.lr}")
    logger.info(f"  批次大小: {config.batch_size}")
    logger.info(f"  早停耐心值: {config.early_stopping_patience}")
    logger.info("=" * 50)
    
    logger.info(f"开始训练，从第 0 轮开始...")
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, (hrrp_data, class_names, angles) in enumerate(progress_bar):
            hrrp_data = hrrp_data.float().to(device, non_blocking=True)
            angles = angles.float().to(device, non_blocking=True)
            
            # 创建目标ID（同一类别在不同角度下的样本应该有相同的ID）
            target_ids = torch.tensor([hash(cls) % 1000 for cls in class_names]).to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            hrrp_features, rotated_features = model(hrrp_data, angles)
            
            # 计算损失
            loss = criterion(rotated_features, target_ids, angles)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            
            optimizer.step()
            
            # 累积损失
            epoch_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Total': f'{loss.item():.4f}',
            })
            
            # 清理缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        val_loss = validate(model, val_loader, device, criterion, config)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        logger.info(f'Epoch {epoch+1}/{config.num_epochs}, '
                   f'Train Loss: {avg_train_loss:.4f}, '
                   f'Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_path / 'best_rotation_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': val_loss,
                'experiment_name': experiment_name
            }, best_model_path)
            logger.info(f"新的最佳模型已保存: {best_model_path}")
        
        # 保存最新检查点
        checkpoint_path = save_path / "latest_checkpoint.pth"
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, logger)
        
        # 早停检查
        if early_stopping(val_loss, model, epoch):
            logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
            logger.info(f"最佳模型在第 {early_stopping.get_best_epoch() + 1} 轮")
            break
    
    logger.info("开始测试模型性能...")
    test_loader = dataloaders['test']  # 使用测试集
    
    test_report = test(model, test_loader, device, criterion, config)
    
    # 保存测试结果
    test_results_path = save_path / "test_results.json"
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)
    
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, [], 
                        save_path, logger)
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
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
    logger.info(f"总训练轮数: {len(train_losses)}")
    logger.info(f"模型保存路径: {save_path}")
    logger.info("=" * 50)
    
    return model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练HRRP旋转模型')
    parser.add_argument('--balanced_data_path', type=str, default="data/zeroshot/balanced_data_enhanced.pt",
                        help='数据路径')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--save_dir', type=str, default="ckpt",
                        help='模型保存目录')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='早停耐心值')
    parser.add_argument('--experiment_name', type=str, default="rotation_training",
                        help='实验名称')
    args = parser.parse_args()
    
    # 创建配置
    config = RotationTrainingConfig(
        balanced_data_path=args.balanced_data_path,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping_patience,
        experiment_name=args.experiment_name,
    )
    
    # 创建统一的实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.experiment_name is None:
        experiment_name = f"rotation_training_{timestamp}"
    else:
        experiment_name = f"{config.experiment_name}_{timestamp}"
    
    # 训练模型
    print("开始训练旋转模型...")
    
    try:
        model = train_rotation_model(config, experiment_name)
        print("旋转模型训练完成！")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()