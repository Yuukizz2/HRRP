import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os

from torch.utils import checkpoint
from model.hrrp_model import HRRPModel
from torch.utils.data import DataLoader
from train_angle import AdvancedDatasetSplitter, TrainingConfig
from typing import Optional, List, Dict, Tuple




class RotationAnalysis:
    """分析旋转融合模块是否有效消除角度影响"""
    
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.config = config
        
    def analyze_rotation_effectiveness(self, dataloader, num_samples_per_class=50):
        """分析旋转融合前后的特征相似性"""
        print("开始分析旋转融合模块的有效性...")
        
        # 存储结果
        results = {
            'before_rotation': {},  # 旋转前特征
            'after_rotation': {},  # 旋转后特征
            'similarity_analysis': {},
            'angle_analysis': {}
        }
        
        # 按类别收集特征
        class_features_before = {}  # {class_id: [features]}
        class_features_after = {}   # {class_id: [features]}
        class_angles = {}          # {class_id: [angles]}
        
        with torch.no_grad():
            for batch_idx, (hrrp_data, labels, angles) in enumerate(dataloader):
                if batch_idx >= 100:  # 限制样本数量
                    break
                    
                hrrp_data = hrrp_data.float().to(self.device, non_blocking=True)
                angles = angles.float().to(self.device, non_blocking=True)
                
                # 获取旋转前后的特征
                encoded_features, hrrp_features, rotated_features = self.model.forward_encoder(hrrp_data, angles)
                
                # 按类别收集特征
                for i in range(len(labels)):
                    class_id = labels[i]
                    angle = angles[i].cpu().numpy()
                    
                    if class_id not in class_features_before:
                        class_features_before[class_id] = []
                        class_features_after[class_id] = []
                        class_angles[class_id] = []
                    
                    # 收集旋转前的特征（使用全局平均池化）
                    before_feat = hrrp_features[i].mean(dim=0).cpu().numpy()  # (feature_dim,)
                    class_features_before[class_id].append(before_feat)
                    
                    # 收集旋转后的特征（使用全局平均池化）
                    after_feat = rotated_features[i].mean(dim=0).cpu().numpy()  # (feature_dim,)
                    class_features_after[class_id].append(after_feat)
                    
                    class_angles[class_id].append(angle)
        
        # 分析每个类别的特征相似性
        for class_id in class_features_before.keys():
            if len(class_features_before[class_id]) < 2:
                continue
                
            before_features = np.array(class_features_before[class_id])
            after_features = np.array(class_features_after[class_id])
            angles = np.array(class_angles[class_id])
            if angles.ndim > 2:
                angles = angles.reshape(-1, 2)
            
            # 计算类内相似性
            before_similarity = self._calculate_intra_class_similarity(before_features)
            after_similarity = self._calculate_intra_class_similarity(after_features)
            
            # 计算角度与特征相似性的相关性
            angle_variance = np.var(angles, axis=0)  # 角度方差
            before_correlation = self._calculate_angle_feature_correlation(angles, before_features)
            after_correlation = self._calculate_angle_feature_correlation(angles, after_features)
            
            results['similarity_analysis'][class_id] = {
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
            
            results['angle_analysis'][class_id] = {
                'angle_variance': {
                    'azimuth': float(angle_variance[0]),
                    'elevation': float(angle_variance[1])
                },
                'before_rotation_correlation': {
                    'azimuth': float(before_correlation[0]),
                    'elevation': float(before_correlation[1])
                },
                'after_rotation_correlation': {
                    'azimuth': float(after_correlation[0]),
                    'elevation': float(after_correlation[1])
                }
            }
        
        return results
    
    def _calculate_intra_class_similarity(self, features):
        """计算类内特征相似性"""
        if len(features) < 2:
            return np.array([1.0])
        
        # 计算余弦相似性矩阵
        similarity_matrix = cosine_similarity(features)
        
        # 提取上三角矩阵（排除对角线）
        n = len(features)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i, j])
        
        return np.array(similarities)
    
    def _calculate_angle_feature_correlation(self, angles, features):
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
    
    def visualize_rotation_effect(self, results, save_dir='rotation_analysis'):
        """可视化旋转效果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 相似性对比图
        self._plot_similarity_comparison(results, save_dir)
        
        # 2. 角度相关性对比图
        self._plot_angle_correlation_comparison(results, save_dir)
        
        # 3. 改进效果统计图
        self._plot_improvement_statistics(results, save_dir)
    
    def _plot_similarity_comparison(self, results, save_dir):
        """绘制相似性对比图"""
        classes = list(results['similarity_analysis'].keys())
        before_means = [results['similarity_analysis'][c]['before_rotation_similarity']['mean'] for c in classes]
        after_means = [results['similarity_analysis'][c]['after_rotation_similarity']['mean'] for c in classes]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(classes))
        width = 0.35
        
        plt.bar(x - width/2, before_means, width, label='Before Rotation', alpha=0.8)
        plt.bar(x + width/2, after_means, width, label='After Rotation', alpha=0.8)
        
        plt.xlabel('Class ID')
        plt.ylabel('Intra-class Similarity')
        plt.title('Intra-class Feature Similarity Comparison Before and After Rotation Fusion')
        plt.legend()
        plt.xticks(x, classes)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'similarity_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_angle_correlation_comparison(self, results, save_dir):
        """绘制角度相关性对比图"""
        classes = list(results['angle_analysis'].keys())
        before_az = [results['angle_analysis'][c]['before_rotation_correlation']['azimuth'] for c in classes]
        after_az = [results['angle_analysis'][c]['after_rotation_correlation']['azimuth'] for c in classes]
        before_el = [results['angle_analysis'][c]['before_rotation_correlation']['elevation'] for c in classes]
        after_el = [results['angle_analysis'][c]['after_rotation_correlation']['elevation'] for c in classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.arange(len(classes))
        width = 0.35
        
        # 方位角相关性
        ax1.bar(x - width/2, before_az, width, label='Before Rotation', alpha=0.8)
        ax1.bar(x + width/2, after_az, width, label='After Rotation', alpha=0.8)
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Correlation with Azimuth Angle')
        ax1.set_title('Azimuth Angle and Feature Correlation Comparison')
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes)
        ax1.grid(True, alpha=0.3)
        
        # 俯仰角相关性
        ax2.bar(x - width/2, before_el, width, label='Before Rotation', alpha=0.8)
        ax2.bar(x + width/2, after_el, width, label='After Rotation', alpha=0.8)
        ax2.set_xlabel('Class ID')
        ax2.set_ylabel('Correlation with Elevation Angle')
        ax2.set_title('Elevation Angle and Feature Correlation Comparison')
        ax2.legend()
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'angle_correlation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_statistics(self, results, save_dir):
        """绘制改进效果统计图"""
        improvements = [results['similarity_analysis'][c]['improvement'] for c in results['similarity_analysis'].keys()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(improvements, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(improvements), color='red', linestyle='--', 
                   label=f'Average Improvement: {np.mean(improvements):.4f}')
        plt.xlabel('Similarity Improvement Value')
        plt.ylabel('Number of Classes')
        plt.title('Improvement Effect Distribution of Rotation Fusion Module')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'improvement_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results, save_dir='rotation_analysis'):
        """生成分析报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 计算总体统计
        all_improvements = [results['similarity_analysis'][c]['improvement'] 
                          for c in results['similarity_analysis'].keys()]
        
        # 计算角度相关性改进
        angle_improvements = []
        for class_id in results['angle_analysis'].keys():
            before_az = results['angle_analysis'][class_id]['before_rotation_correlation']['azimuth']
            after_az = results['angle_analysis'][class_id]['after_rotation_correlation']['azimuth']
            before_el = results['angle_analysis'][class_id]['before_rotation_correlation']['elevation']
            after_el = results['angle_analysis'][class_id]['after_rotation_correlation']['elevation']
            
            # 角度相关性应该降低（改进为负值）
            az_improvement = before_az - after_az
            el_improvement = before_el - after_el
            angle_improvements.extend([az_improvement, el_improvement])
        
        report = {
            'summary': {
                'total_classes': len(results['similarity_analysis']),
                'similarity_improvement': {
                    'mean': float(np.mean(all_improvements)),
                    'std': float(np.std(all_improvements)),
                    'positive_improvements': int(np.sum(np.array(all_improvements) > 0)),
                    'improvement_rate': float(np.sum(np.array(all_improvements) > 0) / len(all_improvements))
                },
                'angle_correlation_improvement': {
                    'mean': float(np.mean(angle_improvements)),
                    'std': float(np.std(angle_improvements)),
                    'positive_improvements': int(np.sum(np.array(angle_improvements) > 0)),
                    'improvement_rate': float(np.sum(np.array(angle_improvements) > 0) / len(angle_improvements))
                }
            },
            'detailed_results': results
        }
        
        # 保存报告
        with open(os.path.join(save_dir, 'rotation_analysis_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        print("\n" + "="*60)
        print("旋转融合模块有效性分析报告")
        print("="*60)
        print(f"分析类别数量: {report['summary']['total_classes']}")
        print(f"\n相似性改进统计:")
        print(f"  平均改进: {report['summary']['similarity_improvement']['mean']:.4f}")
        print(f"  改进率: {report['summary']['similarity_improvement']['improvement_rate']:.2%}")
        print(f"  正改进类别数: {report['summary']['similarity_improvement']['positive_improvements']}")
        
        print(f"\n角度相关性改进统计:")
        print(f"  平均改进: {report['summary']['angle_correlation_improvement']['mean']:.4f}")
        print(f"  改进率: {report['summary']['angle_correlation_improvement']['improvement_rate']:.2%}")
        print(f"  正改进数量: {report['summary']['angle_correlation_improvement']['positive_improvements']}")
        
        print(f"\n分析结果已保存到: {save_dir}")
        print("="*60)
        
        return report

def main():
    """主函数"""
    # 这里需要根据您的实际数据加载方式进行调整
    print("正在加载模型和数据...")
    
    # 假设您有相应的配置和数据加载代码
    # 这里需要根据实际情况修改
    config = TrainingConfig()
    config.label_to_aircraft = {
        0: "A321", 1: "A330", 2: "BY787", 3: "BY737",
        4: "Z8", 5: "ARJ21", 6: "KJ600", 7: "Y20"
    }
    
    # 创建模型（需要根据您的实际参数调整）
    model = HRRPModel(
        hrrp_len=512,  # 根据您的数据调整
        feature_dim=512,  # 根据您的模型调整
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024
    )
    

    checkpoint=torch.load('ckpt/zeroshot_angle_best/best_model.pth', map_location=config.device)
    model.load_state_dict(checkpoint['hrrp_model_state_dict'])
    model.eval()
    
    # 创建分析器
    analyzer = RotationAnalysis(model, config, config.device)
    

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
    
    # 进行分析
    results = analyzer.analyze_rotation_effectiveness(train_loader)
    analyzer.visualize_rotation_effect(results)
    report = analyzer.generate_report(results)

if __name__ == "__main__":
    main()