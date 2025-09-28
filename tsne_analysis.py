import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
from contextlib import contextmanager
import time

# 导入项目模块
from dataset.advanced_dataset_splitter import AdvancedDatasetSplitter
from model.atten_classify import AttentionClassifier
from model.hrrp_model import HRRPModel, SimplifiedHRRPModel
from data.embedding_tensor import build_label_embeddings_tensor
from utils.logger import Logger

# 为兼容包含 TrainingConfig 的旧 checkpoint，提供占位定义或回退导入
try:
    from config.training_config import TrainingConfig as _SavedTrainingConfig
    TrainingConfig = _SavedTrainingConfig
except Exception:
    class TrainingConfig:
        pass

# 将包含 numpy 类型的数据转换为可 JSON 序列化的 Python 原生类型
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

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

def incremental_flatten_pca(features: np.ndarray, target_dim: int = 100, batch_size: int = 256) -> np.ndarray:
    """将三维 [N, n, n] 特征以小批量展平并用 IncrementalPCA 降到 target_dim，返回二维 [N, target_dim]。
    内存友好：不一次性创建 [N, n*n] 大数组。
    """
    assert features.ndim == 3, "features 必须是三维 [N, n, n]"
    num_samples = features.shape[0]
    flattened_dim = features.shape[1] * features.shape[2]
    ipca = IncrementalPCA(n_components=target_dim)
    # 第一遍 partial_fit
    start_idx = 0
    while start_idx < num_samples:
        end_idx = min(start_idx + batch_size, num_samples)
        batch = features[start_idx:end_idx].reshape(end_idx - start_idx, flattened_dim)
        ipca.partial_fit(batch)
        start_idx = end_idx
    # 第二遍 transform
    reduced_list = []
    start_idx = 0
    while start_idx < num_samples:
        end_idx = min(start_idx + batch_size, num_samples)
        batch = features[start_idx:end_idx].reshape(end_idx - start_idx, flattened_dim)
        reduced_list.append(ipca.transform(batch))
        start_idx = end_idx
    return np.vstack(reduced_list)

class TSNEAnalyzer:
    """t-SNE分析器"""
    
    def __init__(self, config: dict, logger: Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config['device'])
        
        # 反向映射: 由飞机名称到标签索引
        self.aircraft_to_label = {v: k for k, v in self.config['label_to_aircraft'].items()}
        
        # 初始化模型
        self._init_models()
        
    def _init_models(self):
        """初始化模型"""
        self.logger.info("初始化模型...")
        
        # 构建标签嵌入张量
        with timer("构建标签嵌入张量", self.logger):
            self.label_embeddings_tensor = build_label_embeddings_tensor(
                self.config['embedding_dir'], 
                self.config['label_jsons'], 
                self.config['aspect_names'],
                self.config['label_to_aircraft']
            )
        
        # 创建HRRP模型
        self.hrrp_model = HRRPModel(
            hrrp_len=self.config['hrrp_len'],
            feature_dim=self.config['feature_dim'],
            nhead=self.config['nhead'],
            num_encoder_layers=self.config['num_layers']
        ).float().to(self.device)
        
        # 创建分类器
        self.classifier = AttentionClassifier(
            hrrp_dim=self.config['feature_dim'],
            num_labels=self.config['num_labels'],
            label_aspects=self.config['num_aspects'],
            label_embedding_dim=self.config['label_embedding_dim'],
            label_embeddings_tensor=self.label_embeddings_tensor,
            trainable_labels=True
        ).float().to(self.device)
        
        # 加载预训练模型
        if self.config.get('ckpt_path') and os.path.exists(self.config['ckpt_path']):
            try:
                self.logger.info(f"加载预训练模型: {self.config['ckpt_path']}")
                checkpoint = torch.load(self.config['ckpt_path'], map_location=self.device)
                
                if 'hrrp_model_state_dict' in checkpoint:
                    self.hrrp_model.load_state_dict(checkpoint['hrrp_model_state_dict'])
                else:
                    self.hrrp_model.load_state_dict(checkpoint['model_state_dict'])
                    
                if 'classifier_state_dict' in checkpoint:
                    self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
                    
                self.hrrp_model.eval()
                self.classifier.eval()
                self.logger.info("模型加载成功")
            except Exception as e:
                self.logger.error(f"加载预训练模型失败: {e}")
                raise
        else:
            self.logger.warning("未找到预训练模型，将使用随机初始化的模型")
    
    def extract_features(self, dataloader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """提取HRRP特征和与样本对齐的标签嵌入"""
        self.hrrp_model.eval()
        self.classifier.eval()
        
        hrrp_features_list = []
        label_embeddings_list = []
        class_names_list = []
        before_project_list = []
        after_project_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取特征"):
                hrrp_data, class_names,angles = batch
                hrrp_data = hrrp_data.float().to(self.device, non_blocking=True)
                angles = angles.float().to(self.device, non_blocking=True)
                # 提取HRRP特征
                hrrp_features,before_project,after_project = self.hrrp_model(hrrp_data,angles)
                
                # 获取标签嵌入（全体类别的投影）
                logits, label_project, hrrp_project = self.classifier(hrrp_features)
                
                # 将 hrrp_project 直接收集（与样本一一对应）
                hrrp_features_list.append(hrrp_project.cpu().numpy())
                before_project_list.append(before_project.cpu().numpy())
                after_project_list.append(after_project.cpu().numpy())
                # 根据当前 batch 的真实类别，为每个样本选择对应的标签投影，做到一一对应
                # class_names 为字符串类别名，需要映射到标签索引
                try:
                    label_indices = [self.aircraft_to_label[str(cls_name)] for cls_name in class_names]
                except KeyError as e:
                    self.logger.error(f"在映射类别名到索引时找不到键: {e}. 可用键: {list(self.aircraft_to_label.keys())}")
                    raise
                label_indices_tensor = torch.tensor(label_indices, device=label_project.device, dtype=torch.long)
                aligned_label_proj = label_project.index_select(dim=0, index=label_indices_tensor)
                label_embeddings_list.append(aligned_label_proj.cpu().numpy())
                
                # 收集类别名（与样本数量一致）
                class_names_list.extend([str(c) for c in class_names])
        
        # 合并所有批次的特征（按样本对齐）
        hrrp_features = np.vstack(hrrp_features_list)
        before_features = np.vstack(before_project_list)
        after_features = np.vstack(after_project_list)
        label_embeddings = np.vstack(label_embeddings_list)
        
        return hrrp_features, label_embeddings, class_names_list, before_features, after_features
    
    def perform_tsne(self, features: np.ndarray, n_components: int = 2, 
                    perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
        """执行t-SNE降维"""
        self.logger.info(f"执行t-SNE降维，perplexity={perplexity}")
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 执行t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            verbose=1
        )
        
        with timer("t-SNE降维", self.logger):
            tsne_result = tsne.fit_transform(features_scaled)
        
        return tsne_result
    
    def perform_pca(self, features: np.ndarray, n_components: int = 2) -> np.ndarray:
        """执行PCA降维"""
        self.logger.info(f"执行PCA降维到{n_components}维")
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 执行PCA
        pca = PCA(n_components=n_components, random_state=42)
        
        with timer("PCA降维", self.logger):
            pca_result = pca.fit_transform(features_scaled)
        
        self.logger.info(f"PCA解释方差比: {pca.explained_variance_ratio_}")
        self.logger.info(f"PCA累积解释方差比: {np.cumsum(pca.explained_variance_ratio_)}")
        
        return pca_result
    
    def plot_tsne_results(self, tsne_result: np.ndarray, class_names: List[str], 
                         seen_classes: List[str], unseen_classes: List[str],
                         save_path: Path, title: str = "t-SNE Visualization"):
        """绘制t-SNE结果"""
        plt.figure(figsize=(12, 8))
        
        # 定义颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(class_names))))
        class_to_color = {cls: colors[i] for i, cls in enumerate(set(class_names))}
        
        # 绘制每个类别
        for cls in set(class_names):
            mask = np.array(class_names) == cls
            points = tsne_result[mask]
            
            # 根据seen/unseen设置不同的标记
            if cls in seen_classes:
                marker = 'o'
                alpha = 0.8
                label = f"{cls} (Seen)"
            elif cls in unseen_classes:
                marker = '^'
                alpha = 0.8
                label = f"{cls} (Unseen)"
            else:
                marker = 's'
                alpha = 0.6
                label = f"{cls} (Other)"
            
            plt.scatter(points[:, 0], points[:, 1], 
                       c=[class_to_color[cls]], 
                       marker=marker, 
                       alpha=alpha,
                       s=50, 
                       label=label)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"t-SNE figure saved to: {save_path}")
    
    def plot_pca_results(self, pca_result: np.ndarray, class_names: List[str], 
                        seen_classes: List[str], unseen_classes: List[str],
                        save_path: Path, title: str = "PCA Visualization"):
        """绘制PCA结果"""
        plt.figure(figsize=(12, 8))
        
        # 定义颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(class_names))))
        class_to_color = {cls: colors[i] for i, cls in enumerate(set(class_names))}
        
        # 绘制每个类别
        for cls in set(class_names):
            mask = np.array(class_names) == cls
            points = pca_result[mask]
            
            # 根据seen/unseen设置不同的标记
            if cls in seen_classes:
                marker = 'o'
                alpha = 0.8
                label = f"{cls} (Seen)"
            elif cls in unseen_classes:
                marker = '^'
                alpha = 0.8
                label = f"{cls} (Unseen)"
            else:
                marker = 's'
                alpha = 0.6
                label = f"{cls} (Other)"
            
            plt.scatter(points[:, 0], points[:, 1], 
                       c=[class_to_color[cls]], 
                       marker=marker, 
                       alpha=alpha,
                       s=50, 
                       label=label)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('PC 1', fontsize=12)
        plt.ylabel('PC 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"PCA figure saved to: {save_path}")
    
    def analyze_separation_metrics(self, features: np.ndarray, class_names: List[str], 
                                 seen_classes: List[str], unseen_classes: List[str]) -> Dict:
        """分析分离度指标"""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        # 将类别名转换为数字标签
        unique_classes = list(set(class_names))
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        labels = np.array([class_to_idx[cls] for cls in class_names])
        
        # 计算聚类指标
        silhouette = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        
        # 计算seen和unseen类别的分离度
        seen_mask = np.array([cls in seen_classes for cls in class_names])
        unseen_mask = np.array([cls in unseen_classes for cls in class_names])
        
        seen_features = features[seen_mask]
        unseen_features = features[unseen_mask]
        
        # 计算类别间距离
        from scipy.spatial.distance import cdist
        if len(seen_features) > 0 and len(unseen_features) > 0:
            inter_class_distances = cdist(seen_features, unseen_features)
            min_inter_class_distance = np.min(inter_class_distances)
            mean_inter_class_distance = np.mean(inter_class_distances)
        else:
            min_inter_class_distance = 0
            mean_inter_class_distance = 0
        
        metrics = {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'min_inter_class_distance': min_inter_class_distance,
            'mean_inter_class_distance': mean_inter_class_distance,
            'num_seen_samples': np.sum(seen_mask),
            'num_unseen_samples': np.sum(unseen_mask),
            'total_samples': len(class_names)
        }
        
        return metrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='t-SNE analysis script')
    parser.add_argument('--embedding_dir', type=str, default="data/embedding/Qwen_QwQ-32B",
                        help='嵌入文件目录')
    parser.add_argument('--balanced_data_path', type=str, default="data/zeroshot/balanced_data_enhanced.pt",
                        help='数据路径')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/zeroshot_2/best_model.pth",
                        help='预训练模型路径')
    parser.add_argument('--save_dir', type=str, default="tsne_analysis",
                        help='分析结果保存目录')
    parser.add_argument('--experiment_name', type=str, default="angle",
                        help='实验名称')
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='t-SNE perplexity参数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--ipca_dim', type=int, default=100,
                        help='IncrementalPCA 降维目标维度（用于 before/after 为三维时）')
    parser.add_argument('--ipca_batch', type=int, default=512,
                        help='IncrementalPCA 分块大小（用于内存友好处理）')
    args = parser.parse_args()
    
    # 创建保存目录
    save_path = Path(args.save_dir) / args.experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = Logger(args.save_dir, args.experiment_name)
    logger.info(f"t-SNE analysis: {args.experiment_name}")
    logger.info(f"Save directory: {save_path}")
    
    # 配置参数
    config = {
        'embedding_dir': args.embedding_dir,
        'balanced_data_path': args.balanced_data_path,
        'ckpt_path': args.ckpt_path,
        'hrrp_len': 512,
        'feature_dim': 512,
        'num_aspects': 11,
        'nhead': 8,
        'num_layers': 3,
        'label_embedding_dim': 768,
        'num_labels': 8,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'label_to_aircraft': {
            0: "A321", 1: "A330", 2: "BY787", 3: "BY737",
            4: "Z8", 5: "ARJ21", 6: "KJ600", 7: "Y20"
        },
        'label_jsons': [
            "A321_hierarchical_embedding.json",
            "A330_hierarchical_embedding.json", 
            "BY787_hierarchical_embedding.json",
            "BY737_hierarchical_embedding.json",
            "Z8_hierarchical_embedding.json",
            "ARJ21_hierarchical_embedding.json",
            "KJ600_hierarchical_embedding.json",
            "Y20_hierarchical_embedding.json"
        ],
        'aspect_names': [
            "flightCharacteristics", "microDopplerSignature", "basicInformation",
            "thermalSignature", "roleAndMission", "geometricCharacteristics",
            "materialsAndCoatings", "aircraftModel", "radarCrossSection",
            "engineAndPerformance", "global"
        ],
        'seen_classes': ["A321", "A330", "ARJ21", "BY737", "KJ600", "Z8"],
        'unseen_classes': ["BY787", "Y20"]
    }
    
    try:
        # 初始化分析器
        analyzer = TSNEAnalyzer(config, logger)
        
        # 加载数据
        logger.info("Loading data...")
        splitter = AdvancedDatasetSplitter(config['balanced_data_path'])
        
        splits = splitter.get_or_create_splits(
            seen_classes=config['seen_classes'],
            unseen_classes=config['unseen_classes'],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            fixed_split_dir="data/splits",
            experiment_name="tsne_analysis"
        )
        
        # 创建DataLoader
        dataloaders = splitter.create_dataloader_splits(splits, batch_size=args.batch_size)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']

        logger.info(f"Test set size: {len(test_loader.dataset)}")
        
        # 提取特征
        logger.info("Extracting HRRP features and label embeddings...")
        hrrp_features, label_embeddings, class_names, before_features, after_features = analyzer.extract_features(train_loader)
        
        logger.info(f"HRRP features shape: {hrrp_features.shape}")
        logger.info(f"Label embeddings shape: {label_embeddings.shape}")
        logger.info(f"Before features shape: {before_features.shape}")
        logger.info(f"After features shape: {after_features.shape}")
        
        # 详细打印维度信息，并将二维化以便后续分析
        logger.info(f"HRRP features ndim: {hrrp_features.ndim}, shape: {hrrp_features.shape}")
        logger.info(f"Before features ndim: {before_features.ndim}, shape: {before_features.shape}")
        logger.info(f"After features ndim: {after_features.ndim}, shape: {after_features.shape}")
        
        # 当 before/after 为 [N, n, n] 时，使用增量PCA按小批量展平+降维为 [N, ipca_dim]
        if before_features.ndim == 3:
            with timer("Incremental PCA for before_features", logger):
                before_features = incremental_flatten_pca(before_features, target_dim=args.ipca_dim, batch_size=args.ipca_batch)
            logger.info(f"Before features reduced to: {before_features.shape}")
        if after_features.ndim == 3:
            with timer("Incremental PCA for after_features", logger):
                after_features = incremental_flatten_pca(after_features, target_dim=args.ipca_dim, batch_size=args.ipca_batch)
            logger.info(f"After features reduced to: {after_features.shape}")
        
        # 分析HRRP特征
        logger.info("=" * 50)
        logger.info("Analyze HRRP features")
        logger.info("=" * 50)
        
        # t-SNE分析HRRP/Before/After 特征
        hrrp_tsne = analyzer.perform_tsne(hrrp_features, perplexity=args.perplexity)
        analyzer.plot_tsne_results(
            hrrp_tsne, class_names, config['seen_classes'], config['unseen_classes'],
            save_path / "hrrp_features_tsne.png", "HRRP Features t-SNE"
        )
        before_tsne = analyzer.perform_tsne(before_features, perplexity=args.perplexity)
        analyzer.plot_tsne_results(
            before_tsne, class_names, config['seen_classes'], config['unseen_classes'],
            save_path / "before_features_tsne.png", "Before Features t-SNE"
        )
        after_tsne = analyzer.perform_tsne(after_features, perplexity=args.perplexity)
        analyzer.plot_tsne_results(
            after_tsne, class_names, config['seen_classes'], config['unseen_classes'],
            save_path / "after_features_tsne.png", "After Features t-SNE"
        )
        
        
        # PCA分析HRRP特征
        hrrp_pca = analyzer.perform_pca(hrrp_features)
        analyzer.plot_pca_results(
            hrrp_pca, class_names, config['seen_classes'], config['unseen_classes'],
            save_path / "hrrp_features_pca.png", "HRRP Features PCA"
        )
        
        # 计算分离度指标
        logger.info("=" * 50)
        logger.info("Compute separation metrics")
        logger.info("=" * 50)
        
        hrrp_metrics = analyzer.analyze_separation_metrics(
            hrrp_features, class_names, config['seen_classes'], config['unseen_classes']
        )
        before_metrics = analyzer.analyze_separation_metrics(
            before_features, class_names, config['seen_classes'], config['unseen_classes']
        )
        after_metrics = analyzer.analyze_separation_metrics(
            after_features, class_names, config['seen_classes'], config['unseen_classes']
        )
        
        # 保存分析结果（移除标签嵌入相关字段）
        analysis_results = {
            'experiment_name': args.experiment_name,
            'config': config,
            'hrrp_metrics': hrrp_metrics,
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'hrrp_features_shape': hrrp_features.shape,
            'before_features_shape': before_features.shape,
            'after_features_shape': after_features.shape,
            'class_distribution': {cls: class_names.count(cls) for cls in set(class_names)},
            'tsne_perplexity': args.perplexity
        }
        
        results_path = save_path / "analysis_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(analysis_results), f, indent=2, ensure_ascii=False)
        
        # 打印结果
        logger.info("=" * 50)
        logger.info("Analysis results")
        logger.info("=" * 50)
        
        logger.info("HRRP features separation metrics:")
        for key, value in hrrp_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("\nBefore features separation metrics:")
        for key, value in before_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("\nAfter features separation metrics:")
        for key, value in after_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info(f"\nAnalysis saved to: {results_path}")
        logger.info(f"Figures saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()