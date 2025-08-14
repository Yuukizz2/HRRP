import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple

class AspectAwareHRRPExtractor(nn.Module):
    """Aspect感知的HRRP特征提取器，为不同aspect学习专门的特征"""
    
    def __init__(self, hrrp_len, feature_dim, num_aspects=11):
        super().__init__()
        self.num_aspects = num_aspects
        self.feature_dim = feature_dim
        
        # 共享的底层特征提取器
        self.shared_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),  # 添加dropout
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),  # 添加dropout
        ).float()  # 确保是float32类型
        
        # 为每个aspect创建专门的特征提取器
        self.aspect_specific_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.15),  # 添加dropout
                nn.Conv1d(256, feature_dim, kernel_size=3, padding=1),
            ).float() for _ in range(num_aspects)
        ])
        
        # Aspect注意力机制
        self.aspect_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout
            nn.Linear(64, num_aspects),
            nn.Softmax(dim=-1)
        ).float()  # 确保是float32类型
        
    def forward(self, hrrp_signal):
        # hrrp_signal: (Batch, SignalLength) -> 需要添加channel维度
        if hrrp_signal.dim() == 2:
            hrrp_signal = hrrp_signal.unsqueeze(1)  # (Batch, 1, SignalLength)
        
        batch_size = hrrp_signal.size(0)
        
        # 共享特征提取
        shared_features = self.shared_conv(hrrp_signal)  # (Batch, 128, SignalLength)
        
        # 计算aspect注意力权重
        pooled_features = F.adaptive_avg_pool1d(shared_features, 1).squeeze(-1)  # (Batch, 128)
        aspect_weights = self.aspect_attention(pooled_features)  # (Batch, num_aspects)
        
        # 为每个aspect提取专门特征
        aspect_features = []
        for i, conv in enumerate(self.aspect_specific_conv):
            aspect_feat = conv(shared_features)  # (Batch, feature_dim, SignalLength)
            aspect_feat = F.adaptive_avg_pool1d(aspect_feat, 1).squeeze(-1)  # (Batch, feature_dim)
            aspect_features.append(aspect_feat)
        
        aspect_features = torch.stack(aspect_features, dim=1)  # (Batch, num_aspects, feature_dim)
        
        return aspect_features, aspect_weights

class AspectAwareTransformer(nn.Module):
    """Aspect感知的Transformer编码器"""
    
    def __init__(self, feature_dim, num_aspects, nhead=8, num_layers=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_aspects = num_aspects
        
        # Aspect位置编码
        self.aspect_pos_embedding = nn.Parameter(torch.randn(1, num_aspects, feature_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).float()
        
        # Aspect交互注意力
        self.aspect_interaction = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True
        ).float()
        
    def forward(self, aspect_features):
        # aspect_features: (Batch, num_aspects, feature_dim)
        batch_size = aspect_features.size(0)
        
        # 添加位置编码
        aspect_features = aspect_features + self.aspect_pos_embedding
        
        # Transformer编码
        encoded_features = self.transformer(aspect_features)
        
        # Aspect间交互
        interacted_features, _ = self.aspect_interaction(
            encoded_features, encoded_features, encoded_features
        )
        
        return interacted_features

class AspectAwareHRRPModel(nn.Module):
    """Aspect感知的HRRP模型"""
    
    def __init__(self, hrrp_len, feature_dim, num_aspects=11, nhead=8, num_layers=3):
        super().__init__()
        self.hrrp_len = hrrp_len
        self.feature_dim = feature_dim
        self.num_aspects = num_aspects
        
        # Aspect感知特征提取器
        self.aspect_extractor = AspectAwareHRRPExtractor(
            hrrp_len, feature_dim, num_aspects
        ).float()  # 确保是float32类型
        
        # Aspect感知Transformer
        self.aspect_transformer = AspectAwareTransformer(
            feature_dim, num_aspects, nhead, num_layers
        ).float()  # 确保是float32类型
        
        # 最终投影层
        self.final_projection = nn.Sequential(
            nn.Linear(feature_dim * num_aspects, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        ).float()  # 确保是float32类型
        
    def forward(self, hrrp_signal):
        # hrrp_signal: (Batch, SignalLength) -> 需要添加channel维度
        
        # 提取aspect特征
        aspect_features, aspect_weights = self.aspect_extractor(hrrp_signal)
        
        # Transformer编码
        encoded_aspects = self.aspect_transformer(aspect_features)
        
        # 加权聚合aspect特征
        weighted_aspects = encoded_aspects * aspect_weights.unsqueeze(-1)  # (Batch, num_aspects, feature_dim)
        
        # 展平并最终投影
        flattened = weighted_aspects.reshape(weighted_aspects.size(0), -1)  # (Batch, num_aspects * feature_dim)
        final_features = self.final_projection(flattened)
        
        return final_features, aspect_features, aspect_weights

class AspectAwareClassifier(nn.Module):
    """Aspect感知的分类器"""
    
    def __init__(self, hrrp_dim, num_labels, label_aspects, label_embedding_dim, 
                 label_embeddings_tensor=None, trainable_labels=True):
        super().__init__()
        self.hrrp_dim = hrrp_dim
        self.num_labels = num_labels
        self.label_aspects = label_aspects
        self.label_embedding_dim = label_embedding_dim
        
        self.dropout=nn.Dropout(0.3)
        self.feature_dropout=nn.Dropout(0.2)
        
        # 标签embedding
        if label_embeddings_tensor is None:
            self.label_embeddings = nn.Parameter(torch.randn(num_labels, label_aspects, label_embedding_dim))
        else:
            self.label_embeddings = nn.Parameter(label_embeddings_tensor, requires_grad=trainable_labels)
        
        # Aspect对齐投影
        self.aspect_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hrrp_dim, label_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2),  # 在投影层中添加dropout
                nn.Linear(label_embedding_dim, label_embedding_dim)
            ) for _ in range(label_aspects)
        ])
        
        # 全局投影（用于aspect_features）
        self.global_projection = nn.Sequential(
            nn.Linear(hrrp_dim, label_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # 在投影层中添加dropout
            nn.Linear(label_embedding_dim, label_embedding_dim)
        )
        
        # 标签交互Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=label_embedding_dim,
            nhead=8,
            dim_feedforward=label_embedding_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.label_interaction = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Aspect权重学习
        self.aspect_weights = nn.Parameter(torch.randn(1, label_aspects) * 0.01)
        
        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(label_embedding_dim, label_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # 增加dropout率
            nn.Linear(label_embedding_dim // 2, num_labels)
        )
        
    def forward(self, hrrp_features, aspect_features=None, aspect_weights=None):
        # hrrp_features: (Batch, hrrp_dim)
        # aspect_features: (Batch, num_aspects, feature_dim) - 可选
        # aspect_weights: (Batch, num_aspects) - 可选
        
        batch_size = hrrp_features.size(0)
        
        # 在输入特征上应用dropout
        hrrp_features = self.dropout(hrrp_features)
        
        # 标签embedding交互
        # label_emb = self.label_embeddings  # (num_labels, label_aspects, label_embedding_dim)
        # label_emb_interacted = self.label_interaction(label_emb)  # (num_labels, label_aspects, label_embedding_dim)
        
        # 计算aspect权重
        if aspect_weights is not None:
            # 使用HRRP的aspect权重
            aspect_importance = aspect_weights.mean(dim=0, keepdim=True)  # (1, num_aspects)
        else:
            # 使用学习的aspect权重
            aspect_importance = F.softmax(self.aspect_weights, dim=1)
        
        # 聚合标签embedding
        # label_project = (label_emb_interacted * aspect_importance.unsqueeze(-1)).sum(dim=1)  # (num_labels, label_embedding_dim)
        
        # 投影HRRP特征到embedding空间
        if aspect_features is not None:
            # 在aspect特征上应用dropout
            aspect_features = self.feature_dropout(aspect_features)
            
            # 使用aspect-specific投影
            hrrp_aspect_projections = []
            for i, proj in enumerate(self.aspect_projection):
                aspect_proj = proj(aspect_features[:, i, :])  # (Batch, label_embedding_dim)
                hrrp_aspect_projections.append(aspect_proj)
            
            # 加权聚合
            hrrp_aspect_projections = torch.stack(hrrp_aspect_projections, dim=1)  # (Batch, num_aspects, label_embedding_dim)
            hrrp_project = (hrrp_aspect_projections * aspect_importance.unsqueeze(-1)).sum(dim=1)  # (Batch, label_embedding_dim)
        else:
            # 使用全局投影
            hrrp_project = self.aspect_projection[0](hrrp_features)  # 使用第一个投影器
        
        # 在投影特征上应用dropout
        hrrp_project = self.dropout(hrrp_project)
        
        # 归一化
        hrrp_project = F.normalize(hrrp_project, p=2, dim=-1)
        # label_project = F.normalize(label_project, p=2, dim=-1)
        
        # # 计算logits
        # logits = torch.matmul(hrrp_project, label_project.T)
        logits = self.classifier(hrrp_project)
        
        return logits, hrrp_project, aspect_importance 