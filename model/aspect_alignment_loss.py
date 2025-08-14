import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class AspectAlignmentLoss(nn.Module):
    """Aspect-aware alignment loss for HRRP features and hierarchical embeddings"""
    
    def __init__(self, label_embedding_dim: int = 768, temperature: float = 0.1):
        super(AspectAlignmentLoss, self).__init__()
        self.label_embedding_dim = label_embedding_dim
        self.temperature = temperature
        
    def forward(self, hrrp_aspect_features, label_aspect_embeddings, aspect_weights, importance_weights=None):
        """
        Args:
            hrrp_aspect_features: (batch_size, num_aspects, feature_dim) - HRRP提取的aspect特征
            label_aspect_embeddings: (num_labels, num_aspects, embedding_dim) - 标签的aspect embedding
            aspect_weights: (batch_size, num_aspects) - 每个样本的aspect权重
            importance_weights: (num_aspects,) - 全局aspect重要性权重，如果为None则使用均匀权重
        """
        # 如果没有提供importance_weights，则使用均匀权重
        if importance_weights is None:
            num_aspects = label_aspect_embeddings.size(1)
            importance_weights = torch.ones(num_aspects, device=hrrp_aspect_features.device) / num_aspects
        batch_size = hrrp_aspect_features.size(0)
        num_labels = label_aspect_embeddings.size(0)
        num_aspects = hrrp_aspect_features.size(1)
        
        # 检查维度是否匹配，如果不匹配则进行投影
        if hrrp_aspect_features.size(-1) != label_aspect_embeddings.size(-1):
            projection = nn.Linear(hrrp_aspect_features.size(-1), label_aspect_embeddings.size(-1)).to(hrrp_aspect_features.device)
            hrrp_aspect_features = projection(hrrp_aspect_features)
        
        # 1. Aspect-wise alignment loss
        aspect_loss = self.compute_aspect_wise_loss(hrrp_aspect_features, label_aspect_embeddings, aspect_weights)
        
        # 2. Global alignment loss
        global_loss = self.compute_global_loss(hrrp_aspect_features, label_aspect_embeddings, importance_weights)
        
        # 3. Consistency loss
        consistency_loss = self.compute_consistency_loss(hrrp_aspect_features, aspect_weights)
        
        # 4. Contrastive loss
        # contrastive_loss = self.compute_contrastive_loss(hrrp_aspect_features, label_aspect_embeddings, importance_weights)
        
        # 总损失
        total_loss = aspect_loss + global_loss + 0.1 * consistency_loss #+ 0.5 * contrastive_loss
        
        loss_components = {
            'total_loss': total_loss.detach().cpu().item(),
            'aspect_loss': aspect_loss.detach().cpu().item(),
            'global_loss': global_loss.detach().cpu().item(),
            'consistency_loss': consistency_loss.detach().cpu().item(),
            # 'contrastive_loss': contrastive_loss.detach().cpu().item()
        }
        
        return total_loss, loss_components
    
    def compute_aspect_wise_loss(self, hrrp_aspect_features, label_aspect_embeddings, aspect_weights):
        """计算aspect-wise对齐损失"""
        batch_size = hrrp_aspect_features.size(0)
        num_labels = label_aspect_embeddings.size(0)
        num_aspects = hrrp_aspect_features.size(1)
        
        # 归一化特征
        hrrp_norm = F.normalize(hrrp_aspect_features, p=2, dim=-1)  # (batch, aspects, dim)
        label_norm = F.normalize(label_aspect_embeddings, p=2, dim=-1)  # (labels, aspects, dim)
        
        # 计算每个aspect的相似度矩阵
        aspect_losses = []
        for aspect_idx in range(num_aspects):
            hrrp_aspect = hrrp_norm[:, aspect_idx, :]  # (batch, dim)
            label_aspect = label_norm[:, aspect_idx, :]  # (labels, dim)
            
            # 计算相似度矩阵
            similarity = torch.matmul(hrrp_aspect, label_aspect.T)  # (batch, labels)
            
            # 使用aspect权重加权
            aspect_weight = aspect_weights[:, aspect_idx].unsqueeze(1)  # (batch, 1)
            weighted_similarity = similarity * aspect_weight
            
            # 计算负对数似然损失
            aspect_loss = -torch.log(torch.softmax(weighted_similarity / self.temperature, dim=1) + 1e-8).mean()
            aspect_losses.append(aspect_loss)
        
        return torch.stack(aspect_losses).mean()
    
    def compute_global_loss(self, hrrp_aspect_features, label_aspect_embeddings, importance_weights):
        """计算全局对齐损失"""
        batch_size = hrrp_aspect_features.size(0)
        num_labels = label_aspect_embeddings.size(0)
        
        # 使用重要性权重聚合aspect特征
        hrrp_global = (hrrp_aspect_features * importance_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (batch, dim)
        label_global = (label_aspect_embeddings * importance_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (labels, dim)
        
        # 归一化
        hrrp_global_norm = F.normalize(hrrp_global, p=2, dim=-1)
        label_global_norm = F.normalize(label_global, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(hrrp_global_norm, label_global_norm.T)  # (batch, labels)
        
        # 计算负对数似然损失
        global_loss = -torch.log(torch.softmax(similarity / self.temperature, dim=1) + 1e-8).mean()
        
        return global_loss
    
    def compute_consistency_loss(self, hrrp_aspect_features, aspect_weights):
        """计算一致性损失，确保aspect权重的一致性"""
        batch_size = hrrp_aspect_features.size(0)
        num_aspects = hrrp_aspect_features.size(1)
        
        # 计算aspect权重的方差，鼓励权重分布的一致性
        weight_variance = torch.var(aspect_weights, dim=1)  # (batch,)
        consistency_loss = weight_variance.mean()
        
        return consistency_loss
    
    # def compute_contrastive_loss(self, hrrp_aspect_features, label_aspect_embeddings, importance_weights):
    #     """计算对比学习损失"""
    #     batch_size = hrrp_aspect_features.size(0)
    #     num_labels = label_aspect_embeddings.size(0)
        
    #     # 检查维度是否匹配
    #     if hrrp_aspect_features.size(-1) != label_aspect_embeddings.size(-1):
    #         projection = nn.Linear(hrrp_aspect_features.size(-1), label_aspect_embeddings.size(-1)).to(hrrp_aspect_features.device)
    #         hrrp_aspect_features = projection(hrrp_aspect_features)
        
    #     # 聚合aspect特征
    #     hrrp_global = (hrrp_aspect_features * importance_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (Batch, embedding_dim)
    #     label_global = (label_aspect_embeddings * importance_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (num_labels, embedding_dim)
        
    #     # 归一化
    #     hrrp_global_norm = F.normalize(hrrp_global, p=2, dim=-1)
    #     label_global_norm = F.normalize(label_global, p=2, dim=-1)
        
    #     # 计算相似度矩阵
    #     similarity_matrix = torch.matmul(hrrp_global_norm, label_global_norm.T)  # (Batch, num_labels)
        
    #     # 温度参数
    #     temperature = 0.1
    #     logits = similarity_matrix / temperature
        
    #     # 保证logits和labels的第一个维度一致
    #     N = min(batch_size, num_labels)
    #     logits = logits[:N, :N]
    #     labels = torch.arange(N, device=logits.device)
    #     contrastive_loss = F.cross_entropy(logits, labels)
        
    #     return contrastive_loss


class HierarchicalAspectLoss(nn.Module):
    """Hierarchical aspect-aware loss for multi-level feature alignment"""
    
    def __init__(self, label_embedding_dim: int = 768, temperature: float = 0.1):
        super(HierarchicalAspectLoss, self).__init__()
        self.label_embedding_dim = label_embedding_dim
        self.temperature = temperature
        
    def forward(self, hrrp_features, hrrp_aspect_features, label_embeddings, label_aspect_embeddings, aspect_weights):
        """
        Args:
            hrrp_features: (batch_size, feature_dim) - HRRP全局特征
            hrrp_aspect_features: (batch_size, num_aspects, feature_dim) - HRRP aspect特征
            label_embeddings: (num_labels, embedding_dim) - 标签全局embedding
            label_aspect_embeddings: (num_labels, num_aspects, embedding_dim) - 标签aspect embedding
            aspect_weights: (batch_size, num_aspects) - aspect权重
        """
        # 1. 全局特征对齐损失
        global_loss = self.compute_global_alignment_loss(hrrp_features, label_embeddings)
        
        # 2. Aspect特征对齐损失
        aspect_loss = self.compute_aspect_alignment_loss(hrrp_aspect_features, label_aspect_embeddings, aspect_weights)
        
        # 3. 层次一致性损失
        hierarchy_loss = self.compute_hierarchy_consistency_loss(hrrp_features, hrrp_aspect_features, aspect_weights)
        
        # 总损失
        total_loss = global_loss + aspect_loss + 0.2 * hierarchy_loss
        
        return {
            'total_loss': total_loss,
            'global_loss': global_loss,
            'aspect_loss': aspect_loss,
            'hierarchy_loss': hierarchy_loss
        }
    
    def compute_global_alignment_loss(self, hrrp_features, label_embeddings):
        """计算全局特征对齐损失"""
        # 检查维度是否匹配
        if hrrp_features.size(-1) != label_embeddings.size(-1):
            projection = nn.Linear(hrrp_features.size(-1), label_embeddings.size(-1)).to(hrrp_features.device)
            hrrp_features = projection(hrrp_features)
        
        # 归一化
        hrrp_norm = F.normalize(hrrp_features, p=2, dim=-1)
        label_norm = F.normalize(label_embeddings, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(hrrp_norm, label_norm.T)
        
        # 计算负对数似然损失
        global_loss = -torch.log(torch.softmax(similarity / self.temperature, dim=1) + 1e-8).mean()
        
        return global_loss
    
    def compute_aspect_alignment_loss(self, hrrp_aspect_features, label_aspect_embeddings, aspect_weights):
        """计算aspect特征对齐损失"""
        batch_size = hrrp_aspect_features.size(0)
        num_aspects = hrrp_aspect_features.size(1)
        
        # 检查维度是否匹配
        if hrrp_aspect_features.size(-1) != label_aspect_embeddings.size(-1):
            projection = nn.Linear(hrrp_aspect_features.size(-1), label_aspect_embeddings.size(-1)).to(hrrp_aspect_features.device)
            hrrp_aspect_features = projection(hrrp_aspect_features)
        
        # 归一化
        hrrp_norm = F.normalize(hrrp_aspect_features, p=2, dim=-1)
        label_norm = F.normalize(label_aspect_embeddings, p=2, dim=-1)
        
        # 计算每个aspect的损失
        aspect_losses = []
        for aspect_idx in range(num_aspects):
            hrrp_aspect = hrrp_norm[:, aspect_idx, :]
            label_aspect = label_norm[:, aspect_idx, :]
            
            # 计算相似度
            similarity = torch.matmul(hrrp_aspect, label_aspect.T)
            
            # 使用aspect权重加权
            aspect_weight = aspect_weights[:, aspect_idx].unsqueeze(1)
            weighted_similarity = similarity * aspect_weight
            
            # 计算损失
            aspect_loss = -torch.log(torch.softmax(weighted_similarity / self.temperature, dim=1) + 1e-8).mean()
            aspect_losses.append(aspect_loss)
        
        return torch.stack(aspect_losses).mean()
    
    def compute_hierarchy_consistency_loss(self, hrrp_features, hrrp_aspect_features, aspect_weights):
        """计算层次一致性损失"""
        batch_size = hrrp_features.size(0)
        num_aspects = hrrp_aspect_features.size(1)
        
        # 检查维度是否匹配
        if hrrp_features.size(-1) != hrrp_aspect_features.size(-1):
            projection = nn.Linear(hrrp_aspect_features.size(-1), hrrp_features.size(-1)).to(hrrp_aspect_features.device)
            hrrp_aspect_features = projection(hrrp_aspect_features)
        
        # 从aspect特征重建全局特征
        reconstructed_global = (hrrp_aspect_features * aspect_weights.unsqueeze(-1)).sum(dim=1)
        
        # 归一化
        hrrp_norm = F.normalize(hrrp_features, p=2, dim=-1)
        reconstructed_norm = F.normalize(reconstructed_global, p=2, dim=-1)
        
        # 计算一致性损失
        consistency_loss = 1 - F.cosine_similarity(hrrp_norm, reconstructed_norm, dim=-1).mean()
        
        return consistency_loss 