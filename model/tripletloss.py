import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossWithMining(nn.Module):
    """
    带有难例挖掘的三元组损失
    实现Semi-Hard Negative Mining策略
    """
    
    def __init__(self, margin=1.0, mining_strategy='semi_hard'):
        """
        初始化三元组损失
        
        Args:
            margin (float): 三元组损失的边界值
            mining_strategy (str): 挖掘策略，支持 'semi_hard', 'hardest', 'random'
        """
        super(TripletLossWithMining, self).__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        
    def forward(self, anchor, positive, all_class_semantics, labels):
        """
        计算带有难例挖掘的三元组损失
        
        Args:
            anchor (torch.Tensor): anchor特征，形状为 [batch_size, feature_dim]
            positive (torch.Tensor): 正样本特征，形状为 [batch_size, feature_dim]
            all_class_semantics (torch.Tensor): 所有类别的语义向量，形状为 [num_classes, feature_dim]
            labels (torch.Tensor): 标签，形状为 [batch_size]
            
        Returns:
            torch.Tensor: 三元组损失值
            dict: 包含损失统计信息的字典
        """
        batch_size = anchor.shape[0]
        num_classes = all_class_semantics.shape[0]
        
        # 确保labels的维度正确
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1:
            labels = labels.squeeze()
        
        # 确保labels的长度与batch_size匹配
        if labels.shape[0] != batch_size:
            raise ValueError(f"labels的形状 {labels.shape} 与batch_size {batch_size} 不匹配")
        
        # 计算anchor与所有类语义向量的距离矩阵
        dist_matrix = torch.cdist(anchor, all_class_semantics, p=2)  # [batch_size, num_classes]
        
        # 获取anchor与positive的距离
        dist_ap = dist_matrix.gather(1, labels.unsqueeze(1)).squeeze()  # [batch_size]
        
        # 根据挖掘策略选择负样本
        if self.mining_strategy == 'semi_hard':
            negative_indices = self._semi_hard_mining(dist_matrix, dist_ap, labels)
        elif self.mining_strategy == 'hardest':
            negative_indices = self._hardest_mining(dist_matrix, labels)
        elif self.mining_strategy == 'random':
            negative_indices = self._random_mining(dist_matrix, labels)
        else:
            raise ValueError(f"不支持的挖掘策略: {self.mining_strategy}")
        
        # 获取负样本
        negative = all_class_semantics[negative_indices]  # [batch_size, feature_dim]
        
        # 计算三元组损失
        dist_an = torch.cdist(anchor, negative, p=2).diagonal()  # [batch_size]
        
        # 三元组损失公式: max(0, dist(a,p) - dist(a,n) + margin)
        triplet_loss = F.relu(dist_ap - dist_an + self.margin)
        
        # 计算平均损失
        avg_loss = triplet_loss.mean()
        
        # 统计信息
        stats = {
            'avg_dist_ap': dist_ap.mean().item(),
            'avg_dist_an': dist_an.mean().item(),
            'num_valid_triplets': (triplet_loss > 0).sum().item(),
            'total_triplets': batch_size,
            'mining_strategy': self.mining_strategy
        }
        
        return avg_loss, stats
    
    def _semi_hard_mining(self, dist_matrix, dist_ap, labels):
        """
        Semi-Hard Negative Mining
        选择满足条件的负样本：dist(a,n) > dist(a,p) 且 dist(a,n) < dist(a,p) + margin
        """
        batch_size = dist_matrix.shape[0]
        
        # 创建mask，排除positive自身
        mask = torch.ones_like(dist_matrix, dtype=torch.bool)
        mask.scatter_(1, labels.unsqueeze(1), False)
        
        # 找到满足semi-hard条件的negative
        # 条件1: dist(a, n) > dist(a, p)
        # 条件2: dist(a, n) < dist(a, p) + margin
        semi_hard_mask = (dist_matrix > dist_ap.unsqueeze(1)) & \
                        (dist_matrix < (dist_ap + self.margin).unsqueeze(1)) & \
                        mask
        
        negative_indices = []
        for i in range(batch_size):
            sample_mask = semi_hard_mask[i]
            if sample_mask.any():
                # 从满足条件的负样本中随机选一个
                possible_neg_indices = torch.where(sample_mask)[0]
                neg_idx = possible_neg_indices[torch.randint(0, len(possible_neg_indices), (1,))]
            else:
                # 没有semi-hard的，就选最难的那个(不包括positive自身)
                dist_with_inf = dist_matrix[i].clone()
                dist_with_inf[labels[i]] = float('inf')
                neg_idx = dist_with_inf.argmin().unsqueeze(0)  # 确保维度一致
            negative_indices.append(neg_idx)
            
        return torch.cat(negative_indices)  # 使用cat而不是stack
    
    def _hardest_mining(self, dist_matrix, labels):
        """
        Hardest Negative Mining
        选择距离anchor最近的负样本（不包括positive）
        """
        batch_size = dist_matrix.shape[0]
        
        # 将positive的距离设为无穷大
        dist_with_inf = dist_matrix.clone()
        dist_with_inf.scatter_(1, labels.unsqueeze(1), float('inf'))
        
        # 选择距离最小的负样本
        negative_indices = dist_with_inf.argmin(dim=1)
        
        return negative_indices
    
    def _random_mining(self, dist_matrix, labels):
        """
        Random Negative Mining
        随机选择负样本（不包括positive）
        """
        batch_size = dist_matrix.shape[0]
        num_classes = dist_matrix.shape[1]
        
        negative_indices = []
        for i in range(batch_size):
            # 创建可选的负样本索引（排除positive）
            valid_indices = list(range(num_classes))
            valid_indices.remove(labels[i].item())
            
            # 随机选择一个负样本
            neg_idx = torch.tensor([valid_indices[torch.randint(0, len(valid_indices), (1,))]], device=dist_matrix.device)
            negative_indices.append(neg_idx)
            
        return torch.cat(negative_indices)


def compute_triplet_loss_with_mining(projected_features, all_class_semantics, labels, margin=1.0, mining_strategy='semi_hard'):
    """
    便捷函数：计算带有难例挖掘的三元组损失
    
    Args:
        projected_features (torch.Tensor): 投影后的特征，形状为 [batch_size, feature_dim]
        all_class_semantics (torch.Tensor): 所有类别的语义向量，形状为 [num_classes, feature_dim]
        labels (torch.Tensor): 标签，形状为 [batch_size]
        margin (float): 三元组损失的边界值
        mining_strategy (str): 挖掘策略
        
    Returns:
        torch.Tensor: 三元组损失值
        dict: 包含损失统计信息的字典
    """
    triplet_loss_fn = TripletLossWithMining(margin=margin, mining_strategy=mining_strategy)
    return triplet_loss_fn(projected_features, all_class_semantics, all_class_semantics, labels)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: model's predictions, shape [N, C] (logits)
        targets: ground truth, shape [N]
        """
        # 首先计算标准的交叉熵损失，但不要进行reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # pt 是模型对正确类别的预测概率
        # inputs.log_softmax(dim=1) 计算 log(softmax(logits))
        # gather 用于根据targets的索引，在log_probs中取出对应的值
        log_probs = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1))).squeeze(1)

        # 计算focal loss的权重
        focal_weight = (1 - pt).pow(self.gamma)

        # 计算最终的focal loss
        loss = focal_weight * ce_loss

        # 如果alpha不为None，则乘以alpha权重
        if self.alpha is not None:
            # 同样地，根据targets取出对应的alpha值
            alpha_t = torch.tensor([self.alpha for _ in range(inputs.shape[0])], device=inputs.device)
            loss = alpha_t * loss

        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss