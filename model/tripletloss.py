import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeTripletLoss(nn.Module):
    """
    一个复合三元组损失函数，结合了类间分离和类内角度排序。

    L_total = L_class + lambda * L_angular

    - L_class: 标准三元组损失，旨在将不同类别的样本推开。
    - L_angular: 角度三元组损失，旨在根据角度差异对同一类别内的样本进行排序。
    """

    def __init__(
        self, margin_class: float, margin_angular: float, lambda_angular: float
    ):
        """
        初始化损失函数。

        Args:
            margin_class (float): 类间分离的边距 (m1)。
            margin_angular (float): 类内角度排序的边距 (m2)。
            lambda_angular (float): 平衡两个损失的权重 (λ)。
        """
        super(CompositeTripletLoss, self).__init__()
        if margin_class <= 0:
            raise ValueError("Class margin must be positive.")
        if margin_angular <= 0:
            raise ValueError("Angular margin must be positive.")

        self.margin_class = margin_class
        self.margin_angular = margin_angular
        self.lambda_angular = lambda_angular

        # 使用平方欧氏距离，可以避免开方操作，计算更稳定高效
        self.distance_function = lambda x, y: torch.sum(torch.pow(x - y, 2), dim=-1)

    def _calculate_angular_distance(self, angles: torch.Tensor) -> torch.Tensor:
        """计算批次内所有样本对之间的周期性角度距离（弧度制）。"""
        # 扩展维度以进行广播
        angles1 = angles.unsqueeze(1)  # (batch_size, 1)
        angles2 = angles.unsqueeze(0)  # (1, batch_size)

        # 计算角度差
        diff = torch.abs(angles1 - angles2)

        # 处理周期性
        return torch.min(diff, 2 * torch.pi - diff)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, angles: torch.Tensor
    ) -> torch.Tensor:
        """
        计算复合损失。

        Args:
            embeddings (torch.Tensor): 编码器输出的特征向量，形状 (batch_size, feature_dim)。
            labels (torch.Tensor): 样本的类别标签，形状 (batch_size,)。
            angles (torch.Tensor): 样本的角度信息（弧度制），形状 (batch_size,)。

        Returns:
            torch.Tensor: 一个标量，代表批次的最终损失值。
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # 1. 计算所有样本对之间的平方欧氏距离
        # (a-b)^2 = a^2 - 2ab + b^2
        dot_product = torch.matmul(embeddings, embeddings.t())
        sq_norm = torch.diag(dot_product)
        dist_sq = sq_norm.unsqueeze(1) - 2.0 * dot_product + sq_norm.unsqueeze(0)
        # 确保对角线为0且非负
        dist_sq = F.relu(dist_sq)

        # 2. 创建掩码
        # mask_positive[i, j] is True if i and j have the same label
        mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0)).fill_diagonal_(
            False
        )
        # mask_negative[i, j] is True if i and j have different labels
        mask_negative = ~mask_positive
        mask_negative.fill_diagonal_(False)

        # ==================================
        # 3. 计算 L_class (类间分离损失)
        # ==================================

        # 使用难例挖掘 (Hard Mining)
        # For each anchor, find the hardest positive and hardest negative

        # Hardest positive: a_p_dist[i] = max_{j where y_i=y_j} D(f_i, f_j)
        # 我们用一个很大的负数填充非正样本对的距离，以便max操作能正确工作
        dist_ap = torch.max(dist_sq * mask_positive.float(), dim=1)[0]

        # Hardest negative: a_n_dist[i] = min_{k where y_i!=y_k} D(f_i, f_k)
        # 我们用一个很大的正数填充正样本对的距离，以便min操作能正确工作
        dist_an = dist_sq.clone()
        dist_an[mask_negative == False] = float("inf")
        dist_an = torch.min(dist_an, dim=1)[0]

        # 计算损失
        class_loss_triplets = F.relu(dist_ap - dist_an + self.margin_class)

        # 只对存在正样本的锚点计算损失
        valid_class_triplets = mask_positive.sum(dim=1) > 0
        class_loss = (
            class_loss_triplets[valid_class_triplets].mean()
            if valid_class_triplets.any()
            else torch.tensor(0.0, device=device)
        )

        # =====================================
        # 4. 计算 L_angular (类内角度序损失)
        # =====================================

        # 计算所有样本对之间的角度距离
        angular_dist_matrix = self._calculate_angular_distance(angles)

        total_angular_loss = torch.tensor(0.0, device=device)
        num_angular_triplets = 0

        # 遍历每个样本作为锚点
        for i in range(batch_size):
            # 找到所有与锚点同类的正样本
            pos_indices = torch.where(mask_positive[i])[0]

            # 至少需要2个其他同类样本才能构成一个角度三元组
            if len(pos_indices) < 2:
                continue

            # 获取这些正样本与锚点的特征距离和角度距离
            anchor_embedding = embeddings[i].unsqueeze(0)
            positive_embeddings = embeddings[pos_indices]

            feat_dist_to_anchor = self.distance_function(
                anchor_embedding, positive_embeddings
            )
            angle_dist_to_anchor = angular_dist_matrix[i, pos_indices]

            # 构建所有可能的 (P_near, P_far) 对
            # P_near 的角度距离更小
            mask_nearer = angle_dist_to_anchor.unsqueeze(
                1
            ) < angle_dist_to_anchor.unsqueeze(0)

            # 对应的特征距离
            feat_dist_near = feat_dist_to_anchor.unsqueeze(1).expand(
                -1, len(pos_indices)
            )
            feat_dist_far = feat_dist_to_anchor.unsqueeze(0).expand(
                len(pos_indices), -1
            )

            # 计算所有有效角度三元组的损失
            angular_loss_pairs = F.relu(
                feat_dist_near[mask_nearer]
                - feat_dist_far[mask_nearer]
                + self.margin_angular
            )

            if angular_loss_pairs.numel() > 0:
                total_angular_loss += angular_loss_pairs.sum()
                num_angular_triplets += angular_loss_pairs.numel()

        angular_loss = total_angular_loss / (num_angular_triplets + 1e-8)

        # ====================
        # 5. 合并总损失
        # ====================
        total_loss = class_loss + self.lambda_angular * angular_loss

        return total_loss


# =============================================
# ===             使用示例                   ===
# =============================================
if __name__ == "__main__":
    # 设置超参数
    BATCH_SIZE = 64
    FEATURE_DIM = 32
    NUM_CLASSES = 10
    MARGIN_CLASS = 1.0
    MARGIN_ANGULAR = 0.2
    LAMBDA_ANGULAR = 0.5

    # 创建损失函数实例
    composite_loss_fn = CompositeTripletLoss(
        margin_class=MARGIN_CLASS,
        margin_angular=MARGIN_ANGULAR,
        lambda_angular=LAMBDA_ANGULAR,
    )

    # 生成模拟数据
    # 使用类平衡采样，例如每10个样本属于一个类
    print("--- 生成模拟数据 ---")
    # 确保在创建张量时就设置requires_grad=True
    mock_embeddings = torch.randn(BATCH_SIZE, FEATURE_DIM, requires_grad=True)
    mock_labels = torch.repeat_interleave(
        torch.arange(BATCH_SIZE // 4), 4
    )  # 4 samples per class
    mock_angles = torch.rand(BATCH_SIZE) * 2 * torch.pi  # 随机角度 (0 to 2π)

    print(f"特征形状: {mock_embeddings.shape}")
    print(f"标签形状: {mock_labels.shape}")
    print(f"角度形状: {mock_angles.shape}")
    print("-" * 20)

    # 计算损失
    print("--- 计算损失 ---")
    total_loss = composite_loss_fn(mock_embeddings, mock_labels, mock_angles)

    print(f"复合损失值: {total_loss.item():.4f}")

    # 计算梯度
    total_loss.backward()
    print("梯度计算成功。")
    print(f"第一个特征向量的梯度（部分）: \n{mock_embeddings.grad[0][:5]}")
