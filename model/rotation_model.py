import torch
import torch.nn as nn

from model.hrrp_model import HRRPExtractor, AngleMLP, RotaryFusionLayer




class RotationModel(nn.Module):
    def __init__(self, hrrp_len, feature_dim, nhead=8, num_encoder_layers=3, dim_feedforward=1024):
        super().__init__()
        self.hrrp_extractor = HRRPExtractor(feature_dim)
        self.angle_mlp = AngleMLP()
        self.rotary_fusion_layer = RotaryFusionLayer()

        
        
    def forward(self, hrrp_data, angle_data):
        
        if hrrp_data.dim() == 2:  # (batch_size, signal_length)
            hrrp_data = hrrp_data.unsqueeze(1)  # (batch_size, 1, signal_length)
        
        hrrp_features = self.hrrp_extractor(hrrp_data)
        angle_features = self.angle_mlp(angle_data)
        
        fused_features = self.rotary_fusion_layer(hrrp_features, angle_features)
        return hrrp_features, fused_features
    
    
class AngleNormalizationLoss(nn.Module):
    """
    角度归一化损失函数
    目标：将不同角度的同一目标HRRP信号旋转到相同的参考角度
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        
    def forward(self, rotated_features, target_ids, angles):
        """
        Args:
            rotated_features: (batch_size, seq_len, feature_dim) 旋转后的特征
            target_ids: (batch_size,) 目标ID，相同ID表示同一目标
            angles: (batch_size, 2) 原始角度信息
        """
        batch_size = rotated_features.size(0)
        
        # 1. 相似性损失：同一目标在不同角度下的特征应该相似
        similarity_loss = 0.0
        similarity_count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if target_ids[i] == target_ids[j]:  # 同一目标
                    # 计算特征相似性（使用全局平均池化后的特征）
                    feat_i = rotated_features[i].mean(dim=0)  # (feature_dim,)
                    feat_j = rotated_features[j].mean(dim=0)  # (feature_dim,)
                    
                    # 余弦相似性损失（希望相似性接近1）
                    similarity = self.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0))
                    similarity_loss += (1 - similarity) ** 2
                    similarity_count += 1
        
        if similarity_count > 0:
            similarity_loss = similarity_loss / similarity_count
        

        
        # 3. 总损失
        total_loss =similarity_loss
        
        return total_loss


if __name__ == "__main__":
    model = RotationModel(hrrp_len=512, feature_dim=512)
    hrrp_data = torch.randn(32,1, 512)
    angle_data = torch.randn(32, 2) 
    output = model(hrrp_data, angle_data)
    print(output.shape)


