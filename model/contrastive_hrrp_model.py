import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 模块一：HRRP特征提取器
class HRRPExtractor(nn.Module):
    """使用1D CNN从HRRP信号中提取特征序列"""

    def __init__(self, output_dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(
                in_channels=64, out_channels=output_dim, kernel_size=3, padding=1
            ),
        )

    def forward(self, hrrp_signal):
        # 输入 hrrp_signal: (Batch, 1, SignalLength)
        features = self.conv_block(hrrp_signal)  # -> (Batch, output_dim, SignalLength)
        # 转换为Transformer期望的格式: (Batch, SeqLen, FeatureDim)
        features = features.permute(0, 2, 1)
        return features


# 模块二：角度信息编码器
class AngleMLP(nn.Module):
    """将(方位角,俯仰角)编码为一个旋转角度因子"""

    def __init__(self):
        super().__init__()
        # 角度预处理为(cos(az), sin(az), cos(el), sin(el))，所以输入维度是4
        self.mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1)  # 输出一个标量作为旋转角度
        )

    def forward(self, angles_encoded):
        # 输入 angles_encoded: (Batch, 4)
        rotation_angle = self.mlp(angles_encoded)
        return rotation_angle


# 模块三：旋转融合层
class RotaryFusionLayer(nn.Module):
    """使用角度因子对HRRP特征序列中的每个向量进行旋转"""

    def __init__(self):
        super().__init__()

    def forward(self, feature_sequence, rotation_angles):
        # feature_sequence: (Batch, SeqLen, FeatureDim)
        # rotation_angles: (Batch, 1) or (Batch,)

        seq_even = feature_sequence[..., 0::2]  # (Batch, SeqLen, FeatureDim/2)
        seq_odd = feature_sequence[..., 1::2]  # (Batch, SeqLen, FeatureDim/2)

        cos_a = torch.cos(rotation_angles).unsqueeze(-1)  # (Batch, 1, 1)
        sin_a = torch.sin(rotation_angles).unsqueeze(-1)  # (Batch, 1, 1)

        # 应用旋转
        out_even = seq_even * cos_a - seq_odd * sin_a
        out_odd = seq_even * sin_a + seq_odd * cos_a

        rotated_sequence = torch.zeros_like(feature_sequence)
        rotated_sequence[..., 0::2] = out_even
        rotated_sequence[..., 1::2] = out_odd

        return rotated_sequence


# 模块四：投影头（用于对比学习）
class ProjectionHead(nn.Module):
    """对比学习中的投影头，将特征映射到对比学习空间"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projection(x)


# 对比学习模型
class ContrastiveHRRPModel(nn.Module):
    def __init__(
        self,
        hrrp_len,
        feature_dim,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=512,
        projection_hidden_dim=256,
        projection_output_dim=128,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # 1. HRRP特征提取器
        self.hrrp_extractor = HRRPExtractor(output_dim=feature_dim)

        # 2. 角度编码器
        self.angle_mlp = AngleMLP()

        # 3. 旋转融合层
        self.rotary_fusion = RotaryFusionLayer()

        # 4. Transformer编码器
        # 创建一个可学习的 [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 5. 投影头（用于对比学习）
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
        )

    def _encode_angles(self, angles):
        # angles: (Batch, 2) -> [[az, el], [az, el], ...]
        # 转换为弧度
        angles_rad = torch.deg2rad(angles)
        az_rad = angles_rad[:, 0]
        el_rad = angles_rad[:, 1]

        # 三角函数编码
        encoded = torch.stack(
            [
                torch.cos(az_rad),
                torch.sin(az_rad),
                torch.cos(el_rad),
                torch.sin(el_rad),
            ],
            dim=-1,
        )
        return encoded.float()

    def forward_encoder(self, hrrp_signal, angles):
        """编码器前向传播，返回特征表示"""
        # hrrp_signal: (Batch, SignalLength)
        # angles: (Batch, 2) -> [[az_deg, el_deg], ...]

        # 0. 预处理
        # 为1D CNN增加通道维度 -> (Batch, 1, SignalLength)
        hrrp_signal = hrrp_signal.unsqueeze(1)
        # 角度三角函数编码 -> (Batch, 4)
        angles_encoded = self._encode_angles(angles)

        # 1. 角度通路：提取旋转角度因子
        rotation_theta = self.angle_mlp(angles_encoded)  # -> (Batch, 1)

        # 2. HRRP通路：提取特征序列
        hrrp_features = self.hrrp_extractor(
            hrrp_signal
        )  # -> (Batch, SeqLen, FeatureDim)

        # 3. 融合：旋转特征序列
        rotated_features = self.rotary_fusion(hrrp_features, rotation_theta)

        # 4. Transformer编码
        # 准备输入给Transformer的序列
        batch_size = rotated_features.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 将CLS token复制B份
        transformer_input = torch.cat(
            (cls_tokens, rotated_features), dim=1
        )  # -> (Batch, 1+SeqLen, FeatureDim)

        # 通过Transformer编码器
        transformer_output = self.transformer_encoder(
            transformer_input
        )  # -> (Batch, 1+SeqLen, FeatureDim)

        # 提取 [CLS] token对应的输出作为整个序列的表示
        cls_output = transformer_output[:, 0, :]  # -> (Batch, FeatureDim)

        return cls_output

    def forward(self, hrrp_signal, angles):
        """完整的前向传播，返回投影后的特征（用于对比学习）"""
        # 获取编码器输出
        encoded_features = self.forward_encoder(hrrp_signal, angles)

        # 通过投影头
        projected_features = self.projection_head(encoded_features)

        return projected_features


# 分类模型（基于对比学习模型）
class ClassificationHRRPModel(nn.Module):
    def __init__(
        self,
        contrastive_model: ContrastiveHRRPModel,
        num_classes: int,
    ):
        super().__init__()

        # 冻结对比学习模型的参数
        for param in contrastive_model.parameters():
            param.requires_grad = False

        self.contrastive_model = contrastive_model

        # 分类头
        self.classifier = nn.Linear(contrastive_model.feature_dim, num_classes)

    def forward(self, hrrp_signal, angles):
        # 获取编码器特征（不经过投影头）
        features = self.contrastive_model.forward_encoder(hrrp_signal, angles)

        # 分类
        logits = self.classifier(features)

        return logits


# --- 使用示例 ---
if __name__ == "__main__":
    # 模型超参数
    BATCH_SIZE = 4
    HRRP_LENGTH = 512
    FEATURE_DIM = 64  # 特征维度必须是偶数
    PROJECTION_DIM = 128

    # 实例化对比学习模型
    contrastive_model = ContrastiveHRRPModel(
        hrrp_len=HRRP_LENGTH,
        feature_dim=FEATURE_DIM,
        nhead=4,
        num_encoder_layers=2,
        projection_hidden_dim=256,
        projection_output_dim=PROJECTION_DIM,
    )

    # 创建模拟输入数据
    dummy_hrrp = torch.randn(BATCH_SIZE, HRRP_LENGTH)
    dummy_angles = torch.tensor(
        [[10.0, 20.0], [45.0, 30.0], [90.0, 5.0], [180.0, 15.0]]
    )  # (Batch, 2)

    print(f"输入HRRP形状: {dummy_hrrp.shape}")
    print(f"输入角度形状: {dummy_angles.shape}\n")

    # 前向传播
    projected_features = contrastive_model(dummy_hrrp, dummy_angles)

    print(
        f"对比学习模型输出形状: {projected_features.shape}"
    )  # 应该为 (BATCH_SIZE, PROJECTION_DIM)

    # 检查参数数量
    total_params = sum(
        p.numel() for p in contrastive_model.parameters() if p.requires_grad
    )
    print(f"对比学习模型总参数量: {total_params:,}")
