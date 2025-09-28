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
    
    def forward(self, angles):
        angles_encoded = self._encode_angles(angles)
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
        # rotation_angles: (Batch, 1) or (Batch,) or may contain extra singleton dims like (Batch,1,1,1)

        # 形状与数值健壮性检查
        assert feature_sequence.dim() == 3, f"feature_sequence 应为3维 (B, S, D)，实际为 {feature_sequence.shape}"
        assert feature_sequence.size(-1) % 2 == 0, f"FeatureDim 必须为偶数，实际为 {feature_sequence.size(-1)}"
        batch_size, seq_len, feat_dim = feature_sequence.shape

        # 将 rotation_angles 规范为 (B, 1, 1)
        # 兼容多余的1维，如 (B,1,1,1)
        if rotation_angles.dim() > 2:
            rotation_angles = rotation_angles.view(batch_size, -1)
        if rotation_angles.dim() == 2 and rotation_angles.size(1) > 1:
            rotation_angles = rotation_angles[:, :1]
        if rotation_angles.dim() == 2 and rotation_angles.size(1) == 1:
            rotation_angles = rotation_angles.squeeze(1)
        if rotation_angles.dim() == 0:
            rotation_angles = rotation_angles.unsqueeze(0)
        assert rotation_angles.dim() in (1, 2), f"rotation_angles 维度应为1或2，实际为 {rotation_angles.shape}"
        rotation_angles = rotation_angles.view(batch_size, 1, 1)

        seq_even = feature_sequence[..., 0::2]  # (Batch, SeqLen, FeatureDim/2)
        seq_odd = feature_sequence[..., 1::2]  # (Batch, SeqLen, FeatureDim/2)

        cos_a = torch.cos(rotation_angles)  # (Batch, 1, 1)
        sin_a = torch.sin(rotation_angles)  # (Batch, 1, 1)

        # 应用旋转（利用广播到 (B, S, D/2)）
        out_even = seq_even * cos_a - seq_odd * sin_a
        out_odd = seq_even * sin_a + seq_odd * cos_a

        rotated_sequence = torch.zeros_like(feature_sequence)
        rotated_sequence[..., 0::2] = out_even
        rotated_sequence[..., 1::2] = out_odd

        return rotated_sequence


# # 模块四：投影头（用于对比学习）
# class ProjectionHead(nn.Module):
#     """对比学习中的投影头，将特征映射到对比学习空间"""

#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim),
#         )

#     def forward(self, x):
#         return self.projection(x)

class HRRPModel(nn.Module):
    def __init__(
        self,
        hrrp_len,
        feature_dim,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
        # projection_hidden_dim=256,
        # projection_output_dim=128,
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

        # # 5. 投影头（用于对比学习）
        # self.projection_head = ProjectionHead(
        #     input_dim=feature_dim,
        #     hidden_dim=projection_hidden_dim,
        #     output_dim=projection_output_dim,
        # )

    
    def forward_encoder(self, hrrp_signal, angles):
        """编码器前向传播，返回特征表示"""
        # hrrp_signal: (Batch, SignalLength)
        # angles: (Batch, 2) -> [[az_deg, el_deg], ...]

        # 0. 预处理
        # 为1D CNN增加通道维度 -> (Batch, 1, SignalLength)
        hrrp_signal = hrrp_signal.unsqueeze(1)

        # 1. 角度通路：提取旋转角度因子
        rotation_theta = self.angle_mlp(angles)  # -> (Batch, 1)

        # 2. HRRP通路：提取特征序列
        hrrp_features = self.hrrp_extractor(
            hrrp_signal
        )  # -> (Batch, SeqLen, FeatureDim)

        # 调试打印：特征提取结果（每100个batch）
        if hasattr(self, 'debug_mode') and self.debug_mode and hasattr(self, 'batch_count') and self.batch_count % 100 == 0:
            print(f"[DEBUG] Batch {self.batch_count} - HRRP特征统计 - 均值: {hrrp_features.mean():.4f}, 标准差: {hrrp_features.std():.4f}")
            print(f"[DEBUG] Batch {self.batch_count} - 旋转角度因子 - 均值: {rotation_theta.mean():.4f}, 标准差: {rotation_theta.std():.4f}")

        # 3. 融合：旋转特征序列
        rotated_features = self.rotary_fusion(hrrp_features, rotation_theta)

        # 4. Transformer编码
        # 准备输入给Transformer的序列
        batch_size = rotated_features.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 将CLS token复制B份
        transformer_input = torch.cat(
            (cls_tokens, rotated_features), dim=1
        )  # -> (Batch, 1+SeqLen, FeatureDim)

        # 调试打印：Transformer输入（每100个batch）
        if hasattr(self, 'debug_mode') and self.debug_mode and hasattr(self, 'batch_count') and self.batch_count % 100 == 0:
            print(f"[DEBUG] Batch {self.batch_count} - Transformer输入统计 - 均值: {transformer_input.mean():.4f}, 标准差: {transformer_input.std():.4f}")

        # 通过Transformer编码器
        transformer_output = self.transformer_encoder(
            transformer_input
        )  # -> (Batch, 1+SeqLen, FeatureDim)

        # 提取 [CLS] token对应的输出作为整个序列的表示
        cls_output = transformer_output[:, 0, :]  # -> (Batch, FeatureDim)

        # 调试打印：最终特征（每100个batch）
        if hasattr(self, 'debug_mode') and self.debug_mode and hasattr(self, 'batch_count') and self.batch_count % 100 == 0:
            print(f"[DEBUG] Batch {self.batch_count} - 编码器输出特征统计 - 均值: {cls_output.mean():.4f}, 标准差: {cls_output.std():.4f}")
            print(f"[DEBUG] Batch {self.batch_count} - 特征形状: {cls_output.shape}")

        return cls_output, hrrp_features, rotated_features

    def forward(self, hrrp_signal, angles):
        """完整的前向传播，返回投影后的特征"""
        # 获取编码器输出
        encoded_features, hrrp_features, rotated_features = self.forward_encoder(hrrp_signal, angles)

        # 通过投影头
        # projected_features = self.projection_head(encoded_features)

        # # 调试打印：投影后特征（每100个batch）
        # if hasattr(self, 'debug_mode') and self.debug_mode and hasattr(self, 'batch_count') and self.batch_count % 100 == 0:
        #     print(f"[DEBUG] Batch {self.batch_count} - 投影后特征统计 - 均值: {projected_features.mean():.4f}, 标准差: {projected_features.std():.4f}")
        #     print(f"[DEBUG] Batch {self.batch_count} - 投影特征形状: {projected_features.shape}")
        #     print(f"[DEBUG] Batch {self.batch_count} - 投影特征范数 - 均值: {torch.norm(projected_features, dim=1).mean():.4f}, 标准差: {torch.norm(projected_features, dim=1).std():.4f}")
        #     print("-" * 50)

        return encoded_features, hrrp_features, rotated_features


class SimplifiedHRRPModel(nn.Module):
    """简化版HRRP模型，去掉角度融合模块，用于测试角度融合是否引入噪声"""
    
    def __init__(
        self,
        hrrp_len,
        feature_dim,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # 1. HRRP特征提取器
        self.hrrp_extractor = HRRPExtractor(output_dim=feature_dim)

        # 2. Transformer编码器
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

    def forward(self, hrrp_signal, angles=None):
        """简化版前向传播，不使用角度信息"""
        # hrrp_signal: (Batch, SignalLength)
        # angles: 忽略，保持接口兼容性

        # 0. 预处理：为1D CNN增加通道维度 -> (Batch, 1, SignalLength)
        hrrp_signal = hrrp_signal.unsqueeze(1)

        # 1. HRRP特征提取
        hrrp_features = self.hrrp_extractor(hrrp_signal)  # -> (Batch, SeqLen, FeatureDim)

        # 2. Transformer编码
        # 准备输入给Transformer的序列
        batch_size = hrrp_features.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 将CLS token复制B份
        transformer_input = torch.cat(
            (cls_tokens, hrrp_features), dim=1
        )  # -> (Batch, 1+SeqLen, FeatureDim)

        # 通过Transformer编码器
        transformer_output = self.transformer_encoder(
            transformer_input
        )  # -> (Batch, 1+SeqLen, FeatureDim)

        # 提取 [CLS] token对应的输出作为整个序列的表示
        cls_output = transformer_output[:, 0, :]  # -> (Batch, FeatureDim)

        return cls_output