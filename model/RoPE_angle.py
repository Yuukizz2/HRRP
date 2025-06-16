import torch
import torch.nn as nn
import math


class RotaryAngleEmbedding(nn.Module):
    """
    一个通用的RoPE模块，可以实现方法一和方法二的变体
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # 生成RoPE的频率 theta_i = base^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, angle_info):
        """
        Args:
            x (torch.Tensor): 输入特征, shape: [batch, ..., dim]
            angle_info (torch.Tensor):
                - 对于方法一, 是一个标量或[batch, 1]的角度值alpha
                - 对于方法二, 是一个整数位置索引m
        Returns:
            torch.Tensor: 旋转后的特征
        """
        # 计算旋转角度 a * theta_i
        # angle_info shape: [batch, ...] -> [batch, ..., 1]
        t = angle_info.float().unsqueeze(-1)
        # freqs shape: [batch, ..., dim/2]
        freqs = torch.einsum("...i,j->...ij", t, self.inv_freq.to(t.device)).squeeze(-1)
        emb = torch.cat((freqs, freqs), dim=-1)  # shape: [batch, ..., dim]

        # 分离出cos和sin
        cos_emb = emb.cos()
        sin_emb = emb.sin()

        # 應用旋轉
        # x[..., 0::2] and x[..., 1::2]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out1 = x1 * cos_emb[..., 0::2] - x2 * sin_emb[..., 0::2]
        out2 = x1 * sin_emb[..., 1::2] + x2 * cos_emb[..., 1::2]

        return torch.cat([out1, out2], dim=-1)


D_MODEL = 64

# === 方法一：全局角度旋转 ===
print("--- 方法一：全局角度旋转 ---")
# 1. 模拟HRRP特征提取
batch_size = 4
hrrp_feature_extractor = nn.Linear(256, D_MODEL)  # 假设HRRP长度256
mock_hrrp_data = torch.randn(batch_size, 256)
h_hrrp = hrrp_feature_extractor(mock_hrrp_data)  # shape: [4, 64]

# 2. 模拟角度信息 (弧度)
mock_angles = torch.tensor([0.1, 0.5, 1.2, 3.14])  # shape: [4]

# 3. 应用全局旋转
# 角度需要一个缩放因子k, 这里简单设为1
angle_k = 1.0
global_rotary_emb = RotaryAngleEmbedding(dim=D_MODEL)
h_rotated = global_rotary_emb(h_hrrp, mock_angles * angle_k)

print("原始HRRP特征 shape:", h_hrrp.shape)
print("全局角度旋转后特征 shape:", h_rotated.shape)


# === 方法二：条件化内部位置编码 ===
print("\n--- 方法二：条件化内部位置编码 (以频率缩放为例) ---")
# 1. 模拟HRRP序列 (batch, seq_len, dim)
seq_len = 128  # HRRP有128个距离单元
mock_hrrp_sequence = torch.randn(batch_size, seq_len, D_MODEL)

# 2. 模拟角度信息和位置索引
mock_angles = torch.tensor([0.1, 0.5, 1.2, 3.14]).view(batch_size, 1)  # shape: [4, 1]
position_ids = torch.arange(seq_len).view(1, seq_len)  # shape: [1, 128]

# 3. 用角度调节位置索引
# 这里g(alpha) = 0.1 * alpha
angle_scaling_factor = 1.0 + 0.1 * mock_angles  # shape: [4, 1]
conditional_position_ids = (
    position_ids * angle_scaling_factor
)  # shape: [4, 128] using broadcasting

# 4. 应用条件化RoPE
internal_rotary_emb = RotaryAngleEmbedding(dim=D_MODEL)
sequence_rotated = internal_rotary_emb(mock_hrrp_sequence, conditional_position_ids)

print("原始HRRP序列 shape:", mock_hrrp_sequence.shape)
print("条件化RoPE后序列 shape:", sequence_rotated.shape)
