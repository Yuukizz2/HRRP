import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self, dim, max_seq_len=2048, base=10000, device=None
    ):  # 增加device参数
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("dim must be even for RoPE")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = (
            device if device is not None else torch.device("cpu")
        )  # 存储device

        # 计算旋转频率
        # inv_freq: (dim / 2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=self.device).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算位置编码
        # t: (max_seq_len)
        t = torch.arange(max_seq_len, device=self.device, dtype=self.inv_freq.dtype)
        # freqs: (max_seq_len, dim / 2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # emb: (max_seq_len, dim) # [cos(m*theta_0), sin(m*theta_0), cos(m*theta_1), sin(m*theta_1), ...] is one way to think,
        # but this cat creates [theta_0, theta_1, ..., theta_0, theta_1, ...] structure for freqs
        # so emb_cos/sin will be [cos(m*theta_0), cos(m*theta_1), ..., cos(m*theta_0), cos(m*theta_1), ...]
        emb = torch.cat(
            (freqs, freqs), dim=-1
        )  # 这使得cos_cached和sin_cached的结构适用于下面的 _rotate_half 方法
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        """
        将输入张量的最后一个维度分成两半，交换它们，并对前半部分取反。
        x_half1 = x[..., :self.dim // 2]
        x_half2 = x[..., self.dim // 2:]
        return torch.cat((-x_half2, x_half1), dim=-1)
        """
        # 更鲁棒的实现，处理任意数量的前导维度
        # x: (..., dim)
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # (..., dim/2, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]  # (..., dim/2)
        # 返回 (-x2, x1) 展平后的形式
        return torch.cat((-x2, x1), dim=-1).reshape(*x.shape)

    def _apply_rotary_emb(self, x, cos_values, sin_values):
        # x: (batch_size, num_heads_optional, seq_len, dim)
        # cos_values, sin_values: (seq_len, dim)

        # 调整cos和sin的形状以进行广播
        # (seq_len, dim) -> (1, 1, seq_len, dim) if x is (B,H,S,D)
        # (seq_len, dim) -> (1, seq_len, dim) if x is (B,S,D)
        # 我们需要根据 x 的维度来 unsqueeze
        cos = cos_values.unsqueeze(0)  # (1, seq_len, dim)
        sin = sin_values.unsqueeze(0)  # (1, seq_len, dim)
        if x.ndim == 4:  # (B, H, S, D)
            cos = cos.unsqueeze(0)  # (1, 1, seq_len, dim)
            sin = sin.unsqueeze(0)  # (1, 1, seq_len, dim)

        # 核心旋转公式: x_rotated = x * cos + _rotate_half(x) * sin
        return x * cos + self._rotate_half(x) * sin

    def forward(self, q, k):
        # q, k 的典型形状: (batch_size, num_heads, seq_len, head_dim)
        # 或者 (batch_size, seq_len, dim) 如果不显式处理头
        # RoPE作用于最后一个维度 (head_dim 或 dim)
        # 我们假设 q.shape[-2] 是序列长度维度
        seq_len = q.shape[
            -2
        ]  # 通常是倒数第二个维度，例如对于 (B, H, S, D) 或 (B, S, D)

        if seq_len > self.max_seq_len:
            # 如果需要处理更长的序列，可以考虑动态生成或报错
            # 动态生成示例 (会慢一些，但更灵活):
            # t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # emb = torch.cat((freqs, freqs), dim=-1)
            # cos = emb.cos()
            # sin = emb.sin()
            # print(f"Warning: seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}. Dynamically creating RoPE.")
            raise ValueError(
                f"seq_len {seq_len} exceeds RoPE's precomputed max_seq_len {self.max_seq_len}"
            )

        # 获取对应序列长度的cos和sin
        # cos_cached/sin_cached 形状 (max_seq_len, dim)
        cos_seq = self.cos_cached[:seq_len, :]  # (seq_len, dim)
        sin_seq = self.sin_cached[:seq_len, :]  # (seq_len, dim)

        # 应用旋转
        q_rotated = self._apply_rotary_emb(q, cos_seq, sin_seq)
        k_rotated = self._apply_rotary_emb(k, cos_seq, sin_seq)

        return q_rotated, k_rotated


if __name__ == "__main__":
    # --- 示例 1: 类似LLM中的用法 (有num_heads维度) ---
    print("--- 示例 1: (B, H, S, D) ---")
    batch_size = 2
    num_heads = 4
    seq_len_example = 50
    head_dim = 8  # RoPE的dim参数

    # 确保 RoPE 模块在正确的设备上
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rope_module_mha = RoPE(dim=head_dim, max_seq_len=100, device=current_device)

    # q_mha, k_mha 的形状: (batch_size, num_heads, seq_len, head_dim)
    q_mha = torch.randn(
        batch_size, num_heads, seq_len_example, head_dim, device=current_device
    )
    k_mha = torch.randn(
        batch_size, num_heads, seq_len_example, head_dim, device=current_device
    )

    q_rotated_mha, k_rotated_mha = rope_module_mha(q_mha, k_mha)

    print("原始 Q (MHA) shape:", q_mha.shape)
    print("旋转后 Q (MHA) shape:", q_rotated_mha.shape)
    print("原始 K (MHA) shape:", k_mha.shape)
    print("旋转后 K (MHA) shape:", k_rotated_mha.shape)
    assert q_rotated_mha.shape == q_mha.shape
    print("第一个样本的第一个头的第一个token的原始Q:", q_mha[0, 0, 0, :4])
    print("第一个样本的第一个头的第一个token的旋转Q:", q_rotated_mha[0, 0, 0, :4])
    print("-" * 30)

    # --- 示例 2: 类似你原始代码的用法 (无num_heads维度) ---
    print("--- 示例 2: (B, S, D) ---")
    dim_example = 8  # RoPE的dim参数
    rope_module_simple = RoPE(dim=dim_example, max_seq_len=100, device=current_device)

    # q, k 的形状: (batch_size, seq_len, dim)
    q_simple = torch.randn(
        batch_size, seq_len_example, dim_example, device=current_device
    )
    k_simple = torch.randn(
        batch_size, seq_len_example, dim_example, device=current_device
    )

    q_rotated_simple, k_rotated_simple = rope_module_simple(q_simple, k_simple)

    print("原始 Q (Simple) shape:", q_simple.shape)
    print("旋转后 Q (Simple) shape:", q_rotated_simple.shape)
    print("原始 K (Simple) shape:", k_simple.shape)
    print("旋转后 K (Simple) shape:", k_rotated_simple.shape)
    assert q_rotated_simple.shape == q_simple.shape
    print("第一个样本的第一个token的原始Q:", q_simple[0, 0, :4])
    print("第一个样本的第一个token的旋转Q:", q_rotated_simple[0, 0, :4])

    # 测试超过max_seq_len的情况
    try:
        q_too_long = torch.randn(2, 150, dim_example, device=current_device)
        rope_module_simple(q_too_long, q_too_long)
    except ValueError as e:
        print(f"\n捕获到错误 (预期之中): {e}")
