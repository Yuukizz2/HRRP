import torch
import torch.nn as nn
from .RoPE import RoPE


class HrrpEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        # 每个HRRP点的幅度值是1维的，所以input_dim通常是1
        # 但如果你的HRRP点本身就有多个特征，则input_dim会不同
        # 这里我们假设HRRP是一个序列，每个点是一个标量值，
        # 我们需要先将其reshape为 (batch_size, sequence_length, 1)
        # 然后用一个线性层将其映射到 d_model
        self.projection = nn.Linear(input_dim, d_model)  # input_dim通常是1

    def forward(self, x):
        # x: (batch_size, sequence_length)
        x = x.unsqueeze(-1)  # -> (batch_size, sequence_length, 1)
        return self.projection(x)  # -> (batch_size, sequence_length, d_model)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope_dim, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE 模块实例化时，dim应该是每个头的维度 self.head_dim
        # 但如果RoPE的实现是期望在分割头之前作用于整个 d_model，则需要调整。
        # 通常RoPE是在分割到每个头之后，对每个头的Q_head, K_head进行旋转。
        # 因此，rope_dim 应该等于 self.head_dim。
        self.rope = RoPE(dim=rope_dim)  # 通常 rope_dim == self.head_dim

    def forward(self, x, mask=None):
        # x: (batch_size, sequence_length, d_model)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head: (batch_size, seq_len, num_heads, head_dim)
        # Then transpose to: (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Scaled Dot-Product Attention
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v)

        # Concatenate heads and project
        # (batch_size, seq_len, num_heads, head_dim)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        output = self.out_proj(context)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope_dim, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, num_heads, rope_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Or ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention block
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward block
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, input_dim, d_model, num_heads, d_ff, num_layers, rope_dim, dropout=0.1
    ):
        super().__init__()
        self.embedding = HrrpEmbedding(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, rope_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class HrrpTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_hrrp_dim,
        sequence_length,
        num_classes,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = HrrpEmbedding(
            input_hrrp_dim, d_model
        )  # input_hrrp_dim often 1

        # RoPE dimension should be head_dim
        head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.encoder = TransformerEncoder(
            input_dim=input_hrrp_dim,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            rope_dim=head_dim,  # 修改这里，使用head_dim作为RoPE维度
            dropout=dropout,
        )

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
        self.sequence_length = sequence_length

    def forward(self, x, mask=None):  # x: (batch_size, sequence_length) HRRP
        # x_embedded: (batch_size, sequence_length, d_model)
        x_embedded = self.embedding(x)

        # If using CLS token (example):
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x_embedded = torch.cat((cls_tokens, x_embedded), dim=1)
        # if mask is not None:
        #     cls_mask = torch.ones(x.shape[0], 1, device=mask.device, dtype=mask.dtype)
        #     mask = torch.cat((cls_mask, mask), dim=1)

        # encoded_features: (batch_size, sequence_length, d_model)
        encoded_features = self.encoder(x_embedded, mask)

        # Classification
        # Option 1: If using [CLS] token (assuming it's at index 0)
        # pooled_output = encoded_features[:, 0]

        # Option 2: Mean pooling
        pooled_output = encoded_features.mean(dim=1)  # (batch_size, d_model)

        logits = self.fc(pooled_output)  # (batch_size, num_classes)
        return logits
