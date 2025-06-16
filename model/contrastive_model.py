import torch
import torch.nn as nn
from hrrp_model import HrrpEmbedding, TransformerEncoder


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class HrrpContrastiveModel(nn.Module):
    def __init__(self, input_hrrp_dim, sequence_length, 
                 num_layers, d_model, num_heads, d_ff, dropout,
                 projection_hidden_dim, projection_output_dim):
        super().__init__()
        self.embedding = HrrpEmbedding(input_hrrp_dim, d_model)
        
        head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, 
                                          rope_dim=head_dim, dropout=dropout)
        
        # 编码器输出后，通常需要一个池化层来获得整个序列的固定大小表示
        # 这里我们简单使用平均池化
        self.pooler = lambda x: x.mean(dim=1) # (B, S, D) -> (B, D)

        self.projection_head = ProjectionHead(d_model, projection_hidden_dim, projection_output_dim)

    def forward_encoder(self, x):
        # x: (batch_size, sequence_length) HRRP
        x_embedded = self.embedding(x) # -> (batch_size, sequence_length, d_model)
        encoded_features = self.encoder(x_embedded) # -> (batch_size, sequence_length, d_model)
        pooled_features = self.pooler(encoded_features) # -> (batch_size, d_model)
        return pooled_features

    def forward(self, x):
        # 用于对比学习的完整前向传播
        encoded_pooled = self.forward_encoder(x)
        projected_features = self.projection_head(encoded_pooled) # -> (batch_size, projection_output_dim)
        return projected_features


class HrrpTransformerClassifier(nn.Module): # 这是你之前的分类器
    def __init__(self, input_hrrp_dim, sequence_length, num_classes,
                 num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = HrrpEmbedding(input_hrrp_dim, d_model)
        head_dim = d_model // num_heads
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff,
                                          rope_dim=head_dim, dropout=dropout)
        self.pooler = lambda x: x.mean(dim=1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x_embedded = self.embedding(x)
        encoded_features = self.encoder(x_embedded, mask)
        pooled_output = self.pooler(encoded_features)
        logits = self.fc(pooled_output)
        return logits