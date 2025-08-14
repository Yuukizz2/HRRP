import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



class AttentionClassifier(nn.Module):
    def __init__(self, hrrp_dim=512, num_labels=8, label_aspects=11, label_embedding_dim=768, label_embeddings_tensor=None, trainable_labels=True):
        super(AttentionClassifier, self).__init__()
        self.hrrp_dim = hrrp_dim
        self.num_labels = num_labels
        self.label_aspects = label_aspects
        self.label_embedding_dim = label_embedding_dim


        # 加载或初始化标签Embedding矩阵
        if label_embeddings_tensor is None:
            # 如果未提供，则随机初始化
            self.label_embeddings = nn.Parameter(torch.randn(num_labels, label_aspects, label_embedding_dim))
        else:
            self.label_embeddings = nn.Parameter(label_embeddings_tensor, requires_grad=trainable_labels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=label_embedding_dim,
            nhead=8,
            dim_feedforward=label_embedding_dim*2,
            dropout=0.1,
            batch_first=True
        )
        self.label_interaction = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        
        self.B = nn.Parameter(torch.randn(1,label_aspects)*0.01)
        
        self.A = nn.Parameter(torch.randn(hrrp_dim, label_embedding_dim)*0.01)  
        

    def forward(self, x_hrrp):
        label_emb = self.label_embeddings
        # 送入transformer，先reshape为[8,11,768]，即batch=8, seq=11, dim=768
        label_emb_interacted = self.label_interaction(label_emb)
        
        weights = F.softmax(self.B, dim=1)
        
        label_project = (label_emb_interacted * weights.unsqueeze(-1)).sum(dim=1)
        hrrp_project = torch.matmul(x_hrrp, self.A)
        
        hrrp_project = F.normalize(hrrp_project, p=2, dim=-1)
        label_project = F.normalize(label_project, p=2, dim=-1)
        logits = torch.matmul(hrrp_project, label_project.T)

        return logits,label_project,hrrp_project