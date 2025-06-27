import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    独立的分类头模块，用于HRRP数据的分类任务
    """

    def __init__(self, feature_dim: int, num_classes: int, dropout_rate: float = 0.3):
        """
        初始化分类头

        Args:
            feature_dim (int): 输入特征维度
            num_classes (int): 分类类别数
            dropout_rate (float): Dropout比率
        """
        super(ClassificationHead, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 分类网络结构
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.67),  # 第二层dropout稍小
            nn.Linear(512, num_classes),
        )

        # 使用标签平滑的交叉熵损失
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features (torch.Tensor): 输入特征，形状 (batch_size, feature_dim)

        Returns:
            torch.Tensor: 分类logits，形状 (batch_size, num_classes)
        """
        return self.classifier(features)

    def compute_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算分类损失

        Args:
            features (torch.Tensor): 输入特征
            labels (torch.Tensor): 真实标签

        Returns:
            torch.Tensor: 分类损失值
        """
        logits = self.forward(features)
        return self.criterion(logits, labels)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测类别

        Args:
            features (torch.Tensor): 输入特征

        Returns:
            torch.Tensor: 预测的类别索引
        """
        with torch.no_grad():
            logits = self.forward(features)
            return torch.argmax(logits, dim=1)

    def compute_accuracy(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算分类准确率

        Args:
            features (torch.Tensor): 输入特征
            labels (torch.Tensor): 真实标签

        Returns:
            float: 准确率
        """
        predictions = self.predict(features)
        accuracy = (predictions == labels).float().mean().item()
        return accuracy
