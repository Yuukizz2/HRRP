from typing import Tuple, Optional, Dict, Any, Union, List
import logging
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
from pathlib import Path


class HrrpDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Dict[str, Any]] = None,
        target_transform: Optional[Dict[str, Any]] = None,
        return_format: str = "tuple",  # "tuple" 或 "dict"
        balance_classes: bool = False,  # 是否平衡类别
        allowed_classes: Optional[List[int]] = None,
    ):
        """
        HRRP数据集类

        Args:
            data_path: 数据路径
            transform: 数据增强配置
            target_transform: 标签转换配置
            return_format: 返回格式，"tuple"返回(data, target)元组，"dict"返回字典
            balance_classes: 是否平衡类别分布
        """
        self.data_path = Path(data_path)
        self.return_format = return_format
        self.balance_classes = balance_classes
        self.allowed_classes = allowed_classes
        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 加载数据
        self.data = self._load_data()
        
        if self.allowed_classes is not None:
            self._filter_by_classes()
            
        # 如果启用类别平衡，计算采样权重
        if self.balance_classes:
            self.sample_weights = self._compute_sample_weights()

    def _load_data(self) -> Dict[str, torch.Tensor]:
        """加载数据"""
        try:
            # 加载.pt格式的数据
            data = torch.load(self.data_path / "hrrp_dataset.pt")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _filter_by_classes(self):
        """只保留属于 allowed_classes 的样本"""
        indices = [i for i, label in enumerate(self.data["class"])
                    if int(label) in self.allowed_classes]
        for key in self.data:
            self.data[key] = self.data[key][indices]


    def __len__(self) -> int:
        return len(self.data["hrrp"])

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        返回单个样本
        
        Returns:
            如果return_format="tuple": (hrrp_data, class_label)
            如果return_format="dict": {"hrrp": ..., "azimuth": ..., "elevation": ..., "class": ...}
        """
        hrrp_data = self.data["hrrp"][idx]
        class_label = self.data["class"][idx]
        
        # 确保class_label是正确的数据类型
        if isinstance(class_label, torch.Tensor):
            class_label = class_label.long()  # 确保是Long类型
        else:
            class_label = torch.tensor(class_label, dtype=torch.long)
                           
        if self.return_format == "tuple":
            return hrrp_data, class_label
        elif self.return_format == "dict":
            return {
                "azimuth": (
                    self.data["azimuth"][idx].item()
                    if hasattr(self.data["azimuth"][idx], "item")
                    else self.data["azimuth"][idx]
                ),
                "elevation": (
                    self.data["elevation"][idx].item()
                    if hasattr(self.data["elevation"][idx], "item")
                    else self.data["elevation"][idx]
                ),
                "distance": self.data["distance"][idx],
                "hrrp": hrrp_data,
                "class": class_label,
            }
        else:
            raise ValueError(f"不支持的返回格式: {self.return_format}")


    def _compute_sample_weights(self):
        """计算采样权重以平衡类别"""
        labels = []
        for i in range(len(self.data["class"])):
            label = self.data["class"][i]
            if isinstance(label, torch.Tensor):
                labels.append(label.item())
            else:
                labels.append(label)
        
        labels = np.array(labels)
        class_counts = np.bincount(labels)
        
        # 计算每个样本的权重（类别数量少的样本权重更高）
        sample_weights = []
        for label in labels:
            weight = 1.0 / class_counts[label]
            sample_weights.append(weight)
        
        return torch.tensor(sample_weights, dtype=torch.float)
    
    def get_sampler(self):
        """获取加权采样器"""
        if self.balance_classes:
            return WeightedRandomSampler(
                weights=self.sample_weights.tolist(),
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return None