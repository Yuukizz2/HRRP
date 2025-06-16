import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging


class HrrpDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Dict[str, Any]] = None,
        target_transform: Optional[Dict[str, Any]] = None,
    ):
        """
        HRRP数据集类

        Args:
            data_path: 数据路径
            transform: 数据增强配置
            target_transform: 标签转换配置
        """
        self.data_path = Path(data_path)
        self.transform = transform or {}
        self.target_transform = target_transform or {}

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 加载数据
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, torch.Tensor]:
        """加载数据"""
        try:
            # 加载.pt格式的数据
            data = torch.load(self.data_path / "hrrp_dataset.pt")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data["hrrp"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "azimuth": self.data["azimuth"][idx],
            "elevation": self.data["elevation"][idx],
            "distance": self.data["distance"][idx],
            "hrrp": self.data["hrrp"][idx],
            "class": self.data["class"][idx],
        }

        # 应用数据增强
        if self.transform.get("augment", False):
            item["hrrp"] = self._augment_data(item["hrrp"])

        return item

    def _augment_data(self, data: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        augmented_data = data.clone()

        # 添加高斯噪声
        if self.transform.get("add_noise", False):
            noise = torch.randn_like(data) * self.transform.get("noise_std", 0.01)
            augmented_data += noise

        # 随机缩放
        if self.transform.get("random_scale", False):
            scale = 1.0 + torch.randn(1) * self.transform.get("scale_std", 0.1)
            augmented_data *= scale

        return augmented_data


# # --- 示例数据 (请替换为你的真实数据) ---
# # 假设你有100个HRRP样本，每个样本长度为64
# num_samples = 100
# sequence_length = 64
# num_classes = 3  # 假设有3个类别

# # 生成一些随机的HRRP信号数据
# dummy_hrrp_signals = np.random.rand(num_samples, sequence_length).astype(np.float32)

# # 生成一些随机的标签 (类别索引从0到num_classes-1)
# dummy_labels = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)

# # --- 实例化 Dataset ---
# # 通常你会将数据分为训练集、验证集和测试集
# # 这里我们用所有虚拟数据创建一个训练集作为示例

# # 假设你有一个transform函数（可选）
# # def example_transform(sample):
# #     # 例如，对信号进行归一化等操作
# #     # sample['signal'] = (sample['signal'] - sample['signal'].mean()) / sample['signal'].std()
# #     # return sample
# #     # 或者更简单地，如果transform只作用于信号:
# #     # signal = sample['signal']
# #     # signal = (signal - signal.mean()) / signal.std()
# #     # return signal, sample['label']
# #     pass


# train_dataset = HrrpDataset(
#     data_path="path_to_your_data", transform=None, target_transform=None
# )  # 如果有transform，在这里传入

# # 检查一个样本
# single_signal, single_label = train_dataset[0]
# print(f"单个HRRP信号形状: {single_signal.shape}, 类型: {single_signal.dtype}")
# print(f"单个标签: {single_label}, 类型: {single_label.dtype}")
# print("-" * 30)

# # --- 2. 实例化 DataLoader ---
# batch_size = 16  # 设置你想要的批次大小

# train_dataloader = torch.DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True,  # 在每个epoch开始时打乱数据，对训练集通常为True
#     num_workers=0,  # 使用多少子进程加载数据。0表示在主进程中加载。
#     # 在Windows上设为0通常更稳定，Linux/macOS可以尝试>0
#     pin_memory=(
#         True if torch.cuda.is_available() else False
#     ),  # 如果使用GPU，可以加快数据传输
# )

# # --- 使用 DataLoader ---
# # 迭代 DataLoader 获取数据批次
# # 通常在训练循环中使用
# print("从DataLoader中取出一个批次的数据:")
# for i, batch in enumerate(train_dataloader):
#     hrrp_batch, labels_batch = batch

#     print(f"批次 {i+1}:")
#     print(f"  HRRP信号批次形状: {hrrp_batch.shape}")  # (batch_size, sequence_length)
#     print(f"  标签批次形状: {labels_batch.shape}")  # (batch_size,)
#     print(f"  HRRP信号批次类型: {hrrp_batch.dtype}")
#     print(f"  标签批次类型: {labels_batch.dtype}")

#     # 在这里，你可以将 hrrp_batch 和 labels_batch 输入到你的模型中进行训练
#     # model(hrrp_batch) ...

#     if i == 0:  # 只演示第一个批次
#         break

# print("-" * 30)
# print("DataLoader 构建完成并可供使用。")
