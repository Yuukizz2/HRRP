import torch
import numpy as np
import json
import random
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset

class HRRPDataset(Dataset):
    """HRRP数据集类"""
    def __init__(self, hrrp_data, class_data):
        self.hrrp_data = hrrp_data
        self.class_data = class_data
    
    def __len__(self):
        return len(self.hrrp_data)
    
    def __getitem__(self, idx):
        hrrp = self.hrrp_data[idx]
        class_name = self.class_data[idx]
        
        # 确保类别名称是字符串
        if isinstance(class_name, (list, np.ndarray)):
            class_name = class_name[0]
        class_name = str(class_name)
        
        return hrrp, class_name

class AdvancedDatasetSplitter:
    """高级数据集划分器"""
    
    def __init__(self, data_path="data/zeroshot/balanced_data.pt"):
        self.data_path = data_path
        self.data = None
        self.label_to_aircraft = {
            "A321": "A321", "A330": "A330", "ARJ21": "ARJ21", "BY737": "BY737",
            "BY787": "BY787", "KJ600": "KJ600", "Y20": "Y20", "Z8": "Z8"
        }
        self.load_data()
    
    def load_data(self):
        """加载平衡后的数据"""
        print(f"Loading balanced data from {self.data_path}")
        self.data = torch.load(self.data_path)
        print(f"Loaded {len(self.data['class'])} samples")
        
        # 显示类别分布
        class_data = self.data['class']
        unique_classes, counts = np.unique(class_data, return_counts=True)
        print("\nClass distribution:")
        for cls, count in zip(unique_classes, counts):
            print(f"  {cls}: {count} samples")
    
    def split_data(self, seen_classes, unseen_classes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        高级数据划分
        
        Args:
            seen_classes: seen类别列表
            unseen_classes: unseen类别列表
            train_ratio: seen类训练集比例
            val_ratio: seen类验证集比例
            test_ratio: seen类测试集比例
            random_seed: 随机种子
        
        Returns:
            dict: 包含train, val, test数据的字典
        """
        print(f"\n=== Advanced Data Splitting ===")
        print(f"Seen classes: {seen_classes}")
        print(f"Unseen classes: {unseen_classes}")
        print(f"Split ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
        
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")
        
        # 验证类别
        all_classes = list(self.label_to_aircraft.keys())
        for cls in seen_classes + unseen_classes:
            if cls not in all_classes:
                raise ValueError(f"Unknown class: {cls}")
        
        # 检查类别重叠
        overlap = set(seen_classes) & set(unseen_classes)
        if overlap:
            raise ValueError(f"Classes cannot appear in both seen and unseen: {overlap}")
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 1. 按seen/unseen划分数据
        class_data = self.data['class']
        seen_indices = []
        unseen_indices = []
        
        for i, class_name in enumerate(class_data):
            if class_name in seen_classes:
                seen_indices.append(i)
            elif class_name in unseen_classes:
                unseen_indices.append(i)
        
        print(f"Seen samples: {len(seen_indices)}")
        print(f"Unseen samples: {len(unseen_indices)}")
        
        # 2. 对seen类进行train/val/test划分
        seen_indices = np.array(seen_indices)
        np.random.shuffle(seen_indices)  # 随机打乱
        
        n_seen = len(seen_indices)
        n_train = int(n_seen * train_ratio)
        n_val = int(n_seen * val_ratio)
        n_test = n_seen - n_train - n_val  # 确保总数正确
        
        train_indices = seen_indices[:n_train]
        val_indices = seen_indices[n_train:n_train + n_val]
        seen_test_indices = seen_indices[n_train + n_val:]
        
        print(f"Seen class splits:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val: {len(val_indices)} samples")
        print(f"  Test: {len(seen_test_indices)} samples")
        
        # 3. 合并seen_test和unseen作为最终test集
        final_test_indices = np.concatenate([seen_test_indices, unseen_indices])
        print(f"Final test (seen_test + unseen): {len(final_test_indices)} samples")
        
        # 4. 提取数据
        splits = {}
        for split_name, indices in [
            ('train', train_indices),
            ('val', val_indices),
            ('test', final_test_indices)
        ]:
            splits[split_name] = {}
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    splits[split_name][key] = value[indices]
                elif isinstance(value, list):
                    # 处理列表类型的数据（如字符串类别列表）
                    splits[split_name][key] = [value[i] for i in indices]
                else:
                    splits[split_name][key] = value[indices]
        
        # 5. 显示每个划分的类别分布
        for split_name, split_data in splits.items():
            split_classes = split_data['class']
            unique_classes, counts = np.unique(split_classes, return_counts=True)
            print(f"\n{split_name.capitalize()} set class distribution:")
            for cls, count in zip(unique_classes, counts):
                print(f"  {cls}: {count} samples")
        
        return splits
    
    def get_split_info(self, splits):
        """获取划分信息"""
        info = {}
        for split_name, split_data in splits.items():
            info[split_name] = {
                'total_samples': len(split_data['class']),
                'hrrp_shape': split_data['hrrp'].shape,
                'classes': list(np.unique(split_data['class']))
            }
        return info
    
    def save_splits(self, splits, save_dir="data/splits", experiment_name="experiment_1"):
        """保存划分结果"""
        save_path = Path(save_dir) / experiment_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存每个划分
        for split_name, split_data in splits.items():
            torch.save(split_data, save_path / f"{split_name}_data.pt")
            print(f"Saved {split_name} data to {save_path / f'{split_name}_data.pt'}")
        
        # 保存划分信息
        split_info = self.get_split_info(splits)
        with open(save_path / "split_info.json", 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"Saved split info to {save_path / 'split_info.json'}")
        
        return save_path

    def load_splits(self, load_dir: str) -> dict:
        """从目录中加载已保存的划分数据。

        目录中应包含: train_data.pt, val_data.pt, test_data.pt
        """
        load_path = Path(load_dir)
        required_files = [
            load_path / "train_data.pt",
            load_path / "val_data.pt",
            load_path / "test_data.pt",
        ]
        for f in required_files:
            if not f.exists():
                raise FileNotFoundError(f"Missing split file: {f}")
        print(f"Loading splits from {load_path}")
        splits = {
            'train': torch.load(load_path / "train_data.pt"),
            'val': torch.load(load_path / "val_data.pt"),
            'test': torch.load(load_path / "test_data.pt"),
        }
        # 简要信息
        info = self.get_split_info(splits)
        for name, i in info.items():
            print(f"{name.capitalize()} loaded: total={i['total_samples']}, classes={i['classes']}")
        return splits

    def get_or_create_splits(
        self,
        seen_classes,
        unseen_classes,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        fixed_split_dir: str = "data/splits",
        experiment_name: str = "fixed_split",
    ) -> dict:
        """若存在固定划分则加载，否则创建并保存后返回。

        返回值与 split_data 一致: {'train': {...}, 'val': {...}, 'test': {...}}
        """
        target_dir = Path(fixed_split_dir) / experiment_name
        try:
            if target_dir.exists():
                print(f"Found existing splits at {target_dir}, loading...")
                return self.load_splits(str(target_dir))
        except Exception as e:
            print(f"Existing split invalid or incomplete: {e}. Recreating splits...")

        # 创建新划分
        splits = self.split_data(
            seen_classes=seen_classes,
            unseen_classes=unseen_classes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )
        # 保存
        self.save_splits(splits, save_dir=fixed_split_dir, experiment_name=experiment_name)
        return splits
    
    def create_dataloader_splits(self, splits, batch_size=32, shuffle_train=True, shuffle_val=False, shuffle_test=False):
        """创建PyTorch DataLoader"""
        from torch.utils.data import DataLoader
        
        dataloaders = {}
        
        for split_name, split_data in splits.items():
            # 创建自定义数据集
            hrrp_tensor = split_data['hrrp']
            class_labels = split_data['class']
            
            dataset = HRRPDataset(hrrp_tensor, class_labels)
            
            # 设置shuffle
            shuffle = False
            if split_name == 'train' and shuffle_train:
                shuffle = True
            elif split_name == 'val' and shuffle_val:
                shuffle = True
            elif split_name == 'test' and shuffle_test:
                shuffle = True
            
            dataloaders[split_name] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True
            )
        
        return dataloaders

def example_usage():
    """使用示例"""
    print("=== Advanced Dataset Splitter Example ===")
    
    # 初始化划分器
    splitter = AdvancedDatasetSplitter()
    
    # 定义seen和unseen类别
    seen_classes = ["A321", "A330", "ARJ21", "BY737", "BY787", "KJ600"]
    unseen_classes = ["Z8", "Y20"]
    
    # 进行划分
    splits = splitter.split_data(
        seen_classes=seen_classes,
        unseen_classes=unseen_classes,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # 显示划分信息
    split_info = splitter.get_split_info(splits)
    print(f"\n=== Split Information ===")
    for split_name, info in split_info.items():
        print(f"{split_name.capitalize()}:")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  HRRP shape: {info['hrrp_shape']}")
        print(f"  Classes: {info['classes']}")
    
    # 保存划分结果
    save_path = splitter.save_splits(splits, experiment_name="zeroshot_experiment_1")
    
    # 创建DataLoader
    dataloaders = splitter.create_dataloader_splits(splits, batch_size=32)
    
    print(f"\n=== DataLoader Information ===")
    for split_name, dataloader in dataloaders.items():
        print(f"{split_name.capitalize()} DataLoader:")
        print(f"  Number of batches: {len(dataloader)}")
        print(f"  Batch size: {dataloader.batch_size}")
        
        # 显示第一个batch的信息
        for batch_idx, (hrrp_batch, class_batch) in enumerate(dataloader):
            print(f"  First batch shape: {hrrp_batch.shape}")
            print(f"  First batch classes: {class_batch[:5].tolist()}")
            break
    
    return splits, dataloaders, save_path

def multiple_experiments():
    """多个实验示例"""
    print("\n=== Multiple Experiments ===")
    
    splitter = AdvancedDatasetSplitter()
    
    # 定义多个实验配置
    experiments = [
        {
            "name": "exp_1_z8_y20_unseen",
            "seen_classes": ["A321", "A330", "ARJ21", "BY737", "BY787", "KJ600"],
            "unseen_classes": ["Z8", "Y20"]
        },
        {
            "name": "exp_2_a321_by737_unseen",
            "seen_classes": ["A330", "ARJ21", "BY787", "KJ600", "Y20", "Z8"],
            "unseen_classes": ["A321", "BY737"]
        },
        {
            "name": "exp_3_kj600_arj21_unseen",
            "seen_classes": ["A321", "A330", "BY737", "BY787", "Y20", "Z8"],
            "unseen_classes": ["KJ600", "ARJ21"]
        }
    ]
    
    for exp in experiments:
        print(f"\n--- {exp['name']} ---")
        
        splits = splitter.split_data(
            seen_classes=exp['seen_classes'],
            unseen_classes=exp['unseen_classes'],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        save_path = splitter.save_splits(splits, experiment_name=exp['name'])
        print(f"Experiment saved to: {save_path}")

if __name__ == "__main__":
    # 运行单个实验示例
    splits, dataloaders, save_path = example_usage()
    
    # 运行多个实验示例
    multiple_experiments()
    
    print(f"\n=== Usage Instructions ===")
    print(f"1. Initialize: splitter = AdvancedDatasetSplitter()")
    print(f"2. Split data: splits = splitter.split_data(seen_classes, unseen_classes)")
    print(f"3. Create DataLoaders: dataloaders = splitter.create_dataloader_splits(splits)")
    print(f"4. Use in training:")
    print(f"   - train_dataloader = dataloaders['train']")
    print(f"   - val_dataloader = dataloaders['val']")
    print(f"   - test_dataloader = dataloaders['test']") 