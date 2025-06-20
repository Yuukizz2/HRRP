import torch
import os
import numpy as np

data = torch.load("data/processed_data/hrrp_dataset.pt")

# 1. 统计所有类别
class_strs = data["class"]
if isinstance(class_strs, torch.Tensor):
    class_strs = class_strs.numpy()
if isinstance(class_strs[0], bytes):
    class_strs = [s.decode() for s in class_strs]
class_strs = [str(s) for s in class_strs]
all_labels = sorted(set(class_strs))
label2id = {name: idx for idx, name in enumerate(all_labels)}
print("类别映射：", label2id)

# 2. 转换为数字
class_ids = [label2id[s] for s in class_strs]
data["class"] = torch.tensor(class_ids, dtype=torch.long)

# 3. 随机划分
N = data["hrrp"].shape[0]
indices = torch.randperm(N)
n_train = int(N * 0.8)
n_val = int(N * 0.1)
n_test = N - n_train - n_val

train_idx = indices[:n_train]
val_idx = indices[n_train : n_train + n_val]
test_idx = indices[n_train + n_val :]


def subset(data, idx):
    return {k: v[idx] for k, v in data.items()}


os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

torch.save(subset(data, train_idx), "data/train/hrrp_dataset.pt")
torch.save(subset(data, val_idx), "data/val/hrrp_dataset.pt")
torch.save(subset(data, test_idx), "data/test/hrrp_dataset.pt")

print(f"划分完成！训练集: {n_train}，验证集: {n_val}，测试集: {n_test}")
