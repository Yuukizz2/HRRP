import torch
import numpy as np
from torch.utils.data import DataLoader
from model.hrrp_model import HRRPModel
from dataset.hrrp_dataset import HrrpDataset
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_model(ckpt_path, device, num_classes, d_model, num_layers, d_ff, nhead):
    # 创建模型
    model = HRRPModel(
        hrrp_len=512,  # 根据你的训练参数调整
        feature_dim=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=d_ff,
        # projection_hidden_dim=512,
        # projection_output_dim=256,
    )
    # 创建分类头
    cls_classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(d_model, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, num_classes)
    )
    # 加载权重
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    cls_classifier.load_state_dict(checkpoint["cls_classifier_state_dict"])
    model.to(device)
    cls_classifier.to(device)
    model.eval()
    cls_classifier.eval()
    return model, cls_classifier

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载测试集
    test_dataset = HrrpDataset(
        data_path=args.test_data_path,
        return_format="dict",
        balance_classes=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # 加载模型
    model, cls_classifier = load_model(
        args.ckpt_path, device, args.num_classes, args.d_model, args.num_layers, args.d_ff, args.nhead
    )
    all_preds = []
    all_labels = []
    # 推理
    with torch.no_grad():
        for batch in test_loader:
            hrrp_data = batch["hrrp"].to(device)
            angles_data = torch.stack([batch["azimuth"], batch["elevation"]], dim=1).to(device)
            labels = batch["class"].to(device)
            # 前向
            cls_features = model.forward_encoder(hrrp_data, angles_data)
            logits = cls_classifier(cls_features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f"测试集准确率: {acc:.4f}")
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵：")
    print(cm)
    # 分类报告
    print("分类报告：")
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比学习模型测试")
    parser.add_argument("--test_data_path", type=str, default="data/test", help="测试数据路径")
    parser.add_argument("--ckpt_path", type=str, default=r"ckpt\20250704_190838\best_model.pth", help="模型权重路径")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载进程数")
    parser.add_argument("--num_classes", type=int, default=8, help="类别数")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--num_layers", type=int, default=3, help="Transformer层数")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=1024, help="前馈网络维度")
    args = parser.parse_args()
    main(args)