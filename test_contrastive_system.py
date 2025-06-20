#!/usr/bin/env python3
"""
测试对比学习系统的各个组件
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from model.contrastive_hrrp_model import ContrastiveHRRPModel, ClassificationHRRPModel
from model.tripletloss import CompositeTripletLoss


def test_contrastive_model():
    """测试对比学习模型"""
    print("=" * 50)
    print("测试对比学习模型")
    print("=" * 50)

    # 模型参数
    BATCH_SIZE = 4
    HRRP_LENGTH = 256
    FEATURE_DIM = 64
    PROJECTION_DIM = 128

    # 创建模型
    model = ContrastiveHRRPModel(
        hrrp_len=HRRP_LENGTH,
        feature_dim=FEATURE_DIM,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        projection_hidden_dim=128,
        projection_output_dim=PROJECTION_DIM,
    )

    # 创建测试数据
    hrrp_data = torch.randn(BATCH_SIZE, HRRP_LENGTH)
    angles_data = torch.tensor([[10.0, 20.0], [45.0, 30.0], [90.0, 5.0], [180.0, 15.0]])

    print(f"输入HRRP形状: {hrrp_data.shape}")
    print(f"输入角度形状: {angles_data.shape}")

    # 前向传播
    projected_features = model(hrrp_data, angles_data)
    print(f"投影特征形状: {projected_features.shape}")

    # 测试编码器
    encoded_features = model.forward_encoder(hrrp_data, angles_data)
    print(f"编码特征形状: {encoded_features.shape}")

    # 检查参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")

    print("✓ 对比学习模型测试通过\n")


def test_triplet_loss():
    """测试三元组损失函数"""
    print("=" * 50)
    print("测试三元组损失函数")
    print("=" * 50)

    # 创建损失函数
    criterion = CompositeTripletLoss(
        margin_class=1.0,
        margin_angular=0.2,
        lambda_angular=0.5,
    )

    # 创建测试数据
    BATCH_SIZE = 8
    FEATURE_DIM = 128

    # 模拟投影后的特征
    embeddings = torch.randn(BATCH_SIZE, FEATURE_DIM, requires_grad=True)

    # 创建标签（4个样本属于类别0，4个样本属于类别1）
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    # 创建角度信息（弧度制）
    angles = torch.rand(BATCH_SIZE) * 2 * torch.pi

    print(f"特征形状: {embeddings.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"角度形状: {angles.shape}")

    # 计算损失
    loss = criterion(embeddings, labels, angles)
    print(f"三元组损失值: {loss.item():.4f}")

    # 测试反向传播
    loss.backward()
    print(f"梯度计算成功，第一个特征梯度范数: {embeddings.grad[0].norm().item():.4f}")

    print("✓ 三元组损失函数测试通过\n")


def test_classification_model():
    """测试分类模型"""
    print("=" * 50)
    print("测试分类模型")
    print("=" * 50)

    # 创建对比学习模型
    contrastive_model = ContrastiveHRRPModel(
        hrrp_len=256,
        feature_dim=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        projection_hidden_dim=128,
        projection_output_dim=128,
    )

    # 创建分类模型
    num_classes = 5
    classification_model = ClassificationHRRPModel(
        contrastive_model=contrastive_model,
        num_classes=num_classes,
    )

    # 创建测试数据
    BATCH_SIZE = 4
    hrrp_data = torch.randn(BATCH_SIZE, 256)
    angles_data = torch.tensor([[10.0, 20.0], [45.0, 30.0], [90.0, 5.0], [180.0, 15.0]])

    print(f"输入HRRP形状: {hrrp_data.shape}")
    print(f"输入角度形状: {angles_data.shape}")

    # 前向传播
    logits = classification_model(hrrp_data, angles_data)
    print(f"分类logits形状: {logits.shape}")

    # 检查参数数量
    total_params = sum(
        p.numel() for p in classification_model.parameters() if p.requires_grad
    )
    print(f"分类模型可训练参数量: {total_params:,}")

    # 检查编码器是否被冻结
    encoder_params = sum(
        p.numel() for p in contrastive_model.parameters() if p.requires_grad
    )
    print(f"编码器可训练参数量: {encoder_params:,} (应该为0)")

    print("✓ 分类模型测试通过\n")


def test_training_components():
    """测试训练组件"""
    print("=" * 50)
    print("测试训练组件")
    print("=" * 50)

    # 创建模型和损失函数
    model = ContrastiveHRRPModel(
        hrrp_len=256,
        feature_dim=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        projection_hidden_dim=128,
        projection_output_dim=128,
    )

    criterion = CompositeTripletLoss(
        margin_class=1.0,
        margin_angular=0.2,
        lambda_angular=0.5,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 创建测试数据
    BATCH_SIZE = 4
    hrrp_data = torch.randn(BATCH_SIZE, 256)
    angles_data = torch.tensor([[10.0, 20.0], [45.0, 30.0], [90.0, 5.0], [180.0, 15.0]])
    labels = torch.tensor([0, 0, 1, 1])

    print("模拟一个训练步骤...")

    # 前向传播
    optimizer.zero_grad()
    projected_features = model(hrrp_data, angles_data)
    angles_rad = torch.deg2rad(angles_data[:, 0])
    loss = criterion(projected_features, labels, angles_rad)

    # 反向传播
    loss.backward()
    optimizer.step()

    print(f"训练损失: {loss.item():.4f}")
    print(
        f"梯度范数: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0):.4f}"
    )

    print("✓ 训练组件测试通过\n")


def test_data_consistency():
    """测试数据一致性"""
    print("=" * 50)
    print("测试数据一致性")
    print("=" * 50)

    # 测试角度编码的一致性
    model = ContrastiveHRRPModel(
        hrrp_len=256,
        feature_dim=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        projection_hidden_dim=128,
        projection_output_dim=128,
    )

    # 创建相同的角度数据
    angles1 = torch.tensor([[0.0, 0.0], [90.0, 0.0], [180.0, 0.0], [270.0, 0.0]])
    angles2 = torch.tensor([[360.0, 0.0], [450.0, 0.0], [540.0, 0.0], [630.0, 0.0]])

    hrrp_data = torch.randn(4, 256)

    # 编码角度
    encoded1 = model._encode_angles(angles1)
    encoded2 = model._encode_angles(angles2)

    print(f"角度1编码: {encoded1}")
    print(f"角度2编码: {encoded2}")

    # 检查周期性一致性
    diff = torch.abs(encoded1 - encoded2)
    print(f"编码差异: {diff}")

    if torch.allclose(encoded1, encoded2, atol=1e-6):
        print("✓ 角度编码周期性一致性测试通过")
    else:
        print("✗ 角度编码周期性一致性测试失败")

    print()


def main():
    """运行所有测试"""
    print("开始测试对比学习系统...\n")

    try:
        test_contrastive_model()
        test_triplet_loss()
        test_classification_model()
        test_training_components()
        test_data_consistency()

        print("=" * 50)
        print("🎉 所有测试通过！系统运行正常。")
        print("=" * 50)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
