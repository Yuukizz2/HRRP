import torch
import torch.nn as nn
from model.classification_head import ClassificationHead
from model.contrastive_hrrp_model import ContrastiveHRRPModel


def test_classification_head():
    """测试分类头模块"""
    print("=== 测试分类头模块 ===")

    # 设置参数
    batch_size = 4
    feature_dim = 512
    num_classes = 8
    sequence_length = 512

    # 创建分类头
    classification_head = ClassificationHead(
        feature_dim=feature_dim, num_classes=num_classes, dropout_rate=0.3
    )

    # 创建模拟特征和标签
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")

    # 测试前向传播
    logits = classification_head(features)
    print(f"Logits形状: {logits.shape}")

    # 测试损失计算
    loss = classification_head.compute_loss(features, labels)
    print(f"分类损失: {loss.item():.4f}")

    # 测试预测
    predictions = classification_head.predict(features)
    print(f"预测形状: {predictions.shape}")
    print(f"预测结果: {predictions}")

    # 测试准确率计算
    accuracy = classification_head.compute_accuracy(features, labels)
    print(f"准确率: {accuracy:.4f}")

    print("分类头测试完成！\n")


def test_model_integration():
    """测试模型集成"""
    print("=== 测试模型集成 ===")

    # 设置参数
    batch_size = 2
    sequence_length = 512
    feature_dim = 512
    num_classes = 8

    # 创建模型
    model = ContrastiveHRRPModel(
        hrrp_len=sequence_length,
        feature_dim=feature_dim,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
        projection_hidden_dim=512,
        projection_output_dim=256,
    )

    # 创建分类头
    classification_head = ClassificationHead(
        feature_dim=feature_dim, num_classes=num_classes
    )

    # 创建模拟数据
    hrrp_data = torch.randn(batch_size, sequence_length)
    angles_data = torch.randn(batch_size, 2)  # azimuth, elevation
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"HRRP数据形状: {hrrp_data.shape}")
    print(f"角度数据形状: {angles_data.shape}")
    print(f"标签形状: {labels.shape}")

    # 测试模型前向传播
    projected_features = model(hrrp_data, angles_data)
    encoder_features = model.forward_encoder(hrrp_data, angles_data)

    print(f"投影特征形状: {projected_features.shape}")
    print(f"编码器特征形状: {encoder_features.shape}")

    # 测试分类头
    logits = classification_head(encoder_features)
    loss = classification_head.compute_loss(encoder_features, labels)
    predictions = classification_head.predict(encoder_features)
    accuracy = classification_head.compute_accuracy(encoder_features, labels)

    print(f"分类Logits形状: {logits.shape}")
    print(f"分类损失: {loss.item():.4f}")
    print(f"分类准确率: {accuracy:.4f}")

    print("模型集成测试完成！\n")


def test_training_workflow():
    """测试训练工作流程"""
    print("=== 测试训练工作流程 ===")

    # 设置参数
    batch_size = 2
    sequence_length = 512
    feature_dim = 512
    num_classes = 8

    # 创建模型和分类头
    model = ContrastiveHRRPModel(
        hrrp_len=sequence_length,
        feature_dim=feature_dim,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
        projection_hidden_dim=512,
        projection_output_dim=256,
    )

    classification_head = ClassificationHead(
        feature_dim=feature_dim, num_classes=num_classes
    )

    # 创建模拟数据
    hrrp_data = torch.randn(batch_size, sequence_length)
    angles_data = torch.randn(batch_size, 2)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 模拟第一阶段：对比学习训练
    print("第一阶段：对比学习训练")
    model.train()

    # 获取投影特征用于对比学习
    projected_features = model(hrrp_data, angles_data)
    print(f"投影特征形状: {projected_features.shape}")

    # 模拟第二阶段：分类头训练
    print("\n第二阶段：分类头训练")
    model.eval()  # 冻结主模型
    classification_head.train()

    # 获取编码器特征用于分类
    with torch.no_grad():
        encoder_features = model.forward_encoder(hrrp_data, angles_data)

    # 训练分类头
    optimizer = torch.optim.Adam(classification_head.parameters(), lr=1e-3)

    for step in range(5):  # 模拟几个训练步骤
        optimizer.zero_grad()

        loss = classification_head.compute_loss(encoder_features, labels)
        loss.backward()
        optimizer.step()

        accuracy = classification_head.compute_accuracy(encoder_features, labels)
        print(f"Step {step+1}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")

    print("训练工作流程测试完成！")


if __name__ == "__main__":
    # 运行所有测试
    test_classification_head()
    test_model_integration()
    test_training_workflow()

    print("所有测试完成！")
