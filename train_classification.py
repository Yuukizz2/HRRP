import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.contrastive_hrrp_model import ContrastiveHRRPModel
from model.classification_head import ClassificationHead
from dataset.hrrp_dataset import HrrpDataset
import datetime
import os


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "classification_training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_pretrained_model(model_path: str, model_config: dict, device: torch.device):
    """
    加载预训练的对比学习模型

    Args:
        model_path: 模型权重文件路径
        model_config: 模型配置参数
        device: 设备

    Returns:
        加载了权重的模型
    """
    logger = logging.getLogger(__name__)

    # 创建模型
    model = ContrastiveHRRPModel(
        hrrp_len=model_config["sequence_length"],
        feature_dim=model_config["d_model"],
        nhead=model_config["num_heads"],
        num_encoder_layers=model_config["num_layers"],
        dim_feedforward=model_config["d_ff"],
        projection_hidden_dim=512,
        projection_output_dim=256,
    )

    # 加载权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"成功加载预训练模型: {model_path}")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"成功加载模型权重: {model_path}")
    else:
        logger.warning(f"模型文件不存在: {model_path}，将使用随机初始化的模型")

    model = model.to(device)
    return model


def train_classification_head(
    model, classification_head, train_loader, val_loader, device, args
):
    """
    训练分类头

    Args:
        model: 预训练的对比学习模型
        classification_head: 分类头
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        args: 训练参数

    Returns:
        tuple: (最佳验证准确率, 训练历史)
    """
    logger = logging.getLogger(__name__)

    # 冻结主模型参数
    for param in model.parameters():
        param.requires_grad = False

    # 只训练分类头
    optimizer = torch.optim.AdamW(
        classification_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        verbose=True,
        min_lr=1e-6,
    )

    best_val_acc = 0.0
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    patience_counter = 0

    def normalize_hrrp(hrrp_data):
        mean = hrrp_data.mean(dim=-1, keepdim=True)
        std = hrrp_data.std(dim=-1, keepdim=True) + 1e-8
        return (hrrp_data - mean) / std

    for epoch in range(args.num_epochs):
        # 训练阶段
        model.eval()
        classification_head.train()
        train_correct = 0
        train_total = 0
        epoch_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            hrrp_data = batch["hrrp"].to(device)
            angles_data = torch.stack([batch["azimuth"], batch["elevation"]], dim=1).to(
                device
            )
            labels = batch["class"].to(device)

            hrrp_data = normalize_hrrp(hrrp_data)

            optimizer.zero_grad()

            # 获取特征
            with torch.no_grad():
                features = model.forward_encoder(hrrp_data, angles_data)

            # 计算分类损失
            loss = classification_head.compute_loss(features, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                classification_head.parameters(), max_norm=1.0
            )

            optimizer.step()

            # 计算准确率
            predictions = classification_head.predict(features)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            epoch_train_loss += loss.item()

            # 每100个batch打印一次进度
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        train_acc = train_correct / train_total
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accs.append(train_acc)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        classification_head.eval()
        val_correct = 0
        val_total = 0
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                hrrp_data = batch["hrrp"].to(device)
                angles_data = torch.stack(
                    [batch["azimuth"], batch["elevation"]], dim=1
                ).to(device)
                labels = batch["class"].to(device)

                hrrp_data = normalize_hrrp(hrrp_data)

                features = model.forward_encoder(hrrp_data, angles_data)
                predictions = classification_head.predict(features)
                loss = classification_head.compute_loss(features, labels)

                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                epoch_val_loss += loss.item()

        val_acc = val_correct / val_total
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accs.append(val_acc)
        val_losses.append(avg_val_loss)

        # 学习率调度
        scheduler.step(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                classification_head.state_dict(),
                Path(args.save_dir) / "best_classification_head.pth",
            )
            logger.info(f"保存最佳分类头模型，验证准确率: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"早停触发，在epoch {epoch+1}停止训练")
            break

    return best_val_acc, (train_accs, val_accs, train_losses, val_losses)


def plot_training_curves(train_accs, val_accs, train_losses, val_losses, save_dir):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 准确率曲线
    ax1.plot(train_accs, label="Train Accuracy", color="blue")
    ax1.plot(val_accs, label="Val Accuracy", color="red")
    ax1.set_title("Classification Head Training Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # 损失曲线
    ax2.plot(train_losses, label="Train Loss", color="blue")
    ax2.plot(val_losses, label="Val Loss", color="red")
    ax2.set_title("Classification Head Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / "classification_training_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(args):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置日志
    logger = setup_logging(Path(args.log_dir))

    # 模型配置
    model_config = {
        "sequence_length": args.sequence_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
    }

    logger.info(f"分类头训练参数: {vars(args)}")

    # 加载预训练模型
    logger.info("加载预训练模型...")
    model = load_pretrained_model(args.model_path, model_config, device)

    # 加载数据
    logger.info("加载数据集...")
    train_dataset = HrrpDataset(
        data_path=args.train_data_path, return_format="dict", balance_classes=True
    )
    val_dataset = HrrpDataset(
        data_path=args.val_data_path, return_format="dict", balance_classes=False
    )

    # 创建数据加载器
    train_sampler = train_dataset.get_sampler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")

    # 创建分类头
    logger.info("创建分类头...")
    classification_head = ClassificationHead(
        feature_dim=args.d_model,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
    ).to(device)

    # 训练分类头
    logger.info("开始训练分类头...")
    best_classification_acc, (train_accs, val_accs, train_losses, val_losses) = (
        train_classification_head(
            model=model,
            classification_head=classification_head,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=args,
        )
    )

    logger.info(f"分类头训练完成！最佳验证准确率: {best_classification_acc:.4f}")

    # 绘制训练曲线
    plot_training_curves(train_accs, val_accs, train_losses, val_losses, args.save_dir)

    # 保存最终模型
    final_checkpoint = {
        "classification_head_state_dict": classification_head.state_dict(),
        "best_classification_acc": best_classification_acc,
        "classification_train_accs": train_accs,
        "classification_val_accs": val_accs,
        "classification_train_losses": train_losses,
        "classification_val_losses": val_losses,
        "model_config": model_config,
        "training_args": vars(args),
    }
    torch.save(final_checkpoint, Path(args.save_dir) / "final_classification_model.pth")

    logger.info(f"最终模型已保存到: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分类头训练")

    # 模型路径
    parser.add_argument(
        "--model_path", type=str, required=True, help="预训练模型权重文件路径"
    )

    # 数据参数
    parser.add_argument(
        "--train_data_path", type=str, default="data/train", help="训练数据路径"
    )
    parser.add_argument(
        "--val_data_path", type=str, default="data/val", help="验证数据路径"
    )
    parser.add_argument("--sequence_length", type=int, default=512, help="序列长度")
    parser.add_argument("--num_classes", type=int, default=8, help="类别数量")

    # 模型参数
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--num_layers", type=int, default=3, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=1024, help="前馈网络维度")

    # 分类头参数
    parser.add_argument(
        "--dropout_rate", type=float, default=0.3, help="分类头dropout比率"
    )

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=15, help="早停耐心值"
    )
    parser.add_argument("--lr_patience", type=int, default=5, help="学习率调度耐心值")

    # 其他参数
    parser.add_argument(
        "--num_workers", type=int, default=8, help="数据加载器工作进程数"
    )
    args = parser.parse_args()

    # 创建输出目录
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = os.path.join("logs", f"classification_{now}")
    save_root = os.path.join("ckpt", f"classification_{now}")
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)
    args.log_dir = log_root
    args.save_dir = save_root

    main(args)
