import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.hrrp_model import HRRPModel
from dataset.hrrp_dataset import HrrpDataset
from utils.trainer import Trainer
import datetime
import os


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def main(args):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置日志
    logger = setup_logging(Path(args.log_dir))

    key_params = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
    }
    logger.info(f"训练参数: {key_params}")

    # 加载数据
    logger.info("加载数据集...")
    train_dataset = HrrpDataset(
        data_path=args.train_data_path,
        return_format="dict",
        balance_classes=True,  # 启用类别平衡
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

    # 创建模型
    logger.info("创建模型...")
    model = HRRPModel(
        hrrp_len=args.sequence_length,
        feature_dim=args.d_model,
        nhead=args.num_heads,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.d_ff,
        # projection_hidden_dim=512,
        # projection_output_dim=256,
    )

    # 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # 创建优化器（使用更小的学习率）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        num_classes=args.num_classes,
        early_stopping_patience=args.early_stopping_patience,
    )

    # 开始对比学习训练
    logger.info("开始模型训练...")
    trainer.train(args.num_epochs)

    logger.info("训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练")

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

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=15, help="早停耐心值"
    )

    # 其他参数
    parser.add_argument(
        "--num_workers", type=int, default=8, help="数据加载器工作进程数"
    )
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = os.path.join("logs", f"{now}")
    save_root = os.path.join("ckpt", f"{now}")
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)
    args.log_dir = log_root
    args.save_dir = save_root

    main(args)
