#!/usr/bin/env python3
"""
HRRP训练流水线脚本

这个脚本展示了如何使用两个独立的训练脚本：
1. train_contrastive.py - 训练对比学习模型
2. train_classification.py - 训练分类头

使用方式：
python run_training_pipeline.py --stage all  # 运行完整流水线
python run_training_pipeline.py --stage contrastive  # 只训练对比学习
python run_training_pipeline.py --stage classification --model_path path/to/model.pth  # 只训练分类头
"""

import subprocess
import argparse
import os
import sys
from pathlib import Path
import datetime


def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n{'='*50}")
    print(f"开始: {description}")
    print(f"命令: {' '.join(command)}")
    print(f"{'='*50}\n")

    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"\n✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 失败")
        print(f"错误代码: {e.returncode}")
        return False


def find_latest_model(checkpoint_dir):
    """找到最新的模型文件"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # 查找最新的best_model.pth文件
    model_files = list(checkpoint_path.glob("**/best_model.pth"))
    if not model_files:
        return None

    # 按修改时间排序，返回最新的
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    return str(latest_model)


def main():
    parser = argparse.ArgumentParser(description="HRRP训练流水线")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["contrastive", "classification", "all"],
        default="all",
        help="训练阶段",
    )
    parser.add_argument(
        "--model_path", type=str, help="预训练模型路径（用于分类头训练）"
    )
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
    parser.add_argument("--sequence_length", type=int, default=512, help="序列长度")
    parser.add_argument("--num_classes", type=int, default=8, help="类别数量")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--num_layers", type=int, default=3, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=1024, help="前馈网络维度")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument(
        "--contrastive_epochs", type=int, default=50, help="对比学习训练轮数"
    )
    parser.add_argument(
        "--classification_epochs", type=int, default=30, help="分类头训练轮数"
    )
    parser.add_argument(
        "--contrastive_lr", type=float, default=5e-4, help="对比学习学习率"
    )
    parser.add_argument(
        "--classification_lr", type=float, default=1e-3, help="分类头学习率"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="数据加载器工作进程数"
    )

    args = parser.parse_args()

    # 检查数据路径
    train_data_path = os.path.join(args.data_root, "train")
    val_data_path = os.path.join(args.data_root, "val")

    if not os.path.exists(train_data_path):
        print(f"❌ 训练数据路径不存在: {train_data_path}")
        sys.exit(1)
    if not os.path.exists(val_data_path):
        print(f"❌ 验证数据路径不存在: {val_data_path}")
        sys.exit(1)

    # 创建时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.stage in ["contrastive", "all"]:
        print("🚀 开始对比学习训练阶段")

        # 构建对比学习训练命令
        contrastive_cmd = [
            sys.executable,
            "train_contrastive.py",
            "--train_data_path",
            train_data_path,
            "--val_data_path",
            val_data_path,
            "--sequence_length",
            str(args.sequence_length),
            "--num_classes",
            str(args.num_classes),
            "--d_model",
            str(args.d_model),
            "--num_layers",
            str(args.num_layers),
            "--num_heads",
            str(args.num_heads),
            "--d_ff",
            str(args.d_ff),
            "--batch_size",
            str(args.batch_size),
            "--learning_rate",
            str(args.contrastive_lr),
            "--num_epochs",
            str(args.contrastive_epochs),
            "--num_workers",
            str(args.num_workers),
        ]

        success = run_command(contrastive_cmd, "对比学习模型训练")
        if not success:
            print("❌ 对比学习训练失败，停止流水线")
            sys.exit(1)

    if args.stage in ["classification", "all"]:
        print("🚀 开始分类头训练阶段")

        # 确定模型路径
        if args.model_path:
            model_path = args.model_path
        elif args.stage == "all":
            # 自动查找最新的模型
            model_path = find_latest_model("ckpt")
            if not model_path:
                print("❌ 未找到预训练模型，请先运行对比学习训练或指定模型路径")
                sys.exit(1)
        else:
            print("❌ 分类头训练需要指定预训练模型路径")
            sys.exit(1)

        print(f"📁 使用预训练模型: {model_path}")

        # 构建分类头训练命令
        classification_cmd = [
            sys.executable,
            "train_classification.py",
            "--model_path",
            model_path,
            "--train_data_path",
            train_data_path,
            "--val_data_path",
            val_data_path,
            "--sequence_length",
            str(args.sequence_length),
            "--num_classes",
            str(args.num_classes),
            "--d_model",
            str(args.d_model),
            "--num_layers",
            str(args.num_layers),
            "--num_heads",
            str(args.num_heads),
            "--d_ff",
            str(args.d_ff),
            "--batch_size",
            str(args.batch_size),
            "--learning_rate",
            str(args.classification_lr),
            "--num_epochs",
            str(args.classification_epochs),
            "--num_workers",
            str(args.num_workers),
        ]

        success = run_command(classification_cmd, "分类头训练")
        if not success:
            print("❌ 分类头训练失败")
            sys.exit(1)

    print("\n🎉 训练流水线完成！")
    print("\n📁 输出文件位置:")
    print("  - 对比学习模型: ckpt/contrastive_*/")
    print("  - 分类头模型: ckpt/classification_*/")
    print("  - 训练日志: logs/contrastive_*/ 和 logs/classification_*/")


if __name__ == "__main__":
    main()
