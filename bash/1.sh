#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# --- 设置参数 ---
# 数据参数
# INPUT_HRRP_DIM=1 # 通常固定为1，可以在train.py中保留默认
SEQ_LEN=64
NUM_CLASSES=3
DATA_PATH="${PROJECT_ROOT}/data/.mat" # (可选) 指定你的数据文件路径

# 模型架构参数
NUM_LAYERS=2
D_MODEL=64
NUM_HEADS=4
# D_FF 可以不在这里设置，让 train.py 根据 d_model 和 d_ff_multiplier 计算
# 或者明确设置 D_FF=256
D_FF_MULTIPLIER=4 # train.py 会用这个来计算 D_FF 如果 D_FF 未指定
DROPOUT_RATE=0.1

# 训练参数
BATCH_SIZE=16
LEARNING_RATE=0.0005
NUM_EPOCHS=30 # 减少epoch用于快速测试
VAL_SPLIT_RATIO=0.2

# 其他
SEED_VAL=42
SAVE_MODEL_PATH="${PROJECT_ROOT}/ckpt/hrrp_transformer_cli_best.pth"


# --- 激活你的conda环境 (如果需要) ---
# 例如:
# source /opt/anaconda3/etc/profile.d/conda.sh # 或者你的conda.sh路径
# conda activate your_env_name_here
# echo "Conda environment 'your_env_name_here' activated."


# --- 执行Python训练脚本 ---
echo "开始执行 train.py 脚本..."
PYTHON_SCRIPT_PATH="${PROJECT_ROOT}/train.py"


python $PYTHON_SCRIPT_PATH \
    --sequence_length $SEQ_LEN \
    --num_classes $NUM_CLASSES \
    --data_path "$DATA_PATH" \
    --num_layers $NUM_LAYERS \
    --d_model $D_MODEL \
    --num_heads $NUM_HEADS \
    --d_ff_multiplier $D_FF_MULTIPLIER \
    # 如果你想直接指定 d_ff，可以取消下面一行的注释，并注释掉 --d_ff_multiplier
    # --d_ff 256 \ 
    --dropout $DROPOUT_RATE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --val_split_ratio $VAL_SPLIT_RATIO \
    --seed $SEED_VAL \
    --save_path "$SAVE_MODEL_PATH"

EXIT_CODE=$? # 获取Python脚本的退出码

if [ $EXIT_CODE -eq 0 ]; then
    echo "train.py 脚本成功执行完毕。"
else
    echo "train.py 脚本执行失败，退出码: $EXIT_CODE"
fi

echo "Bash 脚本结束。"