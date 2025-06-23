#!/bin/bash

python ../train_contrastive.py \
  --train_data_path "../data/train" \
  --val_data_path "../data/val" \
  --sequence_length 512 \
  --num_classes 8 \
  --d_model 512 \
  --num_layers 3 \
  --num_heads 8 \
  --d_ff 1024 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --num_epochs 50 \
  --weight_decay 0.01 \
  --early_stopping_patience 15 \
  --num_workers 8 \
  --save_dir "../ckpt" \
  --log_dir "../logs"