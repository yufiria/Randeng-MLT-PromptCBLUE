#!/bin/bash
# =========================================
# 文件说明：train.sh - Randeng-T5模型训练脚本
#
# 本脚本主要功能：
# 1. 配置分布式训练环境（多GPU）
# 2. 设置训练参数和数据路径
# 3. 启动Randeng-T5模型的训练
#
# 使用方法：
#   bash scripts/train.sh
#
# 配置项说明：
#   - CUDA_VISIBLE_DEVICES: 指定使用的GPU设备
#   - your_data_path: 数据集路径
#   - your_checkpoint_path: 继续训练的检查点路径
#   - output_path: 训练输出路径
#   - model_path: 预训练模型路径
#   - LR: 学习率
#
# 主要训练参数：
#   - max_source_length: 输入序列最大长度
#   - max_target_length: 目标序列最大长度
#   - per_device_train_batch_size: 每个设备的批次大小
#   - gradient_accumulation_steps: 梯度累积步数
#   - max_steps: 最大训练步数
# =========================================

# build_instruction_dataset work with usual format 
export CUDA_VISIBLE_DEVICES='2, 3'
export WANDB_LOG_MODEL=true
# export WANDB_MODE=disabled

# 自动计算GPU数量
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo $gpu_num $CUDA_VISIBLE_DEVICES

# =========================================
# 路径配置（请根据实际情况修改）
# =========================================
your_data_path="datasets/PromptCBLUE"  # 数据集所在文件夹
your_checkpoint_path=checkpoint/randen/new-verb-checkpoint-6000  # 检查点文件夹（用于继续训练）
output_path="checkpoint/randeng-aug-verb"  # 输出文件夹
model_path=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese  # 预训练模型路径

# DeepSpeed配置文件（可选）
deepspeed_config_file="src/chatmed_llama_peft/deepspeed_stage2.json"

# =========================================
# 训练超参数配置
# =========================================
LR=1e-6  # 学习率

# =========================================
# 启动分布式训练
# =========================================
torchrun \
    --nnodes 1 \
    --nproc_per_node $gpu_num \
    --master_port 29500 \
    randeng/main.py \
    --do_train \
    --train_file $your_data_path/aug_train_verb.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_path \
    --checkpoint_path $your_checkpoint_path \
    --output_dir $output_path \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_steps 10000 \
    --logging_steps 100 \
    --save_steps 1000 \
    --lr_scheduler_type constant \
    --save_total_limit 3 \
    --learning_rate $LR \
    --run_name real_final_verb_15k \
    --report_to wandb
    # --lr_scheduler_type constant \
    # --do_eval \
    # --validation_file $your_data_path/dev.json \
    # --evaluation_strategy steps \
    # --eval_steps 500 \

    # --fp16 \
    # --deepspeed ${deepspeed_config_file} \