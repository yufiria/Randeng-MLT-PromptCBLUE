#!/bin/bash
# =========================================
# 文件说明：eval.sh - Randeng-T5模型评估/预测脚本
#
# 本脚本主要功能：
# 1. 配置分布式推理环境（多GPU）
# 2. 加载训练好的检查点进行预测
# 3. 输出预测结果到指定目录
#
# 使用方法：
#   bash scripts/eval.sh
#
# 配置项说明：
#   - CUDA_VISIBLE_DEVICES: 指定使用的GPU设备
#   - your_data_path: 数据集路径
#   - your_checkpoint_path: 训练好的检查点路径
#   - checkpoint_name: 要加载的具体检查点名称
#   - output_path: 预测结果输出路径
#   - model_path: 预训练模型路径
#
# 主要预测参数：
#   - max_source_length: 输入序列最大长度
#   - max_target_length: 生成序列最大长度
#   - per_device_eval_batch_size: 每个设备的批次大小
#   - predict_with_generate: 使用生成模式进行预测
# =========================================

# build_instruction_dataset work with usual format
export CUDA_VISIBLE_DEVICES='0,2,3'
# export WANDB_LOG_MODEL=true
export WANDB_MODE=disabled  # 评估时禁用wandb

# 自动计算GPU数量
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo $gpu_num $CUDA_VISIBLE_DEVICES

# =========================================
# 路径配置（请根据实际情况修改）
# =========================================
your_data_path="datasets/PromptCBLUE"  # 数据集所在文件夹
your_checkpoint_path="checkpoint/randen"  # 检查点存储路径
checkpoint_name=last-last-verb-checkpoint-14000  # 要加载的检查点名称

output_path=output/test  # 预测结果输出路径

model_path=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese  # 预训练模型路径

# 生成配置文件（可选）
generation_config=config/config.json

# =========================================
# 启动分布式预测
# =========================================
torchrun \
    --nnodes 1 \
    --nproc_per_node $gpu_num \
    --master_port 29501 \
    src/randeng/main.py \
    --do_predict \
    --checkpoint_path $your_checkpoint_path/$checkpoint_name \
    --test_file $your_data_path/test.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_path \
    --output_dir $output_path/$checkpoint_name \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 256 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate
    # --generation_config $generation_config \
