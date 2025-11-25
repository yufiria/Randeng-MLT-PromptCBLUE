"""
文件说明：predict.py - 模型预测脚本

本文件主要功能：
1. 使用训练好的Randeng-T5模型进行推理预测
2. 支持按任务类型加载对应的测试数据
3. 使用Accelerate库实现分布式推理加速
4. 输出预测结果为JSON格式

使用方法：
    python randeng/predict.py --task CHIP-STS --model_name_or_path xxx --output_dir xxx

主要参数：
    --task: 任务名称（如CHIP-STS, CMeEE-V2等）
    --model_name_or_path: 模型路径
    --output_dir: 输出目录
    --per_device_eval_batch_size: 每个设备的批次大小

作者：项目开发团队
"""

import logging
import os
import sys
import json

import numpy as np
from tqdm import tqdm

import datasets
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from torch.utils.data import DataLoader

from argparse import ArgumentParser

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    default_data_collator,
    set_seed,
)

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES']='0'  # 指定使用的GPU
os.environ["WANDB_MODE"]='disabled'     # 禁用wandb日志

import sys
sys.path.append("./")

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from randeng.arguments import ModelArguments, DataTrainingArguments
from randeng.instruction import TASK_TO_MAX_NEW_TOKENS, TASK_TO_INSTRUCTION, TASK_TO_TASK_TYPE
from randeng.build_dataset import build_instruction_dataset

from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)


def main(args):
    """
    主函数：执行模型预测
    
    参数：
        args: argparse.Namespace - 命令行参数对象
    
    功能流程：
        1. 初始化Accelerator用于分布式推理
        2. 加载预训练模型和分词器
        3. 构建预测数据集
        4. 执行批量推理
        5. 保存预测结果
    """
    # 初始化Accelerator，支持多GPU推理
    accelerator = Accelerator()
    device = accelerator.device
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 设置日志级别
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # 设置随机种子以确保可复现性
    if args.seed is not None:
        set_seed(args.seed)
        
    # =========================================
    # 加载预训练模型和分词器
    # =========================================
    config = T5Config.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        config=config,
    ).cuda()

    # =========================================
    # 加载数据集
    # =========================================
    data_path = f"data/{args.task}"
    train_file = os.path.join(data_path, 'train.json')
    validation_file = os.path.join(data_path, 'dev.json')
    test_file = os.path.join(data_path, 'test.json')
    
    # 配置数据文件
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
        extension = train_file.split(".")[-1]
    if validation_file is not None:
        data_files["validation"] = validation_file
        extension = validation_file.split(".")[-1]
    if test_file is not None:
        data_files["test"] = test_file
        extension = test_file.split(".")[-1]

    # 设置序列长度
    max_source_length = 512  # 输入序列最大长度
    max_target_length = 10   # 目标序列最大长度
    
    # 构建预测数据集
    predict_dataset = build_instruction_dataset(
        data_path=[validation_file],
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        ignore_pad_token_for_loss=True,
        preprocessing_num_workers=8
    )

    # 创建数据加载器
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    predict_dataloader = DataLoader(
        predict_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    
    # 使用Accelerator准备模型和数据加载器
    model, predict_dataloader = accelerator.prepare(model, predict_dataloader)
    
    model.eval()  # 设置为评估模式
    
    # 计算总批次大小
    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
    logger.info("***** Running predictions *****")
    logger.info(f"  Num examples = {len(predict_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    
    # 读取原始测试文件以获取元数据
    list_test_samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        line = f.readline()
        list_test_samples = json.loads(line)

    # =========================================
    # 执行预测
    # =========================================
    all_pred = []
    for step, batch in enumerate(predict_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        # 根据任务类型设置生成的最大token数
        inputs['max_new_tokens'] = TASK_TO_MAX_NEW_TOKENS[args.task]
        
        # 生成预测结果
        output = model.generate(**inputs)
        predictions = tokenizer.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        all_pred += predictions
        
    # =========================================
    # 保存预测结果
    # =========================================
    output_prediction_file = os.path.join(args.output_dir, f'/test_prediction.json')
        
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        for idx, p in enumerate(all_pred):
            samp = list_test_samples[idx]
            samp["output"] = p
            res = json.dumps(samp, ensure_ascii=False)
            writer.write(f"{res}\n")


# =========================================
# 命令行入口
# =========================================
if __name__ == '__main__':
    parser = ArgumentParser(description='Randeng-T5模型预测脚本')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/CHIP-STS',
        help='预测结果输出目录'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese',
        help='预训练模型路径或HuggingFace模型标识符'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='任务名称（如CHIP-STS, CMeEE-V2等）'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='alpaca',
        help='数据路径'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='是否使用FP16推理'
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=32,
        help='每个设备的批次大小'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于可复现的结果"
    )
    args = parser.parse_args()
    main(args)