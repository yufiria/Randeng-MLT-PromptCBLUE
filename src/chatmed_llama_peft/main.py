#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
文件说明：main.py - ChatMed-LLaMA模型训练与评估主程序（PEFT版本）

本文件主要功能：
1. 基于LLaMA模型使用LoRA进行参数高效微调(PEFT)
2. 支持训练(train)和评估(eval)两种模式
3. 使用因果语言模型(Causal LM)架构处理医疗对话任务
4. 支持DeepSpeed分布式训练和梯度检查点
5. 集成wandb进行训练监控

模型架构：
- 基础模型：LlamaForCausalLM
- 微调方法：LoRA (Low-Rank Adaptation)
- 分词器：LlamaTokenizer

LoRA配置说明：
- target_modules: 应用LoRA的模块（如q_proj, v_proj等）
- r: LoRA秩（影响参数量和表达能力）
- lora_alpha: LoRA缩放因子
- lora_dropout: LoRA层的dropout率

使用方法：
    训练：python src/chatmed_llama_peft/main.py --do_train --train_file xxx --model_name_or_path xxx
    评估：python src/chatmed_llama_peft/main.py --do_eval --validation_file xxx

作者：项目开发团队
"""
# 可以根据自己的序列到序列任务调整此脚本，代码中留有相应的注释说明

import logging
import os
import sys
import json

import math
import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import wandb

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

from typing import Optional, List, Dict, Any, Mapping
sys.path.append("./")
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'  # 指定使用的GPU设备

from transformers import (
    LlamaTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

from src.chatmed_llama_peft.build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset
from src.chatmed_llama_peft.callback import SavePeftModelCallback
from src.chatmed_llama_peft.trainer import Trainer
from src.chatmed_llama_peft.arguments import ModelArguments, DataTrainingArguments
from src.chatmed_llama_peft.instruction import TASK_TO_INSTRUCTION, TASK_TO_MAX_NEW_TOKENS

from peft import PeftModel, LoraConfig, TaskType, PeftModelForCausalLM, get_peft_model, get_peft_model_state_dict

# 初始化日志记录器
logger = logging.getLogger(__name__)


def main():
    """
    主函数：训练、评估的入口函数
    
    功能流程：
    1. 解析命令行参数或JSON配置文件
    2. 初始化wandb进行训练监控
    3. 加载预训练LLaMA模型和分词器
    4. 配置LoRA参数并应用PEFT
    5. 构建训练/验证数据集
    6. 初始化自定义Trainer并执行训练/评估
    7. 保存结果和指标
    """
    # 解析参数：支持JSON文件或命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果只传入一个JSON文件参数，则从该文件解析配置
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化wandb进行实验追踪
    wandb.init(
        project='llama_peft',
        name=training_args.run_name
    )
    
    # 配置日志格式和处理器
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
        
    if training_args.should_log:
        # 默认的log_level是passive，这里设置为info级别
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 记录每个进程的小结
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 在模型初始化前设置随机种子，确保可复现性
    set_seed(training_args.seed)

    # =========================================
    # 加载数据集文件配置
    # =========================================
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # =========================================
    # 加载预训练LLaMA模型和分词器
    # =========================================
    config = LlamaConfig.from_pretrained(model_args.model_name_or_path)

    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    ).cuda()
    
    # 训练时使用半精度以节省显存
    if training_args.do_train:
        model = model.half()
    
    # =========================================
    # 配置LoRA并应用PEFT
    # =========================================
    if model_args.peft_path is not None:
        # 从预训练的PEFT模型加载（用于继续训练或预测）
        logger.info("Peft from pre-trained model")
        logger.info("Only load for prediction")
        peft_config = LoraConfig.from_pretrained(model_args.peft_path)
        model = get_peft_model(model, peft_config)
        model = PeftModelForCausalLM.from_pretrained(model, model_args.peft_path, is_trainable=False)
    else:
        # 初始化新的PEFT模型
        logger.info("Init new peft model")
        target_modules = model_args.trainable.split(',')  # LoRA应用的目标模块
        modules_to_save = model_args.modules_to_save.split(',') if model_args.modules_to_save!="null" else None
        lora_rank = model_args.lora_rank      # LoRA秩
        lora_dropout = model_args.lora_dropout  # Dropout率
        lora_alpha = model_args.lora_alpha      # 缩放因子
        print(target_modules)
        print(lora_rank)
        
        # 创建LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules[0],
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, peft_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
        
    # 量化设置（可选）
    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    # 配置数据整理器
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = None
    train_dataset = None
    
    # =========================================
    # 配置序列长度
    # =========================================
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warn(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    
    # =========================================
    # 构建训练数据集
    # =========================================
    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            files = [data_args.train_file]
            logger.info(f"training files: {' '.join(files)}")
            train_dataset = build_instruction_dataset(
                data_path=files, 
                tokenizer=tokenizer, 
                max_seq_length=data_args.block_size,
                data_cache_dir = None, 
                preprocessing_num_workers = data_args.preprocessing_num_workers)
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
        
    # =========================================
    # 构建验证数据集
    # =========================================
    if training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            files = [data_args.validation_file]
            logger.info(f"validation files: {' '.join(files)}")
            eval_dataset = build_instruction_dataset(
                data_path=files, 
                tokenizer=tokenizer, 
                max_seq_length=data_args.block_size,
                data_cache_dir = None, 
                preprocessing_num_workers = data_args.preprocessing_num_workers)
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("eval example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))
        
    
    def print_dataset_example(example):
        """
        打印数据集样例，用于调试和验证数据预处理
        
        参数：
            example: dict - 包含input_ids和labels的数据样例
        """
        print("input_ids",example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", len(example["labels"]))
        labels = list(map(lambda x: tokenizer.pad_token_id if x == -100 else x, example['labels']))
        print("labels", tokenizer.decode(labels))
    

    def compute_metrics(eval_preds):
        """
        计算评估指标：ROUGE和BLEU
        
        参数：
            eval_preds: tuple - (预测结果, 标签)
        
        返回：
            dict - 包含rouge-1, rouge-2, rouge-l, bleu-4的指标字典
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # 处理padding token
        if data_args.ignore_pad_token_for_loss:
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        # 计算每个样本的指标
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        # 计算平均值
        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    def preprocess_logits_for_metrics(logits, labels):
        """
        预处理logits用于指标计算
        
        参数：
            logits: tensor或tuple - 模型输出的logits
            labels: tensor - 真实标签
        
        返回：
            tensor - argmax后的预测结果
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
        """
        容错数据整理器
        
        在批处理时处理可能的异常情况，确保训练稳定性
        
        参数：
            features: List - 特征列表
        
        返回：
            Dict[str, torch.Tensor] - 批处理后的数据
        """
        import torch

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        # 特殊处理labels字段
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # 处理其他字段
        try:
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
        except ValueError:
            # 容错处理：使用第一个样本填充
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([features[0][k]] * len(features))
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                    else:
                        batch[k] = torch.tensor([features[0][k]] * len(features))

        return batch
    
    # 配置生成参数
    training_args.predict_with_generate = True
    task = data_args.task_name
    data_args.max_new_tokens = TASK_TO_MAX_NEW_TOKENS[task]
    
    # =========================================
    # 初始化自定义Trainer
    # =========================================
    trainer = Trainer(
        model=model,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available() else None,
        callbacks=[SavePeftModelCallback],  # 保存PEFT模型的回调
    )

    # =========================================
    # 执行训练
    # =========================================
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        logger.info(f"load checkpoint from {training_args.resume_from_checkpoint}")
        
        # 启用梯度检查点以节省显存
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # =========================================
    # 执行评估
    # =========================================
    results = {}
    if training_args.do_eval:
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        
        # 计算困惑度
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 预测功能（已注释，可根据需要启用）
    # if training_args.do_predict:
    #     ...

    return results


def _mp_fn(index):
    """
    TPU训练的多进程入口函数
    
    参数：
        index: int - 进程索引
    """
    main()


if __name__ == "__main__":
    main()
