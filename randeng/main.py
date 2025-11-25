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
文件说明：main.py - Randeng-T5模型训练与评估主程序

本文件主要功能：
1. 基于Randeng-T5-784M-MultiTask-Chinese预训练模型进行微调
2. 支持训练(train)、评估(eval)、预测(predict)三种模式
3. 使用Seq2Seq架构处理PromptCBLUE医疗NLP任务
4. 支持从检查点恢复训练
5. 输出预测结果为JSON格式

模型架构：
- 基础模型：T5ForConditionalGeneration
- 分词器：T5Tokenizer
- 训练框架：HuggingFace Transformers Seq2SeqTrainer

使用方法：
    训练：python randeng/main.py --do_train --train_file xxx --model_name_or_path xxx
    预测：python randeng/main.py --do_predict --test_file xxx --checkpoint_path xxx

作者：项目开发团队
"""
# 可以根据自己的序列到序列任务调整此脚本，代码中留有相应的注释说明

import logging
import os
import sys
import json

import sys
sys.path.append("./")

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    set_seed,
)

import evaluate

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from randeng.build_dataset import build_instruction_dataset, build_test_dataset
from randeng.arguments import ModelArguments, DataTrainingArguments


# 初始化日志记录器
logger = logging.getLogger(__name__)

def main():
    """
    主函数：训练、评估和预测的入口函数
    
    功能流程：
    1. 解析命令行参数或JSON配置文件
    2. 设置日志和随机种子
    3. 加载预训练模型和分词器
    4. 构建训练/验证/测试数据集
    5. 初始化Trainer并执行训练/评估/预测
    6. 保存结果和指标
    """
    # 解析参数：支持JSON文件或命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果只传入一个JSON文件参数，则从该文件解析配置
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    # 加载预训练模型和分词器
    # =========================================
    config = T5Config.from_pretrained(
        model_args.model_name_or_path,
    )

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    ).cuda()

    # 如果指定了检查点路径且存在，则加载检查点权重
    if model_args.checkpoint_path != None and os.path.exists(model_args.checkpoint_path):
        logger.info(f'loading checkpoint from {model_args.checkpoint_path}')
        weights_file = os.path.join(model_args.checkpoint_path, 'pytorch_model.bin')
        state_dict = torch.load(weights_file, map_location='cpu')
        model.load_state_dict(state_dict)
        del state_dict
        model = model.cuda()

    # 获取输入/输出列名配置
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    
    # 设置序列长度限制
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length


    def print_dataset_example(example):
        """
        打印数据集样例，用于调试和验证数据预处理
        
        参数：
            example: dict - 包含input_ids和labels的数据样例
        """
        print("input_ids",example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        example["labels"] = [l if l != -100 else tokenizer.pad_token_id for l in example['labels']]
        print("labels", tokenizer.decode(example["labels"]))

    # =========================================
    # 构建训练数据集
    # =========================================
    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            train_dataset = build_instruction_dataset(
                data_path=[data_args.train_file],
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
                preprocessing_num_workers=data_args.preprocessing_num_workers
            )
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        print_dataset_example(train_dataset[0])

    # =========================================
    # 构建验证数据集
    # =========================================
    if training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            eval_dataset = build_instruction_dataset(
                data_path=[data_args.validation_file],
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
                preprocessing_num_workers=data_args.preprocessing_num_workers
            )
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("eval example:")
    
    # =========================================
    # 构建预测数据集
    # =========================================
    if training_args.do_predict:
        with training_args.main_process_first(desc="loading and tokenization"):
            predict_dataset = build_instruction_dataset(
                data_path=[data_args.test_file],
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
                preprocessing_num_workers=data_args.preprocessing_num_workers
            )
        logger.info(f"Num predict_samples  {len(predict_dataset)}")
        logger.info("eval example:")
    
    # 配置数据整理器，用于批处理时的padding
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=True
    )

    def compute_metrics(eval_preds):
        """
        计算评估指标
        
        参数：
            eval_preds: tuple - (预测结果, 标签)
        
        返回：
            dict - 包含F1分数的指标字典
        """
        metric = evaluate.load('f1')
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
            "f1" : [],
        }
        for pred, label in zip(preds, labels):
            results = metric.compute(predictions=pred, references=label, average="micro")
            score_dict['f1'].append(round(results['f1'] * 100, 2))
        
        score_dict["f1"] = float(np.mean(score_dict["f1"]))
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
            # 根据模型和配置，logits可能包含额外的张量（如past_key_values）
            # 但logits总是第一个
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    # 覆盖Seq2SeqTrainer的解码参数
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    
    # 初始化Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None
    )

    # =========================================
    # 执行训练
    # =========================================
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
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
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=512, temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # =========================================
    # 执行预测
    # =========================================
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # 读取原始测试文件以获取元数据
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        # 如果配置了生成参数，则使用自定义生成配置
        if training_args.generation_config != None:
            with open(training_args.generation_config, 'r') as f:
                gen_kwargs = json.load(f)
            logger.info(f'gen_kwargs:{gen_kwargs}')
        
            predict_results = trainer.predict(
                predict_dataset,
                metric_key_prefix="predict",
                max_new_tokens=data_args.max_target_length,
                **gen_kwargs
            )
        else:
            predict_results = trainer.predict(
                predict_dataset,
                metric_key_prefix="predict",
                max_new_tokens=data_args.max_target_length,
            )
            
        metrics = predict_results.metrics
        print(metrics)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # 仅在主进程中保存预测结果
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = np.where(predict_results.predictions==-100, tokenizer.pad_token_id, predict_results.predictions)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                predictions = [pred.strip() for pred in predictions]

                # 将预测结果与原始数据合并并保存
                output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")

                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, p in enumerate(predictions):
                        samp = list_test_samples[idx]
                        samp["target"] = p
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")

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