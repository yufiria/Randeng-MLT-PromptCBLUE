"""
文件说明：build_dataset.py - 数据集构建模块

本文件主要功能：
1. 将原始JSON数据集转换为模型可用的token化数据集
2. 支持训练集和测试集的不同处理逻辑
3. 实现数据缓存机制，加速重复加载
4. 提供自定义的数据整理器(DataCollator)用于批处理

主要函数：
- build_instruction_dataset: 构建带有任务类型前缀的指令数据集
- build_test_dataset: 构建测试数据集
- DataCollatorForSupervisedDataset: 监督学习的数据整理器

数据格式要求：
- 输入JSON文件需包含'task_dataset', 'input', 'target'字段
- 输出为包含'input_ids', 'labels', 'attention_mask'的数据集

作者：项目开发团队
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Union, List
import datasets
import torch
import logging
from datasets import load_dataset, concatenate_datasets
import copy
import transformers
import random

from randeng.instruction import TASK_TO_INSTRUCTION, TASK_TO_MAX_NEW_TOKENS, TASK_TO_TASK_TYPE

# 用于在标签中标记忽略位置的特殊值
IGNORE_INDEX = -100

logger = logging.getLogger('__name__')


def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_source_length: int, max_target_length: int, ignore_pad_token_for_loss=True, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):
    """
    构建指令数据集
    
    将原始JSON数据转换为模型训练所需的token化格式。
    每个样本会添加任务类型前缀，便于模型学习多任务。
    
    参数：
        data_path: str或List[str] - 数据文件路径或路径列表
        tokenizer: PreTrainedTokenizer - 分词器实例
        max_source_length: int - 输入序列的最大长度
        max_target_length: int - 目标序列的最大长度
        ignore_pad_token_for_loss: bool - 是否在损失计算中忽略padding token
        data_cache_dir: str, 可选 - 数据缓存目录
        preprocessing_num_workers: int, 可选 - 预处理使用的进程数
    
    返回：
        Dataset - 处理后的HuggingFace数据集
    
    处理流程：
        1. 加载JSON数据文件
        2. 为每个样本添加任务类型前缀
        3. 对输入和目标进行分词
        4. 缓存处理结果以加速后续加载
    """

    def tokenization(examples):
        """
        批量分词函数
        
        参数：
            examples: dict - 包含多个样本的批次数据
        
        返回：
            dict - 包含input_ids和labels的字典
        """
        sources = []
        targets = []
        for task, input, output in zip(examples['task_dataset'],examples['input'],examples['target']):
            if input is not None and input !="":
                # 添加任务类型前缀，格式："{任务类型}任务：{输入内容}"
                instruction = TASK_TO_TASK_TYPE[task]+'任务：'+input
            source = instruction
            # 在目标末尾添加结束符
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        # 对输入进行分词，启用padding和截断
        model_inputs = tokenizer(sources, max_length=max_source_length, padding=True, truncation=True)
        
        # 对目标进行分词
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=True, truncation=True)

        # 如果需要忽略padding token的损失，将padding位置替换为-100
        if ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    logging.warning("building dataset...")
    all_datasets = []

    # 支持单个文件或多个文件
    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:
        # 设置缓存目录
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)
        try:
            # 尝试从缓存加载已处理的数据集
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            # 缓存不存在，重新处理数据
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            column_names = raw_dataset['train'].column_names
            print(column_names)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            # 保存到缓存
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    # 合并所有数据集
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets



def build_test_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_source_length: int, max_target_length: int, ignore_pad_token_for_loss=True, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):
    """
    构建测试数据集
    
    与训练数据集类似，但不添加任务类型前缀。
    用于模型评估和预测阶段。
    
    参数：
        data_path: str或List[str] - 数据文件路径或路径列表
        tokenizer: PreTrainedTokenizer - 分词器实例
        max_source_length: int - 输入序列的最大长度
        max_target_length: int - 目标序列的最大长度
        ignore_pad_token_for_loss: bool - 是否在损失计算中忽略padding token
        data_cache_dir: str, 可选 - 数据缓存目录
        preprocessing_num_workers: int, 可选 - 预处理使用的进程数
    
    返回：
        Dataset - 处理后的HuggingFace数据集
    """

    def tokenization(examples):
        """
        批量分词函数（测试集版本）
        
        与训练集分词不同，测试集直接使用原始输入，不添加任务前缀
        """
        sources = []
        targets = []
        for input, output in zip(examples['input'],examples['target']):
            if input is not None and input !="":
                instruction = input
            source = instruction
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        
        model_inputs = tokenizer(sources, max_length=max_source_length, padding=True, truncation=True)
        
        # 对目标进行分词
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=True, truncation=True)

        # 处理padding token的标签
        if ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    logging.warning("building predict dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:

        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'eval datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            column_names = raw_dataset['test'].column_names
            print(column_names)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['predict'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    监督学习数据整理器
    
    用于将多个样本整理成一个批次，执行动态padding操作。
    
    属性：
        tokenizer: PreTrainedTokenizer - 分词器实例，用于获取pad_token_id
    
    功能：
        1. 将input_ids列表填充到统一长度
        2. 将labels列表填充到统一长度（使用-100作为padding值）
        3. 生成attention_mask
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理一批样本
        
        参数：
            instances: Sequence[Dict] - 样本列表，每个样本包含input_ids和labels
        
        返回：
            Dict[str, torch.Tensor] - 包含批处理后的input_ids, labels和attention_mask
        """
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 使用RNN的pad_sequence进行填充
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            # 非padding位置的attention_mask为True
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )