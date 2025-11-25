"""
文件说明：alpaca_format.py - Alpaca格式数据转换模块

本文件主要功能：
1. 将PromptCBLUE数据集转换为Alpaca训练格式
2. 为每个任务添加对应的指令（instruction）描述
3. 支持按任务类型拆分数据集

Alpaca格式包含以下字段：
- instruction: 任务指令描述
- input: 输入文本
- output: 目标输出
- sample_id: 样本ID
- task_dataset: 任务数据集名称
- answer_choices: 候选答案列表

支持的任务类型：16种医疗NLP任务

作者：项目开发团队
"""

import json
import os
import sys

import numpy as np
from datasets import load_dataset

# =========================================
# 任务指令映射表
# 为每个任务类型定义详细的中文指令描述
# 这些指令用于引导模型理解任务要求
# =========================================
TASK_TO_INSTRUCTION = {
    "CHIP-CDEE" : "从下列输入中进行临床发现事件抽取任务。输出临床发现事件的主体词，以及发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值",
    "CHIP-CDN" : "诊断实体的语义标准化, 从给定的实体选项中选择与原诊断描述匹配的诊断标准词。从实体选项候选输出结果" ,
    "CHIP-CTC" : "根据输入的句子，确定该句子描述的临床试验筛选标准所属的类型。从类型选项候选输出结果",
    "CHIP-MDCFNPC" : "阴阳性判断的任务，在对话中，给出了一系列临床发现实体，然后根据每个实体判断其阴性或阳性。实体包括症状、疾病或假设可能发生的疾病，以及其他医学检查结果。根据对话内容，需要判断每个实体是已有症状疾病、未患有症状疾病，或者回答不明确或无实际意义。",
    
    "CHIP-STS": "判断输入中的两句话的意思是否相同。如果两句话意思相同输出\"是的\",意思不相同输出\"不是\"",
    "CMeEE-V2" : "抽取出输入中的医学相关命令实体，并根据提供的选项选择特定类型的实体列表。",
    "CMeIE" :  "从给定的文本中找出特定类型的关系，并找出关系的头实体和尾实体。对每个特定关系三元组输出格式，具有**关系的头尾实体对如下：头实体为**，尾实体为**。如果没有找到实体对。输出\"没有指定类型的三元组\". " ,
    "IMCS-V2-DAC" : "判断输入中给定的问诊句子或陈述句的意图类型。根据所提供的选项，选择输出与句子意图相匹配的答案。", 
    "IMCS-V2-MRG" : "根据下输入中给定的问诊对话生成诊疗报告。输出报告需要包括主诉，现病史，辅助检查，既往史，诊断，建议的内容",
    "IMCS-V2-NER" : "根据给定的输入文本，输出对应的实体类型和实体名称。如果没有找到实体对。输出\"上述句子没有指定类型实体\"",
    "IMCS-V2-SR" : "根据给定的对话历史和当前对话，输出每个对话中涉及的症状以及这些症状的阴阳性判断。如果患有该症状输出\"阳性\",没有患有该症状输出\"阴性\",无法根据上下文确定病人是否患有该症状输出\"无法确定\"",
    "KUAKE-IR" : "判断输入中的医疗搜索和回答内容是否相关。如果内容相关输出\"相关\",内容不相关输出\"不相关\"",
    "KUAKE-QIC" : "根据输入中的搜索内容句子，判断搜索的意图类型, 从类型选项候选输出结果",
    "KUAKE-QQR" : "判断输入两个句子之间的语义包含关系。是\"完全一致\"，\"后者是前者的语义子集\"，\"后者是前者的语义父集\"，\"语义无直接关联\"的哪一种",
    "KUAKE-QTR" : "判断输入两个句子之间的语义相似程度。是\"完全不匹配或者没有参考价值\"，\"很少匹配有一些参考价值\"，\"部分匹配\"，\"完全匹配\"中的哪一种" ,
    "MedDG" : "根据输入中给定的问诊对话历史生成医生的下一句回复"
}

# 数据集存储路径
# 数据格式字段说明：
# 'task_dataset': 任务数据集名称
# 'task_type': 任务类型
# 'answer_choice': 候选答案列表（可能为null）
data_path = "datasets/PromptCBLUE"


def split_task(path, dim):
    """
    按任务类型拆分数据集
    
    参数：
        path: str - 数据集文件夹路径（未使用，使用全局folder变量）
        dim: str - 数据集划分类型（'train', 'validation', 'test'）
    
    功能：
        1. 加载原始数据集
        2. 按task_dataset字段分组
        3. 将每个任务的数据保存到单独的文件夹中
    
    输出：
        在./data/{task_name}/目录下生成dev.json文件
    """
    # 加载数据集文件
    train_file =  os.path.join(folder, 'train.json')
    validation_file =  os.path.join(folder, 'dev.json')
    test_file =  os.path.join(folder, 'test.json')
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

    # 使用HuggingFace datasets库加载数据
    dataset = load_dataset(
        extension,
        data_files=data_files,
    )

    # 按任务类型分组数据集
    task_dataset = {}
    for i, task in enumerate(dataset[dim]['task_dataset']):
        if task not in task_dataset:
            task_dataset[task] = [dataset[dim][i]]
        else:
            task_dataset[task].append(dataset[dim][i])
    
    # 为每个任务创建单独的文件夹并保存数据
    for task in task_dataset.keys():
        if not os.path.exists(f'./data/{task}'):
            os.mkdir(f'./data/{task}')
        with open(os.path.join(f'./data/{task}', f'dev.json'), 'w',encoding='utf-8') as file:
            json.dump(task_dataset[task], file, ensure_ascii=False)


def alpaca_format(path):
    """
    将数据集转换为Alpaca格式
    
    参数：
        path: str - 数据集文件夹路径
    
    功能：
        1. 读取测试集文件
        2. 为每个样本添加instruction字段
        3. 重组数据字段为Alpaca格式
        4. 输出转换后的JSON文件
    
    输出格式：
        {
            "instruction": 任务指令,
            "input": 输入文本,
            "output": 目标输出,
            "sample_id": 样本ID,
            "task_dataset": 任务名称,
            "answer_choices": 候选答案列表
        }
    """
    for split in ['test']:
        list_test_samples = []
        print(path+f'/{split}.json')
        # 读取原始数据文件
        with open(path+f'/{split}.json', 'r', encoding='utf-8') as f:
            for line in f:
                list_test_samples.append(json.loads(line))
        
        output_list = []
        for sample in list_test_samples:
            ret_sample = {}
            task_name = sample['task_dataset']
            # 根据任务类型获取对应的指令描述
            ret_sample['instruction'] = TASK_TO_INSTRUCTION[task_name]
            ret_sample['input'] = sample['input']
            ret_sample['output'] = sample['target']
            ret_sample['sample_id'] = sample['sample_id']
            ret_sample['task_dataset'] = sample['task_dataset']
            ret_sample['answer_choices'] = sample['answer_choices']
            
            output_list.append(ret_sample)
        
        # 保存转换后的Alpaca格式数据
        output_path = path
        print(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(output_path+f'/alpaca_{split}.json', 'w', encoding='utf-8') as f:
            res = json.dumps(output_list, ensure_ascii=False)
            f.write(f"{res}\n")