"""
文件说明：verbaliser.py - 数据言语化处理模块

本文件主要功能：
1. 读取aug_data目录下的增强数据集文件
2. 对不同类型的医疗NLP任务数据进行言语化（Verbalisation）处理
3. 将原始标签转换为更适合模型学习的格式
4. 最终输出处理后的训练数据集

支持的任务类型包括：
- CMeIE: 医学关系抽取
- CHIP-CDEE: 临床发现事件抽取
- CHIP-MDCFNPC: 临床发现阴阳性判断
- CHIP-STS: 语义相似度判断
- CHIP-CTC: 临床试验标准分类
- KUAKE-QIC: 查询意图分类
- KUAKE-IR: 信息检索相关性判断
- IMCS-V2-SR: 症状识别

作者：项目开发团队
"""

import re
import os
import sys
import json
import csv
import random

# 数据集所在目录路径
path = './aug_data'

# 存储所有样本的列表
samples = []

# 遍历aug_data目录下的所有文件，读取并解析JSON数据
# 注意：跳过IMCS-V2-DAC任务的数据（意图识别任务）
for file in os.listdir(path):
    new_path = os.path.join(path, file)
    with open(new_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            # 排除IMCS-V2-DAC任务数据
            if sample['task_dataset'] == 'IMCS-V2-DAC': continue
            samples.append(sample)

# 定义需要特殊处理的任务类型列表
# NULL_keys: 可能输出"非上述类型"的任务
NULL_keys = ['CHIP-CTC' , 'KUAKE-QIC']
# with_first_sent_keys: 输出中包含引导句的任务
with_first_sent_keys = ["CMeEE-V2", "IMCS-V2-MRG",]

# 遍历所有样本，根据不同任务类型进行言语化处理
for sample in samples:
    # 添加非上述类型
    # if sample['task_dataset'] in NULL_keys:
    #     prompt = sample['input']
    #     if prompt[-1] == '：':
    #         prompt = prompt[:-3]
    #     prompt += "，非上述类型"
    #     sample['input'] = prompt

    # =========================================
    # CMeIE任务：医学关系抽取
    # 将原始的关系三元组格式转换为更结构化的输出格式
    # 输出格式：关系名称：(头实体，[尾实体1|尾实体2|...])
    # =========================================
    if sample['task_dataset'] == 'CMeIE':
        output = ""
        if sample['target'] == '':
            continue
        lines = sample['target'].split('\n')
        # 解析每一行关系描述
        for line in lines:
            if line.startswith('没有'):
                # 如果没有找到关系，输出所有候选关系
                for choice in sample['answer_choices']:
                    output += f"\n{choice}关系："
            else:
                # 使用正则表达式提取关系类型和实体对
                rel_pat = r"具有(.*?)关系的头尾实体"
                rel = re.findall(rel_pat, line)[0].strip()
                output += f"\n{rel}关系："
                pat = r"头实体为(.*?)，尾实体为(.*?)。"
                matches = re.findall(pat, line)
                # 将相同头实体的尾实体合并
                dict = {}
                for match in matches:
                    sub, obj = match[0].strip(), match[1].strip()
                    if sub in dict:
                        dict[sub].append(obj)
                    else:
                        dict[sub] = [obj]
                # 格式化输出：(头实体，[尾实体1|尾实体2|...])
                for k, v in dict.items():
                    output += f'({k}，[{"|".join(v)}])。'
        sample['target'] = output[1:]

    # =========================================
    # CHIP-CDEE任务：临床发现事件抽取
    # 提取事件的主体词、发生状态、描述词和解剖部位
    # 输出格式：(主体词；发生状态；描述词；解剖部位)
    # =========================================
    elif sample['task_dataset'] == 'CHIP-CDEE':
        output = ""
        lines = sample['target'].split('\n')[1:]
        # 定义事件属性分隔符
        delimiters = ["主体词：", "发生状态：", "描述词：", "解剖部位："]
        for line in lines:
            triple = []
            for delimiter in delimiters:
                # 按照分隔符提取每个属性的值
                split_string = line.split(delimiter, 1)
                if len(split_string) > 1:
                    value = split_string[1].split("；", 1)[0].strip()
                else:
                    value = ""
                triple.append(value)
            # 格式化为元组形式
            output += "\n(" + '；'.join(triple) +")"
        sample['target'] = output[1:]
    
    # =========================================
    # CHIP-MDCFNPC任务：临床发现阴阳性判断
    # 将原始标签转换为简化的阴阳性标记
    # 阳阳阳=已有症状, 阴阴阴=未患有, 其他的=回答不明确, 不标注=无实际意义
    # =========================================
    elif sample['task_dataset'] == 'CHIP-MDCFNPC':
        output = ""
        lines = sample['target'].split('\n')[1:] 
        for line in lines:
            sym, label = line.split('：')[0], line.split('：')[1]
            # 将详细描述转换为简化标记，便于模型学习
            if label.startswith("已有"):
                label = "阳阳阳"  # 表示已有症状/疾病
            elif label.startswith("未患有"):
                label = '阴阴阴'  # 表示未患有
            elif label.startswith("没有回答"):
                label = '其他的'  # 表示回答不明确
            elif label.startswith('无实际'):
                label = '不标注'  # 表示无实际意义
            output += f"\n{sym}：{label}"
        sample['target'] = output[1:]
    
    # =========================================
    # CHIP-STS任务：语义相似度判断
    # 将标签重复10次以增强模型对该任务的学习
    # "是"或"不"重复10次
    # =========================================
    if sample['task_dataset'] == 'CHIP-STS':
        label = sample['target']
        # 将"相同/不同"转换为"是/不"并重复10次
        if label in ['不同', '不是']:
            label = '不'
        else:
            label = '是'
        sample['target'] = ''.join([label for _ in range(10)])
    
    # =========================================
    # CHIP-CTC任务：临床试验标准分类
    # 将分类标签重复3次以增强模型学习
    # =========================================
    elif sample['task_dataset'] == 'CHIP-CTC':
        label = sample['target']
        sample['target'] = '；'.join([label for _ in range(3)])
    
    
    # =========================================
    # KUAKE-QIC任务：查询意图分类
    # 将分类标签重复3次以增强模型学习
    # =========================================
    elif sample['task_dataset'] == 'KUAKE-QIC':
        label = sample['target']
        sample['target'] = '；'.join([label for _ in range(3)])

    
    # =========================================
    # KUAKE-IR任务：信息检索相关性判断
    # 将相关性标签重复3次以增强模型学习
    # =========================================
    elif sample['task_dataset'] == 'KUAKE-IR':
        label = sample['target']
        sample['target'] = '；'.join([label for _ in range(3)])
    
    # =========================================
    # IMCS-V2-SR任务：症状识别
    # 将症状阴阳性标签转换为简化标记
    # 阳阳阳=患有, 阴阴阴=没有患有, 不确定=无法确定
    # =========================================
    elif sample['task_dataset'] == 'IMCS-V2-SR':
        output = ""
        lines = sample['target'].split('\n')[1:] 
        for line in lines:
            sym, label = line.split('：')[0], line.split('：')[1]
            # 将详细描述转换为简化标记
            if label.startswith("患有"):
                label = "阳阳阳"  # 表示患有该症状
            elif label.startswith("没有患有"):
                label = '阴阴阴'  # 表示没有患有
            elif label.startswith("无法根据"):
                label = '不确定'  # 表示无法确定
            output += f"\n{sym}：{label}"
        sample['target'] = output[1:]
    
    # 以下注释代码用于处理带有引导句的任务，可根据需要启用
    # elif sample['task_dataset'] in with_first_sent_keys:
    #     lines = sample['target'].split('\n')[1:]
    #     sample['target'] = '\n'.join(lines)

# 随机打乱样本顺序，增加训练的随机性
random.shuffle(samples)

# =========================================
# 输出处理后的数据集
# 将言语化处理后的样本写入JSON文件
# =========================================
# output_path = './datasets/toy_examples/toy_aug_train_verb.json'  # 测试用小数据集
output_path = './datasets/PromptCBLUE/aug_train_verb.json'  # 正式训练数据集
with open(output_path, 'w', encoding='utf-8') as f:
    for sample in samples:
        str = json.dumps(sample, ensure_ascii=False)
        f.write(str+'\n')
