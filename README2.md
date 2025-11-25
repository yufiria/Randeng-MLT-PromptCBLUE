# Randeng-MLT-PromptCBLUE 项目说明文档

## 项目概述

本项目是一个面向中文医疗自然语言处理（NLP）的多任务学习框架，基于PromptCBLUE数据集进行开发。项目实现了使用Randeng-T5和LLaMA系列模型对16种不同的医疗NLP任务进行微调和评估。

### 主要功能

1. **多任务医疗NLP处理**：支持16种医疗NLP任务的统一训练和评估
2. **双模型架构支持**：
   - **Randeng-T5**：基于T5的Encoder-Decoder架构，适合序列到序列任务
   - **ChatMed-LLaMA**：基于LLaMA的Decoder-only架构，使用LoRA进行参数高效微调
3. **数据处理流水线**：完整的数据预处理、言语化和格式转换工具
4. **分布式训练支持**：支持DeepSpeed和多GPU训练

### 支持的任务类型

| 任务代码 | 任务名称 | 任务类型 |
|---------|---------|---------|
| CMeEE-V2 | 医学命名实体识别 | 实体识别 |
| CMeIE | 医学关系抽取 | 关系抽取 |
| CHIP-CDN | 诊断标准化 | 实体标准化 |
| CHIP-CDEE | 临床事件抽取 | 事件抽取 |
| CHIP-STS | 语义相似度判断 | 文本匹配 |
| CHIP-CTC | 临床试验标准分类 | 文本分类 |
| CHIP-MDCFNPC | 临床发现阴阳性判断 | 属性分类 |
| KUAKE-IR | 医疗搜索相关性 | 信息检索 |
| KUAKE-QIC | 查询意图分类 | 意图识别 |
| KUAKE-QQR | 查询语义关系 | 语义推理 |
| KUAKE-QTR | 查询匹配度 | 语义匹配 |
| IMCS-V2-MRG | 诊疗报告生成 | 文本生成 |
| IMCS-V2-NER | 对话实体识别 | 实体识别 |
| IMCS-V2-DAC | 对话意图识别 | 意图分类 |
| IMCS-V2-SR | 症状识别 | 属性抽取 |
| MedDG | 医学对话生成 | 对话生成 |

---

## 项目结构

```
Randeng-MLT-PromptCBLUE/
├── README.md                    # 原始说明文档
├── README2.md                   # 本详细中文说明文档
├── requirements.txt             # 项目依赖包列表
│
├── data_utils/                  # 数据处理工具目录
│   ├── verbaliser.py           # 数据言语化处理模块
│   ├── alpaca_format.py        # Alpaca格式数据转换模块
│   └── zero_to_fp32.py         # DeepSpeed检查点转换工具
│
├── randeng/                     # Randeng-T5模型训练模块
│   ├── main.py                 # 训练主程序入口
│   ├── arguments.py            # 命令行参数定义
│   ├── build_dataset.py        # 数据集构建模块
│   ├── instruction.py          # 任务指令配置
│   ├── post_generate_process.py # 模型输出后处理
│   └── predict.py              # 预测推理脚本
│
├── src/
│   └── chatmed_llama_peft/      # ChatMed-LLaMA PEFT训练模块
│       ├── main.py             # 训练主程序入口
│       ├── arguments.py        # 命令行参数定义
│       ├── build_dataset.py    # 数据集构建模块
│       ├── instruction.py      # 任务指令配置
│       ├── trainer.py          # 自定义Trainer类
│       ├── callback.py         # 训练回调函数
│       ├── merge_llama_with_chinese_lora.py  # LoRA权重合并工具
│       ├── run_clm_pt_with_peft.py          # 预训练脚本
│       └── run_clm_sft_with_peft.py         # 监督微调脚本
│
├── scripts/                     # 训练和评估脚本
│   ├── train.sh                # 训练启动脚本
│   └── eval.sh                 # 评估启动脚本
│
├── datasets/                    # 数据集目录
│   └── PromptCBLUE/            # PromptCBLUE数据集
│       ├── train.json          # 训练集
│       ├── dev.json            # 验证集
│       └── test.json           # 测试集
│
├── aug_data/                    # 增强数据目录
│
└── exp/                         # 实验输出目录
    └── [experiment_name]/       # 各实验的输出
        ├── checkpoint-xxx/      # 检查点文件
        ├── test_predictions.json # 预测结果
        └── metrics.json         # 评估指标
```

---

## 文件详细说明

### data_utils/ 目录

#### verbaliser.py
**功能**：数据言语化处理模块

将原始的PromptCBLUE数据集标签转换为更适合模型学习的格式。主要处理包括：
- CMeIE任务的关系三元组格式化
- CHIP-CDEE任务的事件属性提取
- CHIP-MDCFNPC/IMCS-V2-SR任务的阴阳性标记简化
- CHIP-STS等分类任务的标签重复增强

**使用方法**：
```bash
python data_utils/verbaliser.py
# 需要在脚本中配置输入输出路径
```

#### alpaca_format.py
**功能**：Alpaca格式数据转换模块

将PromptCBLUE数据集转换为Alpaca训练格式，为每个任务添加对应的指令描述。

**输出格式**：
```json
{
    "instruction": "任务指令描述",
    "input": "输入文本",
    "output": "目标输出",
    "sample_id": "样本ID",
    "task_dataset": "任务名称"
}
```

#### zero_to_fp32.py
**功能**：DeepSpeed ZeRO检查点转换工具

将DeepSpeed ZeRO Stage 2/3的分布式检查点转换为标准PyTorch FP32模型文件。

**使用方法**：
```bash
python data_utils/zero_to_fp32.py <checkpoint_dir> <output_file>
# 例如：python data_utils/zero_to_fp32.py ./checkpoint pytorch_model.bin
```

---

### randeng/ 目录

#### main.py
**功能**：Randeng-T5模型训练与评估主程序

基于Randeng-T5-784M-MultiTask-Chinese预训练模型进行微调，支持训练、评估和预测三种模式。

**主要参数**：
- `--do_train`：执行训练
- `--do_eval`：执行评估
- `--do_predict`：执行预测
- `--model_name_or_path`：预训练模型路径
- `--train_file`：训练数据文件
- `--validation_file`：验证数据文件
- `--test_file`：测试数据文件

#### arguments.py
**功能**：命令行参数定义模块

定义两个数据类：
- `ModelArguments`：模型相关参数（模型路径、检查点、量化设置等）
- `DataTrainingArguments`：数据相关参数（数据文件路径、序列长度、beam搜索参数等）

#### build_dataset.py
**功能**：数据集构建模块

将原始JSON数据转换为模型可用的token化数据集，支持：
- 自动缓存机制，加速重复加载
- 多进程并行预处理
- 自定义数据整理器(DataCollator)

#### instruction.py
**功能**：任务指令配置模块

定义三个核心配置：
- `TASK_TO_MAX_NEW_TOKENS`：各任务的最大生成token数
- `TASK_TO_INSTRUCTION`：各任务的详细指令描述
- `TASK_TO_TASK_TYPE`：各任务的简化任务类型

#### post_generate_process.py
**功能**：模型输出后处理模块

将模型的原始预测结果转换为PromptCBLUE评测所需的标准格式，包括：
- 实体去重
- 标签还原（将简化标记还原为完整描述）
- 格式修正

**使用方法**：
```bash
python randeng/post_generate_process.py <input_file> <output_file>
```

#### predict.py
**功能**：模型预测脚本

使用Accelerate库实现分布式推理加速。

**主要参数**：
- `--task`：任务名称
- `--model_name_or_path`：模型路径
- `--output_dir`：输出目录
- `--per_device_eval_batch_size`：批次大小

---

### src/chatmed_llama_peft/ 目录

#### main.py
**功能**：ChatMed-LLaMA模型训练主程序（PEFT版本）

基于LLaMA模型使用LoRA进行参数高效微调(PEFT)，集成wandb进行训练监控。

**LoRA配置参数**：
- `--trainable`：应用LoRA的模块（如q_proj,v_proj）
- `--lora_rank`：LoRA秩
- `--lora_alpha`：LoRA缩放因子
- `--lora_dropout`：Dropout率

#### callback.py
**功能**：训练回调函数模块

`SavePeftModelCallback`类：
- 在每次保存检查点时额外保存PEFT适配器模型
- 自动清理大型检查点文件以节省磁盘空间

#### trainer.py
**功能**：自定义Trainer类

扩展HuggingFace Trainer，支持：
- 自定义评估逻辑（使用model.generate）
- 特定的损失计算方式

#### merge_llama_with_chinese_lora.py
**功能**：LoRA权重合并工具

将基础LLaMA模型与中文LoRA适配器合并，支持：
- 多个LoRA模型顺序合并
- 输出HuggingFace格式或pth格式

**使用方法**：
```bash
python src/chatmed_llama_peft/merge_llama_with_chinese_lora.py \
    --base_model <基础模型路径> \
    --lora_model <LoRA模型路径1>,<LoRA模型路径2> \
    --output_type huggingface \
    --output_dir <输出目录>
```

#### run_clm_pt_with_peft.py
**功能**：因果语言模型预训练脚本（PEFT版本）

用于对LLaMA模型进行持续预训练。

#### run_clm_sft_with_peft.py
**功能**：因果语言模型监督微调脚本（PEFT版本）

用于对LLaMA模型进行指令微调。

---

## 运行指南

### 环境配置

1. **安装依赖**：
```bash
pip install -r requirements.txt
```

2. **主要依赖包**：
- transformers >= 4.28.0
- peft >= 0.3.0
- torch >= 1.13.0
- datasets
- accelerate
- deepspeed（可选，用于分布式训练）
- wandb（可选，用于实验追踪）

### 数据准备

1. 下载PromptCBLUE数据集并放置在`datasets/PromptCBLUE/`目录下
2. 运行数据预处理（如需要）：
```bash
python data_utils/verbaliser.py
```

### 使用Randeng-T5训练

#### 训练模型
```bash
python randeng/main.py \
    --do_train \
    --model_name_or_path IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese \
    --train_file datasets/PromptCBLUE/train.json \
    --validation_file datasets/PromptCBLUE/dev.json \
    --output_dir exp/randeng_t5_experiment \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3
```

#### 评估模型
```bash
python randeng/main.py \
    --do_eval \
    --model_name_or_path IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese \
    --checkpoint_path exp/randeng_t5_experiment/checkpoint-xxx \
    --validation_file datasets/PromptCBLUE/dev.json \
    --output_dir exp/randeng_t5_experiment
```

#### 预测
```bash
python randeng/main.py \
    --do_predict \
    --model_name_or_path IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese \
    --checkpoint_path exp/randeng_t5_experiment/checkpoint-xxx \
    --test_file datasets/PromptCBLUE/test.json \
    --output_dir exp/randeng_t5_experiment
```

#### 后处理预测结果
```bash
python randeng/post_generate_process.py \
    exp/randeng_t5_experiment/test_predictions.json \
    exp/randeng_t5_experiment/final_results.json
```

### 使用ChatMed-LLaMA (PEFT) 训练

#### 训练模型（使用LoRA）
```bash
python src/chatmed_llama_peft/main.py \
    --do_train \
    --model_name_or_path <LLaMA模型路径> \
    --train_file datasets/PromptCBLUE/train.json \
    --validation_file datasets/PromptCBLUE/dev.json \
    --output_dir exp/llama_peft_experiment \
    --task_name CMeEE-V2 \
    --trainable q_proj,v_proj \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --block_size 512 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3
```

#### 合并LoRA权重
```bash
python src/chatmed_llama_peft/merge_llama_with_chinese_lora.py \
    --base_model <基础LLaMA模型路径> \
    --lora_model exp/llama_peft_experiment/checkpoint-xxx/adapter_model \
    --output_type huggingface \
    --output_dir exp/merged_model
```

### 使用DeepSpeed分布式训练

```bash
deepspeed --num_gpus=4 randeng/main.py \
    --do_train \
    --deepspeed ds_config.json \
    --model_name_or_path IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese \
    --train_file datasets/PromptCBLUE/train.json \
    --output_dir exp/deepspeed_experiment \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

---

## 常见问题

### Q1: 显存不足怎么办？

A: 可以尝试以下方法：
1. 减小`per_device_train_batch_size`
2. 增大`gradient_accumulation_steps`
3. 使用半精度训练（`--fp16`）
4. 使用梯度检查点（代码中已默认启用）
5. 使用LoRA等PEFT方法减少可训练参数

### Q2: 如何选择使用哪个模型？

A: 
- **Randeng-T5**：适合需要编码器-解码器架构的任务，如摘要生成、翻译等
- **ChatMed-LLaMA**：适合对话生成、指令遵循等任务，使用LoRA可以大幅减少训练资源需求

### Q3: 如何添加新的任务？

A: 
1. 在`instruction.py`中添加任务的指令描述和配置
2. 在`post_generate_process.py`中添加对应的后处理逻辑
3. 准备符合格式要求的训练数据

---

## 参考资源

- [PromptCBLUE数据集](https://github.com/michael-wzhu/PromptCBLUE)
- [Randeng-T5模型](https://huggingface.co/IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese)
- [PEFT库文档](https://github.com/huggingface/peft)
- [DeepSpeed文档](https://www.deepspeed.ai/)

---

## 许可证

本项目采用Apache License 2.0许可证。详见LICENSE文件。

---

## 贡献

欢迎提交Issue和Pull Request来改进本项目！

---

*本文档由项目开发团队编写，如有问题请提交Issue。*
