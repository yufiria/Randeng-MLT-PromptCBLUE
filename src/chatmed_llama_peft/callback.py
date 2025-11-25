"""
文件说明：callback.py - 训练回调函数模块

本文件主要功能：
1. 定义SavePeftModelCallback回调类
2. 在每次保存检查点时，额外保存PEFT适配器模型
3. 自动清理不需要的大型检查点文件以节省磁盘空间

回调功能：
- 在on_save事件触发时执行
- 保存PEFT适配器到adapter_model子目录
- 删除完整模型的state_dict文件（pytorch_model.bin）
- 删除DeepSpeed的分布式检查点文件

使用场景：
- 与HuggingFace Trainer配合使用
- 特别适用于LoRA等PEFT方法的训练

作者：项目开发团队
"""

import os
import shutil
import sys

import wandb

import transformers
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    TrainingArguments,
    TrainerControl,
    TrainerState,
)


class SavePeftModelCallback(TrainerCallback):
    """
    PEFT模型保存回调类
    
    在训练过程中的每个保存点，自动保存PEFT适配器模型，
    并清理不需要的大型检查点文件以节省磁盘空间。
    
    继承自TrainerCallback，实现on_save方法。
    """
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        保存检查点时的回调函数
        
        参数：
            args: TrainingArguments - 训练参数配置
            state: TrainerState - 训练器当前状态
            control: TrainerControl - 训练控制对象
            **kwargs: 其他参数，包含model对象
        
        返回：
            TrainerControl - 训练控制对象
        
        功能：
            1. 根据当前步数构建检查点目录路径
            2. 保存PEFT适配器模型到adapter_model子目录
            3. 删除完整模型的pytorch_model.bin文件以节省空间
            4. 删除DeepSpeed的分布式检查点目录
        """
        # 构建检查点目录路径
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        # 保存PEFT适配器模型
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # 删除完整模型的state_dict以节省磁盘空间
        # 因为PEFT只需要保存适配器权重，完整模型权重可以从原始预训练模型恢复
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        
        # 删除DeepSpeed的分布式state_dict以节省磁盘空间
        # DeepSpeed会保存每个进程的状态，这些文件通常很大
        deepspeed_path = os.path.join(checkpoint_folder, f"global_step{state.global_step}")
        if os.path.exists(deepspeed_path):
            shutil.rmtree(deepspeed_path, ignore_errors=True)
            
        return control