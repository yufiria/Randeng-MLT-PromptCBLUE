#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
文件说明：zero_to_fp32.py - DeepSpeed ZeRO检查点转换工具

本文件主要功能：
1. 将DeepSpeed ZeRO Stage 2/3的分布式检查点转换为标准PyTorch FP32模型
2. 支持从多个GPU分片恢复完整模型权重
3. 转换后的模型可以在不依赖DeepSpeed的环境中使用

使用方法：
    python zero_to_fp32.py <checkpoint_dir> <output_file>
    例如：python zero_to_fp32.py ./checkpoint pytorch_model.bin

支持的ZeRO阶段：
- ZeRO Stage 2: 优化器状态分片
- ZeRO Stage 3: 参数分片

注意事项：
- 需要安装DeepSpeed库（用于解析检查点格式）
- 转换后的模型权重会加载到CPU内存中
- 对于大模型，确保有足够的CPU内存

作者：Microsoft DeepSpeed Team
"""

# 使用示例: python zero_to_fp32.py . pytorch_model.bin

import argparse
import torch
import glob
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

# 虽然此脚本不使用DeepSpeed来恢复数据，但由于检查点是使用DeepSpeed数据结构序列化的，
# 因此需要在当前Python环境中安装DeepSpeed
from deepspeed.utils import logger
from deepspeed.checkpoint.constants import (DS_VERSION, OPTIMIZER_STATE_DICT, SINGLE_PARTITION_OF_FP32_GROUPS,
                                            FP32_FLAT_GROUPS, ZERO_STAGE, PARTITION_COUNT, PARAM_SHAPES, BUFFER_NAMES,
                                            FROZEN_PARAM_SHAPES, FROZEN_PARAM_FRAGMENTS)


@dataclass
class zero_model_state:
    """
    ZeRO模型状态数据类
    
    用于存储从检查点中解析出的模型状态信息
    
    属性：
        buffers: dict - 模型缓冲区（如BatchNorm的running_mean等）
        param_shapes: dict - 参数形状信息
        shared_params: list - 共享参数列表
        ds_version: int - DeepSpeed版本号
        frozen_param_shapes: dict - 冻结参数的形状信息
        frozen_param_fragments: dict - 冻结参数的片段数据
    """
    buffers: dict()
    param_shapes: dict()
    shared_params: list
    ds_version: int
    frozen_param_shapes: dict()
    frozen_param_fragments: dict()


# 调试模式开关
debug = 0

# 将模型加载到CPU
device = torch.device('cpu')


def atoi(text):
    """
    将文本转换为整数（如果是数字）
    
    参数：
        text: str - 输入文本
    
    返回：
        int或str - 如果是数字则返回整数，否则返回原字符串
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    用于自然排序的键函数
    
    实现人类友好的排序方式，例如：
    file1, file2, file10 而不是 file1, file10, file2
    
    参数：
        text: str - 待排序的文本
    
    返回：
        list - 用于排序比较的键列表
    
    参考：http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_model_state_file(checkpoint_dir, zero_stage):
    """
    获取模型状态文件路径
    
    参数：
        checkpoint_dir: str - 检查点目录路径
        zero_stage: int - ZeRO阶段（2或3）
    
    返回：
        str - 模型状态文件的完整路径
    
    异常：
        FileNotFoundError - 目录或文件不存在时抛出
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")

    # 根据ZeRO阶段确定模型状态文件名
    if zero_stage == 2:
        file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    elif zero_stage == 3:
        file = os.path.join(checkpoint_dir, "zero_pp_rank_0_mp_rank_00_model_states.pt")

    if not os.path.exists(file):
        raise FileNotFoundError(f"can't find model states file at '{file}'")

    return file


def get_checkpoint_files(checkpoint_dir, glob_pattern):
    """
    获取匹配指定模式的检查点文件列表
    
    参数：
        checkpoint_dir: str - 检查点目录路径
        glob_pattern: str - 文件匹配模式
    
    返回：
        list - 按自然顺序排序的检查点文件列表
    
    异常：
        FileNotFoundError - 未找到匹配文件时抛出
    """
    # TODO: 需要测试此简单的glob规则是否适用于多节点设置
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files


def get_optim_files(checkpoint_dir):
    """
    获取优化器状态文件列表
    
    参数：
        checkpoint_dir: str - 检查点目录路径
    
    返回：
        list - 优化器状态文件路径列表
    """
    return get_checkpoint_files(checkpoint_dir, "*_optim_states.pt")


def get_model_state_files(checkpoint_dir):
    """
    获取模型状态文件列表
    
    参数：
        checkpoint_dir: str - 检查点目录路径
    
    返回：
        list - 模型状态文件路径列表
    """
    return get_checkpoint_files(checkpoint_dir, "*_model_states.pt")


def parse_model_states(files):
    """
    解析模型状态文件
    
    参数：
        files: list - 模型状态文件路径列表
    
    返回：
        list[zero_model_state] - 解析后的模型状态对象列表
    
    功能：
        1. 加载模型状态字典
        2. 提取缓冲区数据并转换为FP32
        3. 收集参数形状信息
        4. 处理共享参数和冻结参数
    """
    zero_model_states = []
    for file in files:
        state_dict = torch.load(file, map_location=device)

        if BUFFER_NAMES not in state_dict:
            raise ValueError(f"{file} is not a model state checkpoint")
        buffer_names = state_dict[BUFFER_NAMES]
        if debug:
            print("Found buffers:", buffer_names)

        # 恢复缓冲区数据，如果是FP16则转换为FP32
        buffers = {k: v.float() for k, v in state_dict["module"].items() if k in buffer_names}
        param_shapes = state_dict[PARAM_SHAPES]

        # 收集包含在param_shapes中的参数名称
        param_names = []
        for s in param_shapes:
            for name in s.keys():
                param_names.append(name)

        # 处理冻结参数
        frozen_param_shapes = state_dict.get(FROZEN_PARAM_SHAPES, None)
        if frozen_param_shapes is not None:
            if debug:
                print(f"Found frozen_param_shapes: {frozen_param_shapes}")
            param_names += list(frozen_param_shapes.keys())

        # 处理共享参数
        shared_params = [[k, v] for k, v in state_dict["shared_params"].items()]

        ds_version = state_dict.get(DS_VERSION, None)

        frozen_param_fragments = state_dict.get(FROZEN_PARAM_FRAGMENTS, None)

        z_model_state = zero_model_state(buffers=buffers,
                                         param_shapes=param_shapes,
                                         shared_params=shared_params,
                                         ds_version=ds_version,
                                         frozen_param_shapes=frozen_param_shapes,
                                         frozen_param_fragments=frozen_param_fragments)
        zero_model_states.append(z_model_state)

    return zero_model_states


def parse_optim_states(files, ds_checkpoint_dir):
    """
    解析优化器状态文件
    
    参数：
        files: list - 优化器状态文件路径列表
        ds_checkpoint_dir: str - DeepSpeed检查点目录路径
    
    返回：
        tuple - (zero_stage, world_size, fp32_flat_groups)
            - zero_stage: int - ZeRO阶段（2或3）
            - world_size: int - 分布式训练的世界大小
            - fp32_flat_groups: list - FP32扁平化参数组
    
    异常：
        ValueError - 检查点格式错误或文件数量不匹配时抛出
    """
    total_files = len(files)
    state_dicts = []
    for f in files:
        state_dicts.append(torch.load(f, map_location=device))

    if not ZERO_STAGE in state_dicts[0][OPTIMIZER_STATE_DICT]:
        raise ValueError(f"{files[0]} is not a zero checkpoint")
    zero_stage = state_dicts[0][OPTIMIZER_STATE_DICT][ZERO_STAGE]
    world_size = state_dicts[0][OPTIMIZER_STATE_DICT][PARTITION_COUNT]

    # 对于ZeRO-2，每个参数组可能有不同的partition_count
    # 专家参数的数据并行度可能与非专家参数不同
    # 使用最大的partition_count作为dp world_size
    if type(world_size) is list:
        world_size = max(world_size)

    if world_size != total_files:
        raise ValueError(
            f"Expected {world_size} of '*_optim_states.pt' under '{ds_checkpoint_dir}' but found {total_files} files. "
            "Possibly due to an overwrite of an old checkpoint, or a checkpoint didn't get saved by one or more processes."
        )

    # 不同阶段的参数组命名不同
    if zero_stage == 2:
        fp32_groups_key = SINGLE_PARTITION_OF_FP32_GROUPS
    elif zero_stage == 3:
        fp32_groups_key = FP32_FLAT_GROUPS
    else:
        raise ValueError(f"unknown zero stage {zero_stage}")

    if zero_stage == 2:
        fp32_flat_groups = [state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key] for i in range(len(state_dicts))]
    elif zero_stage == 3:
        # 如果有多个参数组，会有多个扁平化张量
        # 为简化处理，将它们合并为单个张量
        fp32_flat_groups = [
            torch.cat(state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key], 0) for i in range(len(state_dicts))
        ]

    return zero_stage, world_size, fp32_flat_groups


def _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir):
    """
    从DeepSpeed检查点恢复FP32状态字典（内部函数）
    
    参数：
        ds_checkpoint_dir: str - DeepSpeed检查点目录路径
    
    返回：
        OrderedDict - 重建的FP32模型状态字典
    
    功能：
        1. 解析优化器状态文件获取ZeRO阶段和分片信息
        2. 解析模型状态文件获取参数形状
        3. 根据ZeRO阶段调用相应的恢复函数
    """
    print(f"Processing zero checkpoint '{ds_checkpoint_dir}'")

    optim_files = get_optim_files(ds_checkpoint_dir)
    zero_stage, world_size, fp32_flat_groups = parse_optim_states(optim_files, ds_checkpoint_dir)
    print(f"Detected checkpoint of type zero stage {zero_stage}, world_size: {world_size}")

    model_files = get_model_state_files(ds_checkpoint_dir)

    zero_model_states = parse_model_states(model_files)
    print(f'Parsing checkpoint created by deepspeed=={zero_model_states[0].ds_version}')

    if zero_stage == 2:
        return _get_fp32_state_dict_from_zero2_checkpoint(world_size, fp32_flat_groups, zero_model_states)
    elif zero_stage == 3:
        return _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups, zero_model_states)


def _zero2_merge_frozen_params(state_dict, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    frozen_param_shapes = zero_model_states[0].frozen_param_shapes
    frozen_param_fragments = zero_model_states[0].frozen_param_fragments

    if debug:
        num_elem = sum(s.numel() for s in frozen_param_shapes.values())
        print(f'rank 0: {FROZEN_PARAM_SHAPES}.numel = {num_elem}')

        wanted_params = len(frozen_param_shapes)
        wanted_numel = sum(s.numel() for s in frozen_param_shapes.values())
        avail_numel = sum([p.numel() for p in frozen_param_fragments.values()])
        print(f'Frozen params: Have {avail_numel} numels to process.')
        print(f'Frozen params: Need {wanted_numel} numels in {wanted_params} params')

    total_params = 0
    total_numel = 0
    for name, shape in frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        state_dict[name] = frozen_param_fragments[name]

        if debug:
            print(f"{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} ")

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


def _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states):
    param_shapes = zero_model_states[0].param_shapes

    # Reconstruction protocol:
    #
    # XXX: document this

    if debug:
        for i in range(world_size):
            for j in range(len(fp32_flat_groups[0])):
                print(f"{FP32_FLAT_GROUPS}[{i}][{j}].shape={fp32_flat_groups[i][j].shape}")

    # XXX: memory usage doubles here (zero2)
    num_param_groups = len(fp32_flat_groups[0])
    merged_single_partition_of_fp32_groups = []
    for i in range(num_param_groups):
        merged_partitions = [sd[i] for sd in fp32_flat_groups]
        full_single_fp32_vector = torch.cat(merged_partitions, 0)
        merged_single_partition_of_fp32_groups.append(full_single_fp32_vector)
    avail_numel = sum(
        [full_single_fp32_vector.numel() for full_single_fp32_vector in merged_single_partition_of_fp32_groups])

    if debug:
        wanted_params = sum([len(shapes) for shapes in param_shapes])
        wanted_numel = sum([sum(shape.numel() for shape in shapes.values()) for shapes in param_shapes])
        # not asserting if there is a mismatch due to possible padding
        print(f"Have {avail_numel} numels to process.")
        print(f"Need {wanted_numel} numels in {wanted_params} params.")

    # params
    # XXX: for huge models that can't fit into the host's RAM we will have to recode this to support
    # out-of-core computing solution
    total_numel = 0
    total_params = 0
    for shapes, full_single_fp32_vector in zip(param_shapes, merged_single_partition_of_fp32_groups):
        offset = 0
        avail_numel = full_single_fp32_vector.numel()
        for name, shape in shapes.items():

            unpartitioned_numel = shape.numel()
            total_numel += unpartitioned_numel
            total_params += 1

            if debug:
                print(f"{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} ")
            state_dict[name] = full_single_fp32_vector.narrow(0, offset, unpartitioned_numel).view(shape)
            offset += unpartitioned_numel

        # Z2 started to align to 2*world_size to improve nccl performance. Therefore both offset and
        # avail_numel can differ by anywhere between 0..2*world_size. Due to two unrelated complex
        # paddings performed in the code it's almost impossible to predict the exact numbers w/o the
        # live optimizer object, so we are checking that the numbers are within the right range
        align_to = 2 * world_size

        def zero2_align(x):
            return align_to * math.ceil(x / align_to)

        if debug:
            print(f"original offset={offset}, avail_numel={avail_numel}")

        offset = zero2_align(offset)
        avail_numel = zero2_align(avail_numel)

        if debug:
            print(f"aligned  offset={offset}, avail_numel={avail_numel}")

        # Sanity check
        if offset != avail_numel:
            raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(f"Reconstructed fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero2_checkpoint(world_size, fp32_flat_groups, zero_model_states):
    state_dict = OrderedDict()

    # buffers
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print(f"added {len(buffers)} buffers")

    _zero2_merge_frozen_params(state_dict, zero_model_states)

    _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)

    # recover shared parameters
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def zero3_partitioned_param_info(unpartitioned_numel, world_size):
    remainder = unpartitioned_numel % world_size
    padding_numel = (world_size - remainder) if remainder else 0
    partitioned_numel = math.ceil(unpartitioned_numel / world_size)
    return partitioned_numel, padding_numel


def _zero3_merge_frozen_params(state_dict, world_size, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    if debug:
        for i in range(world_size):
            num_elem = sum(s.numel() for s in zero_model_states[i].frozen_param_fragments.values())
            print(f'rank {i}: {FROZEN_PARAM_SHAPES}.numel = {num_elem}')

        frozen_param_shapes = zero_model_states[0].frozen_param_shapes
        wanted_params = len(frozen_param_shapes)
        wanted_numel = sum(s.numel() for s in frozen_param_shapes.values())
        avail_numel = sum([p.numel() for p in zero_model_states[0].frozen_param_fragments.values()]) * world_size
        print(f'Frozen params: Have {avail_numel} numels to process.')
        print(f'Frozen params: Need {wanted_numel} numels in {wanted_params} params')

    total_params = 0
    total_numel = 0
    for name, shape in zero_model_states[0].frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        param_frags = tuple(model_state.frozen_param_fragments[name] for model_state in zero_model_states)
        state_dict[name] = torch.cat(param_frags, 0).narrow(0, 0, unpartitioned_numel).view(shape)

        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print(
                f"Frozen params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


def _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states):
    param_shapes = zero_model_states[0].param_shapes
    avail_numel = fp32_flat_groups[0].numel() * world_size
    # Reconstruction protocol: For zero3 we need to zip the partitions together at boundary of each
    # param, re-consolidating each param, while dealing with padding if any

    # merge list of dicts, preserving order
    param_shapes = {k: v for d in param_shapes for k, v in d.items()}

    if debug:
        for i in range(world_size):
            print(f"{FP32_FLAT_GROUPS}[{i}].shape={fp32_flat_groups[i].shape}")

        wanted_params = len(param_shapes)
        wanted_numel = sum(shape.numel() for shape in param_shapes.values())
        # not asserting if there is a mismatch due to possible padding
        avail_numel = fp32_flat_groups[0].numel() * world_size
        print(f"Trainable params: Have {avail_numel} numels to process.")
        print(f"Trainable params: Need {wanted_numel} numels in {wanted_params} params.")

    # params
    # XXX: for huge models that can't fit into the host's RAM we will have to recode this to support
    # out-of-core computing solution
    offset = 0
    total_numel = 0
    total_params = 0
    for name, shape in param_shapes.items():

        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        total_params += 1

        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print(
                f"Trainable params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

        # XXX: memory usage doubles here
        state_dict[name] = torch.cat(
            tuple(fp32_flat_groups[i].narrow(0, offset, partitioned_numel) for i in range(world_size)),
            0).narrow(0, 0, unpartitioned_numel).view(shape)
        offset += partitioned_numel

    offset *= world_size

    # Sanity check
    if offset != avail_numel:
        raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(f"Reconstructed Trainable fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups, zero_model_states):
    state_dict = OrderedDict()

    # buffers
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print(f"added {len(buffers)} buffers")

    _zero3_merge_frozen_params(state_dict, world_size, zero_model_states)

    _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)

    # recover shared parameters
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=None):
    """
    从ZeRO 2或3检查点转换为单个FP32合并状态字典
    
    该函数将分布式训练保存的检查点转换为可以使用load_state_dict()加载的
    标准PyTorch状态字典，适用于不依赖DeepSpeed的训练或与他人共享模型。
    
    参数：
        checkpoint_dir: str - 检查点文件夹路径
        tag: str, 可选 - 检查点标签（如'global_step14'）
                        如果未提供，将尝试从'latest'文件读取
    
    返回：
        dict - PyTorch状态字典
    
    注意：
        如果应用程序没有足够的CPU内存，此方法可能无法工作。
        可以使用离线方式，通过检查点保存的zero_to_fp32.py脚本进行转换。
    
    使用示例：
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        # 训练并保存检查点后
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)  # 已在CPU上
        model = model.cpu()  # 移动到CPU
        model.load_state_dict(state_dict)
        # 提交到模型中心或保存模型
    
    警告：
        使用此函数后，模型将无法在同一应用程序的DeepSpeed上下文中继续使用。
        需要重新初始化DeepSpeed引擎。
        如果希望自动完成所有操作，请使用load_state_dict_from_zero_checkpoint。
    """
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(ds_checkpoint_dir):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")

    return _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir)


def convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, output_file, tag=None):
    """
    将ZeRO 2或3检查点转换为单个FP32合并状态字典文件
    
    转换后的文件可以使用torch.load()加载，然后用load_state_dict()加载到模型中，
    无需DeepSpeed即可进行训练。
    
    参数：
        checkpoint_dir: str - 检查点文件夹路径（包含tag文件夹的目录，如包含global_step14的目录）
        output_file: str - PyTorch FP32状态字典输出文件路径（如path/pytorch_model.bin）
        tag: str, 可选 - 检查点标签。如果未提供，将尝试从检查点文件夹中的'latest'文件加载
    
    功能：
        1. 从检查点恢复FP32状态字典
        2. 将状态字典保存到指定的输出文件
    """
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    print(f"Saving fp32 state dict to {output_file}")
    torch.save(state_dict, output_file)


def load_state_dict_from_zero_checkpoint(model, checkpoint_dir, tag=None):
    """
    从ZeRO检查点加载状态字典到模型
    
    完整流程：
    1. 将模型移动到CPU
    2. 将ZeRO 2或3检查点转换为单个FP32合并状态字典
    3. 将状态字典加载到模型中
    
    参数：
        model: nn.Module - 要更新的模型对象
        checkpoint_dir: str - 检查点文件夹路径（包含tag文件夹的目录）
        tag: str, 可选 - 检查点标签。如果未提供，将尝试从'latest'文件加载
    
    返回：
        nn.Module - 加载了权重的模型
    
    注意：
        调用此函数前请确保有足够的CPU内存。
        如果内存不足，请使用检查点文件夹中的zero_to_fp32.py工具进行离线转换。
    
    使用示例：
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
        # 提交到模型中心或保存模型
    
    警告：
        运行此函数后，模型将无法在同一应用程序的DeepSpeed上下文中继续使用。
        需要重新初始化DeepSpeed引擎，因为load_state_dict()会移除所有DeepSpeed相关功能。
    """
    logger.info(f"Extracting fp32 weights")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)

    logger.info(f"Overwriting model with fp32 weights")
    model = model.cpu()
    model.load_state_dict(state_dict, strict=False)

    return model


# =========================================
# 命令行入口
# 使用方法：python zero_to_fp32.py <checkpoint_dir> <output_file>
# =========================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir",
                        type=str,
                        help="检查点文件夹路径，例如：path/checkpoint-12")
    parser.add_argument(
        "output_file",
        type=str,
        help="PyTorch FP32状态字典输出文件路径，例如：path/checkpoint-12/pytorch_model.bin")
    parser.add_argument("-d", "--debug", action='store_true', help="启用调试模式")
    args = parser.parse_args()

    debug = args.debug

    convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint_dir, args.output_file)
