"""
文件说明：arguments.py - 命令行参数定义模块

本文件主要功能：
1. 定义ModelArguments数据类，包含模型相关的配置参数
2. 定义DataTrainingArguments数据类，包含数据相关的配置参数
3. 这些参数类与HfArgumentParser配合使用，支持命令行和JSON文件解析

参数类说明：
- ModelArguments: 模型路径、检查点、分词器等配置
- DataTrainingArguments: 训练数据、验证数据、序列长度等配置

作者：项目开发团队
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    模型参数配置类
    
    包含与模型、配置和分词器相关的参数，用于微调预训练模型。
    
    属性：
        model_name_or_path: str - 预训练模型路径或HuggingFace模型标识符
        checkpoint_path: str, 可选 - 用于恢复训练的检查点路径
        config_name: str, 可选 - 预训练配置名称或路径（如果与model_name不同）
        tokenizer_name: str, 可选 - 预训练分词器名称或路径（如果与model_name不同）
        cache_dir: str, 可选 - 存储从HuggingFace下载的预训练模型的目录
        use_fast_tokenizer: bool - 是否使用快速分词器（基于tokenizers库）
        model_revision: str - 使用的具体模型版本（分支名、标签名或commit id）
        use_auth_token: bool - 是否使用huggingface-cli login生成的token
        resize_position_embeddings: bool, 可选 - 是否自动调整位置编码大小
        quantization_bit: int, 可选 - 模型量化位数
    """
    model_name_or_path: str = field(
        metadata={"help": "预训练模型路径或HuggingFace模型标识符"}
    )
    checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "用于加载的检查点路径"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练配置名称或路径（如果与model_name不同）"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练分词器名称或路径（如果与model_name不同）"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "存储从HuggingFace下载的预训练模型的目录"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速分词器（基于tokenizers库）"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "使用的具体模型版本（分支名、标签名或commit id）"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "是否使用huggingface-cli login生成的token（用于私有模型）"
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "如果max_source_length超过模型的位置编码大小，是否自动调整位置编码"
            )
        },
    )
    quantization_bit: Optional[int] = field(
        default=None
    )


@dataclass
class DataTrainingArguments:
    """
    数据训练参数配置类
    
    包含用于模型训练和评估的数据相关参数。
    
    属性：
        lang: str, 可选 - 语言ID（用于摘要任务）
        dataset_name: str, 可选 - 数据集名称（通过datasets库使用）
        dataset_config_name: str, 可选 - 数据集配置名称
        prompt_column: str, 可选 - 数据集中包含输入文本的列名
        response_column: str, 可选 - 数据集中包含目标输出的列名
        history_column: str, 可选 - 数据集中包含对话历史的列名
        train_file: str, 可选 - 训练数据文件路径（jsonlines或csv格式）
        validation_file: str, 可选 - 验证数据文件路径
        test_file: str, 可选 - 测试数据文件路径
        overwrite_cache: bool - 是否覆盖缓存的训练和评估集
        preprocessing_num_workers: int, 可选 - 预处理使用的进程数
        max_source_length: int - 分词后输入序列的最大长度
        max_target_length: int - 分词后目标序列的最大长度
        val_max_target_length: int, 可选 - 验证目标序列的最大长度
        pad_to_max_length: bool - 是否将所有样本填充到最大长度
        max_train_samples: int, 可选 - 训练样本数量限制（用于调试）
        max_eval_samples: int, 可选 - 评估样本数量限制（用于调试）
        max_predict_samples: int, 可选 - 预测样本数量限制（用于调试）
        num_beams: int, 可选 - 束搜索的beam数量
        ignore_pad_token_for_loss: bool - 损失计算时是否忽略padding token
        source_prefix: str - 每个输入文本前添加的前缀（适用于T5模型）
        forced_bos_token: str, 可选 - 强制作为第一个生成token的标记
    """

    lang: Optional[str] = field(default=None, metadata={"help": "语言ID（用于摘要任务）"})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "数据集名称（通过datasets库使用）"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "数据集配置名称（通过datasets库使用）"}
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "数据集中包含完整输入文本的列名"},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "数据集中包含目标摘要/输出的列名"},
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={"help": "数据集中包含对话历史的列名"},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "训练数据文件路径（jsonlines或csv格式）"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "可选的验证数据文件路径，用于计算rouge等指标（jsonlines或csv格式）"
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "可选的测试数据文件路径，用于计算rouge等指标（jsonlines或csv格式）"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖缓存的训练和评估集"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使用的进程数"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "分词后输入序列的最大总长度。超过此长度的序列将被截断，"
                "较短的序列将被填充。"
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "分词后目标文本的最大总长度。超过此长度的序列将被截断，"
                "较短的序列将被填充。"
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "验证目标文本分词后的最大总长度。超过此长度的序列将被截断，"
                "较短的序列将被填充。如果未设置，将使用max_target_length的值。"
                "此参数也用于覆盖model.generate的max_length参数，"
                "在evaluate和predict时使用。"
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "是否将所有样本填充到模型的最大句子长度。"
                "如果为False，将在批处理时动态填充到批次中的最大长度。"
                "在GPU上更高效，但对TPU不友好。"
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "用于调试或快速训练，限制训练样本的数量。"
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "用于调试或快速训练，限制评估样本的数量。"
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "用于调试或快速训练，限制预测样本的数量。"
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "用于评估的beam数量。将传递给model.generate，"
                "在evaluate和predict时使用。"
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "损失计算时是否忽略与填充标签对应的token"
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "每个输入文本前添加的前缀（适用于T5模型）"}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "强制作为decoder_start_token_id之后第一个生成token的标记。"
                "对于多语言模型（如mBART）很有用，第一个生成的token需要是目标语言token。"
            )
        },
    )

    

    def __post_init__(self):
        """
        参数后处理验证
        
        验证内容：
        1. 必须提供dataset_name或train_file/validation_file/test_file之一
        2. 训练/验证文件必须是csv或json格式
        3. 如果未设置val_max_target_length，则使用max_target_length的值
        """
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("需要提供数据集名称或训练/验证/测试文件")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "train_file必须是csv或json文件"
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "validation_file必须是csv或json文件"
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

