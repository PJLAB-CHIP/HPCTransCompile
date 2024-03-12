from dataclasses import dataclass,field
from typing import Union,Optional

@dataclass
class ModelArguments:
    model_name_or_path: Union[str,None] = field(
        default='meta-llama/Llama-2-7b'
    )
    max_seq_len: int = field(
        default=6144
    )
    source_max_len: int = field(
        default=2048
    )
    target_max_len: int = field(
        default=4096
    )

@dataclass
class DataArguments:
    dataset_name_or_path: str = field(
        default='alpaca',
        metadata={'help':'dataset path or name'}
    )
    dataset_format: Optional[str] = field(
        default='input-output',
        metadata={'help':'alpaca|'}
    )


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default='./output',
        metadata={'help':'training output directory, include checkpoints and final results'}
    )

    train_size: float = field(
        default=0.8,
        metadata={'help':'train dataset size'}
    )

    num_epochs: int = field(
        default=1,
        metadata={'help':'training epochs'}
    )

    batch_size: int = field(
        default=1,
        metadata={'help':'The training batch size per GPU. Increase for better speed.'}
    )

    learning_rate: int = field(
        default=0.0001,
        metadata={'help':'learning rate'}
    )

    device_map: str = field(
        default='cuda',
        metadata={'help':'which device is used'}
    )

    use_lora: bool = field(
        default=False,
        metadata={'help':'whether to use lora'}
    )

    full_finetune: bool = field(
        default=False,
        metadata={'help':'whether to perform full fine-tuning'}
    )

    bits: int = field(
        default=4,
        metadata={'help':'the type of model data'}
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )

    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )

    group_by_length: str = field(
        default=False,
        metadata={'help':'whether group dataset by length'}
    )

    use_wandb: bool = field(
        default=False,
        metadata={'help':'whether to use wandb'}
    )

    wandb_run_name: str = field(
        default='test',
        metadata={'help':'wandb project name'}
    )

    eval_steps: int = field(
        default='200',
        metadata={'help':'how many steps to adjust the hyperparameter'}
    )

    save_steps: int = field(
        default='200',
        metadata={'help':'how many steps to save the model'}
    )