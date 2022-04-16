from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class ModelArguments : 
    PLM: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    save_path: str = field(
        default="./checkpoints",
        metadata={
            "help": "Path to save checkpoint from fine tune model"
        },
    )
    
@dataclass
class DataTrainingArguments:
    max_length: int = field(
        default=128,
        metadata={
            "help": "Max length of input sequence"
        },
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={
            "help": "The number of preprocessing workers"
        }
    )
    
@dataclass
class MyTrainingArguments(TrainingArguments):
    report_to: Optional[str] = field(
        default='wandb',
    )
    fold_size : Optional[int] = field(
        default=5,
        metadata={"help" : "The number of folds"}
    )
    use_lstm: bool = field(
        default=False,
        metadata={
            "help" : "using lstm model"
        }
    )
    use_noam: bool = field(
        default=False,
        metadata={
            "help" : "using noam scheduler"
        }
    )
    use_rdrop: bool = field(
        default=False,
        metadata={
            "help" : "rdop trinaer"
        }
    )
    model_type: str = field(
        default='base',
        metadata={
            'help' : 'model type'
        }
    )

@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )
    project_name: Optional[str] = field(
        default="Industry Classification",
        metadata={"help": "project name"},
    )

@dataclass
class InferenceArguments:
    dir_path : Optional[str] = field(
        default='./results',
        metadata={"help" : "The csv file for test dataset"}
    )