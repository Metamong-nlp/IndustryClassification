import os
import wandb
import torch
import random
import pickle
import importlib
import numpy as np
from dotenv import load_dotenv
from datasets import load_metric, load_dataset
from utils.encoder import Encoder
from utils.preprocessor import Preprocessor
from sklearn.model_selection import StratifiedKFold
from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    LoggingArguments
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    Trainer,
)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

def compute_metrics(EvalPrediction):
    preds, labels = EvalPrediction
    preds = np.argmax(preds, axis=1)

    f1_metric = load_metric('f1')    
    f1 = f1_metric.compute(predictions = preds, references = labels, average="macro")

    acc_metric = load_metric('accuracy')
    acc = acc_metric.compute(predictions = preds, references = labels)
    acc.update(f1)
    return acc

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    # -- Loading datasets
    dset = load_dataset('sh110495/IndustryClassification')
    dset = dset['train'].shuffle(training_args.seed)
    print(dset)

    # -- Preprocessing datasets
    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)

    with open('data/labels/label_to_index.pickle', 'rb') as f:
        label_dict = pickle.load(f)

    preprocessor = Preprocessor(tokenizer, label_dict, train_flag=True)
    dset = dset.map(preprocessor, batched=True, num_proc=4,remove_columns=dset.column_names)
    print(dset)

    # -- Tokenizing & Encoding
    encoder = Encoder(tokenizer, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
    print(dset)

    config = AutoConfig.from_pretrained(model_args.PLM)
    config.num_labels = 225
    
    # -- Model Class
    if training_args.model_type == 'base' :
        model_class = AutoModelForSequenceClassification
    else :
        model_type_str = 'model'
        model_lib = importlib.import_module(model_type_str)

        if training_args.model_type == 'average' :
            model_class = getattr(model_lib, 'RobertaWeighAverage')
        elif training_args.model_type == 'lstm' :
            model_class = getattr(model_lib, 'RobertaLSTM')
        elif training_args.model_type == 'cnn' :
            model_class = getattr(model_lib, 'RobertaCNN')
        elif training_args.model_type == 'rbert' :
            model_class = getattr(model_lib, 'RobertaRBERT')
        else :
            raise NotImplementedError

    # -- Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    if training_args.do_train:
        skf = StratifiedKFold(n_splits=training_args.fold_size, shuffle=True)

        for i, (train_idx, valid_idx) in enumerate(skf.split(dset, dset['labels'])):
            model = model_class.from_pretrained(model_args.PLM, config=config)
            train_dataset = dset.select(train_idx.tolist())
            valid_dataset = dset.select(valid_idx.tolist())
                        
            # -- Wandb
            load_dotenv(dotenv_path=logging_args.dotenv_path)
            WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
            wandb.login(key=WANDB_AUTH_KEY)

            group_name = training_args.model_type + '-' + str(training_args.fold_size) + '-fold-training'
            name = f"EP:{training_args.num_train_epochs}\
                _LR:{training_args.learning_rate}\
                _BS:{training_args.per_device_train_batch_size}\
                _WR:{training_args.warmup_ratio}\
                _WD:{training_args.weight_decay}"
        
            wandb.init(
                entity="sangha0411",
                project=logging_args.project_name,
                group=group_name,
                name=name
            )
            wandb.config.update(training_args)

            if training_args.use_rdrop :
                trainer_lib = importlib.import_module('trainer')
                trainer_class = getattr(trainer_lib, 'RdropTrainer')
            else :
                trainer_class = Trainer

            trainer = trainer_class(                    # the instantiated 🤗 Transformers model to be trained
                model=model,                            # model
                args=training_args,                     # training arguments, defined above
                train_dataset=train_dataset,            # training dataset
                eval_dataset=valid_dataset,             # evaluation dataset
                data_collator=data_collator,            # collator
                tokenizer=tokenizer,                    # tokenizer
                compute_metrics=compute_metrics,        # define metrics function
            )

            # -- Training
            trainer.train()
            save_path = os.path.join(model_args.save_path, f'fold{i}')
            
            if training_args.do_eval:
                trainer.evaluate()
            trainer.save_model(save_path)
            
            wandb.finish()
  
if __name__ == '__main__':
    main()