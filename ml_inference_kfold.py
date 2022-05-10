import os
import random
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
)

from utils.encoder import Encoder
from utils.preprocessor import Preprocessor
from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments,
)
from model import ThreeRoberta, ThreeRobertaWithLSTM

with open('./data/map/large_to_medium.pickle', 'rb') as f:
    LARGE_TO_MEDIUM = pickle.load(f)
with open('./data/map/medium_to_small.pickle', 'rb') as f:
    MEDIUM_TO_SMALL = pickle.load(f)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)
    
    if data_args.use_spaced:
        test_df = pd.read_csv('./input/spaced_test.csv', index_col=False)
    else :
        test_df = pd.read_csv('./input/test.csv', index_col=False)
        
    dset = Dataset.from_pandas(test_df)
    print(dset)

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    preprocessor = Preprocessor(tokenizer, mode_test=True)
    dset = dset.map(preprocessor, 
        batched=True, 
        num_proc=4,
        remove_columns=dset.column_names
    )
    print(dset)

    encoder = Encoder(tokenizer, data_args.max_length, mode_test=True)
    test_dataset = dset.map(encoder, batched=True, remove_columns=dset.column_names)
    print(test_dataset)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
    
    test_dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    soft_large = 0
    soft_medium = 0
    soft_small = 0
    for i in range(5):
        kfold_path = os.path.join(model_args.model_name_or_path, f'_fold{i}')
        if 'lstm' in model_args.model_name_or_path.lower():
            model = ThreeRobertaWithLSTM.from_pretrained(kfold_path)
        else :
            model = ThreeRoberta.from_pretrained(kfold_path)
        model = model.to(device)

        # evaluation per eval_steps
        large_preds = None
        medium_preds = None
        small_preds = None
        if training_args.do_predict:
            model.eval()                
            for step, test_batch in enumerate(tqdm(test_dataloader, desc="***** Running Prediction *****")):
                with torch.no_grad():
                    test_inputs={k : v.to(device) for k, v in test_batch.items()}
                    outputs = model(**test_inputs)
                    logits = outputs.logits
                large_preds = logits['large'].detach().cpu() if large_preds is None else torch.cat((large_preds, logits['large'].detach().cpu()), 0)
                medium_preds = logits['medium'].detach().cpu() if medium_preds is None else torch.cat((medium_preds, logits['medium'].detach().cpu()), dim=0)
                small_preds = logits['small'].detach().cpu() if small_preds is None else torch.cat((small_preds, logits['small'].detach().cpu()), dim=0)

            folder_name = os.path.join('./output', model_args.model_name_or_path.split('/')[-1])
            kfold_output_path = os.path.join(folder_name, f'fold{i}')
            os.makedirs(kfold_output_path, exist_ok=True)
            
            with open(os.path.join(kfold_output_path, 'large_logit.pickle'), 'wb') as f:
                pickle.dump(large_preds, f)
            with open(os.path.join(kfold_output_path, 'medium_logit.pickle'), 'wb') as f:
                pickle.dump(medium_preds, f)
            with open(os.path.join(kfold_output_path, 'small_logit.pickle'), 'wb') as f:
                pickle.dump(small_preds, f)
            
            # For soft voting
            soft_large += F.softmax(large_preds, dim=1)
            soft_medium += F.softmax(medium_preds, dim=1)
            soft_small += F.softmax(small_preds, dim=1)
    
    with open(os.path.join(folder_name, 'soft_large_logit.pickle'), 'wb') as f:
        pickle.dump(large_preds, f)
    with open(os.path.join(folder_name, 'soft_medium_logit.pickle'), 'wb') as f:
        pickle.dump(medium_preds, f)
    with open(os.path.join(folder_name, 'soft_small_logit.pickle'), 'wb') as f:
        pickle.dump(small_preds, f)

def mapping_function(example):
    for k, v in LARGE_TO_MEDIUM.items():
        if example in v:
            return k
    
if __name__ == "__main__" :
    main()