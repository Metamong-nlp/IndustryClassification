import os
import wandb
import random
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import math
from tqdm.auto import tqdm, trange
import warnings
warnings.filterwarnings(action='ignore')

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from datasets import Dataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from utils.encoder import Encoder
from utils.preprocessor import Preprocessor
from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    LoggingArguments
)
from model import ThreeRoberta, ThreeRobertaWithLSTM

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

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)
    
    os.makedirs(data_args.save_path, exist_ok=True)

    if data_args.use_spaced:
        dataset = pd.read_csv('./input/spaced_train.csv', index_col=False)
    else :
        dataset = pd.read_csv('./input/train.csv', index_col=False)
        
    dset = Dataset.from_pandas(dataset)
    print(dset)
    
    with open('./data/labels/large_label_to_num.pickle', 'rb') as f:
        large_label = pickle.load(f)
    with open('./data/labels/medium_label_to_num.pickle', 'rb') as f:
        medium_label = pickle.load(f)
    with open('./data/labels/small_label_to_num.pickle', 'rb') as f:
        small_label = pickle.load(f)
            
    label_dict = [large_label, medium_label, small_label]

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    preprocessor = Preprocessor(tokenizer, label_dict)
    dset = dset.map(preprocessor, 
        batched=True, 
        num_proc=4,
        remove_columns=dset.column_names,
    )
    print(dset)

    encoder = Encoder(tokenizer, data_args.max_length)
    dset = dset.map(encoder, batched=True, remove_columns=dset.column_names)
    dset.set_format('torch')
    print(dset)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
    
    if training_args.do_eval:
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        
        for i, (train_idx, valid_idx) in enumerate(skf.split(dset, dset['labels'])):
            if i == 0 or i==1 or i==2 or i == 3 : continue
            print(f"######### Fold : {i} !!! ######### ")
            train_dataset = dset.select(train_idx.tolist())
            valid_dataset = dset.select(valid_idx.tolist())
            train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
            eval_dataloader = DataLoader(valid_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator)
        
            # # wandb
            load_dotenv(dotenv_path=logging_args.dotenv_path)
            WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
            wandb.login(key=WANDB_AUTH_KEY)
                
            wandb.init(
                entity="metamong",
                project=logging_args.project_name,
                group='Multi-label',
                name=training_args.run_name + f"_fold{i}"
            )
            wandb.config.update(training_args)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            num_update_steps_per_epoch = max(len(train_dataloader) // training_args.gradient_accumulation_steps, 1)
            num_training_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
            
            wandb.define_metric("train/global_step")
            wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
            
            if training_args.use_lstm :
                model = ThreeRobertaWithLSTM.from_pretrained(model_args.model_name_or_path, num_labels=225)
            else :
                model = ThreeRoberta.from_pretrained(model_args.model_name_or_path, num_labels=225)
            print(model)
            model = model.to(device)
            wandb.watch(model)
            
            optimizer = torch.optim.Adam(model.parameters(), lr = training_args.learning_rate, eps=training_args.adam_epsilon)
            
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=(
                    training_args.warmup_steps if training_args.warmup_steps > 0 else math.ceil(num_training_steps * training_args.warmup_ratio)
                ),
                num_training_steps=(len(train_dataloader) * training_args.num_train_epochs) // training_args.gradient_accumulation_steps,
            )
            
            scaler = GradScaler()

            progress_bar = trange(int(training_args.num_train_epochs * len(train_dataloader)), desc="***** Running Training *****")
            total_step = 0
            train_loss = 0
            best_acc = 0
            best_f1 = 0
            for epoch in range(int(training_args.num_train_epochs)):
                for step, train_batch in enumerate(train_dataloader):
                    model.train()
                    train_inputs= {k : v.to(device) for k, v in train_batch.items()}
                    with autocast():
                        outputs = model(**train_inputs)
                        loss = outputs.loss
                        logits = outputs.logits
                        loss = loss / training_args.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    
                    if step % training_args.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        
                    total_step += 1    
                    
                    train_loss += loss.item()    
                    if (step+1) % training_args.logging_steps == 0:
                        print(f" {total_step} step | loss = {train_loss / training_args.logging_steps}")
                        wandb.log({'train/loss' : train_loss / training_args.logging_steps, 'train/global_step' : total_step, 'train/learning_rate' : lr_scheduler.get_last_lr()[0]})
                        train_loss = 0
                    
                    # evaluation per eval_steps
                    if training_args.do_eval and total_step % training_args.eval_steps == 0:
                        eval_loss = 0
                        large_acc = 0
                        medium_acc = 0
                        small_acc = 0
                        large_f1 = 0
                        medium_f1 = 0
                        small_f1 = 0

                        model.eval()                
                        for step, eval_batch in enumerate(tqdm(eval_dataloader, desc="***** Running Evaluation *****", leave=False)):
                            with torch.no_grad():
                                eval_inputs={k : v.to(device) for k, v in eval_batch.items()}
                                outputs = model(**eval_inputs)
                                loss = outputs.loss
                                logits = outputs.logits                        
                                eval_loss += loss.item()       
                            large_preds = logits['large'].argmax(dim=-1).cpu()
                            medium_preds = logits['medium'].argmax(dim=-1).cpu()
                            samll_preds = logits['small'].argmax(dim=-1).cpu()
                            large_acc += accuracy_score(eval_batch['large_labels'], large_preds)
                            medium_acc += accuracy_score(eval_batch['medium_labels'], medium_preds)
                            small_acc += accuracy_score(eval_batch['labels'], samll_preds)
                            large_f1 +=  f1_score(eval_batch['large_labels'], large_preds, average='macro')
                            medium_f1 += f1_score(eval_batch['medium_labels'], medium_preds, average='macro')
                            small_f1 += f1_score(eval_batch['labels'], samll_preds, average='macro')         
                        eval_dict = {
                            'eval/large_acc' : large_acc / len(eval_dataloader),
                            'eval/large_f1' : large_f1 / len(eval_dataloader),
                            'eval/medium_acc' : medium_acc / len(eval_dataloader),
                            'eval/medium_f1' : medium_f1 / len(eval_dataloader),
                            'eval/accuracy' : small_acc / len(eval_dataloader),                        
                            'eval/f1' : small_f1 / len(eval_dataloader),
                            'eval/loss' : eval_loss / len(eval_dataloader),
                            'train/global_step' : total_step
                        }
                        print(f" {total_step} step | ", eval_dict)
                        wandb.log(eval_dict)
                        if best_acc < small_acc / len(eval_dataloader) :
                            best_acc = small_acc / len(eval_dataloader)
                            # Saving
                            file_path = os.path.join(data_args.save_path, f'fold{i}')
                            model.save_pretrained(file_path)
                            tokenizer.save_pretrained(file_path)
                        print(f" ** Best score is Update ** | Best acc : {best_acc}, Best f1 : {best_f1}")
            wandb.finish()
            
    
    
    
if __name__ == '__main__':
    main()