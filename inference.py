import os
import torch
import pickle
import random
import importlib
import collections
import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from functools import partial
from utils.encoder import Encoder
from utils.preprocessor import Preprocessor

from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    InferenceArguments
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    Trainer
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, InferenceArguments)
    )
    model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()

    # -- Loading datasets
    dset = load_dataset('sh110495/IndustryClassification')
    dset = dset['test'].shuffle(training_args.seed)
    print(dset)

    # -- Preprocessing datasets
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    preprocessor = Preprocessor(tokenizer, label_dict=None, train_flag=False)
    dset = dset.map(preprocessor, batched=True, num_proc=4, remove_columns=dset.column_names)
    print(dset)

    # -- Tokenizing & Encoding datasets
    encoder = Encoder(tokenizer, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
    print(dset)
    
    # -- Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
    
    # -- Model Class
    model_lib = importlib.import_module('model')
    if training_args.model_type == 'average' :
        model_class = getattr(model_lib, 'RobertaWeighAverage')
    elif training_args.model_type == 'lstm' :
        model_class = getattr(model_lib, 'RobertaLSTM')
    elif training_args.model_type == 'cnn' :
        model_class = getattr(model_lib, 'RobertaCNN')
    elif training_args.model_type == 'rbert' :
        model_class = getattr(model_lib, 'RobertaRBERT')
    elif training_args.model_type == 'arcface' :
        model_class = getattr(model_lib, 'RobertaArcface')
    else :
        model_class = AutoModelForSequenceClassification

    pred_ids = []
    pred_probs = []
    for i in tqdm(range(training_args.fold_size)) :
        PLM = os.path.join(model_args.PLM, f'fold{i}')

        # -- Config & Model
        config = AutoConfig.from_pretrained(PLM)
        model = model_class.from_pretrained(PLM, config=config)

        trainer = Trainer(                       # the instantiated 🤗 Transformers model to be trained
            model=model,                         # trained model
            args=training_args,                  # training arguments, defined above
            data_collator=data_collator,         # collator
        )

        # -- Inference
        outputs = trainer.predict(dset)
        pred_probs.append(outputs[0])
        pred_ids.append(outputs[0].argmax(axis=1))

    with open('data/labels/index_to_label.pickle', 'rb') as f:
        index_to_label = pickle.load(f)

    with open('./data/map/large_to_medium.pickle', 'rb') as f:
        large_groupset = pickle.load(f)

    def mapping_function(example, groupset):
        for k, v in groupset.items():
            if example in v:
                return k

    map_fn = partial(mapping_function, groupset=large_groupset)

    # -- Soft voting
    print('Soft Voting')
    soft_prediction = np.sum(pred_probs, axis=0)
    soft_submission = pd.read_csv('./data/test.csv', index_col=False)
    soft_submission.digit_3 = list(soft_prediction.argmax(axis=1))
    soft_submission.digit_3 = soft_submission.digit_3.map(index_to_label).astype(str)
    soft_submission.digit_2 = soft_submission.digit_3.map(lambda x : x[:-1])
    soft_submission.digit_1 = soft_submission.digit_2.map(map_fn)
    
    soft_voting_path = os.path.join(inference_args.dir_path, 'softvoting.csv')
    soft_submission.to_csv(soft_voting_path, index=False)

    # -- Hard voting
    print('Hard Voting')
    voted_labels = []
    counter = collections.Counter()
    hard_submission = pd.read_csv('./data/test.csv', index_col=False)
    for i in tqdm(range(len(hard_submission))) :
        labels = [id_list[i] for id_list in pred_ids]
        counter.update(labels)
        counter_dict = dict(counter)

        items = sorted(counter_dict.items(), key=lambda x : x[1], reverse=True)
        voted_labels.append(items[0][0])
        counter.clear()

    hard_submission.digit_3 = voted_labels
    hard_submission.digit_3 = hard_submission.digit_3.map(index_to_label).astype(str)
    hard_submission.digit_2 = hard_submission.digit_3.map(lambda x : x[:-1])
    hard_submission.digit_1 = hard_submission.digit_2.map(map_fn)

    hard_voting_path = os.path.join(inference_args.dir_path, 'hardvoting.csv')
    hard_submission.to_csv(hard_voting_path, index=False)

if __name__ == "__main__" :
    main()