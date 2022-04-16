from typing import Optional

class Preprocessor :
    def __init__(self, tokenizer, label_dict:Optional[dict] = None, train_flag:bool = False) :
        self.tokenizer = tokenizer        
        self.label_dict = label_dict
        self.train_flag = train_flag

    def __call__(self, dataset) :
        inputs = []
        labels = []
        sep_token = self.tokenizer.sep_token

        for i in range(len(dataset['AI_id'])) :
            obj = dataset['text_obj'][i]
            mthd = dataset['text_mthd'][i]
            deal = dataset['text_deal'][i]
            
            if self.train_flag == True :
                label_index = self.label_dict[dataset['digit_3'][i]]
                labels.append(label_index)

            input_sen = obj + sep_token + mthd + sep_token + deal
            inputs.append(input_sen)

        dataset['inputs'] = inputs
        if self.train_flag == True :
            dataset['labels'] = labels
        return dataset