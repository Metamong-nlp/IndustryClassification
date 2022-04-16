# IndustryClassification

## Goal
  1. 통계청 인공지능 활용대회 - 자연어 기반 인공지능 산업분류 자동화

## Dataset
  1. 산업분류 통계데이터 
  2. 주소 : https://data.kostat.go.kr/sbchome/contents/cntPage.do?cntntsId=CNTS_000000000000575&curMenuNo=OPT_09_03_00_0
  3. 훈련 데이터 갯수 : 1000000개
  4. 테스트 데이터 갯수 : 100000개  
      
## Environment
  1. Platform : GCP - Debian Linux OS (Virtual Machine)
  2. GPU : Nvidia Tesla A100 GPU 1개
  3. CPU : Intel cascade lake 1개

## Dependency 
  1. datasets : 1.18.0
  2. transformers : 4.17.0

## Baseline
  1. Huggingface의 transformers 라이브러리르 기반으로 basline 코드 작성
  2. RobertaModelForSequenceClassification을 상속받아서 다양한 모델을 만들고 성능 비교하기
  4. K-fold Validation & Ensemable을 통한 성능 향상

## Data Preprocessing
  1. 데이터 변형
      * KoSpacing 라이브러리를 활용해서 기존 데이터에서 띄어쓰기 적용
  2. 데이터 전처리
      * [SEP] : klue/roberta-large tokenizer의 sep_token
      * Text_obj + [SEP] + Text_mthd + [SEP] + Text_deal 형식으로 데이터를 하나의 문장으로 전처리
  3. 데이터 토크나이징 & 인코딩
      * klue/roberta-large의 tokenizer를 가지고서 하나의 문장이 된 데이터르 토큰화 및 인코딩 진행

## Model
  1. Base Model
      * klue/roberta-large 모델을 그대로 사용
  3. WeightAverage
      * klue/roberta-large에서 output hidden states를 하여서 최종 Layer 3개를 Weight Average한 모델
  5. LSTM
      * klue/roberta-large 모델에서 head에 LSTM layer를 추가한 모델
  7. CNN 
      * Source : https://aclanthology.org/2020.semeval-1.271.pdf
      * Structre 
        * 논문과의 차이점 : **sigmoid가 아닌 softmax로 변형**
        * Image
          ![스크린샷 2022-04-16 오후 4 24 24](https://user-images.githubusercontent.com/48673702/163665979-1eba991a-1f1a-42c6-96ab-2eb07fb4db24.png)
  6. RBERT
      * Source : https://arxiv.org/pdf/1905.08284.pdf
      * Structure 
        * 논문과의 차이점 : **경계 토큰 사이에 있는 토큰들을 가져오는 것이 아니라 경계 토큰([CLS], [SEP])의 정보를 가져와서 전달**
        * Image
          ![136745007-699b42eb-5338-43a5-815c-3c681a63e8e4](https://user-images.githubusercontent.com/48673702/163665931-0315f49a-c009-4e8a-a72e-fe430f83581d.png)

## Objective function
  1. Optimizer : AdamW
  2. Learning Rate : Linear warmup decay scheduler
  3. Objective : R-Drop
      * Source : https://proceedings.neurips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf
      * Structure
        ![스크린샷 2022-04-16 오후 4 34 45](https://user-images.githubusercontent.com/48673702/163666309-05b5b2b2-e6e9-4f3b-8943-486c196b0a8b.png)
     
## Terminal Command Example
  ```
  # training 
  python train.py \
  --save_total_limit 5 \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --fold_size 10 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-3 \
  --max_length 32 \
  --output_dir ./exp \
  --logging_dir ./logs \
  --save_strategy steps \
  --evaluation_strategy steps \
  --logging_steps 200 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --load_best_model_at_end \
  --metric_for_best_model accuracy \
  --model_type lstm \
  --use_rdrop \
  --fp16
  
  # inference 
  python inference.py \
  --PLM ./checkpoints \
  --fold_size 10 \
  --max_length 32 \
  --model_type lstm \
  --per_device_eval_batch_size 128 \
  --output_dir results
  ```

## Wandb Log
  * Results
    ![스크린샷 2022-04-16 오후 4 54 29](https://user-images.githubusercontent.com/48673702/163667031-ed0f9897-77e2-4941-b590-36241afd0856.png)

## Final Hyperparameter 
  1. 10-fold training
      * Epochs : 3
      * Learning rate : 3e-5
      * Train batch size : 256
      * Max input length : 32
      * Warmup steps : 0.05
      * Weight decay : 1e-3
      * Model type : LSTM
      * Objective : R-Drop

## Result
|Model|Accuracy|F1-score|
|-----|----|----|
|10-fold Ensemable(lstm + rdrop)|91.33|81.57|


