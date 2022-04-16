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

python inference.py \
--PLM ./checkpoints \
--fold_size 10 \
--max_length 32 \
--model_type lstm \
--per_device_eval_batch_size 128 \
--output_dir results