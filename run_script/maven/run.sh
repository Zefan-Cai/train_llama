

export EXPERIMENT_NAME=llama_rationale_tuning
export DATASET_NAME=maven
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export MODEL_DIR=/home/haozhezhao/model/
export MODEL_NAME=llama-7b


bs=4
eval_bs=6
lr=5e-5
dropout=0.1
epoch=2
seed=1234
do_train=True
do_test=True
do_valid=True
master_port=29502
deepspeed --master_port $master_port run.py \
--experiment_name ${EXPERIMENT_NAME} \
--dataset_name ${DATASET_NAME} \
--dataset_config_name None \
--overwrite_cache True \
--pad_to_max_length True \
--train_file /home/haozhezhao/Confuse-Classes/data/re_maven.jsonl \
--validation_file /home/haozhezhao/Confuse-Classes/data/re_fewrel.jsonl \
--test_file /home/haozhezhao/Confuse-Classes/data/re_fewrel.jsonl \
--do_train \
--do_eval \
--do_predict \
--bf16 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size ${eval_bs} \
--gradient_accumulation_steps 1 \
--num_train_epochs ${epoch} \
--output_dir checkpoints/${EXPERIMENT_NAME} \
--overwrite_output_dir \
--learning_rate ${lr} \
--weight_decay 0.0005 \
--seed ${seed} \
--warmup_ratio 0.2 \
--evaluation_strategy steps \
--eval_steps 250 \
--remove_unused_columns False \
--model_name_or_path ${MODEL_DIR}${MODEL_NAME} \
--use_fast_tokenizer True \
--model_revision main \
--eval_type val \
--generation_max_length 32 \
--do_full_training True \
--max_eval_samples 1000 \
--max_predict_samples 1000 \
--deepspeed config/deepspeed_config.json
# --load_best_model_at_end \
# --multiple_choice True
# --max_seq_length 512 \
