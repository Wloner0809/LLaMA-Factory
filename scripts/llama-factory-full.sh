#!/usr/bin/env bash

export WANDB_PROJECT="Post-Train"
export WANDB_ENTITY="MSRA-YuWang"
wandb login "211edf92584710e4ccb6e0bf073a549c7404a723"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
if [ "$PROC_PER_NODE" != "1" ] && [ "$RANK" != "0" ]; then
    exit 0
fi

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_dpo_ds3.yaml

# deepspeed --num_nodes "${WORLD_SIZE}" --master_port "${MASTER_PORT}" --master_addr "${MASTER_ADDR}" \
#     src/train.py \
#     --model_name_or_path /mnt/default/finetuned-model/llama3.1-8b-full-sft-Infinity_Instruct/checkpoint-25900 \
#     --stage dpo \
#     --do_train True \
#     --finetuning_type full \
#     --deepspeed examples/deepspeed/ds_z3_config.json \
#     --pref_beta 0.1 \
#     --pref_loss sigmoid \
#     --dataset Infinity_Preference-sharegpt_format \
#     --template llama3 \
#     --cutoff_len 2048 \
#     --preprocessing_num_workers 16 \
#     --output_dir /mnt/default/finetuned-model/llama3.1-8b-full-dpo-Infinity_Preference-sft_Infinity_Instruct_ckpt25900-sharegpt_format \
#     --logging_steps 1 \
#     --save_steps 100 \
#     --save_only_model True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.0e-6 \
#     --num_train_epochs 1.0 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.1 \
#     --bf16 True \
#     --ddp_timeout 180000000 \
#     --report_to wandb \
#     --run_name llama-factory-dpo-full-Infinity_Preference-sharegpt_format \
#     --val_size 0.05 \
#     --per_device_eval_batch_size 1 \
#     --eval_strategy steps \
#     --eval_steps 100

# deepspeed --num_nodes "${WORLD_SIZE}" --master_port "${MASTER_PORT}" --master_addr "${MASTER_ADDR}" \
#     src/train.py \
#     --model_name_or_path /mnt/default/model/llama3.1-8b \
#     --stage sft \
#     --do_train True \
#     --finetuning_type full \
#     --deepspeed examples/deepspeed/ds_z3_config.json \
#     --dataset Infinity_Instruct \
#     --template llama3 \
#     --cutoff_len 2048 \
#     --preprocessing_num_workers 16 \
#     --output_dir /mnt/default/finetuned-model/llama3.1-8b-full-sft-Infinity_Instruct-epoch2-lr1e-5 \
#     --logging_steps 1 \
#     --save_steps 500 \
#     --save_only_model True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.0e-5 \
#     --num_train_epochs 1.0 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.1 \
#     --bf16 True \
#     --ddp_timeout 180000000 \
#     --report_to wandb \
#     --run_name llama-factory-sft-full-Infinity_Instruct-epoch2-lr1e-5 \
#     --val_size 0.01 \
#     --per_device_eval_batch_size 1 \
#     --eval_strategy steps \
#     --eval_steps 500




# accelerate启动

# if [[ $RANK -eq 0 ]]; then
#     export MAIN_IP_ADDR=$(hostname -I | awk '{print $1}')
# fi
# export MAIN_IP_ADDR=$(hostname -I | awk '{print $1}')
# echo $MAIN_IP_ADDR

# sed -i "s/^main_process_ip:.*$/main_process_ip: $MAIN_IP_ADDR/" examples/accelerate/fsdp_config.yaml

# accelerate launch \
#     --config_file examples/accelerate/fsdp_config.yaml \
#     src/train.py examples/train_full/llama3_full_dpo_ds3.yaml