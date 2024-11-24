#!/usr/bin/env bash

# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_dpo_ds3.yaml


# deepspeed --num_nodes "${WORLD_SIZE}" --master_port "${MASTER_PORT}" --master_addr "${MASTER_ADDR}" \
#     src/train.py \
#     --model_name_or_path /mnt/default/finetuned-model/llama3.1-8b-full-sft-Infinity_Instruct/checkpoint-25900 \
#     --stage dpo \
#     --do_train True \
#     --finetuning_type full \
#     --deepspeed examples/deepspeed/ds_z3_config.json \
#     --pref_beta 0.1 \
#     --pref_loss sigmoid \
#     --dataset Infinity_Preference \
#     --template llama3 \
#     --cutoff_len 2048 \
#     --preprocessing_num_workers 16 \
#     --output_dir /mnt/default/finetuned-model/llama3.1-8b-full-dpo-Infinity_Preference-sft_Infinity_Instruct_ckpt25900-amd \
#     --logging_steps 1 \
#     --save_steps 100 \
#     --save_only_model True \
#     --plot_loss True \
#     --overwrite_output_dir True \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1.0e-6 \
#     --num_train_epochs 1.0 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.1 \
#     --bf16 True \
#     --ddp_timeout 180000000 \
#     --report_to wandb \
#     --run_name llama-factory-dpo-full-Infinity_Preference-amd \
#     --val_size 0.05 \
#     --per_device_eval_batch_size 1 \
#     --eval_strategy steps \
#     --eval_steps 100