#!/bin/bash

# ============================================================
# Config: edit these paths before running
# ============================================================
CKPT_BASE=
DATA_BASE=

MODEL_PATH=${CKPT_BASE}/llava-v1.5-7b
VISION_TOWER=${CKPT_BASE}/clip-vit-large-patch14-336
PRETRAIN_ADAPTER=${CKPT_BASE}/llava-v1.5-7b/mm_projector.bin
DATA_PATH=${DATA_BASE}/LLaVAb_z/LLaVAb_dataset/large_attack_dataset522.json
IMAGE_FOLDER=${DATA_BASE}/data/mscoco/train2017
OUTPUT_DIR=${CKPT_BASE}/train_encoder_decoder
# ============================================================

# Auto-create output directory if not exists
mkdir -p ${OUTPUT_DIR}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port=20002 llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_ADAPTER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --tune_mm_mlp_adapter True \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
