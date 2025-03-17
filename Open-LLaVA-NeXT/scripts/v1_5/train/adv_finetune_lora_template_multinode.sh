#!/bin/bash
set -x

VISION_TOWER=$1
IMAGE_PROCESSOR_NAME_OR_PATH=$2
EPSILON=$3
STEP_SIZE=$4
NUM_STEPS=$5
VISION_TOWER_LR=$6
NNODES=$7
NODE_RANK=$8
MASTER_ADDR=$9
MM_PROJECDTOR_PATH=${10}

# Define the allowed values for NNODES
ALLOWED_NNODES_VALUES=("1" "2" "4")

# Check if NNODES is one of the allowed values
is_allowed=0
for value in "${ALLOWED_NNODES_VALUES[@]}"; do
    if [[ "$NNODES" == "$value" ]]; then
        is_allowed=1
        break
    fi
done

# If NNODES is not one of the allowed values, exit with an error
if [[ $is_allowed == 0 ]]; then
    echo "Error: The value '$NNODES' is not allowed."
    echo "Allowed values are: ${ALLOWED_NNODES_VALUES[*]}"
    exit 1
fi

# If NNODES is not one of the allowed values, exit with an error
if [[ $NODE_RANK -ge $NNODES ]]; then
    echo "Error: The value NODE_RANK '$NODE_RANK' is not allowed with NNODES '$NNODES'."
    exit 1
fi


BASE_DIR=[/path/to/Double_Visual_Defense/Open-LLaVA-NeXT]
cd ${BASE_DIR}
export PYTHONPATH="${BASE_DIR}":"${PYTHONPATH}"

export CONDAPATH=[/path/to/conda]
eval "$(${CONDAPATH} shell.bash hook)"
conda activate double_visual_defense
wandb login c1f4743fe9573f6414122263a434fbc09a07a16c
export WANDB_PROJECT=llava_v1.5_7b_finetune_lora_adv

export BASE_DATA_DIR=[/path/to/base/data]
export DATA_PATH=${BASE_DATA_DIR}/LLaVA-Instruct-150K/llava_v1_5_mix665k-689852649.json
export IMAGE_FOLDER=${BASE_DATA_DIR}/Open-LLaVA-NeXT/data
export BASE_OUTPUT_DIR=[/path/to/base/output]
export OUTPUT_DIR=${BASE_OUTPUT_DIR}/llava_v1.5_7b_finetune_lora_adv/open_clip_bv/eps${EPSILON}_stepsize${STEP_SIZE}_numsteps${NUM_STEPS}_vitlr${VISION_TOWER_LR}_${NNODES}nodes
# if an empty string
if [ -z "$MM_PROJECDTOR_PATH" ]; then
    export MM_PROJECDTOR_PATH=${BASE_OUTPUT_DIR}/llava_v1.5_7b_pretrain_adv/open_clip_bv/eps${EPSILON}_stepsize${STEP_SIZE}_numsteps${NUM_STEPS}_${NNODES}nodes/mm_projector.bin
    echo "No MM_PROJECDTOR_PATH supplied, using default path "$MM_PROJECDTOR_PATH
fi

export BASE_LR=2e-4
export PER_DEVICE_TRAIN_BATCH_SIZE="$((8 / NNODES))"
export PER_DEVICE_EVAL_BATCH_SIZE=1
export GRADIENT_ACCU_STEPS=2
export SAVE_STEPS=1200

mkdir -p ${OUTPUT_DIR}

torchrun --nnodes $NNODES --nproc_per_node 8 --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port 25009 llava/train/train_adv_grad_accum_fa2.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2_nooc.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower $VISION_TOWER \
    --image_processor_name_or_path ${IMAGE_PROCESSOR_NAME_OR_PATH} \
    --pretrain_mm_mlp_adapter ${MM_PROJECDTOR_PATH} \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr ${VISION_TOWER_LR} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --epsilon ${EPSILON} \
    --step_size ${STEP_SIZE} \
    --num_steps ${NUM_STEPS} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name eps${EPSILON}_stepsize${STEP_SIZE}_numsteps${NUM_STEPS}_vitlr${VISION_TOWER_LR}_${NNODES}nodes 2>&1 | tee ${OUTPUT_DIR}/output.log
