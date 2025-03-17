#!/bin/bash

# Get the command-line argument
MODEL_PATH="$1"
MODEL_BASE="$2"
TARGET_STR="$3"
EPS="$4"
NUM_STEPS="$5"
EXP_NAME="$6"
NUM_SAMPLES="$7"
TRIAL_SEEDS="$8"
##################

export CONDAPATH=[/path/to/conda]
eval "$(${CONDAPATH} shell.bash hook)"
conda activate double_visual_defense

BASE_DIR=[/path/to/Double_Visual_Defense/RobustVLM]
cd ${BASE_DIR}
export OUTPUT_PATH=${BASE_DIR}/output/stealthy_targeted_attack/${EXP_NAME}
export DATA_BASE_PATH=[/path/to/robust_vlm_data]
mkdir -p ${OUTPUT_PATH}


python -m vlm_eval.run_evaluation \
--eval_coco \
--verbose \
--attack apgd --eps ${EPS} --steps ${NUM_STEPS} --mask_out none \
--targeted --target_str "${TARGET_STR}" \
--precision float32 \
--attn_implementation eager \
--num_samples ${NUM_SAMPLES} \
--trial_seeds ${TRIAL_SEEDS} \
--shots 0 \
--batch_size 1 \
--results_file llava-v1.5-7b \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--out_base_path ${OUTPUT_PATH} \
--model_path ${MODEL_PATH} \
--model_base ${MODEL_BASE} \
--model_name llava-v1.5-7b-lora \
--coco_train_image_dir_path ${DATA_BASE_PATH}/coco/train2014 \
--coco_val_image_dir_path ${DATA_BASE_PATH}/coco/val2014 \
--coco_karpathy_json_path ${DATA_BASE_PATH}/eval_benchmark/mscoco_karpathy/karpathy_coco.json \
--coco_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/mscoco_karpathy/annotations/captions_val2014.json \
2>&1 | tee ${OUTPUT_PATH}/output.log
