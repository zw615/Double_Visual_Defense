#!/bin/bash

# Get the command-line argument
MODEL_PATH="$1"
MODEL_BASE="$2"
EPS="$3"
EXP_NAME="$4"
EVAL_STRING="$5"
NUM_SAMPLES="$6"
START_SAMPLE_IDX="$7"
END_SAMPLE_IDX="$8"
ATTACK="$9"
##################
# Define the allowed values for EVAL_STRING
allowed_eval_values=("--eval_coco" "--eval_flickr30" "--eval_vqav2" "--eval_vizwiz" "--eval_textvqa")

# Check if EVAL_STRING is one of the allowed values
is_allowed=0
for value in "${allowed_eval_values[@]}"; do
    if [[ "$EVAL_STRING" == "$value" ]]; then
        is_allowed=1
        break
    fi
done

# If EVAL_STRING is not one of the allowed values, exit with an error
if [[ $is_allowed == 0 ]]; then
    echo "Error: The value '$EVAL_STRING' is not allowed."
    echo "Allowed values are: ${allowed_eval_values[*]}"
    exit 1
fi

# If EVAL_STRING is allowed, proceed with the rest of the script
echo "The eval string passed is: $EVAL_STRING"
##################

export CONDAPATH=[/path/to/conda]
eval "$(${CONDAPATH} shell.bash hook)"
conda activate double_visual_defense

BASE_DIR=[/path/to/Double_Visual_Defense/RobustVLM]
cd ${BASE_DIR}
export OUTPUT_PATH=${BASE_DIR}/output/robustness_evaluation/${EXP_NAME}/${START_SAMPLE_IDX}_to_${END_SAMPLE_IDX}_in_${NUM_SAMPLES}
export DATA_BASE_PATH=[/path/to/robust_vlm_data]
mkdir -p ${OUTPUT_PATH}

if [ -z "$ATTACK" ]; then
    ATTACK=ensemble
    echo "No ATTACK specified, using default attack "$ATTACK
fi


# LLaVA evaluation script
python -m vlm_eval.run_evaluation \
--attack $ATTACK --eps ${EPS} --steps 100 --mask_out none \
--precision float16 \
--attn_implementation eager \
--num_samples ${NUM_SAMPLES} \
--start_sample_idx ${START_SAMPLE_IDX} \
--end_sample_idx ${END_SAMPLE_IDX} \
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
${EVAL_STRING} \
--coco_train_image_dir_path ${DATA_BASE_PATH}/coco/train2014 \
--coco_val_image_dir_path ${DATA_BASE_PATH}/coco/val2014 \
--coco_karpathy_json_path ${DATA_BASE_PATH}/eval_benchmark/mscoco_karpathy/karpathy_coco.json \
--coco_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/mscoco_karpathy/annotations/captions_val2014.json \
--flickr_image_dir_path ${DATA_BASE_PATH}/flickr30k/flickr30k-images \
--flickr_karpathy_json_path ${DATA_BASE_PATH}/flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path ${DATA_BASE_PATH}/vizwiz/train \
--vizwiz_test_image_dir_path ${DATA_BASE_PATH}/vizwiz/val \
--vizwiz_train_questions_json_path ${DATA_BASE_PATH}/eval_benchmark/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path ${DATA_BASE_PATH}/eval_benchmark/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/vizwiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path ${DATA_BASE_PATH}/coco/train2014 \
--vqav2_train_questions_json_path ${DATA_BASE_PATH}/eval_benchmark/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/vqav2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path ${DATA_BASE_PATH}/coco/val2014 \
--vqav2_test_questions_json_path ${DATA_BASE_PATH}/eval_benchmark/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/vqav2/v2_mscoco_val2014_annotations.json \
--textvqa_image_dir_path ${DATA_BASE_PATH}/textvqa/train_images \
--textvqa_train_questions_json_path ${DATA_BASE_PATH}/eval_benchmark/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path ${DATA_BASE_PATH}/eval_benchmark/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path ${DATA_BASE_PATH}/eval_benchmark/textvqa/val_annotations_vqa_format.json \
2>&1 | tee ${OUTPUT_PATH}/output.log