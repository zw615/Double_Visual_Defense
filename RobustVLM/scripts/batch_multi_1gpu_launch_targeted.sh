#!/bin/bash
TARGET_STRINGS_FILE_PATH=[/path/to/Double_Visual_Defense/RobustVLM/target_strings.txt]
BASE_DIR=[/path/to/Double_Visual_Defense/RobustVLM]
cd ${BASE_DIR}
export PYTHONPATH=${BASE_DIR}:PYTHONPATH

################ custom clip llava ##########
BASE_WEIGHT_PATH=[/path/to/base_weight]
MODEL_NAME=delta2_llava_8
EXP_NAME=delta2_llava_8_targeted_reproduce
MODEL_PATH=${BASE_WEIGHT_PATH}/${MODEL_NAME}

EPS=4
NUM_STEPS=10000
EXP_NAME_PREFIX=${EXP_NAME}
NUM_SAMPLES=10
TRIAL_SEEDS=2024


#############################################
SCRIPT_PATH=[/path/to/Double_Visual_Defense/RobustVLM/scripts_private/llava_lora_eval_targeted.sh]
MODEL_BASE=lmsys/vicuna-7b-v1.5

idx=0
while IFS= read -r target_str; do
    echo "Text read from file: $target_str"
    CUDA_VISIBLE_DEVICES=${idx} bash ${SCRIPT_PATH} "${MODEL_PATH}" "${MODEL_BASE}" "${target_str}" "${EPS}" "${NUM_STEPS}" "${EXP_NAME_PREFIX}_target${idx}_numsamples${NUM_SAMPLES}_eps${EPS}_numsteps${NUM_STEPS}" $NUM_SAMPLES $TRIAL_SEEDS &
    ((idx++))
done < ${TARGET_STRINGS_FILE_PATH}

wait
