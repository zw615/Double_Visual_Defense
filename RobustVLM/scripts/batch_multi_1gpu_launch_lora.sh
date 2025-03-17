#!/bin/bash
SCRIPT_PATH=scripts_private/llava_lora_dist_eval.sh
POSTPROCESS_SCRIPT_PATH=scripts_private/merge_results.py
BASE_DIR=[/path/to/Double_Visual_Defense/RobustVLM]
cd ${BASE_DIR}
export PYTHONPATH=${BASE_DIR}:PYTHONPATH

DATA_BASE_PATH=[/path/to/robust_vlm_data]

EPS=4
NUM_GPUS=4
NUM_SAMPLES=500
NUM_SAMPLES_EACH_GPU=$((NUM_SAMPLES / NUM_GPUS))

################ custom clip llava ##########
BASE_WEIGHT_PATH=[/path/to/base_weight]
MODEL_NAME=delta2_llava_8
EXP_NAME=delta2_llava_8_reproduce

MODEL_BASE=lmsys/vicuna-7b-v1.5

MODEL_PATH=${BASE_WEIGHT_PATH}/${MODEL_NAME}
#############################################

export CONDAPATH=[/path/to/conda]
eval "$(${CONDAPATH} shell.bash hook)"
conda activate double_visual_defense

################# eval_captioning #######################
SECONDS=0
echo "Eval start!"
idx=0
while ((idx < NUM_GPUS));
do
    CUDA_VISIBLE_DEVICES=$idx bash ${SCRIPT_PATH} ${MODEL_PATH} ${MODEL_BASE} ${EPS} ${EXP_NAME} --eval_coco ${NUM_SAMPLES} $((idx * NUM_SAMPLES_EACH_GPU)) $(((idx+1) * NUM_SAMPLES_EACH_GPU)) &
    ((idx++))
    echo $idx
done

START_SAMPLE_IDS=()
END_SAMPLE_IDS=()
idx=0
while ((idx < NUM_GPUS));
do
    CUDA_VISIBLE_DEVICES=$((idx+NUM_GPUS)) bash ${SCRIPT_PATH} ${MODEL_PATH} ${MODEL_BASE} ${EPS} ${EXP_NAME} --eval_flickr30 ${NUM_SAMPLES} $((idx * NUM_SAMPLES_EACH_GPU)) $(((idx+1) * NUM_SAMPLES_EACH_GPU)) &
    START_SAMPLE_IDS+=($((idx * NUM_SAMPLES_EACH_GPU)))
    END_SAMPLE_IDS+=($(((idx+1) * NUM_SAMPLES_EACH_GPU)))
    ((idx++))
    echo $idx
done

wait

echo "Eval finished! Took" $SECONDS "seconds!"
echo "Start merging results!"

python3 ${POSTPROCESS_SCRIPT_PATH} \
--base-dir ${BASE_DIR}/output/${EXP_NAME} \
--start-sample-ids ${START_SAMPLE_IDS[@]} \
--end-sample-ids ${END_SAMPLE_IDS[@]} \
--num-samples ${NUM_SAMPLES} \
--file-prefix cocoresults-best \
--eval-metric cider \
--annotations-json-path ${DATA_BASE_PATH}/eval_benchmark/mscoco_karpathy/annotations/captions_val2014.json

python3 ${POSTPROCESS_SCRIPT_PATH} \
--base-dir ${BASE_DIR}/output/${EXP_NAME} \
--start-sample-ids ${START_SAMPLE_IDS[@]} \
--end-sample-ids ${END_SAMPLE_IDS[@]} \
--num-samples ${NUM_SAMPLES} \
--file-prefix flickrresults-best \
--eval-metric cider \
--annotations-json-path ${DATA_BASE_PATH}/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json
#########################################################

################# eval_vqa #######################
 SECONDS=0
 echo "Eval start!"
 idx=0
 while ((idx < NUM_GPUS));
 do
     CUDA_VISIBLE_DEVICES=$idx bash ${SCRIPT_PATH} ${MODEL_PATH} ${MODEL_BASE} ${EPS} ${EXP_NAME} --eval_vqav2 ${NUM_SAMPLES} $((idx * NUM_SAMPLES_EACH_GPU)) $(((idx+1) * NUM_SAMPLES_EACH_GPU)) &
     ((idx++))
 done

 START_SAMPLE_IDS=()
 END_SAMPLE_IDS=()
 idx=0
 while ((idx < NUM_GPUS));
 do
     CUDA_VISIBLE_DEVICES=$((idx+NUM_GPUS)) bash ${SCRIPT_PATH} ${MODEL_PATH} ${MODEL_BASE} ${EPS} ${EXP_NAME} --eval_textvqa ${NUM_SAMPLES} $((idx * NUM_SAMPLES_EACH_GPU)) $(((idx+1) * NUM_SAMPLES_EACH_GPU)) &
     START_SAMPLE_IDS+=($((idx * NUM_SAMPLES_EACH_GPU)))
     END_SAMPLE_IDS+=($(((idx+1) * NUM_SAMPLES_EACH_GPU)))
     ((idx++))
 done

 wait

 echo "Eval finished! Took" $SECONDS "seconds!"
 echo "Start merging results!"

 python3 ${POSTPROCESS_SCRIPT_PATH} \
 --base-dir ${BASE_DIR}/output/${EXP_NAME} \
 --start-sample-ids ${START_SAMPLE_IDS[@]} \
 --end-sample-ids ${END_SAMPLE_IDS[@]} \
 --num-samples ${NUM_SAMPLES} \
 --file-prefix vqav2results-best \
 --eval-metric vqa_accuracy \
 --test-questions-json-path ${DATA_BASE_PATH}/eval_benchmark/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
 --test-annotations-json-path ${DATA_BASE_PATH}/eval_benchmark/vqav2/v2_mscoco_val2014_annotations.json

 python3 ${POSTPROCESS_SCRIPT_PATH} \
 --base-dir ${BASE_DIR}/output/${EXP_NAME} \
 --start-sample-ids ${START_SAMPLE_IDS[@]} \
 --end-sample-ids ${END_SAMPLE_IDS[@]} \
 --num-samples ${NUM_SAMPLES} \
 --file-prefix textvqaresults-best \
 --eval-metric vqa_accuracy \
 --test-questions-json-path ${DATA_BASE_PATH}/eval_benchmark/textvqa/val_questions_vqa_format.json \
 --test-annotations-json-path ${DATA_BASE_PATH}/eval_benchmark/textvqa/val_annotations_vqa_format.json
#########################################################