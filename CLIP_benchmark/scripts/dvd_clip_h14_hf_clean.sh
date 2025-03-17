OUTPUT_PATH=output/dvd_clip_h14_hf
mkdir -p ${OUTPUT_PATH}

CUDA_VISIBLE_DEVICES=0 python3 -m clip_benchmark.cli eval \
    --model "hf-hub:zw123/delta_clip_h14_336" \
    --model_type 'open_clip' \
    --dataset benchmark/webdatasets.txt \
    --num_workers 4 \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "${OUTPUT_PATH}/clean_benchmark_{dataset}_{language}_{task}_{n_samples}_bs{bs}_{attack}_{eps}_{iterations}.json"

python3 -m clip_benchmark.cli build ${OUTPUT_PATH}/clean_benchmark_*.json --output ${OUTPUT_PATH}/clean_benchmark.csv