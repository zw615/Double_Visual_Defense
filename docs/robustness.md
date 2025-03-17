## **Robustness Evaluation**

### VLM Captioning and VQA
In `RobustVLM/scripts` update your local paths for the datasets. The required annotation files for the datasets can be obtained from this [HuggingFace repository](https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main).
We have also prepared a sample data preparation script in `RobustVLM/scripts/prepare_data.sh`.
Set `BASE_WEIGHT_PATH` and `MODEL_NAME` to specify LLaVA model to be evaluated.

Then run
```
cd RobustVLM
bash scripts/batch_multi_1gpu_launch_lora.sh
```
The LLaVA model will be automatically downloaded from HuggingFace.

Note that for faster evaluation, our script partitions the dataset into multiple subset on different GPUs and merge the final results once process on very GPU is finished.


### VLM Targeted Attacks
For targeted attacks on COCO, run
```shell
bash scripts/batch_multi_1gpu_launch_targeted.sh
```
Set `BASE_WEIGHT_PATH` and `MODEL_NAME` to specify LLaVA model to be evaluated.
With 10,000 iterations it takes about 2 hours per image on an A5000 GPU.