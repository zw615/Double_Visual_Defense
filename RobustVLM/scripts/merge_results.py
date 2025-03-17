import os
import json
import glob
import argparse

from open_flamingo.eval.coco_metric import compute_cider

from open_flamingo.eval.vqa_metric import compute_vqa_accuracy


def merge_results(args):
    assert len(args.start_sample_ids) == len(args.end_sample_ids)

    all_captions_or_answers = []
    all_ids = set()

    for idx in range(len(args.start_sample_ids)):

        folder_name = f"{args.start_sample_ids[idx]}_to_{args.end_sample_ids[idx]}_in_{args.num_samples}"
        caption_or_answer_path_list = glob.glob(os.path.join(args.base_dir, folder_name, "captions-json",  f"{args.file_prefix}_*.json"))

        assert len(caption_or_answer_path_list) == 1, "there should only be one caption_or_answer in each dist eval fodler!"
        caption_or_answer_path = caption_or_answer_path_list[0]

        with open(caption_or_answer_path, 'r') as file:
            caption_or_answer = json.load(file)
        assert isinstance(caption_or_answer, list)

        if args.eval_metric == "cider":
            for item in caption_or_answer:
                assert isinstance(item, dict) and "image_id" in item.keys()
                if not item["image_id"] in all_ids:
                    all_captions_or_answers.append(item)
                    all_ids.add(item["image_id"])
        elif args.eval_metric == "vqa_accuracy":
            for item in caption_or_answer:
                assert isinstance(item, dict) and "question_id" in item.keys()
                if not item["question_id"] in all_ids:
                    all_captions_or_answers.append(item)
                    all_ids.add(item["question_id"])
        else:
            raise ValueError
    
    
    if args.eval_metric == "cider":
        all_captions_path = os.path.join(args.base_dir, f"{args.file_prefix}_all_captions.json")
        with open(all_captions_path, "w") as f:
            f.write(json.dumps(all_captions_or_answers, indent=4))

        metrics = compute_cider(
            result_path=all_captions_path,
            annotations_path=args.annotations_json_path,
        )
        print(f'CIDER score of {args.file_prefix}: {metrics["CIDEr"] * 100.0}')
        metrics_dict=dict(CIDEr=metrics["CIDEr"])
    elif args.eval_metric == "vqa_accuracy":
        all_answers_path = os.path.join(args.base_dir, f"{args.file_prefix}_all_answers.json")
        with open(all_answers_path, "w") as f:
            f.write(json.dumps(all_captions_or_answers, indent=4))

        vqa_accuracy = compute_vqa_accuracy(
            all_answers_path,
            args.test_questions_json_path,
            args.test_annotations_json_path,
        )
        print(f"vqa_accuracy {args.file_prefix}: {vqa_accuracy}")
        metrics_dict=dict(vqa_accuracy=vqa_accuracy)
    else:
        raise ValueError
    
    results_path = os.path.join(args.base_dir, f"{args.file_prefix}_all_results.json")
    with open(results_path, "w") as f:
        f.write(json.dumps(metrics_dict))
                                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str)
    parser.add_argument('--start-sample-ids', nargs='+')
    parser.add_argument('--end-sample-ids', nargs='+')
    parser.add_argument('--num-samples', type=int)
    parser.add_argument("--file-prefix", type=str, choices=["flickrresults-best", "cocoresults-best", "vqav2results-best", "textvqaresults-best",
                                                            "flickrresults", "cocoresults", "vqav2results", "textvqaresults",])
    parser.add_argument("--eval-metric", type=str, choices=["cider", "vqa_accuracy"])
    parser.add_argument("--annotations-json-path", type=str, default=None)
    parser.add_argument("--test-questions-json-path", type=str, default=None)
    parser.add_argument("--test-annotations-json-path", type=str, default=None)
    args = parser.parse_args()

    merge_results(args)