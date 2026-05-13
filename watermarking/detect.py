import os
import json
import argparse
from transformers import AutoTokenizer

from config import MODEL_NAME, OUTPUT_DIR, GENERATIONS_FILE, DETECTIONS_FILE
from watermarkers import KGWWatermarker, SynthIDWatermarker


def run_detection(generations_path=None, detections_path=None):
    if generations_path is None:
        generations_path = os.path.join(OUTPUT_DIR, GENERATIONS_FILE)
    if detections_path is None:
        detections_path = os.path.join(OUTPUT_DIR, DETECTIONS_FILE)

    if not os.path.exists(generations_path):
        raise FileNotFoundError(f"Generations file not found: {generations_path}")

    generations = []
    with open(generations_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("output") is not None and len(item["output"]) > 0:
                    generations.append(item)
            except Exception:
                pass

    print(f"Loaded {len(generations)} valid generations.")

    completed = set()
    if os.path.exists(detections_path):
        with open(detections_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    completed.add((item["prompt_idx"], item["condition"]))
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} detections already done.")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing KGW detector...")
    kgw_wm = KGWWatermarker(tokenizer)

    print("Initializing SynthID detector...")
    synthid_wm = SynthIDWatermarker(tokenizer)

    total = len(generations)
    done  = len(completed)

    print(f"\nRunning detection: {total - done} remaining out of {total} total\n")

    with open(detections_path, "a") as out_f:
        for gen in generations:
            prompt_idx = gen["prompt_idx"]
            condition  = gen["condition"]

            if (prompt_idx, condition) in completed:
                continue

            text = gen["output"]
            kgw_result = kgw_wm.detect(text)
            synthid_result = synthid_wm.detect(text)

            record = {
                "prompt_idx":             prompt_idx,
                "condition":              condition,
                "category":               gen["category"],
                "num_tokens":             gen["num_tokens"],
                "kgw_z_score":            kgw_result.get("z_score"),
                "kgw_prediction":         kgw_result.get("prediction"),
                "kgw_green_fraction":     kgw_result.get("green_fraction"),
                "kgw_num_tokens_scored":  kgw_result.get("num_tokens_scored"),
                "synthid_score":          synthid_result.get("score"),
                "synthid_prediction":     synthid_result.get("prediction"),
                "synthid_threshold":      synthid_result.get("threshold"),
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            done += 1

            if done % 100 == 0:
                print(f"  [{done}/{total}] prompt={prompt_idx} condition={condition}")

    print(f"\nDetection complete. Results saved to: {detections_path}")
    _print_summary(detections_path)


def _print_summary(detections_path: str):
    from collections import defaultdict

    kgw_scores    = defaultdict(list)
    synthid_scores = defaultdict(list)

    with open(detections_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                cond = item["condition"]
                if item.get("kgw_z_score") is not None:
                    kgw_scores[cond].append(item["kgw_z_score"])
                if item.get("synthid_score") is not None:
                    synthid_scores[cond].append(item["synthid_score"])
            except Exception:
                pass

    conditions = ["baseline", "kgw_static", "kgw_entropy", "synthid_static", "synthid_entropy"]

    print(f"\n{'Condition':<25} {'KGW mean z':>12} {'SynthID mean score':>20}")
    print("-" * 60)
    for cond in conditions:
        kz  = f"{sum(kgw_scores[cond])/len(kgw_scores[cond]):.3f}" if kgw_scores[cond] else "N/A"
        ss  = f"{sum(synthid_scores[cond])/len(synthid_scores[cond]):.4f}" if synthid_scores[cond] else "N/A"
        print(f"  {cond:<23} {kz:>12} {ss:>20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=os.path.join(OUTPUT_DIR, GENERATIONS_FILE))
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, DETECTIONS_FILE))
    args = parser.parse_args()
    run_detection(generations_path=args.input, detections_path=args.output)