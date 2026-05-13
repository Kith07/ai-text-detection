import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"
MAX_LENGTH = 512
OUTPUT_DIR = "./outputs"
PERPLEXITY_FILE = "perplexity.jsonl"


def load_reference_model():
    print(f"Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()
    print("Reference model loaded.")
    return model, tokenizer


def compute_perplexity(text: str, model, tokenizer, device: str = DEVICE, max_length: int = MAX_LENGTH) -> float | None:
    if not text or len(text.strip()) == 0:
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)

    if inputs["input_ids"].shape[1] < 10:
        return None

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"],
        )
        perplexity = torch.exp(outputs.loss).item()

    if not torch.isfinite(torch.tensor(perplexity)):
        return None

    return perplexity


def run_perplexity_scoring(input_path: str):
    output_path = os.path.join(OUTPUT_DIR, PERPLEXITY_FILE)

    generations = []
    with open(input_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("output") and len(item["output"].strip()) > 0:
                    generations.append(item)
            except Exception:
                pass

    print(f"Loaded {len(generations)} generations to score.")

    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    completed.add((item["prompt_idx"], item["condition"]))
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} already scored.")

    model, tokenizer = load_reference_model()

    total = len(generations)
    done = len(completed)

    print(f"Scoring {total - done} outputs...\n")

    with open(output_path, "a") as out_f:
        for i, gen in enumerate(generations):
            prompt_idx = gen["prompt_idx"]
            condition  = gen["condition"]

            if (prompt_idx, condition) in completed:
                continue

            ppl = compute_perplexity(gen["output"], model, tokenizer)

            record = {
                "prompt_idx": prompt_idx,
                "condition":  condition,
                "category":   gen["category"],
                "num_tokens": gen["num_tokens"],
                "perplexity": ppl,
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            done += 1

            if done % 100 == 0:
                ppl_str = f"{ppl:.2f}" if ppl else "N/A"
                print(f"  [{done}/{total}] prompt={prompt_idx} condition={condition} ppl={ppl_str}")

    print(f"\nPerplexity scoring complete. Results saved to: {output_path}")
    _print_summary(output_path)


def _print_summary(output_path: str):
    from collections import defaultdict

    scores = defaultdict(list)
    with open(output_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item["perplexity"] is not None:
                    scores[item["condition"]].append(item["perplexity"])
            except Exception:
                pass

    print("\nMean perplexity by condition (lower = better quality):")
    print(f"  {'Condition':<30} {'Mean PPL':>10} {'Median PPL':>12} {'N':>6}")
    print(f"  {'-'*60}")

    import statistics
    for condition in ["baseline", "kgw_static", "kgw_entropy", "synthid_static", "synthid_entropy"]:
        vals = scores.get(condition, [])
        if vals:
            mean = sum(vals) / len(vals)
            median = statistics.median(vals)
            print(f"  {condition:<30} {mean:>10.2f} {median:>12.2f} {len(vals):>6}")
        else:
            print(f"  {condition:<30} {'N/A':>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perplexity scoring with base model")
    parser.add_argument(
        "--input",
        default=os.path.join(OUTPUT_DIR, "generations.jsonl"),
        help="Path to generations JSONL file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    run_perplexity_scoring(args.input)