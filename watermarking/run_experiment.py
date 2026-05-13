import argparse
import json
import os
import torch

from config import OUTPUT_DIR, GENERATIONS_FILE, DEVICE, TEMPERATURE


def run_check():
    print("=" * 60)
    print("CHECK — 5 prompts x 5 conditions")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
    from config import MODEL_NAME
    from data import load_prompts, format_prompt_for_llama
    from watermarkers import get_all_watermarkers

    import config
    original = config.CATEGORY_SAMPLES.copy()
    config.CATEGORY_SAMPLES = {k: 1 for k in list(config.CATEGORY_SAMPLES.keys())[:5]}
    prompts = load_prompts()[:5]
    config.CATEGORY_SAMPLES = original

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()

    watermarkers = get_all_watermarkers(tokenizer)
    conditions = ["baseline", "kgw_static", "kgw_entropy", "synthid_static", "synthid_entropy"]
    trial_records = []

    for p_idx, prompt_item in enumerate(prompts):
        instruction = prompt_item["instruction"]
        formatted = format_prompt_for_llama(instruction, tokenizer)
        print(f"\n{'_'*60}")
        print(f"PROMPT {p_idx}: [{prompt_item['category']}] {instruction[:80]}...")

        for condition in conditions:
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(DEVICE)
            input_len = inputs["input_ids"].shape[1]
            wm = watermarkers[condition]

            generate_kwargs = dict(
                **inputs,
                max_new_tokens=400,
                min_new_tokens=50,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )

            with torch.no_grad():
                if condition == "baseline":
                    output_ids = model.generate(**generate_kwargs)
                elif condition == "synthid_static":
                    output_ids = model.generate(
                        **generate_kwargs,
                        watermarking_config=wm.get_watermarking_config()
                    )
                else:
                    output_ids = model.generate(
                        **generate_kwargs,
                        logits_processor=LogitsProcessorList([wm.get_logits_processor()])
                    )

            generated = output_ids[0][input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            num_tokens = len(generated)

            kgw_result = watermarkers["kgw_static"].detect(text)
            z = kgw_result.get("z_score", "N/A")
            z_str = f"{z:.2f}" if isinstance(z, float) else str(z)

            print(f"\n  [{condition}] z={z_str} tokens={num_tokens}")
            print(f"  {text[:150]}...")

            trial_records.append({
                "prompt_idx":  p_idx,
                "instruction": instruction,
                "category":    prompt_item["category"],
                "condition":   condition,
                "output":      text,
                "num_tokens":  num_tokens,
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trial_path = os.path.join(OUTPUT_DIR, "trial_generations.jsonl")
    with open(trial_path, "w") as f:
        for record in trial_records:
            f.write(json.dumps(record) + "\n")

    print("\n" + "=" * 60)
    print("trial check complete.")
    print(f"Outputs saved to: {trial_path}")
    print("\nNext steps:")
    print(f"  Perplexity:      python score_perplexity.py --input {trial_path}")
    print(f"  Full generation: python run_experiment.py --mode generate")
    print("=" * 60)


def print_quick_stats():
    path = os.path.join(OUTPUT_DIR, GENERATIONS_FILE)
    if not os.path.exists(path):
        print("No generations file found yet.")
        return

    from collections import Counter
    conditions = Counter()
    errors = 0
    with open(path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("output"):
                    conditions[item["condition"]] += 1
                else:
                    errors += 1
            except Exception:
                pass

    print("\nGeneration progress:")
    for cond, count in sorted(conditions.items()):
        print(f"  {cond:<30} {count:>5} / 1000")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropy-Adaptive Watermarking Experiment")
    parser.add_argument(
        "--mode",
        choices=["trial", "generate", "detect", "all", "stats"],
        default="trial",
    )
    args = parser.parse_args()

    if args.mode == "trial":
        run_check()

    elif args.mode == "generate":
        from generate import run_generation
        run_generation()

    elif args.mode == "detect":
        from detect import run_detection
        run_detection()

    elif args.mode == "all":
        from generate import run_generation
        from detect import run_detection
        run_generation()
        run_detection()

    elif args.mode == "stats":
        print_quick_stats()