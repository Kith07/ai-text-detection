import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ATTACKER_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE          = "cuda"
OUTPUT_DIR      = "./outputs"
INPUT_FILE      = "generations.jsonl"
OUTPUT_FILE     = "generations_attacked.jsonl"

CONDITIONS_TO_ATTACK = ["kgw_static", "kgw_entropy", "synthid_static", "synthid_entropy"]

MAX_NEW_TOKENS  = 450
TEMPERATURE     = 0.7
DO_SAMPLE       = True
BATCH_SIZE      = 8

SYSTEM_PROMPT = "You are a helpful assistant that rewrites text."

def make_paraphrase_prompt(text: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Rewrite the following text in your own words. "
                "Preserve the meaning and all key information, but use different "
                "vocabulary, sentence structure, and phrasing. "
                "Do not add new information or commentary. "
                "Output only the rewritten text, nothing else.\n\n"
                f"Text to rewrite:\n{text}"
            ),
        },
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted

def load_attacker():
    print(f"Loading attacker model: {ATTACKER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ATTACKER_MODEL,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()
    print(f"Attacker loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return model, tokenizer

def paraphrase_batch(texts: list[str], model, tokenizer) -> list[str | None]:
    valid = [(i, t) for i, t in enumerate(texts) if t and len(t.strip()) >= 20]
    results = [None] * len(texts)

    if not valid:
        return results

    indices, valid_texts = zip(*valid)
    prompts = [make_paraphrase_prompt(t, tokenizer) for t in valid_texts]

    tokenizer.padding_side = "left"

    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        truncation=True,
        max_length=768,
        padding=True,
    ).to(DEVICE)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    for batch_i, orig_i in enumerate(indices):
        generated = output_ids[batch_i][input_len:]
        paraphrased = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results[orig_i] = paraphrased if len(paraphrased) >= 20 else None

    return results

def run_attack():
    input_path  = os.path.join(OUTPUT_DIR, INPUT_FILE)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    generations = []
    with open(input_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("output") and len(item["output"].strip()) > 0:
                    generations.append(item)
            except Exception:
                pass

    print(f"Loaded {len(generations)} generations.")

    to_attack = [g for g in generations if g["condition"] in CONDITIONS_TO_ATTACK]
    baseline  = [g for g in generations if g["condition"] == "baseline"]

    print(f"Conditions to attack: {CONDITIONS_TO_ATTACK}")
    print(f"Outputs to paraphrase: {len(to_attack)}")
    print(f"Baseline outputs (kept as-is): {len(baseline)}")

    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("output"):
                        completed.add((item["prompt_idx"], item["condition"]))
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} already attacked.")

    baseline_done = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item["condition"] == "baseline":
                        baseline_done.add(item["prompt_idx"])
                except Exception:
                    pass

    with open(output_path, "a") as out_f:
        for item in baseline:
            if item["prompt_idx"] not in baseline_done:
                out_f.write(json.dumps(item) + "\n")
                out_f.flush()

    print("Baseline entries written. Loading attacker model...")

    model, tokenizer = load_attacker()

    total = len(to_attack)
    done  = len(completed)
    errors = 0

    print(f"\nStarting paraphrasing: {total - done} remaining\n")

    remaining = [g for g in to_attack if (g["prompt_idx"], g["condition"]) not in completed]

    print(f"  {len(remaining)} texts to paraphrase in batches of {BATCH_SIZE}")

    with open(output_path, "a") as out_f:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start:batch_start + BATCH_SIZE]

            try:
                texts = [g["output"] for g in batch]
                paraphrased_list = paraphrase_batch(texts, model, tokenizer)

                for gen, paraphrased in zip(batch, paraphrased_list):
                    prompt_idx = gen["prompt_idx"]
                    condition  = gen["condition"]

                    if paraphrased is None:
                        paraphrased = gen["output"]

                    record = {
                        "prompt_idx":      prompt_idx,
                        "instruction":     gen["instruction"],
                        "category":        gen["category"],
                        "condition":       condition,
                        "output":          paraphrased,
                        "num_tokens":      len(tokenizer(paraphrased)["input_ids"]),
                        "attacked":        True,
                        "original_output": gen["output"],
                    }

                    out_f.write(json.dumps(record) + "\n")
                    done += 1

                out_f.flush()

                if done % 100 == 0:
                    last = batch[-1]
                    print(f"  [{done}/{total}] prompt={last['prompt_idx']} condition={last['condition']}")

            except Exception as e:
                errors += 1
                print(f"  ERROR batch starting at {batch_start}: {e}")
                import traceback; traceback.print_exc()
                for gen in batch:
                    fallback = dict(gen)
                    fallback["attacked"] = False
                    fallback["error"] = str(e)
                    out_f.write(json.dumps(fallback) + "\n")
                    done += 1
                out_f.flush()

    print(f"\nAttack complete.")
    print(f"  Total paraphrased: {done}")
    print(f"  Errors: {errors}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    run_attack()