import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList

from config import (
    MODEL_NAME, DEVICE, TORCH_DTYPE,
    MAX_NEW_TOKENS, MIN_NEW_TOKENS, TEMPERATURE, DO_SAMPLE,
    OUTPUT_DIR, GENERATIONS_FILE,
)
from data import load_prompts, format_prompt_for_llama
from watermarkers import get_all_watermarkers


def load_model_and_tokenizer():
    print(f"Loading model: {MODEL_NAME}")
    dtype = torch.float16 if TORCH_DTYPE == "float16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded on {DEVICE}. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


def generate_one(model, tokenizer, formatted_prompt: str, condition: str, watermarkers: dict,) -> tuple[str, int]:
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    wm = watermarkers[condition]

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        if condition == "synthid_static":
            output_ids = model.generate(
                **generate_kwargs,
                watermarking_config=wm.get_watermarking_config(),
            )
        elif condition == "synthid_entropy":
            output_ids = model.generate(
                **generate_kwargs,
                logits_processor=LogitsProcessorList([wm.get_logits_processor()]),
            )
        else:
            output_ids = model.generate(
                **generate_kwargs,
                logits_processor=LogitsProcessorList([wm.get_logits_processor()]),
            )

    generated_ids = output_ids[0][input_len:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    num_tokens = len(generated_ids)

    return output_text, num_tokens


def run_generation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, GENERATIONS_FILE)

    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    completed.add((item["prompt_idx"], item["condition"]))
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} generations already saved.")

    prompts = load_prompts()
    model, tokenizer = load_model_and_tokenizer()
    watermarkers = get_all_watermarkers(tokenizer)

    conditions = ["baseline", "kgw_static", "kgw_entropy", "synthid_static", "synthid_entropy"]

    total = len(prompts) * len(conditions)
    done = len(completed)

    print(f"\nStarting generation: {len(prompts)} prompts × {len(conditions)} conditions = {total} total")
    print(f"Already done: {done} | Remaining: {total - done}\n")

    with open(output_path, "a") as out_f:
        for prompt_idx, prompt_item in enumerate(prompts):
            instruction = prompt_item["instruction"]
            category = prompt_item["category"]
            formatted = format_prompt_for_llama(instruction, tokenizer)

            for condition in conditions:
                if (prompt_idx, condition) in completed:
                    continue

                try:
                    output_text, num_tokens = generate_one(
                        model, tokenizer, formatted, condition, watermarkers
                    )

                    record = {
                        "prompt_idx":  prompt_idx,
                        "instruction": instruction,
                        "category":    category,
                        "condition":   condition,
                        "output":      output_text,
                        "num_tokens":  num_tokens,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    done += 1
                    if done % 50 == 0:
                        print(f"  [{done}/{total}] prompt={prompt_idx} condition={condition} tokens={num_tokens}")

                except Exception as e:
                    print(f"  ERROR at prompt={prompt_idx} condition={condition}: {e}")
                    error_record = {
                        "prompt_idx":  prompt_idx,
                        "instruction": instruction,
                        "category":    category,
                        "condition":   condition,
                        "output":      None,
                        "num_tokens":  0,
                        "error":       str(e),
                    }
                    out_f.write(json.dumps(error_record) + "\n")
                    out_f.flush()

    print(f"\nGeneration complete. Output saved to: {output_path}")


if __name__ == "__main__":
    run_generation()