import random
from datasets import load_dataset
from config import DATASET_NAME, CATEGORY_SAMPLES, RANDOM_SEED


def load_prompts() -> list[dict]:
    random.seed(RANDOM_SEED)

    print(f"Loading dataset: {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train")

    by_category: dict[str, list[str]] = {}
    for item in ds:
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = []
        instruction = item["instruction"].strip()
        if len(instruction) >= 10:
            by_category[cat].append(instruction)

    selected = []
    for cat, n in CATEGORY_SAMPLES.items():
        pool = by_category.get(cat, [])
        if len(pool) < n:
            raise ValueError(
                f"Category '{cat}' only has {len(pool)} items but {n} requested."
            )
        sampled = random.sample(pool, n)
        for instruction in sampled:
            selected.append({"instruction": instruction, "category": cat})

    random.shuffle(selected)

    print(f"Loaded {len(selected)} prompts across {len(CATEGORY_SAMPLES)} categories.")
    _print_category_counts(selected)
    return selected


def _print_category_counts(prompts: list[dict]):
    from collections import Counter
    counts = Counter(p["category"] for p in prompts)
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<30} {n}")


def format_prompt_for_llama(instruction: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question clearly and concisely."
        },
        {
            "role": "user",
            "content": instruction
        }
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted


if __name__ == "__main__":
    prompts = load_prompts()
    print(f"\nExample prompt:\n{prompts[0]['instruction']}")
    print(f"Category: {prompts[0]['category']}")