import torch
from transformers import AutoTokenizer

from config import (
    KGW_GAMMA, KGW_DELTA, KGW_SEEDING_SCHEME,
    SYNTHID_KEYS, SYNTHID_NGRAM_LEN, SYNTHID_SAMPLING_TABLE_SIZE,
    SYNTHID_CONTEXT_HISTORY_SIZE, ENTROPY_SCALING, ENTROPY_THRESHOLD,
    ENTROPY_EPS, MIN_DELTA_FRACTION,
)

def compute_h_norm(scores: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(scores, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + ENTROPY_EPS), dim=-1)
    vocab_size = scores.shape[-1]
    H_max = torch.log(torch.tensor(vocab_size, dtype=torch.float, device=scores.device))
    return entropy / H_max


def entropy_scale(H_norm: torch.Tensor) -> torch.Tensor:
    if ENTROPY_SCALING == "linear":
        return H_norm
    elif ENTROPY_SCALING == "quadratic":
        return H_norm ** 2
    elif ENTROPY_SCALING == "sqrt":
        return H_norm ** 0.5
    elif ENTROPY_SCALING == "threshold":
        t = ENTROPY_THRESHOLD
        return torch.clamp((H_norm - t) / (1.0 - t), min=0.0, max=1.0)
    else:
        raise ValueError(f"Unknown ENTROPY_SCALING: {ENTROPY_SCALING}")

try:
    from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
except ImportError:
    raise ImportError(
        "Could not import extended_watermark_processor. "
        "Download from: https://github.com/jwkirchenbauer/lm-watermarking"
    )

try:
    from transformers import SynthIDTextWatermarkingConfig
    from transformers.generation.watermarking import SynthIDTextWatermarkLogitsProcessor
except ImportError:
    raise ImportError(
        "SynthID requires transformers >= 4.46.0. "
        "Run: pip install --upgrade transformers"
    )

def _make_synthid_config():
    return SynthIDTextWatermarkingConfig(
        keys=SYNTHID_KEYS,
        ngram_len=SYNTHID_NGRAM_LEN,
        sampling_table_size=SYNTHID_SAMPLING_TABLE_SIZE,
        context_history_size=SYNTHID_CONTEXT_HISTORY_SIZE,
    )

def _synthid_mean_detect(text: str, tokenizer, processor) -> dict:
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(processor.device)
        seq_len = input_ids.shape[1]

        if seq_len < processor.ngram_len + 1:
            return {"error": "text too short", "score": None, "prediction": None}

        g_values = processor.compute_g_values(input_ids).float()
        g_seq_len = g_values.shape[1]

        if g_seq_len == 0:
            return {"error": "no tokens to score", "score": None, "prediction": None}

        mask = torch.ones(1, g_seq_len, device=processor.device)

        if tokenizer.eos_token_id is not None:
            for pos in range(g_seq_len):
                src_pos = pos + (processor.ngram_len - 1)
                if src_pos < seq_len and input_ids[0, src_pos] == tokenizer.eos_token_id:
                    mask[0, pos] = 0.0

        num_unmasked = mask.sum().item()
        if num_unmasked == 0:
            return {"error": "no tokens to score", "score": None, "prediction": None}

        watermarking_depth = g_values.shape[-1]

        # score = sum(g_values * mask_expanded) / (depth * num_unmasked)
        mask_expanded = mask.unsqueeze(-1)
        score = (g_values * mask_expanded).sum() / (watermarking_depth * num_unmasked)
        mean_score = score.item()

        vocab_size = tokenizer.vocab_size
        threshold = processor.expected_mean_g_value(vocab_size)

        return {
            "score": mean_score,
            "prediction": bool(mean_score > threshold),
            "threshold": float(threshold),
        }
    except Exception as e:
        return {"error": str(e), "score": None, "prediction": None}

class BaselineWatermarker:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.name = "baseline"
        self._kgw_detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=KGW_GAMMA,
            seeding_scheme=KGW_SEEDING_SCHEME,
            device="cuda",
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
        self._synthid_config = _make_synthid_config()
        self._synthid_processor_for_detection = SynthIDTextWatermarkLogitsProcessor(
            **self._synthid_config.to_dict(), device="cuda"
        )

    def get_logits_processor(self):
        return None

    def detect_kgw(self, text: str) -> dict:
        try:
            result = self._kgw_detector.detect(text)
            return {
                "z_score": result["z_score"],
                "prediction": result["prediction"],
                "green_fraction": result["green_fraction"],
                "num_tokens_scored": result["num_tokens_scored"],
            }
        except Exception as e:
            return {"error": str(e), "z_score": None, "prediction": None}

    def detect_synthid(self, text: str) -> dict:
        return _synthid_mean_detect(text, self.tokenizer, self._synthid_processor_for_detection)

class KGWWatermarker:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=KGW_GAMMA,
            delta=KGW_DELTA,
            seeding_scheme=KGW_SEEDING_SCHEME,
        )
        self.detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=KGW_GAMMA,
            seeding_scheme=KGW_SEEDING_SCHEME,
            device="cuda",
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
        self.name = "kgw_static"

    def get_logits_processor(self):
        return self.processor

    def detect(self, text: str) -> dict:
        try:
            result = self.detector.detect(text)
            return {
                "z_score": result["z_score"],
                "prediction": result["prediction"],
                "num_tokens_scored": result["num_tokens_scored"],
                "num_green_tokens": result["num_green_tokens"],
                "green_fraction": result["green_fraction"],
            }
        except Exception as e:
            return {"error": str(e), "z_score": None, "prediction": None}

class EntropyKGWWatermarker(KGWWatermarker):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        self.name = "kgw_entropy"
        self.processor = self._make_entropy_processor(tokenizer)

    def _make_entropy_processor(self, tokenizer):
        base_delta = KGW_DELTA
        min_frac = MIN_DELTA_FRACTION

        class EntropyScaledKGWProcessor(WatermarkLogitsProcessor):

            def _bias_greenlist_logits(self, scores, greenlist_mask, greenlist_bias):
                H_norm = compute_h_norm(scores)                        
                scale = entropy_scale(H_norm)                          
                effective_scale = min_frac + (1.0 - min_frac) * scale # main change 
                scaled_delta = base_delta * effective_scale            

                for b_idx in range(scores.shape[0]):
                    scores[b_idx][greenlist_mask[b_idx]] += scaled_delta[b_idx].item()

                return scores

        return EntropyScaledKGWProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=KGW_GAMMA,
            delta=KGW_DELTA,
            seeding_scheme=KGW_SEEDING_SCHEME,
        )

class SynthIDWatermarker:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.config = _make_synthid_config()
        self.name = "synthid_static"
        self._detection_processor = SynthIDTextWatermarkLogitsProcessor(
            **self.config.to_dict(), device="cuda"
        )

    def get_watermarking_config(self):
        return self.config

    def detect(self, text: str) -> dict:
        return _synthid_mean_detect(text, self.tokenizer, self._detection_processor)


class EntropySynthIDWatermarker(SynthIDWatermarker):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        self.name = "synthid_entropy"
        self._entropy_processor = self._make_entropy_processor()

    def _make_entropy_processor(self):
        config = self.config
        class EntropySynthIDProcessor(SynthIDTextWatermarkLogitsProcessor):
            def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                H_norm = compute_h_norm(scores)     
                lam = entropy_scale(H_norm)

                apply_mask = torch.bernoulli(lam).bool()  

                original_scores = scores.clone()

                watermarked_scores = super().__call__(input_ids, scores)

                output = torch.where(
                    apply_mask.unsqueeze(-1),
                    watermarked_scores,
                    original_scores,
                )
                return output

        return EntropySynthIDProcessor(**self.config.to_dict(), device="cuda")

    def get_logits_processor(self):
        return self._entropy_processor


def get_all_watermarkers(tokenizer: AutoTokenizer) -> dict:
    watermarkers = {
        "baseline":        BaselineWatermarker(tokenizer),
        "kgw_static":      KGWWatermarker(tokenizer),
        "kgw_entropy":     EntropyKGWWatermarker(tokenizer),
        "synthid_static":  SynthIDWatermarker(tokenizer),
        "synthid_entropy": EntropySynthIDWatermarker(tokenizer),
    }
    print(f"Initialized: {list(watermarkers.keys())}")
    return watermarkers