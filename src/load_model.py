"""Model loading helpers with pinned revisions for reproducibility."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PINNED_MODEL_REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "unsloth/Llama-3.2-1B-Instruct": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
}


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict[str, str]]:
    revision = PINNED_MODEL_REVISIONS.get(model_name)
    kwargs = {"revision": revision} if revision else {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, **kwargs
        ).to("cuda")
        device = "cuda"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.to("cpu")
        device = "cpu"

    info = {
        "model_name": model_name,
        "revision": revision or "unversioned",
        "device": device,
    }
    return model, tokenizer, info

