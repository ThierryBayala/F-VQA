"""Frozen causal LLM (e.g. TinyLlama) for VQA generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_frozen_llm(model_name: str, dtype: torch.dtype = torch.float16):
    """Load LLM and tokenizer; freeze all parameters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
    )
    for p in llm.parameters():
        p.requires_grad = False
    return llm, tokenizer
