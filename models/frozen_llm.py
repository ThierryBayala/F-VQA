"""Frozen causal LLM (e.g. TinyLlama) for VQA generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_frozen_llm(model_name: str, dtype=None):
    """Load LLM and tokenizer; freeze all parameters.
    dtype: torch.dtype or str 'float32'/'float16'/'bfloat16'. Default float32 for training stability.
    """
    if dtype is None:
        dtype = torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype.strip().lower())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    for p in llm.parameters():
        p.requires_grad = False
    return llm, tokenizer
