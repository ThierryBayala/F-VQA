"""Benchmark the three projection approaches (linear, qformer, cross_attention) on response time and memory."""

import argparse
import gc
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vision_encoder import VisionEncoder
from models.frozen_llm import load_frozen_llm
from models.fvqa_model import LinearVisionToLLM, FrozenVQA
from models.query_transformer import QueryTransformer
from models.vqa_cross_attn_llm import FrozenVQACrossAttn, CrossAttnInjector
from data.vqa_dataset import load_vqa_subset, VQADataset
from utils import load_config, load_checkpoint


def build_and_load_model(projection_type: str, checkpoint_path: str, config: dict, llm, tokenizer):
    """Build vision encoder + projector/injector + attach to LLM. Load checkpoint into trainable part."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    vision_dim = model_cfg.get("vision_dim", 768)
    projection_hidden_dim = model_cfg.get("projection_hidden_dim", 0)
    projection_dropout = train_cfg.get("projection_dropout", 0.0)

    vision_encoder = VisionEncoder()

    if projection_type == "qformer":
        num_query_tokens = model_cfg.get("qformer_num_query_tokens", 16)
        num_layers = model_cfg.get("qformer_num_layers", 2)
        num_heads = model_cfg.get("qformer_num_heads", 8)
        qformer_dropout = model_cfg.get("qformer_dropout", projection_dropout)
        projector = QueryTransformer(
            vision_dim=vision_dim,
            llm_dim=llm.config.hidden_size,
            num_query_tokens=num_query_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=qformer_dropout,
        )
        load_checkpoint(checkpoint_path, model=projector)
        model = FrozenVQA(vision_encoder=vision_encoder, projector=projector, llm=llm)
    elif projection_type == "cross_attention":
        injector = CrossAttnInjector.from_llm_config(
            llm,
            vision_dim=vision_dim,
            num_heads=model_cfg.get("cross_attention_num_heads", 8),
            dropout=model_cfg.get("cross_attention_dropout", projection_dropout),
        )
        load_checkpoint(checkpoint_path, model=injector)
        model = FrozenVQACrossAttn(vision_encoder=vision_encoder, injector=injector, llm=llm)
    else:
        projector = LinearVisionToLLM(
            vision_dim=vision_dim,
            llm_dim=llm.config.hidden_size,
            dropout=projection_dropout,
            hidden_dim=projection_hidden_dim,
        )
        load_checkpoint(checkpoint_path, model=projector)
        model = FrozenVQA(vision_encoder=vision_encoder, projector=projector, llm=llm)

    main_device = next(llm.parameters()).device
    llm_dtype = next(llm.parameters()).dtype
    vision_encoder.to(main_device)
    if projection_type == "cross_attention":
        model.injector.to(main_device).to(llm_dtype)
    else:
        model.projector.to(main_device).to(llm_dtype)

    return model, main_device, llm_dtype


def run_inference_batch(
    model,
    projection_type: str,
    loader,
    tokenizer,
    main_device,
    max_length: int,
    max_new_tokens: int,
    num_samples: int,
):
    """Run inference for up to num_samples; return list of (elapsed_sec, num_tokens_generated)."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    prompt_prefix = "Question: "
    prompt_suffix = " Answer:"
    results = []  # (time_per_sample, tokens_generated)
    count = 0

    with torch.no_grad():
        for batch in loader:
            if count >= num_samples:
                break
            images = batch["image"].to(main_device)
            questions = batch["question"]
            bs = images.size(0)

            if projection_type == "cross_attention":
                for i in range(bs):
                    if count >= num_samples:
                        break
                    prompt = prompt_prefix + questions[i] + prompt_suffix
                    tok = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    )
                    input_ids = tok["input_ids"].to(main_device)
                    prompt_len = input_ids.shape[1]
                    num_tokens = 0
                    t0 = time.perf_counter()
                    for _ in range(max_new_tokens):
                        attn = (input_ids != pad_id).long().to(main_device)
                        out = model(
                            image=images[i : i + 1],
                            question_ids=input_ids,
                            attention_mask=attn,
                            labels=None,
                        )
                        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_id], dim=1)
                        num_tokens += 1
                        if next_id.item() == tokenizer.eos_token_id:
                            break
                    if main_device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    results.append((elapsed, num_tokens))
                    count += 1
            else:
                vision_feat = model.vision(images)
                vision_emb = model.projector(vision_feat)
                for i in range(bs):
                    if count >= num_samples:
                        break
                    prompt = prompt_prefix + questions[i] + prompt_suffix
                    tok = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    )
                    input_ids = tok["input_ids"].to(main_device)
                    attention_mask = tok["attention_mask"].to(main_device)
                    inputs_embeds = model.llm.get_input_embeddings()(input_ids)
                    inputs_embeds[:, 0, :] = vision_emb[i : i + 1]
                    t0 = time.perf_counter()
                    out = model.llm.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
                    if main_device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    num_tokens = out.shape[1] - input_ids.shape[1]
                    results.append((elapsed, num_tokens))
                    count += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark linear, qformer, cross_attention: time & memory.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="validation[:500]")
    parser.add_argument("--md", type=str, default=None, help="Write markdown table to this file (e.g. benchmark_results.md)")
    args = parser.parse_args()

    config = load_config(args.config) if Path(args.config).exists() else {}
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    llm_name = model_cfg.get("llm_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    image_size = data_cfg.get("image_size", 224)
    max_length = data_cfg.get("max_length", 256)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class EvalDataset(VQADataset):
        def __getitem__(self, idx):
            item = self.data[idx]
            question = item.get("question", "")
            out = super().__getitem__(idx)
            img = out["image"]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
            out["image"] = transform(img)
            out["question"] = question
            return out

    hf_data = load_vqa_subset(
        dataset_name=data_cfg.get("dataset_name", "HuggingFaceM4/VQAv2"),
        split=args.split,
    )
    # Load LLM once and reuse for all approaches
    print("Loading LLM and tokenizer (shared across approaches)...")
    llm, tokenizer = load_frozen_llm(llm_name)
    llm.eval()

    dataset = EvalDataset(hf_data, tokenizer=tokenizer, image_size=image_size, max_length=max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    base_checkpoint_dir = Path(args.checkpoint_dir)
    approaches = ["linear", "qformer", "cross_attention"]
    checkpoint_name = "projector_last.pt"

    results_table = []

    for projection_type in approaches:
        ckpt_path = base_checkpoint_dir / projection_type / checkpoint_name
        if not ckpt_path.exists():
            print(f"Skipping {projection_type}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n--- Benchmarking {projection_type} ---")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

        try:
            model, main_device, _ = build_and_load_model(
                projection_type, str(ckpt_path), config, llm, tokenizer
            )
            model.eval()
        except Exception as e:
            print(f"  Failed to load {projection_type}: {e}")
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        run_results = run_inference_batch(
            model,
            projection_type,
            loader,
            tokenizer,
            main_device,
            max_length,
            args.max_new_tokens,
            args.num_samples,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / (1024 ** 2)
        else:
            peak_mb = 0.0

        if not run_results:
            print(f"  No samples run for {projection_type}")
            continue

        total_time = sum(r[0] for r in run_results)
        total_tokens = sum(r[1] for r in run_results)
        n = len(run_results)
        mean_time_per_sample = total_time / n
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

        results_table.append({
            "approach": projection_type,
            "samples": n,
            "total_time_s": total_time,
            "mean_time_per_sample_s": mean_time_per_sample,
            "total_tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "peak_gpu_mb": peak_mb,
        })
        print(f"  Samples: {n}, Total time: {total_time:.2f}s, Mean time/sample: {mean_time_per_sample:.3f}s")
        print(f"  Total tokens: {total_tokens}, Tokens/s: {tokens_per_sec:.1f}, Peak GPU: {peak_mb:.1f} MB")

        # Free model so next approach gets clean memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY: Response time and memory (three approaches)")
    print("=" * 70)
    if not results_table:
        print("No results.")
        return

    header = f"{'Approach':<18} {'Mean time/sample (s)':>22} {'Total time (s)':>14} {'Peak GPU (MB)':>14} {'Tokens/s':>10}"
    print(header)
    print("-" * 70)
    for r in results_table:
        print(
            f"{r['approach']:<18} {r['mean_time_per_sample_s']:>22.3f} {r['total_time_s']:>14.2f} "
            f"{r['peak_gpu_mb']:>14.1f} {r['tokens_per_sec']:>10.1f}"
        )
    print("=" * 70)

    # Markdown table
    md_lines = [
        "",
        "| Approach | Samples | Mean time/sample (s) | Total time (s) | Total tokens | Tokens/s | Peak GPU (MB) |",
        "|----------|---------|----------------------|----------------|--------------|----------|---------------|",
    ]
    for r in results_table:
        md_lines.append(
            f"| {r['approach']} | {r['samples']} | {r['mean_time_per_sample_s']:.3f} | "
            f"{r['total_time_s']:.2f} | {r['total_tokens']} | {r['tokens_per_sec']:.1f} | {r['peak_gpu_mb']:.1f} |"
        )
    md_table = "\n".join(md_lines)
    print("\nMarkdown table:\n" + md_table)

    if args.md:
        Path(args.md).write_text("# Benchmark results\n\n" + md_table + "\n", encoding="utf-8")
        print(f"\nMarkdown table written to {args.md}")


if __name__ == "__main__":
    main()
