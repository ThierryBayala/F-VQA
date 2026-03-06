"""Evaluate task performance metrics (EM, BLEU-1/4, CIDEr, ROUGE-L) for all three models and print a comparison table."""

import argparse
import gc
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
from utils import load_config, load_checkpoint, compute_all_metrics, print_metrics_report


def build_and_load_model(projection_type: str, checkpoint_path: str, config: dict, llm, tokenizer):
    """Build model for given projection type and load checkpoint. Returns (model, main_device)."""
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

    return model, main_device


def run_generation_and_metrics(
    model,
    projection_type: str,
    loader: DataLoader,
    tokenizer,
    main_device: torch.device,
    max_length: int,
    max_new_tokens: int,
):
    """Run autoregressive generation for all batches and compute task metrics. Returns metrics dict."""
    prompt_prefix = "Question: "
    prompt_suffix = " Answer:"
    all_references = []
    all_predictions = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(main_device)
            questions = batch["question"]
            answers = batch["answer"]
            bs = images.size(0)

            if projection_type == "cross_attention":
                for i in range(bs):
                    prompt = prompt_prefix + questions[i] + prompt_suffix
                    tok = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    )
                    input_ids = tok["input_ids"].to(main_device)
                    prompt_len = input_ids.shape[1]
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
                        if next_id.item() == tokenizer.eos_token_id:
                            break
                    generated = tokenizer.decode(
                        input_ids[0][prompt_len:], skip_special_tokens=True
                    ).strip()
                    all_references.append(answers[i])
                    all_predictions.append(generated)
            else:
                vision_feat = model.vision(images)
                vision_emb = model.projector(vision_feat)
                for i in range(bs):
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
                    out = model.llm.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
                    generated = tokenizer.decode(
                        out[0][input_ids.shape[1] :], skip_special_tokens=True
                    ).strip()
                    all_references.append(answers[i])
                    all_predictions.append(generated)

    if not all_references or not all_predictions:
        return None
    return compute_all_metrics(all_references, all_predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate task metrics (EM, BLEU-1/4, CIDEr, ROUGE-L) for linear, qformer, cross_attention and print comparison table."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--split", type=str, default="validation[:1000]")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--md", type=str, default=None, help="Write markdown comparison table to this file")
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
            answer = item.get(
                "multiple_choice_answer",
                item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else str(item.get("answer", "")),
            )
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            out = super().__getitem__(idx)
            img = out["image"]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
            out["image"] = transform(img)
            out["question"] = question
            out["answer"] = answer
            return out

    print("Loading dataset and LLM (shared across models)...")
    hf_data = load_vqa_subset(
        dataset_name=data_cfg.get("dataset_name", "HuggingFaceM4/VQAv2"),
        split=args.split,
    )
    llm, tokenizer = load_frozen_llm(llm_name)
    llm.eval()

    dataset = EvalDataset(hf_data, tokenizer=tokenizer, image_size=image_size, max_length=max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    base_checkpoint_dir = Path(args.checkpoint_dir)
    approaches = ["linear", "qformer", "cross_attention"]
    checkpoint_name = "projector_last.pt"

    results = []  # list of {"approach": str, "metrics": dict}

    for projection_type in approaches:
        ckpt_path = base_checkpoint_dir / projection_type / checkpoint_name
        if not ckpt_path.exists():
            print(f"Skipping {projection_type}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n--- Evaluating {projection_type} ---")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        try:
            model, main_device = build_and_load_model(
                projection_type, str(ckpt_path), config, llm, tokenizer
            )
        except Exception as e:
            print(f"  Failed to load {projection_type}: {e}")
            continue

        metrics = run_generation_and_metrics(
            model,
            projection_type,
            loader,
            tokenizer,
            main_device,
            max_length,
            args.max_new_tokens,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if metrics is None:
            print(f"  No predictions for {projection_type}")
            continue

        results.append({"approach": projection_type, "metrics": metrics})
        print_metrics_report(metrics, title=f"Task Performance — {projection_type}")

    if not results:
        print("No results to compare.")
        return

    # ----- Comparison table (console) -----
    metric_names = ["Exact Match (EM)", "BLEU-1", "BLEU-4", "CIDEr", "ROUGE-L"]
    pct_metrics = {"Exact Match (EM)", "BLEU-1", "BLEU-4", "ROUGE-L"}

    print("\n" + "=" * 80)
    print("TASK PERFORMANCE COMPARISON (three models)")
    print("=" * 80)

    col_width = 14
    header = "Model".ljust(18) + "".join(m.ljust(col_width) for m in metric_names)
    print(header)
    print("-" * 80)
    for r in results:
        row = r["approach"].ljust(18)
        for name in metric_names:
            val = r["metrics"].get(name, 0.0)
            if name in pct_metrics:
                row += f"{val:.2f}%".ljust(col_width)
            else:
                row += f"{val:.4f}".ljust(col_width)
        print(row)
    print("=" * 80)

    # ----- Markdown table -----
    n_cols = 1 + len(metric_names)
    md_lines = [
        "",
        "| Model | " + " | ".join(metric_names) + " |",
        "|" + "|".join(["---"] * n_cols) + "|",
    ]
    for r in results:
        cells = [r["approach"]]
        for name in metric_names:
            val = r["metrics"].get(name, 0.0)
            if name in pct_metrics:
                cells.append(f"{val:.2f}%")
            else:
                cells.append(f"{val:.4f}")
        md_lines.append("| " + " | ".join(cells) + " |")
    md_table = "\n".join(md_lines)
    print("\nMarkdown table:\n" + md_table)

    if args.md:
        Path(args.md).write_text(
            "# Task performance comparison (EM, BLEU, CIDEr, ROUGE-L)\n\n" + md_table + "\n",
            encoding="utf-8",
        )
        print(f"\nMarkdown table written to {args.md}")


if __name__ == "__main__":
    main()
