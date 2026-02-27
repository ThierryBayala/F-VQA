"""Evaluate: validation loss and Task Performance Metrics (EM, BLEU-1/4, CIDEr, ROUGE-L)."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vision_encoder import VisionEncoder
from models.frozen_llm import load_frozen_llm
from models.fvqa_model import LinearVisionToLLM, FrozenVQA
from models.query_transformer import QueryTransformer
from data.vqa_dataset import load_vqa_subset, VQADataset
from utils import load_config, load_checkpoint
from metrics import compute_all_metrics, print_metrics_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Projector checkpoint")
    parser.add_argument("--split", type=str, default="validation[:1000]")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Max tokens for answer generation")
    parser.add_argument("--metrics_only", action="store_true", help="Skip validation loss, only run metrics")
    args = parser.parse_args()

    config = load_config(args.config) if Path(args.config).exists() else {}
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    llm_name = model_cfg.get("llm_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
      # Projector type: "linear" (LinearVisionToLLM) or "qformer" (QueryTransformer)
    projection_type = model_cfg.get("projection_type", "qformer").lower()
    vision_dim = model_cfg.get("vision_dim", 768)
    projection_hidden_dim = model_cfg.get("projection_hidden_dim", 0)
    projection_dropout = train_cfg.get("projection_dropout", 0.0)

    llm, tokenizer = load_frozen_llm(llm_name)
    llm.eval()
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
    else:
        projector = LinearVisionToLLM(
            vision_dim=vision_dim,
            llm_dim=llm.config.hidden_size,
            dropout=projection_dropout,
            hidden_dim=projection_hidden_dim,
        )
    load_checkpoint(args.checkpoint, model=projector)
    model = FrozenVQA(vision_encoder=vision_encoder, projector=projector, llm=llm)

    main_device = next(llm.parameters()).device
    vision_encoder.to(main_device)
    projector.to(main_device)

    image_size = data_cfg.get("image_size", 224)
    max_length = data_cfg.get("max_length", 256)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    hf_data = load_vqa_subset(
        dataset_name=data_cfg.get("dataset_name", "HuggingFaceM4/VQAv2"),
        split=args.split,
    )

    class EvalDataset(VQADataset):
        """Add raw question/answer and transformed image for evaluation."""

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

    dataset = EvalDataset(hf_data, tokenizer=tokenizer, image_size=image_size, max_length=max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()

    if not args.metrics_only:
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(main_device)
                input_ids = batch["input_ids"].to(main_device)
                attention_mask = batch["attention_mask"].to(main_device)
                labels = batch["labels"].to(main_device)
                outputs = model(
                    image=images,
                    question_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item() * images.size(0)
                n += images.size(0)
        val_loss = total_loss / max(n, 1)
        print(f"Validation loss: {val_loss:.4f}")

    # Generation for task metrics (EM, BLEU, CIDEr, ROUGE-L)
    prompt_prefix = "Question: "
    prompt_suffix = " Answer:"
    all_references = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(main_device)
            questions = batch["question"]
            answers = batch["answer"]
            bs = images.size(0)

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
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(
                    out[0][input_ids.shape[1] :], skip_special_tokens=True
                ).strip()
                all_references.append(answers[i])
                all_predictions.append(generated)

    if all_references and all_predictions:
        metrics = compute_all_metrics(all_references, all_predictions)
        print_metrics_report(metrics)
    else:
        print("No samples to compute metrics.")


if __name__ == "__main__":
    main()
