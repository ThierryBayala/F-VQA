"""Train only the projector; frozen vision + frozen LLM. Validates pipeline."""

import argparse
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
from utils import load_config, get_device, save_checkpoint, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = load_config(args.config) if Path(args.config).exists() else {}
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    dataset_name = data_cfg.get("dataset_name", "HuggingFaceM4/VQAv2")
    split = data_cfg.get("train_split", "train[:5000]")
    batch_size = data_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 0)
    image_size = data_cfg.get("image_size", 224)
    max_length = data_cfg.get("max_length", 256)

    model_cfg = config.get("model", {})
    llm_name = model_cfg.get("llm_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    llm_dtype_str = model_cfg.get("llm_dtype", "float32")
    llm_dtype = getattr(torch, llm_dtype_str.strip().lower()) if isinstance(llm_dtype_str, str) else llm_dtype_str
    # Projector type: "linear" | "qformer" | "cross_attention" (injection inside LLM)
    projection_type = model_cfg.get("projection_type", "qformer").lower()
    vision_dim = model_cfg.get("vision_dim", 768)
    projection_hidden_dim = model_cfg.get("projection_hidden_dim", 0)
    lr = train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    projection_dropout = train_cfg.get("projection_dropout", 0.0)
    epochs = train_cfg.get("epochs", 3)
    log_every = train_cfg.get("log_every", 10)
    save_every = train_cfg.get("save_every", 1)
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    base_checkpoint_dir = Path(args.checkpoint_dir or paths_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir = base_checkpoint_dir / projection_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and frozen LLM...")
    llm, tokenizer = load_frozen_llm(llm_name, dtype=llm_dtype)
    llm.eval()

    print("Loading vision encoder and projector...")
    vision_encoder = VisionEncoder()
    llm_hidden = llm.config.hidden_size

    if projection_type == "qformer":
        num_query_tokens = model_cfg.get("qformer_num_query_tokens", 16)
        num_layers = model_cfg.get("qformer_num_layers", 2)
        num_heads = model_cfg.get("qformer_num_heads", 8)
        qformer_dropout = model_cfg.get("qformer_dropout", projection_dropout)
        projector = QueryTransformer(
            vision_dim=vision_dim,
            llm_dim=llm_hidden,
            num_query_tokens=num_query_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=qformer_dropout,
        )
        model = FrozenVQA(vision_encoder=vision_encoder, projector=projector, llm=llm)
    elif projection_type == "cross_attention":
        injector = CrossAttnInjector.from_llm_config(
            llm,
            vision_dim=vision_dim,
            num_heads=model_cfg.get("cross_attention_num_heads", 8),
            dropout=model_cfg.get("cross_attention_dropout", projection_dropout),
            residual_scale=model_cfg.get("cross_attention_residual_scale", 0.1),
        )
        projector = injector  # saved as "projector" checkpoint
        model = FrozenVQACrossAttn(vision_encoder=vision_encoder, injector=injector, llm=llm)
    else:
        projector = LinearVisionToLLM(
            vision_dim=vision_dim,
            llm_dim=llm_hidden,
            dropout=projection_dropout,
            hidden_dim=projection_hidden_dim,
        )
        model = FrozenVQA(vision_encoder=vision_encoder, projector=projector, llm=llm)

    main_device = next(llm.parameters()).device
    llm_dtype = next(llm.parameters()).dtype
    vision_encoder = vision_encoder.to(main_device)
    projector = projector.to(main_device).to(llm_dtype)

    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)

    print("Loading VQA subset...")
    hf_split = split if "[" in split else f"{split}[:5000]"
    hf_data = load_vqa_subset(dataset_name=dataset_name, split=hf_split)

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class TransformDataset(VQADataset):
        def __getitem__(self, idx):
            out = super().__getitem__(idx)
            out["image"] = image_transform(out["image"].convert("RGB"))
            return out

    train_dataset = TransformDataset(
        hf_data,
        tokenizer=tokenizer,
        image_size=image_size,
        max_length=max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Run training only when no checkpoint exists
    last_ckpt = checkpoint_dir / "projector_last.pt"
    if last_ckpt.exists():
        print(f"Checkpoint found at {last_ckpt}; loading and skipping training.")
        load_checkpoint(str(last_ckpt), model=projector, device=main_device)
        print("Done. Use this checkpoint for evaluation.")
        return

    print("No checkpoint found. Training (projector only)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
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
            loss = outputs.loss
            if not torch.isfinite(loss).item():
                print(f"Epoch {epoch + 1} step {step + 1} loss=nan (skipping step)")
                continue
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), grad_clip_norm)
            optimizer.step()

            if (step + 1) % log_every == 0:
                print(f"Epoch {epoch + 1} step {step + 1} loss={loss.item():.4f}")

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1} avg_loss={avg_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            path = checkpoint_dir / f"projector_epoch_{epoch + 1}.pt"
            save_checkpoint(projector, optimizer, epoch + 1, str(path))

    save_checkpoint(projector, optimizer, epochs, str(checkpoint_dir / "projector_last.pt"))
    print("Training done. Pipeline validated: loss decreased, only projector trained.")


if __name__ == "__main__":
    main()
