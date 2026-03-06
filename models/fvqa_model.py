"""Visual → LLM projection and full Frozen VQA model."""

import torch
import torch.nn as nn


class LinearVisionToLLM(nn.Module):
    """Projects vision features into LLM embedding space. Only trainable part.
    Single linear (hidden_dim=0) or bottleneck 768→hidden_dim→llm_dim with ReLU (hidden_dim>0).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        llm_dim: int = 2048,
        dropout: float = 0.0,
        hidden_dim: int = 0,
    ):
        super().__init__()
        if hidden_dim is None or hidden_dim <= 0:
            self.proj = nn.Linear(vision_dim, llm_dim)
            self.fc1 = None
            self.act = None
            self.fc2 = None
        else:
            self.proj = None
            self.fc1 = nn.Linear(vision_dim, hidden_dim)
            self.act = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_dim, llm_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, vision_feat: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            return self.dropout(self.proj(vision_feat))
        x = self.fc1(vision_feat)
        x = self.act(x)
        return self.dropout(self.fc2(x))


class FrozenVQA(nn.Module):
    """Minimal VQA: frozen vision + trainable projector + frozen LLM. Visual token as prefix."""

    def __init__(self, vision_encoder, projector: LinearVisionToLLM, llm):
        super().__init__()
        self.vision = vision_encoder
        self.projector = projector
        self.llm = llm

    def forward(self, image, question_ids, attention_mask, labels=None):
        vision_feat = self.vision(image)
        proj_dtype = next(self.projector.parameters()).dtype
        vision_emb = self.projector(vision_feat.to(proj_dtype))

        inputs_embeds = self.llm.get_input_embeddings()(question_ids)
        inputs_embeds[:, 0, :] = vision_emb.to(inputs_embeds.dtype)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs
