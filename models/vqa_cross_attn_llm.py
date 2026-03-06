"""VQA with cross-attention injection inside the LLM (vision as K,V in each decoder layer)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_attention_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(1, 1, L, L) causal mask: 0 attend, -inf don't."""
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)


class FrozenVQACrossAttn(nn.Module):
    """
    VQA with vision injected inside the LLM via cross-attention.
    After each decoder layer's self-attention + MLP, we add a trainable cross-attention:
    Q = decoder hidden states, K/V = projected vision token(s). LLM weights stay frozen.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        injector: "CrossAttnInjector",
        llm,
    ):
        super().__init__()
        self.vision = vision_encoder
        self.injector = injector
        self.llm = llm

    def forward(self, image, question_ids, attention_mask, labels=None):
        # Vision: (B, 768) -> (B, 1, llm_dim)
        vision_feat = self.vision(image)
        if vision_feat.dim() == 2:
            vision_feat = vision_feat.unsqueeze(1)
        proj_dtype = next(self.injector.vision_proj.parameters()).dtype
        vision_embeds = self.injector.vision_proj(vision_feat.to(proj_dtype))  # (B, N_v, llm_dim)

        # Text embeddings (no prefix replacement)
        inputs_embeds = self.llm.get_input_embeddings()(question_ids)
        B, L, D = inputs_embeds.shape

        base = self.llm.model
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        vision_embeds = vision_embeds.to(dtype)

        # Position ids and causal mask (training: no cache)
        position_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(L, device=device, dtype=torch.long)
        causal_mask = _causal_attention_mask(L, device, dtype)  # (1,1,L,L)
        if attention_mask is not None:
            causal_mask = causal_mask.expand(B, -1, -1, -1).clone()
            causal_mask = causal_mask.masked_fill(
                (1 - attention_mask).view(B, 1, 1, L).to(torch.bool),
                float("-inf"),
            )

        # Rotary embeddings from inner model
        position_embeddings = base.rotary_emb(inputs_embeds, position_ids=position_ids)

        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(base.layers):
            # Frozen decoder layer (self-attn + MLP)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # Trainable cross-attention: Q = hidden, K/V = vision
            # Scale injector output so we don't add large random values into the frozen LLM stream (avoids NaN).
            residual = hidden_states
            attn_out, _ = self.injector.cross_attn_layers[i](
                hidden_states,
                vision_embeds,
                vision_embeds,
                need_weights=False,
            )
            hidden_states = residual + self.injector.residual_scale.to(attn_out.dtype) * attn_out

        hidden_states = base.norm(hidden_states)
        logits = self.llm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            valid = shift_labels != -100
            if valid.any():
                # Compute loss in float32 and clamp logits to avoid overflow/NaN (e.g. with fp16/bf16).
                shift_logits_f = shift_logits.float().clamp(-50.0, 50.0)
                loss = F.cross_entropy(
                    shift_logits_f,
                    shift_labels,
                    ignore_index=-100,
                )
            else:
                # All labels ignored -> cross_entropy would return NaN; use 0 to avoid.
                loss = shift_logits.sum() * 0.0

        class Out:
            pass
        out = Out()
        out.logits = logits
        out.loss = loss
        return out


class CrossAttnInjector(nn.Module):
    """
    Trainable vision projection + one cross-attention layer per LLM decoder layer.
    Saved/loaded as a single 'projector' checkpoint for projection_type='cross_attention'.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_layers: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("residual_scale", torch.tensor(residual_scale, dtype=torch.float32))
        self.vision_proj = nn.Linear(vision_dim, llm_dim)
        # Small init so vision embeddings don't blow up attention scores (avoids NaN in fp16)
        nn.init.xavier_uniform_(self.vision_proj.weight, gain=0.01)
        if self.vision_proj.bias is not None:
            nn.init.zeros_(self.vision_proj.bias)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=llm_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self._zero_init_cross_attn_out_proj()

    def _zero_init_cross_attn_out_proj(self):
        """Zero-initialize output projection of each cross-attn so at init vision adds nothing (avoids NaN)."""
        for layer in self.cross_attn_layers:
            nn.init.zeros_(layer.out_proj.weight)
            if layer.out_proj.bias is not None:
                nn.init.zeros_(layer.out_proj.bias)

    @classmethod
    def from_llm_config(
        cls,
        llm,
        vision_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        llm_dim = llm.config.hidden_size
        num_layers = llm.config.num_hidden_layers
        return cls(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            residual_scale=residual_scale,
        )
