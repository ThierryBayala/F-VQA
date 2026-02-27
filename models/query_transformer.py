"""Query-Former style projector: vision features → LLM embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTransformer(nn.Module):
    """Q-Former style projector.

    Consumes vision features and produces a single LLM-dim embedding that can be
    used as the visual prefix token in the FrozenVQA model, making it a
    drop-in replacement for the existing linear projector.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        llm_dim: int = 2048,
        num_query_tokens: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_query_tokens = num_query_tokens

        # Learnable query tokens (K, Dl) broadcast across batch
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, llm_dim)
        )

        # Project vision features to LLM dim
        self.vision_proj = nn.Linear(vision_dim, llm_dim)

        # Transformer blocks operating on the query tokens, attending to vision
        self.layers = nn.ModuleList(
            [
                QueryBlock(
                    dim=llm_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            vision_features: Tensor of shape (B, Dv) or (B, N, Dv)

        Returns:
            Tensor of shape (B, Dl) – pooled query embedding suitable as a
            single visual prefix token for the LLM.
        """
        # Ensure we have a sequence dimension: (B, N, Dv)
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(1)  # (B, 1, Dv)

        B = vision_features.size(0)

        # Project vision tokens to LLM dimension
        vision_embeds = self.vision_proj(vision_features)  # (B, N, Dl)

        # Expand learnable queries for batch: (B, K, Dl)
        queries = self.query_tokens.expand(B, -1, -1)

        # Cross-/self-attention layers
        for layer in self.layers:
            queries = layer(queries, vision_embeds)

        queries = self.norm(queries)  # (B, K, Dl)

        # Pool queries to a single vector per sample.
        # Using the first query token is standard in Q-Former-style designs.
        pooled = queries[:, 0, :]  # (B, Dl)
        return pooled


class QueryBlock(nn.Module):
    """One Q-Former block with cross-attention, self-attention and FFN."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()

        # Cross-attention: queries attend to vision tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, vision_embeds: torch.Tensor) -> torch.Tensor:
        # Cross-attention: queries ← vision
        attn_output, _ = self.cross_attn(
            query=queries,
            key=vision_embeds,
            value=vision_embeds,
        )
        queries = self.norm1(queries + self.dropout(attn_output))

        # Self-attention over queries
        attn_output, _ = self.self_attn(
            query=queries,
            key=queries,
            value=queries,
        )
        queries = self.norm2(queries + self.dropout(attn_output))

        # Feed-forward network
        ffn_output = self.ffn(queries)
        queries = self.norm3(queries + self.dropout(ffn_output))

        return queries

