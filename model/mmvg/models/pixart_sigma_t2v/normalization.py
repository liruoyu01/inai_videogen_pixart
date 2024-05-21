from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings


class AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim, size_emb_dim=embedding_dim // 3)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
