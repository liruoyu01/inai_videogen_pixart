from typing import Any, Dict, Optional

import torch
from torch import nn
from einops import rearrange

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward, _chunked_feed_forward


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
        )
        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        attn_output = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


@maybe_allow_in_graph
class TemporalBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, activation_fn="geglu")

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        self._chunk_size = chunk_size
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.Tensor,
        temporal_pos_embed: torch.Tensor,
        num_frame: int,
    ) -> torch.FloatTensor:
        num_patchs = hidden_states.shape[1]

        hidden_states = rearrange(hidden_states, '(b f) p c -> (b p) f c', f=num_frame)

        batch_size = hidden_states.shape[0]
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        
        norm_hidden_states = self.norm1(hidden_states + temporal_pos_embed)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output
        
        hidden_states = ff_output + hidden_states
            
        hidden_states = rearrange(hidden_states, '(b p) f c -> (b f) p c', p=num_patchs)

        return hidden_states
