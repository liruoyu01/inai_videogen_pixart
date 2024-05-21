from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.models.modeling_utils import ModelMixin

from .attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from .embeddings import PatchEmbed, TemporalEmbedding, PixArtAlphaTextProjection
from .normalization import AdaLayerNormSingle


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        interpolation_scale: float = None,
        temporal_interpolation_scale: float = 1.0,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False
        self.patch_size = patch_size
        
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )
        
        self.temporal_pos_embed = TemporalEmbedding(inner_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.adaln_single = AdaLayerNormSingle(inner_dim)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frame: int = 0,
        num_image: int = 0,
        return_dict: bool = True,
    ):
        batch_size = hidden_states.shape[0] // (num_frame + num_image)
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        num_patchs = height * width
        
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        
        hidden_states = self.pos_embed(hidden_states)
        timestep, embedded_timestep = self.adaln_single(timestep, hidden_dtype=hidden_states.dtype)
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)

        spatial_timestep = timestep
        temporal_timestep = rearrange(timestep, '(b f) c -> b f c', b=batch_size)[:, 0, :]
        temporal_timestep = repeat(temporal_timestep, 'b c -> b p c', p=num_patchs)
        temporal_timestep = rearrange(temporal_timestep, 'b p c -> (b p) c')

        temporal_pos_embed = self.temporal_pos_embed(num_frame)
        
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    spatial_timestep,
                    **ckpt_kwargs,
                )
                hidden_states = rearrange(hidden_states, '(b f) p c -> b f p c', b=batch_size)
                hidden_states_frames = hidden_states[:, :num_frame, :, :]
                hidden_states_images = hidden_states[:, num_frame:, :, :]
                hidden_states_frames = rearrange(hidden_states_frames, 'b f p c -> (b f) p c')
                hidden_states_frames = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temporal_block),
                    hidden_states_frames,
                    temporal_timestep,
                    temporal_pos_embed,
                    num_frame,
                    **ckpt_kwargs,
                )                    
                hidden_states_frames = rearrange(hidden_states_frames, '(b f) p c -> b f p c', b=batch_size)
                hidden_states = torch.cat((hidden_states_frames, hidden_states_images), dim=1)
                hidden_states = rearrange(hidden_states, 'b f p c -> (b f) p c')
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    spatial_timestep,
                )
                hidden_states = rearrange(hidden_states, '(b f) p c -> b f p c', b=batch_size)
                hidden_states_frames = hidden_states[:, :num_frame, :, :]
                hidden_states_images = hidden_states[:, num_frame:, :, :]
                hidden_states_frames = rearrange(hidden_states_frames, 'b f p c -> (b f) p c')
                hidden_states_frames = temporal_block(
                    hidden_states_frames,
                    temporal_timestep,
                    temporal_pos_embed,
                    num_frame,
                )
                hidden_states_frames = rearrange(hidden_states_frames, '(b f) p c -> b f p c', b=batch_size)
                hidden_states = torch.cat((hidden_states_frames, hidden_states_images), dim=1)
                hidden_states = rearrange(hidden_states, 'b f p c -> (b f) p c')

        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)
        hidden_states = hidden_states.reshape(shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels))
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size))

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)
