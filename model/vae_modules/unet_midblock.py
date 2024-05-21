import torch
from torch import nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from diffusers.utils import logging
from diffusers.models.attention_processor import Attention

from .resnet import ResnetBlock2D
from .temporal_layers import TemporalConvBlock

logger = logging.get_logger(__name__)


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states

class UNetMidBlock3DConvAtten(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvBlock(
                in_channels,
                in_channels,
                dropout=0.1,
            )
        ]
        attentions = []
        temp_attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            temp_attentions.append(
                TransformerTemporalModel(
                    num_attention_heads=in_channels // attention_head_dim,
                    attention_head_dim=attention_head_dim,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=None,
                    norm_num_groups=resnet_groups,
                )
            )

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            temp_convs.append(
                TemporalConvBlock(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
    
    def _set_partial_grad(self):
        for temp_conv in self.temp_convs:
            temp_conv.requires_grad_(True)
        for temp_attention in self.temp_attentions:
            temp_attention.requires_grad_(True)

    def forward(
        self, 
        hidden_states, 
        temb=None,
        num_frames=1,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, temp_attn, resnet, temp_conv in zip(
            self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            hidden_states = attn(hidden_states, temb=temb)
            hidden_states = temp_attn(hidden_states, num_frames=num_frames, return_dict=False)[0]
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        return hidden_states
    
class UNetMidBlock3DConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvBlock(
                in_channels,
                in_channels,
                dropout=0.1,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            temp_convs.append(
                TemporalConvBlock(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
    
    def _set_partial_grad(self):
        for temp_conv in self.temp_convs:
            temp_conv.requires_grad_(True)

    def forward(
        self, 
        hidden_states, 
        temb=None,
        num_frames=1,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        if num_frames > 1:
            hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, resnet, temp_conv in zip(
            self.attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)
            if num_frames > 1:
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        return hidden_states
    
# class UNetMidBlockAtten3DFusionFrame(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         temb_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         add_attention: bool = True,
#         attention_head_dim=1,
#         output_scale_factor=1.0,
#     ):
#         super().__init__()
#         resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
#         self.add_attention = add_attention

#         # there is always at least one resnet
#         resnets = [
#             ResnetBlock2D(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 temb_channels=temb_channels,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 dropout=dropout,
#                 time_embedding_norm=resnet_time_scale_shift,
#                 non_linearity=resnet_act_fn,
#                 output_scale_factor=output_scale_factor,
#                 pre_norm=resnet_pre_norm,
#             )
#         ]

#         attentions = []
#         temp_attentions = []

#         if attention_head_dim is None:
#             logger.warn(
#                 f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
#             )
#             attention_head_dim = in_channels

#         for i in range(num_layers):
#             if self.add_attention:
#                 attentions.append(
#                     Attention(
#                         in_channels,
#                         heads=in_channels // attention_head_dim,
#                         dim_head=attention_head_dim,
#                         rescale_output_factor=output_scale_factor,
#                         eps=resnet_eps,
#                         norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
#                         spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
#                         residual_connection=True,
#                         bias=True,
#                         upcast_softmax=True,
#                         _from_deprecated_attn_block=True,
#                     )
#                 )
#             else:
#                 attentions.append(None)

#             temp_attentions.append(
#                 TransformerTemporalModel(
#                     num_attention_heads=in_channels // attention_head_dim,
#                     attention_head_dim=attention_head_dim,
#                     in_channels=in_channels,
#                     num_layers=1,
#                     cross_attention_dim=None,
#                     norm_num_groups=resnet_groups,
#                 )
#             )

#             if not i==num_layers-1:
#                 resnets.append(
#                     ResnetBlock2D(
#                         in_channels=in_channels,
#                         out_channels=in_channels,
#                         temb_channels=temb_channels,
#                         eps=resnet_eps,
#                         groups=resnet_groups,
#                         dropout=dropout,
#                         time_embedding_norm=resnet_time_scale_shift,
#                         non_linearity=resnet_act_fn,
#                         output_scale_factor=output_scale_factor,
#                         pre_norm=resnet_pre_norm,
#                     )
#                 )
#             else:
#                 # fusion frames in the last resblock
#                 resnets.append(
#                     ResnetFrameFusionBlock(
#                         in_channels=in_channels,
#                         out_channels=in_channels,
#                         temb_channels=temb_channels,
#                         eps=resnet_eps,
#                         groups=resnet_groups,
#                         dropout=dropout,
#                         time_embedding_norm=resnet_time_scale_shift,
#                         non_linearity=resnet_act_fn,
#                         output_scale_factor=output_scale_factor,
#                         pre_norm=resnet_pre_norm,
#                     )
#                 )
   

#         self.resnets = nn.ModuleList(resnets)
#         self.attentions = nn.ModuleList(attentions)
#         self.temp_attentions = nn.ModuleList(temp_attentions)

#     def _set_partial_grad(self):
#         self.resnets[-1].requires_grad_(True)
#         for temp_atten in self.temp_attentions:
#             temp_atten.requires_grad_(True)

#     def forward(
#         self, 
#         hidden_states, 
#         temb=None,
#         num_frames=1,
#     ):
#         hidden_states = self.resnets[0](hidden_states, temb)
#         for attn, temp_attn, resnet in zip(
#             self.attentions, self.temp_attentions, self.resnets[1:]
#         ):
#             hidden_states = attn(hidden_states, temb=temb)
#             hidden_states = temp_attn(hidden_states, num_frames=num_frames, return_dict=False)[0]
#             hidden_states = resnet(hidden_states, temb)

#         return hidden_states

# class UNetMidBlockSwinRecur2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         temb_channels: int=None,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         add_attention: bool = True,
#         attention_head_dim=1,
#         output_scale_factor=1.0,
#         enable_recurrent=True,
#         recur_use_atten=False,
#         conv_recur_mid_scale=1,
#         detach_pre=False,
#         detach_hidden=False
#     ):
#         super().__init__()
#         resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
#         self.add_attention = add_attention
#         self.enable_recurrent=enable_recurrent

#         # there is always at least one resnet
#         resnets = [
#             ResnetBlock2D(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 temb_channels=temb_channels,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 dropout=dropout,
#                 time_embedding_norm=resnet_time_scale_shift,
#                 non_linearity=resnet_act_fn,
#                 output_scale_factor=output_scale_factor,
#                 pre_norm=resnet_pre_norm,
#             )
#         ]
#         attentions = []

#         if attention_head_dim is None:
#             logger.warn(
#                 f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
#             )
#             attention_head_dim = in_channels

#         for _ in range(num_layers):
#             if self.add_attention:
#                 attentions.append(
#                     Attention(
#                         in_channels,
#                         heads=in_channels // attention_head_dim,
#                         dim_head=attention_head_dim,
#                         rescale_output_factor=output_scale_factor,
#                         eps=resnet_eps,
#                         norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
#                         spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
#                         residual_connection=True,
#                         bias=True,
#                         upcast_softmax=True,
#                         _from_deprecated_attn_block=True,
#                     )
#                 )
#             else:
#                 attentions.append(None)

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=in_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                 )
#             )

#         self.attentions = nn.ModuleList(attentions)
#         self.resnets = nn.ModuleList(resnets)

#         if self.enable_recurrent:
#             self.recurrent_block = SwinAttenRecurrentBlock(
#                 in_channels=in_channels,
#                 dropout = dropout,
#                 use_atten = recur_use_atten, 
#                 attention_head_dim = attention_head_dim,
#                 mid_channels_scale=conv_recur_mid_scale,
#                 mid_layers=2,
#                 groups = resnet_groups,
#                 non_linearity = resnet_act_fn,
#                 pre_norm = resnet_pre_norm,
#                 detach_pre=detach_pre,
#                 detach_hidden=detach_hidden
#             )
    
#     def _set_partial_grad(self):
#         if self.enable_recurrent:
#             self.recurrent_block.requires_grad_(True)

#     def forward(self, feature_in, num_frames=1, temb=None):
#         '''
#             feature_in: (bf) c h w
#         '''
#         feature_in = self.resnets[0](feature_in, temb=temb)
#         for attn, resnet in zip(self.attentions, self.resnets[1:]):
#             if attn is not None:
#                 feature_in = attn(feature_in, temb=temb)
#             feature_in = resnet(feature_in, temb)
        
#         if self.enable_recurrent:
#             feature_in = self.recurrent_block(feature_in, num_frames)

#         return feature_in

# class UNetMidBlockDualRecur2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         temb_channels: int=None,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         add_attention: bool = True,
#         attention_head_dim=1,
#         output_scale_factor=1.0,
#         enable_recurrent=True,
#         recur_use_atten=False,
#         conv_recur_mid_scale=1,
#         detach_pre=False,
#         detach_hidden=False
#     ):
#         super().__init__()
#         resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
#         self.add_attention = add_attention
#         self.enable_recurrent=enable_recurrent

#         # there is always at least one resnet
#         resnets = [
#             ResnetBlock2D(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 temb_channels=temb_channels,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 dropout=dropout,
#                 time_embedding_norm=resnet_time_scale_shift,
#                 non_linearity=resnet_act_fn,
#                 output_scale_factor=output_scale_factor,
#                 pre_norm=resnet_pre_norm,
#             )
#         ]
#         attentions = []

#         if attention_head_dim is None:
#             logger.warn(
#                 f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
#             )
#             attention_head_dim = in_channels

#         for _ in range(num_layers):
#             if self.add_attention:
#                 attentions.append(
#                     Attention(
#                         in_channels,
#                         heads=in_channels // attention_head_dim,
#                         dim_head=attention_head_dim,
#                         rescale_output_factor=output_scale_factor,
#                         eps=resnet_eps,
#                         norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
#                         spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
#                         residual_connection=True,
#                         bias=True,
#                         upcast_softmax=True,
#                         _from_deprecated_attn_block=True,
#                     )
#                 )
#             else:
#                 attentions.append(None)

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=in_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                 )
#             )

#         self.attentions = nn.ModuleList(attentions)
#         self.resnets = nn.ModuleList(resnets)

#         if self.enable_recurrent:
#             self.recurrent_block = ConvAttenRecurrentBlock(
#                 in_channels=in_channels,
#                 dropout = dropout,
#                 use_atten = recur_use_atten, 
#                 attention_head_dim = attention_head_dim,
#                 mid_channels_scale=conv_recur_mid_scale,
#                 mid_layers=2,
#                 groups = resnet_groups,
#                 non_linearity = resnet_act_fn,
#                 pre_norm = resnet_pre_norm,
#                 detach_pre=detach_pre,
#                 detach_hidden=detach_hidden
#             )
    
#     def _set_partial_grad(self):
#         if self.enable_recurrent:
#             self.recurrent_block.requires_grad_(True)

#     def forward(self, feature_in, num_frames=1, temb=None):
#         '''
#             feature_in: (bf) c h w
#         '''
#         feature_in = self.resnets[0](feature_in, temb=temb)
#         for attn, resnet in zip(self.attentions, self.resnets[1:]):
#             if attn is not None:
#                 feature_in = attn(feature_in, temb=temb)
#             feature_in = resnet(feature_in, temb)
        
#         if self.enable_recurrent:
#             feature_in = self.recurrent_block(feature_in, num_frames)

#         return feature_in
    

# if __name__=='__main__':
#     block = UNetMidBlockRecur2D(256)
#     feature_in = torch.rand((32,256,32,32))
#     feature_in = block(feature_in,16)
#     print(feature_in.shape)
