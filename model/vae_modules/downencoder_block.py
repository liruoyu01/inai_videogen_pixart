import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import os

from diffusers.models.attention_processor import Attention

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from .resnet import ResnetBlock2D, Downsample2D
from .temporal_layers import TemporalConvBlock
# from vae_modules.recurrent_blocks import SwinAttenRecurrentBlock, ConvRecurrentBlock, ConvAttenRecurrentBlock
# from modules.vae_modules.recurrent_blocks import SwinNewAttenRecurrentBlock as SwinAttenRecurrentBlock


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        add_temp_downsample=False,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if add_temp_downsample and i == num_layers-1:
                temp_convs.append(
                    TemporalConvBlock(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    down_sample=True
                    )
                )
            else:
                temp_convs.append(
                    TemporalConvBlock(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        self.add_temp_downsample = add_temp_downsample

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None
    
    def _set_partial_grad(self):
        for temp_conv in self.temp_convs:
            temp_conv.requires_grad_(True)
        if self.downsamplers:
            for down_layer in self.downsamplers:
                down_layer.requires_grad_(True)

    # def forward(self, hidden_states, scale: float = 1.0, num_frames=1):
    def forward(self, hidden_states, num_frames=1):
        # for resnet in self.resnets:
        #     hidden_states = resnet(hidden_states, temb=None, scale=scale)
        
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb=None)
            if num_frames > 1:
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states