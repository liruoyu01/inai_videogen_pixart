from torch import nn
# from einops import rearrange
import torch.nn.functional as F
import os

# from diffusers.models.attention_processor import Attention

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from .resnet import Upsample2D, ResnetBlock2D
from .temporal_layers import TemporalConvBlock
# from modules.vae_modules.recurrent_blocks import SwinAttenRecurrentBlock, ConvRecurrentBlock, ConvAttenRecurrentBlock
# from modules.vae_modules.recurrent_blocks import SwinNewAttenRecurrentBlock as SwinAttenRecurrentBlock


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        temb_channels=None,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
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

        self.resnets = nn.ModuleList(resnets)

        if self.add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, temb=None, scale: float = 1.0):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        add_temp_upsample=False,
        temb_channels=None,
    ):
        super().__init__()
        self.add_upsample = add_upsample

        resnets = []
        temp_convs = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
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
            if add_temp_upsample and i==num_layers-1:
                temp_convs.append(
                    TemporalConvBlock(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    up_sample=True
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

        self.add_temp_upsample = add_temp_upsample

        if self.add_upsample:
            # self.upsamplers = nn.ModuleList([PSUpsample2D(out_channels, use_conv=True, use_pixel_shuffle=True, out_channels=out_channels)])
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
    
    def _set_partial_grad(self):
        for temp_conv in self.temp_convs:
            temp_conv.requires_grad_(True)
        if self.add_upsample:
            self.upsamplers.requires_grad_(True)

    def forward(self, hidden_states, temb=None, num_frames=1):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb=temb)
            if num_frames > 1:
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
    
# class RecurrentUpDecoderBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         output_scale_factor=1.0,
#         add_upsample=True,
#         temb_channels=None,
#         attention_head_dim=1,
#         enable_recurrent = False,
#         recurrent_type = 'cross_atten', # 'cross_atten', 'conv',
#         conv_recur_mid_scale = 1,
#     ):
#         super().__init__()
#         self.enable_recurrent = enable_recurrent
#         self.add_upsample = add_upsample

#         resnets = []
#         for i in range(num_layers):
#             input_channels = in_channels if i == 0 else out_channels

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=input_channels,
#                     out_channels=out_channels,
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
#         self.resnets = nn.ModuleList(resnets)

#         if self.enable_recurrent:
#             if recurrent_type=='cross_atten':
#                 self.recurrent_block = AttenRecurrentBlock(
#                     in_channels=out_channels,
#                     dropout = dropout,
#                     resnet_eps = resnet_eps,
#                     resnet_time_scale_shift = resnet_time_scale_shift,
#                     resnet_act_fn = resnet_act_fn,
#                     resnet_groups = resnet_groups,
#                     resnet_pre_norm = resnet_pre_norm,
#                     output_scale_factor = output_scale_factor,
#                     temb_channels = temb_channels,
#                     attention_head_dim = attention_head_dim,
#                 )
#             elif recurrent_type=='conv':
#                 self.recurrent_block = ConvRecurrentBlock(
#                     in_channels = out_channels,
#                     mid_channels_scale = conv_recur_mid_scale,
#                     dropout = dropout,
#                     groups = resnet_groups,
#                     pre_norm = resnet_pre_norm,
#                     eps = resnet_eps,
#                     non_linearity = resnet_act_fn,
#                 )
#             else:
#                 raise ValueError(f"unknown recurrent_type : {recurrent_type} ")

#         if self.add_upsample:
#             self.upsamplers = nn.ModuleList([PSUpsample2D(out_channels, use_conv=True, use_pixel_shuffle=True, out_channels=out_channels)])
#         else:
#             self.upsamplers = None
    
#     def _set_partial_grad(self):
#         if self.enable_recurrent:
#             self.recurrent_block.requires_grad_(True)
#         if self.add_upsample:
#             self.upsamplers.requires_grad_(True)

#     def forward(self, feature_in, hidden_states=None, temb=None):
#         for resnet in self.resnets:
#             feature_in = resnet(feature_in, temb=temb)  # b c h w
        
#         if self.enable_recurrent:
#             feature_in, hidden_states = self.recurrent_block(feature_in, hidden_states)

#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 feature_in = upsampler(feature_in)

#         return feature_in, hidden_states

# class AttenRecurrentBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int=None,
#             dropout: float = 0.0,
#             resnet_eps: float = 1e-6,
#             resnet_time_scale_shift: str = "default",  # default, spatial
#             resnet_act_fn: str = "swish",
#             resnet_groups: int = 32,
#             resnet_pre_norm: bool = True,
#             output_scale_factor=1.0,
#             temb_channels=None,
#             attention_head_dim=1,
#         ):
#         super().__init__()
#         out_channels = in_channels if not out_channels else out_channels
#         self.cross_atten=Attention(
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

#         self.resnet_main=ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                 )

#         self.resnet_hidden= ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                 )
        
#         # zero init
#         nn.init.zeros_(self.cross_atten.to_out[0].weight)
#         nn.init.zeros_(self.cross_atten.to_out[0].bias)
#         nn.init.zeros_(self.resnet_main.conv2.weight)
#         nn.init.zeros_(self.resnet_main.conv2.bias)
#         nn.init.zeros_(self.resnet_hidden.conv2.weight)
#         nn.init.zeros_(self.resnet_hidden.conv2.bias)
    
#     def forward(self, feature_in, hidden_states=None, temb=None):
#         if hidden_states==None:
#             hidden_states = torch.zeros(feature_in.shape).to(feature_in.device).to(feature_in.dtype)

#         # recurrent block
#         _, _, h, w = feature_in.shape
#         feature_in = rearrange(feature_in, 'b c h w -> b (h w) c')
#         hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')
#         cross_out = self.cross_atten(hidden_states=feature_in, encoder_hidden_states=hidden_states, temb=temb)
#         cross_out = rearrange(cross_out, 'b (h w) c -> b c h w', h=h)

#         feature_in = self.resnet_main(cross_out, temb=temb)
#         hidden_states = self.resnet_hidden(cross_out, temb=temb)
#         return feature_in, hidden_states

# class DualBranchRecurrentUpDecoderBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         output_scale_factor=1.0,
#         add_upsample=True,
#         temb_channels=None,
#         attention_head_dim=64,
#         enable_recurrent = True,
#         use_atten = True,
#         conv_recur_mid_scale = 1,
#         detach_pre=False,
#         detach_hidden=False
#     ):
#         super().__init__()
#         self.enable_recurrent = enable_recurrent
#         self.add_upsample = add_upsample

#         resnets = []
#         for i in range(num_layers):
#             input_channels = in_channels if i == 0 else out_channels

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=input_channels,
#                     out_channels=out_channels,
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
#         self.resnets = nn.ModuleList(resnets)

#         if self.enable_recurrent:
#             self.recurrent_block = ConvAttenRecurrentBlock(
#                 in_channels=out_channels,
#                 dropout = dropout,
#                 attention_head_dim = attention_head_dim,
#                 mid_channels_scale=conv_recur_mid_scale,
#                 mid_layers=2,
#                 groups = resnet_groups,
#                 non_linearity = resnet_act_fn,
#                 pre_norm = resnet_pre_norm,
#                 use_atten = use_atten,
#                 detach_pre=detach_pre,
#                 detach_hidden=detach_hidden
#             )

#         if self.add_upsample:
#             self.upsamplers = nn.ModuleList([PSUpsample2D(out_channels, use_conv=True, use_pixel_shuffle=True, out_channels=out_channels)])
#         else:
#             self.upsamplers = None
    
#     def _set_partial_grad(self):
#         if self.enable_recurrent:
#             self.recurrent_block.requires_grad_(True)
#         if self.add_upsample:
#             self.upsamplers.requires_grad_(True)

#     def forward(self, feature_in, num_frames=1):
#         for resnet in self.resnets:
#             feature_in = resnet(feature_in)  # bf c h w

#         if self.enable_recurrent:
#             feature_in = self.recurrent_block(
#                                     feature_in=feature_in,
#                                     num_frames=num_frames
#                                 )

#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 feature_in = upsampler(feature_in)

#         return feature_in

# class SwinRecurrentUpDecoderBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         output_scale_factor=1.0,
#         add_upsample=True,
#         temb_channels=None,
#         attention_head_dim=64,
#         enable_recurrent = True,
#         recurrent_type = 'swin_atten', # 'cross_atten', 'conv',
#         use_atten = True,
#         conv_recur_mid_scale = 1,
#         swin_win_size = 7,
#         swin_v2 = False,
#         detach_pre=False,
#         detach_hidden=False
#     ):
#         super().__init__()
#         self.enable_recurrent = enable_recurrent
#         self.add_upsample = add_upsample

#         resnets = []
#         for i in range(num_layers):
#             input_channels = in_channels if i == 0 else out_channels

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=input_channels,
#                     out_channels=out_channels,
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
#         self.resnets = nn.ModuleList(resnets)

#         if self.enable_recurrent:
#             if recurrent_type=='swin_atten':
#                 self.recurrent_block = SwinAttenRecurrentBlock(
#                     in_channels=out_channels,
#                     dropout = dropout,
#                     attention_head_dim = attention_head_dim,
#                     mid_channels_scale=conv_recur_mid_scale,
#                     mid_layers=2,
#                     groups = resnet_groups,
#                     non_linearity = resnet_act_fn,
#                     pre_norm = resnet_pre_norm,
#                     use_atten = use_atten,
#                     window_size=swin_win_size,
#                     swin_v2=swin_v2,
#                     detach_pre=detach_pre,
#                     detach_hidden=detach_hidden
#                 )
#             elif recurrent_type=='conv':
#                 self.recurrent_block = ConvRecurrentBlock(
#                     in_channels = out_channels,
#                     mid_channels_scale = conv_recur_mid_scale,
#                     dropout = dropout,
#                     groups = resnet_groups,
#                     pre_norm = resnet_pre_norm,
#                     eps = resnet_eps,
#                     non_linearity = resnet_act_fn,
#                 )
#             else:
#                 raise ValueError(f"unknown recurrent_type : {recurrent_type} ")

#         if self.add_upsample:
#             self.upsamplers = nn.ModuleList([PSUpsample2D(out_channels, use_conv=True, use_pixel_shuffle=True, out_channels=out_channels)])
#         else:
#             self.upsamplers = None
    
#     def _set_partial_grad(self):
#         if self.enable_recurrent:
#             self.recurrent_block.requires_grad_(True)
#         if self.add_upsample:
#             self.upsamplers.requires_grad_(True)

#     def forward(self, feature_in, num_frames=1):
#         for resnet in self.resnets:
#             feature_in = resnet(feature_in)  # bf c h w

#         if self.enable_recurrent:
#             feature_in = self.recurrent_block(
#                                     feature_in=feature_in,
#                                     num_frames=num_frames
#                                 )

#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 feature_in = upsampler(feature_in)

#         return feature_in
    
# class SwinRecurrentUpDecoderBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         output_scale_factor=1.0,
#         add_upsample=True,
#         temb_channels=None,
#         attention_head_dim=64,
#         enable_recurrent = True,
#         recurrent_type = 'swin_atten', # 'cross_atten', 'conv',
#         use_atten = True,
#         conv_recur_mid_scale = 1,
#         swin_win_size = 7,
#         swin_v2 = False
#     ):
#         super().__init__()
#         self.enable_recurrent = enable_recurrent
#         self.add_upsample = add_upsample

#         resnets = []
#         for i in range(num_layers):
#             input_channels = in_channels if i == 0 else out_channels

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=input_channels,
#                     out_channels=out_channels,
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
#         self.resnets = nn.ModuleList(resnets)

#         if self.enable_recurrent:
#             if recurrent_type=='swin_atten':
#                 self.recurrent_block = SwinAttenRecurrentBlock(
#                     in_channels=out_channels,
#                     dropout = dropout,
#                     attention_head_dim = attention_head_dim,
#                     mid_channels_scale=conv_recur_mid_scale,
#                     mid_layers=2,
#                     groups = resnet_groups,
#                     non_linearity = resnet_act_fn,
#                     pre_norm = resnet_pre_norm,
#                     use_atten = use_atten,
#                     window_size=swin_win_size,
#                     swin_v2=swin_v2
#                 )
#             elif recurrent_type=='conv':
#                 self.recurrent_block = ConvRecurrentBlock(
#                     in_channels = out_channels,
#                     mid_channels_scale = conv_recur_mid_scale,
#                     dropout = dropout,
#                     groups = resnet_groups,
#                     pre_norm = resnet_pre_norm,
#                     eps = resnet_eps,
#                     non_linearity = resnet_act_fn,
#                 )
#             else:
#                 raise ValueError(f"unknown recurrent_type : {recurrent_type} ")

#         if self.add_upsample:
#             self.upsamplers = nn.ModuleList([PSUpsample2D(out_channels, use_conv=True, use_pixel_shuffle=True, out_channels=out_channels)])
#         else:
#             self.upsamplers = None
    
#     def _set_partial_grad(self):
#         if self.enable_recurrent:
#             self.recurrent_block.requires_grad_(True)
#         if self.add_upsample:
#             self.upsamplers.requires_grad_(True)

#     def forward(self, feature_in, feature_pre=None, hidden_states=None):
#         for resnet in self.resnets:
#             feature_in = resnet(feature_in)  # b c h w

#         if feature_pre!=None:
#             assert feature_pre.shape==feature_in.shape, \
#                 f"'feature_pre' must has the same size as 'feature_in', but got{feature_pre.shape} and {feature_in.shape}"
#         if hidden_states!=None:
#             assert hidden_states.shape==feature_in.shape, \
#                 f"'hidden_states' must has the same size as 'feature_in', but got{hidden_states.shape} and {feature_in.shape}"
        
#         if self.enable_recurrent:
#             feature_in, hidden_states = self.recurrent_block(
#                                     feature_in=feature_in,
#                                     feature_pre=feature_pre, 
#                                     hidden_states=hidden_states
#                                 )

#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 feature_in = upsampler(feature_in)

#         return feature_in, hidden_states
    
# if __name__=='__main__':
#     recurr = DualBranchRecurrentUpDecoderBlock2D(in_channels=256, out_channels=128, num_layers=2, add_upsample=True)
#     x = torch.rand((16,256,64,64))

#     feature_out = recurr(x, 16)
#     print(feature_out.shape)
   