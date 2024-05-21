import torch
import torch.nn as nn
from einops import rearrange

from diffusers.utils import is_torch_version
from diffusers.models.attention_processor import SpatialNorm

from .downencoder_block import DownEncoderBlock2D
from .unet_midblock import UNetMidBlock3DConv
from .updecoder_block import UpDecoderBlock3D

'''
    2D Encoder & 3D Dcoder
'''

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_blocks=4,
        blocks_temp_li=[False, False, False, False],
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.blocks_temp_li = blocks_temp_li

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.temp_conv_in = nn.Conv3d(
            block_out_channels[0],
            block_out_channels[0],
            (3,1,1),
            padding = (1, 0, 0)
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                add_temp_downsample=blocks_temp_li[i],
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.down_blocks.append(down_block)

        # mid
        # self.mid_block = UNetMidBlock2D(
        self.mid_block = UNetMidBlock3DConv(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels

        self.temp_conv_out = nn.Conv3d(block_out_channels[-1], block_out_channels[-1], (3,1,1), padding = (1, 0, 0))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        nn.init.zeros_(self.temp_conv_in.weight)
        nn.init.zeros_(self.temp_conv_in.bias)
        nn.init.zeros_(self.temp_conv_out.weight)
        nn.init.zeros_(self.temp_conv_out.bias)

        self.gradient_checkpointing = False
    
    def _set_partial_grad(self):
        self.mid_block._set_partial_grad()
        for down_block in self.down_blocks:
            down_block._set_partial_grad()
        self.temp_conv_in.requires_grad_(True)
        self.temp_conv_out.requires_grad_(True)

    def forward(self, x, num_frames, latent_embeds=None):
        '''
            x: (b f), c, h, w
        '''
        sample = x
        sample = self.conv_in(sample)
        if num_frames > 1:
            identity = sample
            sample = rearrange(sample, '(b f) c h w -> b c f h w', f=num_frames)
            sample = self.temp_conv_in(sample)
            sample = rearrange(sample, 'b c f h w -> (b f) c h w')
            sample = identity + sample

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down

            if is_torch_version(">=", "1.11.0"):
                # for down_block in self.down_blocks:
                for b_id, down_block in enumerate(self.down_blocks):
                    # import ipdb; ipdb.set_trace()
                    # sample = torch.utils.checkpoint.checkpoint(
                    #     create_custom_forward(down_block), sample, num_frames=num_frames, use_reentrant=False
                    # )
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, num_frames, use_reentrant=False
                    )
                    if self.blocks_temp_li[b_id] and num_frames > 1:
                        num_frames = int(num_frames//2)
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds, num_frames, use_reentrant=False
                )
            else:
                for b_id, down_block in enumerate(self.down_blocks):
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample, num_frames)
                    if self.blocks_temp_li[b_id] and num_frames > 1:
                        num_frames = int(num_frames//2)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample, latent_embeds, num_frames)

        else:
            # down
            for b_id, down_block in enumerate(self.down_blocks):
                sample = down_block(sample, num_frames=num_frames)
                if self.blocks_temp_li[b_id] and num_frames > 1:
                    num_frames = int(num_frames//2)

            # middle
            sample = self.mid_block(sample, num_frames=num_frames)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        
        if num_frames > 1:
            identity = sample
            sample = rearrange(sample, '(b f) c h w -> b c f h w', f=num_frames)
            sample = self.temp_conv_out(sample)
            sample = rearrange(sample, 'b c f h w -> (b f) c h w')
            sample = identity + sample

        sample = self.conv_out(sample)

        return sample
    
class Decoder3D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        num_blocks=4,
        blocks_temp_li=[False, False, False, False],
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.blocks_temp_li = blocks_temp_li
        self.temp_stride = blocks_temp_li.count(True)

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.temp_conv_in = nn.Conv3d(
            block_out_channels[-1],
            block_out_channels[-1],
            (3,1,1),
            padding = (1, 0, 0)
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock3DConv(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(num_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                add_temp_upsample=blocks_temp_li[i],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        self.temp_conv_out = nn.Conv3d(block_out_channels[0], block_out_channels[0], (3,1,1), padding = (1, 0, 0))
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        nn.init.zeros_(self.temp_conv_in.weight)
        nn.init.zeros_(self.temp_conv_in.bias)
        nn.init.zeros_(self.temp_conv_out.weight)
        nn.init.zeros_(self.temp_conv_out.bias)

        self.gradient_checkpointing = False

    def _set_partial_grad(self):
        self.mid_block._set_partial_grad()
        for up_block in self.up_blocks:
            up_block._set_partial_grad()
        self.temp_conv_in.requires_grad_(True)
        self.temp_conv_out.requires_grad_(True)

    def forward(self, z, num_frames=1, latent_embeds=None):
        sample = z
        sample = self.conv_in(sample)
        
        if num_frames > 1:
            num_frames = int(num_frames / (2**self.temp_stride))
            identity = sample
            sample = rearrange(sample, '(b f) c h w -> b c f h w', f=num_frames)
            sample = self.temp_conv_in(sample)
            sample = rearrange(sample, 'b c f h w -> (b f) c h w')
            sample = identity + sample

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds, num_frames, use_reentrant=False
                )
                sample = sample.to(upscale_dtype)

                # up
                for b_id, up_block in enumerate(self.up_blocks):
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), sample, latent_embeds, num_frames, use_reentrant=False
                    )
                    if self.blocks_temp_li[b_id] and num_frames > 1:
                        num_frames = int(num_frames*2)
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds, num_frames, 
                )
                sample = sample.to(upscale_dtype)

                # up
                # for up_block in self.up_blocks:
                for b_id, up_block in enumerate(self.up_blocks):
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds, num_frames)
                    if self.blocks_temp_li[b_id] and num_frames > 1:
                        num_frames = int(num_frames*2)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds, num_frames=num_frames)
            sample = sample.to(upscale_dtype)

            # up
            # for up_block in self.up_blocks:
            for b_id, up_block in enumerate(self.up_blocks):
                sample = up_block(sample, latent_embeds, num_frames=num_frames)
                if self.blocks_temp_li[b_id] and num_frames > 1:
                    num_frames = int(num_frames*2)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)

        if num_frames > 1:
            identity = sample 
            sample = rearrange(sample, '(b f) c h w -> b c f h w', f=num_frames)
            sample = self.temp_conv_out(sample)
            sample = rearrange(sample, 'b c f h w -> (b f) c h w')
            sample = identity + sample

        sample = self.conv_out(sample)

        return sample



