# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
# from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models import AutoencoderKL
from diffusers.loaders import FromSingleFileMixin
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.modeling_utils import ModelMixin

from model.vae_modules.vae_3d_compress import Decoder3D, Encoder
from model.utils.decoder_utils import DecoderOutput, DiagonalGaussianDistribution

# @dataclass
# class AutoencoderKLOutput(BaseOutput):
#     """
#     Output of AutoencoderKL encoding method.

#     Args:
#         latent_dist (`DiagonalGaussianDistribution`):
#             Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
#             `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
#     """

#     latent_dist: "DiagonalGaussianDistribution"

@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of encoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The encoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class AutoencoderKL_3D(ModelMixin, ConfigMixin, FromSingleFileMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix

    encode过程支持spatial tile (encode)
    decode过程运算量更大, 支持spatial tile(_decode) & batch slice(decode)
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_num: int = 4,
        up_block_num: int = 4,
        block_out_channels: Tuple[int] = (128,256,512,512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 768,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        blocks_tempdown_li=[False, False, False, False],
        blocks_tempup_li=[False, False, False, False],
    ):
        super().__init__()

        self.blocks_tempdown_li = blocks_tempdown_li
        self.blocks_tempup_li = blocks_tempup_li
        self.temp_stride = blocks_tempup_li.count(True)
        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=down_block_num,
            blocks_temp_li=blocks_tempdown_li,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=up_block_num,
            blocks_temp_li=blocks_tempup_li,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        sample_size = (
            sample_size[0]
            if isinstance(sample_size, (list, tuple))
            else sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def _init_from_sd_hf(self, pretrained, subfolder=None):
        '''
            load spatial layers from pretrained huggingface sd vae model.
        '''
        img_ae = AutoencoderKL.from_pretrained(pretrained, subfolder=subfolder)
        img_ae_state_dict = img_ae.state_dict()
        del img_ae
        
        missing_keys, unexpected_keys = self.load_state_dict(img_ae_state_dict, strict=False)
        return missing_keys, unexpected_keys

    def _set_partial_grad(self):
        self.encoder._set_partial_grad()
        self.decoder._set_partial_grad()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder3D)):
            module.gradient_checkpointing = value

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, sample_posterior: bool = False, 
        generator: Optional[torch.Generator] = None, num_frames: int = 1, temp_compress: bool = False,return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, torch.FloatTensor]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, sample_posterior, num_frames, generator, temp_compress, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice, num_frames=num_frames, temp_compress=temp_compress) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x, num_frames=num_frames, temp_compress=temp_compress)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        if not return_dict:
            return (z,)

        return AutoencoderKLOutput(sample=z)

    def _decode(self, z: torch.FloatTensor, num_frames: int=1, temp_compress: bool = False, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, num_frames=num_frames, temp_compress=temp_compress, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z, num_frames=num_frames, temp_compress=temp_compress)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor, num_frames: int=1, temp_compress: bool = False, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice, num_frames=num_frames, temp_compress=temp_compress).sample for z_slice in z.split(1)]  # batch维度拆开infer
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, num_frames=num_frames, temp_compress=temp_compress).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor, sample_posterior: bool = False,  num_frames: int =1,
        generator: Optional[torch.Generator] = None, temp_compress: bool = False, return_dict: bool = True) -> Union[AutoencoderKLOutput, torch.FloatTensor]:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile, num_frames=num_frames, temp_compress=temp_compress)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        if not return_dict:
            return (z,)

        return AutoencoderKLOutput(sample=z)

    def tiled_decode(self, z: torch.FloatTensor, num_frames: int=1, temp_compress: bool = False, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile, num_frames=num_frames, temp_compress=temp_compress)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        num_frames: int = 1,
        sample_posterior: bool = False,
        return_dict: bool = True,
        temp_compress: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        enc = self.encode(x, sample_posterior=sample_posterior, generator=generator, num_frames=num_frames, temp_compress=temp_compress).sample  # class DiagonalGaussianDistribution
        dec = self.decode(enc, num_frames=num_frames, temp_compress=temp_compress).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    
# if __name__ == '__main__':
    # from diffusers.models.autoencoder_kl import AutoencoderKL
    # import jsonlines
    # import random
    # import os
    # import tqdm
    # from decord import VideoReader
    # import math
    # from torchvision import transforms as T
    # from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    # img_ae = AutoencoderKL.from_pretrained(
    #                 '/ML-A100/team/mm/yanghuan/huggingface/stabilityai/sd-vae-ft-ema',
    #                 subfolder="vae",
    #             )
    # img_ae_state_dict = img_ae.state_dict()
    
    # video_ae = AutoencoderKL_3D()
    # missing_keys, unexpected_keys = video_ae.load_state_dict(img_ae_state_dict, strict=False)
    # # print(f'Miss_keys:{missing_keys}')
    # # print(f'Unexpected_keys:{unexpected_keys}')

    # for param in img_ae.parameters():
    #     param.requires_grad = False
    # img_ae.eval()
    # img_ae.to('cuda')

    # for param in video_ae.parameters():
    #     param.requires_grad = False
    # video_ae.eval()
    # video_ae.to('cuda')

    # ############### test ###############
    # len_video = 100
    # out_h = 288
    # out_w = 512
    # len_split = 16

    # def read_meta(meta_in, num_sample=100):
    #     item_list = []
    #     with open(meta_in, 'r', encoding="utf-8") as f:
    #         for line in jsonlines.Reader(f):
    #             a_ratio = float(line['height'])/float(line['width'])
    #             if a_ratio<0.68:
    #                 item_list.append(line)
        
    #     return random.sample(item_list, k=num_sample)
    
    # in_meta = '/ML-A100/team/mm/yanghuan/data/pexels/pexels_video_20231016_val.jsonl'
    # in_video_root = '/ML-A100/team/mm/yanghuan/data/pexels/'
    # out_video_root = './test_autoencoder_init'
    # os.makedirs(out_video_root, exist_ok=True)

    # video_metas = read_meta(in_meta, num_sample=50)

    # # ---infrence---
    # for meta in tqdm.tqdm(video_metas):
    #     v_id = meta['id']
    #     video_path = os.path.join(in_video_root, meta['file'])
    #     ori_h = meta['height']
    #     ori_w = meta['width']

    #     out_video_path = os.path.join(out_video_root, str(v_id)+'.mp4')

    #     vr = VideoReader(video_path)
    #     video_fps = vr.get_avg_fps()

    #     start = (len(vr)-len_video)//2
    #     end = start+len_video
    #     video_idxs = torch.arange(start, end)
    #     video = torch.tensor(vr.get_batch(video_idxs).asnumpy()).float()/255.0
    #     video = video.permute([0, 3, 1, 2]).contiguous() # f * c * h * w

    #     #---resize---
    #     ratio = max(float(out_h/ori_h), float(out_w/ori_w))
    #     new_size = tuple([math.ceil(ori_h*ratio), math.ceil(ori_w*ratio)])
    #     Resize = T.Resize(new_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    #     video = Resize(video)

    #     #---crop video---
    #     # y_min = random.randint(0, video.size(2)-out_h)
    #     # x_min = random.randint(0, video.size(3)-out_w)
    #     y_min = 0
    #     x_min = 0
    #     video_crop = video[
    #         :,:,
    #         y_min:y_min+out_h,
    #         x_min:x_min+out_w
    #     ].contiguous().to("cuda")
    #     del video

    #     video_crop = video_crop*2-1
    #     #---split video---
    #     video_splits = torch.split(video_crop, len_split, dim=0)

    #     out_2d = []
    #     out_3d = []
    #     for i, video_sp in enumerate(video_splits):
    #         num_frames = video_sp.shape[0]

    #         rec_ori = img_ae(video_sp).sample
    #         out_2d.append(rec_ori)
    #         print(rec_ori.shape)

    #         rec_3d = video_ae(video_sp, num_frames=num_frames, sample_posterior=True).sample
    #         out_3d.append(rec_3d)
    #         print(rec_3d.shape)
            
    #     video_rec_2d = torch.cat(out_2d, dim=0)
    #     video_rec_3d = torch.cat(out_3d, dim=0)
    #     del out_2d,out_3d, video_splits

    #     out_stich = torch.cat((video_rec_2d, video_rec_3d), dim=-1).contiguous()
    #     video64 = ((out_stich + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0,2,3,1).cpu() # (t, h, w, c)
    #     video64 = video64.numpy()
    #     imgs = [img for img in video64]
    #     video_clip = ImageSequenceClip(imgs, fps=video_fps)
    #     video_clip.write_videofile(out_video_path, video_fps, audio=False)




