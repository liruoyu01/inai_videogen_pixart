
import torch 
import os 
from typing import List

from model.vae_modules.autoencoder_kl_3d_compress import AutoencoderKL_3D as video_vae_tempcompress

def load_pretrain_vae(
        ckpt_path:str, 
        down_block_num: int=4,
        up_block_num: int=4,
        sample_size: int=256,
        block_out_channels: List[int]=[128,256,512,512],
        blocks_tempdown_li: List[bool]=[True, True, False, False], # control temporal compress at each block
        blocks_tempup_li: List[bool]=[False, True, True, False],
        cuda_device=None,
        debug_print=False,
    ):
    vae = video_vae_tempcompress(
        in_channels=3,
        out_channels=3,
        down_block_num=down_block_num,
        up_block_num=up_block_num,
        block_out_channels=block_out_channels,
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=32,
        blocks_tempdown_li=blocks_tempdown_li,
        blocks_tempup_li=blocks_tempup_li,
        sample_size=sample_size,
    )

    if cuda_device:
        vae = vae.to(device=cuda_device)

    
    print(f"########### Load img vae ckpt ###########") #488
    old_p = "/ML-A100/team/mm/wangqiuyue/experiments/SD_videoVAE_deepspeed/0510_compress_conti25k_conti25k6k_skip2_conv4out2C_2way_34kconti_201ktemponly_full38k_35k_19k_333temponlyfo/global_step27000_full_ori.pt"
    old_state = torch.load(old_p)
    vae.load_state_dict(old_state, strict=False)

    print(f"########### Load ckpt {ckpt_path} ###########")
    assert os.path.exists(ckpt_path)
    vae_state = torch.load(ckpt_path)
    vae.load_state_dict(vae_state, strict=False)
    return vae