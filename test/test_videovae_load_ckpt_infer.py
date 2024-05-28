
import torch 
import os 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from einops import rearrange

from model.vae_modules.autoencoder_kl_3d_compress import AutoencoderKL_3D
from data.raw_data.pexel_dataset import PexelsDataset
from model.utils.data_utils import export_to_gif

from typing import List

def load_vae(
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
    vae = AutoencoderKL_3D(
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

def save_to_local(save_root, save_file_name, frames, duration):
    # frames range [0,1]
    frames_forgif = rearrange(frames, 'f c h w -> f h w c')

    debug_video_path = os.path.join(save_root, str(save_file_name)+'.png')
    save_image(frames, debug_video_path)

    debug_video_path_gif = os.path.join(save_root, str(save_file_name)+'.gif')
    export_to_gif(torch.unbind(frames_forgif, dim=0), debug_video_path_gif, duration=duration)

def main():
    cuda_device = 'cuda:4'
    ckpt_paths = {
        "video_488_temponly333_add4conv_72k":"/ML-A100/team/mm/wangqiuyue/experiments/SD_videoVAE_deepspeed/0511_compress_conti25k_conti25k6k_skip2_conv4out2C_2way_333temponlyfo_27k_add4conv/global_step72000.pt",
    }
    dtype=torch.bfloat16

    vae = load_vae(ckpt_path=ckpt_paths['video_488_temponly333_add4conv_72k'], cuda_device=cuda_device).to(dtype=dtype)
    vae.eval()

    latent_scale_factor = 0.13025
    data_dir = '/ML-A100/team/mm/yanghuan/data/pexels_clip'
    data_val_meta = '/ML-A100/team/mm/zixit/data/pexles_20240405/pexels_meta_val.jsonl'
    num_frames = 16
    fps = 15
    pexles_val = PexelsDataset(data_dir, data_val_meta, [num_frames,3,128,128], [[512, 512]], [1.], [fps])
    video_val_dataloader = DataLoader(
        dataset=pexles_val,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    save_root = './video_debug'
    os.makedirs(save_root, exist_ok=True)

    for b_idx, batch in enumerate(video_val_dataloader):
        frames = rearrange(batch['video'], 'b c f h w -> (b f) c h w')
        num_frames = int(batch['num_frames'][0])  

        frames_to_save = ((frames + 1.0)/2.0).clamp(0,1)
        # frames_to_save must be range [0,1]
        save_to_local(save_root, f'{b_idx}_org', frames_to_save, num_frames * 1.0//fps)

        frames = frames.to(dtype=dtype, device=cuda_device)
        latent = vae.encode(frames, sample_posterior=True, num_frames=num_frames, temp_compress=True).sample
        print('latent shape', latent.shape)

        recon_video = vae.decode(latent, num_frames=num_frames, temp_compress=True).sample
        recon_video = recon_video.detach().cpu()

        recon_video = ((recon_video + 1.0)/2.0).clamp(0,1)
        save_to_local(save_root, f'{b_idx}_recon', recon_video, num_frames * 1.0//fps)

        if b_idx == 5:
            break

if __name__ == '__main__':
    main()
