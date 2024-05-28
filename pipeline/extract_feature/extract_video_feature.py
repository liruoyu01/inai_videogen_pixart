import os
import argparse
import multiprocessing as mp
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
from einops import rearrange

from accelerate import Accelerator
from accelerate.utils.dataclasses import InitProcessGroupKwargs

from model.mmvg.utils.logger import tprint

from data.raw_data.pexel_dataset import PexelsDataset
from model.vae_modules.build_vae import load_pretrain_vae


def save_feature(fpath, fdict):
    torch.save(fdict, fpath)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/ML-A100/team/mm/yanghuan/data/pexels_clip')
parser.add_argument('--data_meta', type=str, default='/ML-A100/team/mm/zixit/data/pexles_20240405/pexels_meta_train.jsonl')
parser.add_argument('--vae_ckpt_dir', type=str, default='/ML-A100/team/mm/wangqiuyue/experiments/SD_videoVAE_deepspeed/0511_compress_conti25k_conti25k6k_skip2_conv4out2C_2way_333temponlyfo_27k_add4conv/global_step72000.pt')
parser.add_argument('--base_save_path', type=str, default='/ML-A100/team/mm/liruoyu/data/feature')
parser.add_argument('--dataset_name', type=str, default='pexel_video_vae_f120')
parser.add_argument('--resolution', type=str, default='256x256')
parser.add_argument('--fps', type=int, default=30)

parser.add_argument('--max_frame_num', type=int, default=60) # max frame to pass vae each time, matter of efficiency
parser.add_argument('--num_frames', type=int, default=120)  # frames to extract at dataloader
parser.add_argument('--max_num_sample', type=int, default=10000)

args = parser.parse_args()

accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(5400))])

device = accelerator.device
dtype = torch.bfloat16

num_frames = args.num_frames
fps = args.fps
h, w = list(map(int, args.resolution.split('x')))
accelerator.print(f'resolution: {h}x{w}')

pexles_ds = PexelsDataset(
    args.data_dir, 
    args.data_meta, 
    [num_frames,3,h,w], 
    [[512, 512]], 
    [1.], 
    [fps],
    max_num_sample=args.max_num_sample,
)
video_dataloader = DataLoader(
    dataset=pexles_ds,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=False,
)

# load vae
vae = load_pretrain_vae(ckpt_path=args.vae_ckpt_dir, cuda_device=device).to(dtype=dtype)
vae.eval()
vae.requires_grad_(False)

loader = accelerator.prepare(video_dataloader)

full_save_path = os.path.join(
    args.base_save_path,
    args.dataset_name,
    args.resolution
)
os.makedirs(full_save_path, exist_ok=True)

if accelerator.is_local_main_process:
    accelerator.print('Start %s' % (full_save_path))


pool = mp.Pool(processes=32)
for idx, batch in enumerate(loader):
    if batch is None:
        continue
    video, path = batch['video'], batch['v_feat_path']
    video = video.to(dtype=dtype)
    video = torch.squeeze(video[0])
    if accelerator.is_local_main_process:
        accelerator.print('video shape before rearrage', video.shape)
    video = rearrange(video, 'c f h w -> f c h w')
    if accelerator.is_local_main_process:
        accelerator.print('video shape after rearrange', video.shape)

    path = path[0]
    with torch.no_grad():
        mean_list = []
        std_list = []
        sidx = 0
        while sidx < video.size(0):
            eidx = min(video.size(0), sidx + args.max_frame_num)
            clip=video[sidx:eidx]
            latent_dist = vae.encode(clip, num_frames=clip.shape[0], temp_compress=True).posterior
            mean, std = latent_dist.mean_std()
            mean_list.append(mean)
            std_list.append(std)
            sidx = eidx
        mean = torch.cat(mean_list, dim=0)
        std = torch.cat(std_list, dim=0)
    mean = mean.cpu()
    std = std.cpu()
    fdict = {'vae_mean': mean.clone(), 'vae_std': std.clone()}
    if accelerator.is_local_main_process:
        accelerator.print('mean shape', mean.shape)
        accelerator.print('std shape', std.shape)

    fpath = '%s/%s' % (full_save_path, path)
    pool.apply_async(save_feature, args=(fpath, fdict))
    
    if accelerator.is_local_main_process:
        accelerator.print('Process %d/%s' % (idx+1, len(loader)))
pool.close()
pool.join()

if accelerator.is_local_main_process:
    tprint('End %s' % (full_save_path))
