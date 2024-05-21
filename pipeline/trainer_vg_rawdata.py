from datetime import timedelta
import os
import gc
import argparse

from einops import rearrange, repeat
from typing import List

import torch
from torch.optim import AdamW

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from diffusers.utils.torch_utils import randn_tensor
from transformers import T5Tokenizer, T5EncoderModel

from model.utils import *
from model.mmvg.models.pixart_sigma_t2v.transformer import Transformer3DModel
from model.mmvg.diffusion.iddpm import IDDPM
from model.mmvg.utils.logger import ctime
from model.vae_modules.autoencoder_kl_3d_compress import AutoencoderKL_3D

from data.feature_data.mj_feature_dataset import ImageJsonFeatureDataset
from data.raw_data.pexel_dataset import PexelsDataset
# from pipeline.inference_vae3d import load_vae


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default='/ML-A100/team/mm/liruoyu/code/inai_videogen_pixart/exp_dir', help='the dir to save logs and models')
    parser.add_argument('--project_name', type=str, default='vg_replacevae_3d_488_t1')
    parser.add_argument('--num_train_samples', type=int, default=100)

    parser.add_argument('--img_dataset_name', type=str, default='mj_256')
    parser.add_argument('--video_dataset_name', type=str, default='pexel_video')
    parser.add_argument('--num_frame', type=int, default=60)
    parser.add_argument('--num_image', type=int, default=60)
    parser.add_argument('--video_target_fps', type=int, default=15)

    parser.add_argument('--null_text', type=str, default='/ML-A100/team/mm/yanghuan/data/null_feature/pixart_sigma_text.pt')
    parser.add_argument('--null_text_prob', type=float, default=0.1)
    parser.add_argument('--max_text_length', type=int, default=300)

    parser.add_argument('--latent_scale_factor', type=float, default=0.13025)
    parser.add_argument('--resolution', type=str, default='256x256')

    parser.add_argument('--transformer_model_path', type=str, default='/ML-A100/team/mm/yanghuan/huggingface/PixArt-alpha/PixArt-XL-2-256x256/transformer')
    parser.add_argument('--pretrained_vae_ckpt_path', type=str, default='/ML-A100/team/mm/wangqiuyue/experiments/SD_videoVAE_deepspeed/0511_compress_conti25k_conti25k6k_skip2_conv4out2C_2way_333temponlyfo_27k_add4conv/global_step72000.pt')
    parser.add_argument('--pipeline_load_from', type=str, default='/ML-A100/team/mm/yanghuan/huggingface/PixArt-alpha/PixArt-XL-2-512x512')

    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--train_step', type=int, default=1000000)
    parser.add_argument('--print_step', type=int, default=10)
    parser.add_argument('--save_step', type=int, default=1000)
    parser.add_argument('--grad_norm', type=float, default=1.0)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--seed', type=int, default=19910511)
    args = parser.parse_args()
    return args

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
    
    print(f"########### Load ckpt {ckpt_path} ###########")
    assert os.path.exists(ckpt_path)
    vae_state = torch.load(ckpt_path)
    vae.load_state_dict(vae_state, strict=True)

    return vae

def prepare_image_dataset(args):
    img_dataset_path = {
        'sigma_512': [
            '/ML-A100/team/mm/yanghuan/data/', 
            '/ML-A100/team/mm/yanghuan/data/pixart-sigma_hq-image-512x512_llava-text_feature_20240425.jsonl',
        ],
        'mj_512': [
            '/ML-A100/team/mm/yanghuan/data/', 
            '/ML-A100/team/mm/yanghuan/data/pixart-sigma_midjourney-peter-image-512x512_user-llava-text_feature_20240430.jsonl', 
            ['llava_text_feature', ('user_text_feature', 0.5)],
        ],
        'mj_256': [
            '/ML-A100/team/mm/yanghuan/data/', 
            '/ML-A100/team/mm/yanghuan/data/pixart-sigma_midjourney-peter-image-256x256_user-llava-text_feature_20240430.jsonl'
        ],
    }
    assert args.img_dataset_name in img_dataset_path

    data_dir = img_dataset_path[args.img_dataset_name][0]
    meta_json_dir = img_dataset_path[args.img_dataset_name][1]

    mj_feature_dataset = ImageJsonFeatureDataset(
        data_dir, 
        meta_json_dir, 
        num_samples=args.num_train_samples,
        text_feat_name='llava_text_feature', 
        extra_text_feat=('user_text_feature', 0.5),
        sample_latent=False,  # return mean, std, text, mask
    )
    image_dataloader = torch.utils.data.DataLoader(
        dataset=mj_feature_dataset,
        batch_size=args.train_batch_size * args.num_image,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=False,
    )

    return image_dataloader

def prepare_video_dataset(args):
    video_dataset_path = {
        'pexel_video': [
            '/ML-A100/team/mm/yanghuan/data/pexels_clip', 
            '/ML-A100/team/mm/zixit/data/pexles_20240405/pexels_meta_val.jsonl',
        ]
    }
    assert args.video_dataset_name in video_dataset_path

    data_dir = video_dataset_path[args.video_dataset_name][0]
    meta_json_dir = video_dataset_path[args.video_dataset_name][1]

    h,w = list(map(int, args.resolution.split('x')))
    print(f'height/width : {h}/{w}')

    pexles_dataset = PexelsDataset(
        data_dir,
        meta_json_dir, 
        [args.num_frame, 3, h, w], # train_size
        [[512, 512]], # resize_size
        [1.], #resize prob
        [args.video_target_fps], #train fps
        include_caption=True
    )
    video_dataloader = torch.utils.data.DataLoader(
        dataset=pexles_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=False,
    )

    return video_dataloader


class DiTVideoGenTrainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator(
            mixed_precision=args.precision,
            gradient_accumulation_steps=args.grad_accum,
            log_with='wandb',
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(5400))],
        )

        # if self.accelerator.is_main_process:
        self.exp_dir = os.path.join(args.work_dir, args.project_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.results_dir = os.path.join(self.exp_dir, 'results')
        self.logs_dir = os.path.join(self.exp_dir, 'logs')
        self.cache_dir = os.path.join(self.exp_dir, 'cache')

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.accelerator.init_trackers(
            project_name=args.project_name,
            init_kwargs={
                'wandb': {
                    'name': args.project_name,
                    'dir': self.logs_dir,
                    'config': vars(args)
                }
            }
        )
        
        match self.accelerator.mixed_precision:
            case 'no':
                self.dtype = torch.float32
            case 'fp16':
                self.dtype = torch.float16
            case 'bf16':
                self.dtype = torch.bfloat16

        set_seed(args.seed)

        model = Transformer3DModel.from_pretrained(args.transformer_model_path, low_cpu_mem_usage=False, device_map=None)
        model.enable_gradient_checkpointing()
        model.train()

        trainable_params = []
        trainable_parama_count = 0
        for n, p in model.named_parameters():
            p.requires_grad_(True)
            trainable_params.append(p)
            trainable_parama_count += p.numel()
        
        optimizer = AdamW(trainable_params, lr=args.lr)

        img_loader = prepare_image_dataset(args)
        video_loader = prepare_video_dataset(args)

        self.model, self.optim, self.img_loader, self.video_loader = self.accelerator.prepare(
            model, 
            optimizer, 
            img_loader,
            video_loader,
        )

        self.diffusion = IDDPM()

        # load model for video batch feature extraction process
        self.load_text_encoder()
        self.load_pretrained_vae()

    @property
    def device(self):
        return self.accelerator.device
    
    def load_pretrained_vae(self):
        self.vae = load_vae(self.args.pretrained_vae_ckpt_path, cuda_device=self.device).to(dtype=self.dtype)
    
    def load_text_encoder(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.pipeline_load_from, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(self.args.pipeline_load_from, subfolder="text_encoder", torch_dtype=self.dtype).to(self.device)
        self.text_encoder.requires_grad_(False)

    def get_caption_embed(self, caption_batch):
        caption_embeds_list = []
        caption_embeds_attention_mask_list = []
        for caption in caption_batch:
            tokens = self.tokenizer(caption, max_length=self.args.max_text_length, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            caption_embeds = self.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
            caption_embeds_attention_mask = tokens.attention_mask

            caption_embeds_list.append(caption_embeds.squeeze(dim=0))
            caption_embeds_attention_mask_list.append(caption_embeds_attention_mask.squeeze(dim=0))
        caption_embeds_list_t = torch.stack(caption_embeds_list)
        caption_embeds_attention_mask_list_t = torch.stack(caption_embeds_attention_mask_list)

        return caption_embeds_list_t.to(dtype=self.dtype), caption_embeds_attention_mask_list_t.to(dtype=self.dtype)

    def process_video_batch(self, batch):
        video = batch['video']
        num_frames = int(batch['num_frames'][0])
        video = rearrange(video, 'b c f h w -> (b f) c h w').to(device=self.device, dtype=self.dtype)
        posterior = self.vae.encode(video, num_frames=num_frames).posterior
        v_mean, v_std = posterior.mean_std()
        
        caption_text = batch['caption_text']
        v_text, v_mask = self.get_caption_embed(caption_text)
        return v_mean, v_std, v_text, v_mask

    def img_video_joint_latent(self, imean, istd, itext, imask, vmean, vstd, vtext, vmask):
        batch_size = self.args.train_batch_size
        num_frame = self.args.num_frame

        imean = rearrange(imean, '(b f) c h w -> b f c h w', b=batch_size)
        istd = rearrange(istd, '(b f) c h w -> b f c h w', b=batch_size)

        vmean = rearrange(vmean, '(b f) c h w -> b f c h w', b=batch_size)
        vstd = rearrange(vstd, '(b f) c h w -> b f c h w', b=batch_size)

        print(imean.shape)
        print(vmean.shape)

        mean = torch.cat((vmean, imean), dim=1)
        std = torch.cat((vstd, istd), dim=1)
        print(mean.shape)
        print(std.shape)
        
        mean = rearrange(mean, 'b f c h w -> (b f) c h w')
        std = rearrange(std, 'b f c h w -> (b f) c h w')
        print(mean.shape)
        print(std.shape)

        vtext = repeat(vtext, 'b t c -> b f t c', f=num_frame//4)
        vmask = repeat(vmask, 'b t -> b f t', f=num_frame//4)
        itext = rearrange(itext, '(b f) t c -> b f t c', b=batch_size)
        imask = rearrange(imask, '(b f) t -> b f t', b=batch_size)
        
        text = torch.cat((vtext, itext), dim=1)
        mask = torch.cat((vmask, imask), dim=1)
        text = rearrange(text, 'b f t c -> (b f) t c')
        mask = rearrange(mask, 'b f t -> (b f) t')


        sample = randn_tensor(mean.shape, generator=None, device=self.device)
        latent = (mean + std * sample) * self.args.latent_scale_factor
        return latent, text, mask
    
    def train(self):
        num_step = 0
        loss_accum = 0.0
        loss_count = 0
        img_loader_iter = iter(self.img_loader)
        video_loader_iter = iter(self.video_loader)

        self.accelerator.print('[%s] start training' % (ctime()))
        while num_step < self.args.train_step:
            try:
                i_mean, i_std, i_text, i_mask = next(img_loader_iter)
                video_batch = next(video_loader_iter)
                v_mean, v_std, v_text, v_mask = self.process_video_batch(video_batch)

            except StopIteration:
                img_loader_iter = iter(self.img_loader)
                video_loader_iter = iter(self.video_loader)
                i_mean, i_std, i_text, i_mask = next(img_loader_iter)
                video_batch = next(video_loader_iter)
                v_mean, v_std, v_text, v_mask = self.process_video_batch(video_batch)

            latent, text, mask = self.img_video_joint_latent(i_mean, i_std, i_text, i_mask, v_mean, v_std, v_text, v_mask)
            latent = latent * self.args.latent_scale_factor
            
            bs = latent.shape[0]    
            # bs =         
            timestep = torch.randint(0, 1000, (bs,), device=self.device).long()
            # timestep = repeat(timestep, 'b -> (b f)', f=self.args.num_frame // 4 + self.args.num_image)

            model_kwargs = {
                'encoder_hidden_states': text.to(self.dtype),
                'attention_mask': None,
                'encoder_attention_mask': mask,
                'num_frame': self.args.num_frame // 4,
                'num_image': self.args.num_image,
            }
            
            with self.accelerator.accumulate(self.model):
                loss_term = self.diffusion.training_losses_diffusers(self.model, latent.to(self.dtype), timestep, model_kwargs=model_kwargs)
                loss = loss_term['loss'].mean()
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                self.optim.step()
                self.optim.zero_grad()
            
            if (num_step + 1) % self.args.print_step == 0:
                loss_accum += self.accelerator.gather(loss).mean().item()
                loss_count += 1
            
            if self.accelerator.sync_gradients:   
                num_step += 1
                
                if num_step % self.args.print_step == 0:
                    loss_accum /= loss_count
                    self.accelerator.print('[%s] step %08d, loss=%.8f' % (ctime(), num_step, loss_accum))
                    self.accelerator.log({'loss': loss_accum}, step=num_step)

                if num_step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    unwrap_model = self.accelerator.unwrap_model(self.model)
                    unwrap_model.save_pretrained(os.path.join(self.checkpoints_dir, '%08d' % num_step), safe_serialization=True)
                
                loss_accum = 0.0
                loss_count = 0

            torch.cuda.empty_cache()
            gc.collect()

        self.accelerator.print('[%s] end training' % (ctime()))

        self.accelerator.end_training()


def main():
    args = prepare_args()
    trainer = DiTVideoGenTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

# for local test on huoshan devserver
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file distribute_configs/pixart_t2i_huoshan.yaml pipeline/trainer_vg_rawdata.py
