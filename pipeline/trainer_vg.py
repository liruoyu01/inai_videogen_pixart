from datetime import timedelta
import os
import gc

import torch
import torch.nn as nn
from torch.optim import AdamW

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed

# from accelerate import FullyShardedDataParallelPlugin
# from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, FullyShardedDataParallel

from pixart_modules.utils import *
from pixart_modules.data import ImageJsonFeatureDataset

# from diffusers.utils.torch_utils import randn_tensor
from pixart_modules.mmvg.models.pixart_sigma_t2v.transformer import Transformer3DModel
from pixart_modules.mmvg.diffusion.iddpm import IDDPM
from pixart_modules.mmvg.utils.logger import ctime
from pixart_modules.mmvg.datasets.video_feature_dataset import VideoFeatureDataset
from pixart_modules.mmvg.datasets.image_feature_dataset import ImageFeatureDataset


def prepare_video_image_feature_dataset(args):

    img_dataset_path = {
        'sigma_256': ['/ML-A100/team/mm/yanghuan/data/', 'pixart-sigma_hq-image-256x256_llava-text_feature_20240425.jsonl'],
        'sigma_512': ['/ML-A100/team/mm/yanghuan/data/', 'pixart-sigma_hq-image-512x512_llava-text_feature_20240425.jsonl'],
        # 'mj_512': [
        #     '/ML-A100/team/mm/yanghuan/data/', 
        #     'pixart-sigma_midjourney-peter-image-512x512_user-llava-text_feature_20240430.jsonl', 
        #     ['llava_text_feature', ('user_text_feature', 0.5)]
        # ]
    }

    video_dataset_path = {
        'pexel': ['/ML-A100/team/mm/yanghuan/data', 'pixart-sigma_hq-video-512x512_llava-text_feature_debug.jsonl']
    }
    assert args.img_dataset_name in img_dataset_path
    assert args.video_dataset_name in video_dataset_path

    ##################################################################################
    data_root = img_dataset_path[args.img_dataset_name][0]
    json_file_name = img_dataset_path[args.img_dataset_name][1]
    meta_path = os.path.join(data_root, json_file_name)

    img_dataset = ImageFeatureDataset(
        data_path=data_root,
        meta_file=meta_path,
        null_text=args.null_text, 
        null_text_prob=args.null_text_prob
    )

    img_train_loader = torch.utils.data.DataLoader(
        img_dataset,
        num_workers=4,
        drop_last=True,
        # pin_memory=True,
        shuffle=True, 
        batch_size=args.train_batch_size,
    )

    ##################################################################################
    data_root = video_dataset_path[args.video_dataset_name][0]
    json_file_name = video_dataset_path[args.video_dataset_name][1]
    meta_path = os.path.join(data_root, json_file_name)

    video_dataset = VideoFeatureDataset(
        data_path=data_root,
        meta_file=meta_path,
        num_frame=args.num_frame, 
        frame_skip=args.frame_skip, 
        null_text=args.null_text, 
        null_text_prob=args.null_text_prob
    )

    video_train_loader = torch.utils.data.DataLoader(
        video_dataset, 
        batch_size=args.train_batch_size,
        shuffle=True, 
        drop_last=True, 
        # pin_memory=True,
        num_workers=4,
    )

    return img_train_loader, video_train_loader

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

        model = Transformer3DModel.from_pretrained(args.model_path, low_cpu_mem_usage=False, device_map=None)
        model.enable_gradient_checkpointing()
        model.train()

        trainable_params = []
        trainable_parama_count = 0
        for n, p in model.named_parameters():
            p.requires_grad_(True)
            trainable_params.append(p)
            trainable_parama_count += p.numel()
        
        optimizer = AdamW(trainable_params, lr=args.lr)

        img_loader, video_loader = prepare_video_image_feature_dataset(args)

        self.model, self.optim, self.img_loader, self.video_loader = self.accelerator.prepare(
            model, 
            optimizer, 
            img_loader,
            video_loader,
        )

        self.diffusion = IDDPM()

    @property
    def device(self):
        return self.accelerator.device

    def train(self):
        num_step = 0
        loss_accum = 0.0
        loss_count = 0
        dl_iter = iter(self.loader)

        self.accelerator.print('[%s] start training' % (ctime()))
        while num_step < self.args.train_step:
            try:
                latent, text, mask = next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.loader)
                latent, text, mask = next(dl_iter)

            latent = latent * self.args.latent_scale_factor
            
            bs = latent.shape[0]            
            timestep = torch.randint(0, 1000, (bs,), device=self.device).long()

            model_kwargs = {
                'encoder_hidden_states': text.to(self.dtype),
                'attention_mask': None,
                'encoder_attention_mask': mask,
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
