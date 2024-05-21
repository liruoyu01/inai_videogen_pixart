import os
import gc
import argparse
from datetime import timedelta

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops import rearrange, repeat

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import InitProcessGroupKwargs

from diffusers.utils.torch_utils import randn_tensor

from model.mmvg.datasets.video_feature_dataset import VideoFeatureDataset
from model.mmvg.datasets.image_feature_dataset import ImageFeatureDataset
from model.mmvg.models.pixart_sigma_t2v.transformer import Transformer3DModel
from model.mmvg.diffusion.iddpm import IDDPM
from model.mmvg.utils.logger import ctime

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='t2v_pixart-sigma')
# parser.add_argument('--expr_path', type=str, default='/ML-A100/team/mm/yanghuan/expr/t2v_pixart-sigma_120x512x512_15fps')
parser.add_argument('--expr_path', type=str, default='debug')
parser.add_argument('--log_path', type=str, default='/ML-A100/team/mm/yanghuan')
parser.add_argument('--video_path', type=str, default='/ML-A100/team/mm/yanghuan/data')
# parser.add_argument('--video_meta', type=str, default='/ML-A100/team/mm/yanghuan/data/pixart-sigma_hq-video-512x512_llava-text_feature_20240425.jsonl')
parser.add_argument('--video_meta', type=str, default='/ML-A100/team/mm/yanghuan/data/pixart-sigma_hq-video-512x512_llava-text_feature_debug.jsonl')
parser.add_argument('--num_frame', type=int, default=60)
parser.add_argument('--frame_skip', type=int, default=1)
parser.add_argument('--image_path', type=str, default='/ML-A100/team/mm/yanghuan/data')
# parser.add_argument('--image_meta', type=str, default='/ML-A100/team/mm/yanghuan/data/pixart-sigma_hq-image-512x512_llava-text_feature_20240425.jsonl')
parser.add_argument('--image_meta', type=str, default='/ML-A100/team/mm/yanghuan/data/pixart-sigma_hq-image-512x512_llava-text_feature_debug.jsonl')
parser.add_argument('--num_image', type=int, default=15)
parser.add_argument('--null_text', type=str, default='/ML-A100/team/mm/yanghuan/data/null_feature/pixart_sigma_text.pt')
parser.add_argument('--null_text_prob', type=float, default=0.1)
parser.add_argument('--latent_scale_factor', type=float, default=0.13025)
parser.add_argument('--resolution', type=str, default='512x512')
parser.add_argument('--model_path', type=str, default='/ML-A100/team/mm/yanghuan/huggingface/PixArt-alpha/PixArt-Sigma-XL-2-512-MS/transformer')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--grad_accum', type=int, default=1)
parser.add_argument('--train_step', type=int, default=1000000)
parser.add_argument('--print_step', type=int, default=1)
parser.add_argument('--save_step', type=int, default=1000)
parser.add_argument('--grad_norm', type=float, default=1.0)
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--seed', type=int, default=19910511)
args = parser.parse_args()

accelerator = Accelerator(
    mixed_precision=args.precision,
    gradient_accumulation_steps=args.grad_accum,
    log_with='wandb',
    kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(5400))],
)

accelerator.init_trackers(
    project_name=args.project_name,
    init_kwargs={
        'wandb': {
            'name': os.path.basename(args.expr_path),
            'dir': args.log_path,
            'config': vars(args)
        }
    }
)

if accelerator.is_main_process:
    os.makedirs(args.expr_path, exist_ok=True)

device = accelerator.device
match accelerator.mixed_precision:
    case 'no':
        dtype = torch.float32
    case 'fp16':
        dtype = torch.float16
    case 'bf16':
        dtype = torch.bfloat16

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

optim = AdamW(trainable_params, lr=args.lr)
vdata = VideoFeatureDataset(data_path=args.video_path, meta_file=args.video_meta, num_frame=args.num_frame, frame_skip=args.frame_skip, null_text=args.null_text, null_text_prob=args.null_text_prob)
idata = ImageFeatureDataset(data_path=args.image_path, meta_file=args.image_meta, null_text=args.null_text, null_text_prob=args.null_text_prob)
vloader = DataLoader(vdata, args.batch_size, shuffle=True, drop_last=True, num_workers=2)
iloader = DataLoader(idata, args.batch_size * args.num_image, shuffle=True, drop_last=True, num_workers=2)

accelerator.print('[%s] video dataset: %d' % (ctime(), len(vdata)))
accelerator.print('[%s] image dataset: %d' % (ctime(), len(idata)))

model, optim, vloader, iloader = accelerator.prepare(model, optim, vloader, iloader)

diffusion = IDDPM()

viter = iter(vloader)
iiter = iter(iloader)

num_step = 0
loss_accum = 0.0
loss_count = 0

accelerator.print('[%s] start training' % (ctime()))

while num_step < args.train_step:
    try:
        vmean, vstd, vtext, vmask = next(viter)
    except StopIteration:
        viter = iter(vloader) 
        vmean, vstd, vtext, vmask = next(viter)

    try:
        imean, istd, itext, imask = next(iiter)
    except StopIteration:
        iiter = iter(iloader)
        imean, istd, itext, imask = next(iiter)

    batch_size = args.batch_size
    num_frame = args.num_frame
    num_image = args.num_image

    print('vmean shape', vmean.shape)
    print('vtext shape', vtext.shape)
    print('vmask shape', vmask.shape)
    print('imean shape', imean.shape)
    print('itext shape', itext.shape)
    print('imask shape', imask.shape)

    imean = rearrange(imean, '(b f) c h w -> b f c h w', b=batch_size)
    print('imean shape after rearrange', imean.shape)

    istd = rearrange(istd, '(b f) c h w -> b f c h w', b=batch_size)
    mean = torch.cat((vmean, imean), dim=1)
    std = torch.cat((vstd, istd), dim=1)
    mean = rearrange(mean, 'b f c h w -> (b f) c h w')
    std = rearrange(std, 'b f c h w -> (b f) c h w')

    

    print('mean shape', mean.shape)
    print('std shape', std.shape)

    vtext = repeat(vtext, 'b t c -> b f t c', f=num_frame)
    itext = rearrange(itext, '(b f) t c -> b f t c', b=batch_size)
    vmask = repeat(vmask, 'b t -> b f t', f=num_frame)
    imask = rearrange(imask, '(b f) t -> b f t', b=batch_size)

    print('vtext shape after repeating', vtext.shape)
    print('vmask shape after repeating', vmask.shape)
    print('itext shape after rearrange', itext.shape)
    print('imask shape after rearrange', imask.shape)

    text = torch.cat((vtext, itext), dim=1)
    mask = torch.cat((vmask, imask), dim=1)
    text = rearrange(text, 'b f t c -> (b f) t c')
    mask = rearrange(mask, 'b f t -> (b f) t')
    print('text shape', text.shape)
    print('mask shape', mask.shape)

    sample = randn_tensor(mean.shape, generator=None, device=device)
    latent = (mean + std * sample) * args.latent_scale_factor
    
    timestep = torch.randint(0, 1000, (batch_size,), device=device).long()
    timestep = repeat(timestep, 'b -> (b f)', f=num_frame+num_image)

    model_kwargs = {
        'encoder_hidden_states': text.to(dtype),
        'attention_mask': None,
        'encoder_attention_mask': mask,
        'num_frame': args.num_frame,
        'num_image': args.num_image,
    }
    
    with accelerator.accumulate(model):
        loss_term = diffusion.training_losses_diffusers(model, latent.to(dtype), timestep, model_kwargs=model_kwargs)
        loss = loss_term['loss'].mean()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.grad_norm)
        optim.step()
        optim.zero_grad()
    
    if (num_step + 1) % args.print_step == 0:
        loss_accum += accelerator.gather(loss).mean().item()
        loss_count += 1
    
    if accelerator.sync_gradients:   
        num_step += 1
        
        if num_step % args.print_step == 0:
            loss_accum /= loss_count
            accelerator.print('[%s] step %08d, loss=%.8f' % (ctime(), num_step, loss_accum))
            accelerator.log({'loss': loss_accum}, step=num_step)

        if num_step % args.save_step == 0 and accelerator.is_main_process:
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(os.path.join(args.expr_path, '%08d' % num_step), safe_serialization=True)
        
        loss_accum = 0.0
        loss_count = 0

    torch.cuda.empty_cache()
    gc.collect()

accelerator.print('[%s] end training' % (ctime()))

accelerator.end_training()
