import os 
import torch 
from einops import rearrange

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from transformers import T5Tokenizer, T5EncoderModel

from data.raw_data.pexel_dataset import PexelsDataset
from model.utils.data_utils import export_to_gif

data_dir = '/ML-A100/team/mm/yanghuan/data/pexels_clip'
train_meta = '/ML-A100/team/mm/zixit/data/pexles_20240405/pexels_meta_train.jsonl'
val_meta = '/ML-A100/team/mm/zixit/data/pexles_20240405/pexels_meta_val.jsonl'


# pexles_train = PexelsDataset(data_dir, train_meta, [16,3,128,256], [[384, 672]], [1.], [30])
# for idx, data in enumerate(pexles_train):
#     print(data['fps'])
#     print(data['num_frames'])

#     print(type(data['video']))
#     print(data['video'].shape)
#     if idx ==2:
#         break

pipeline_load_from = '/ML-A100/team/mm/yanghuan/huggingface/PixArt-alpha/PixArt-XL-2-512x512'
device = 'cuda:1'
dtype = torch.bfloat16

tokenizer = T5Tokenizer.from_pretrained(pipeline_load_from, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pipeline_load_from, subfolder="text_encoder", torch_dtype=dtype).to(device)
text_encoder.requires_grad_(False)

def get_caption_embed(caption_batch):
    caption_embeds_list = []
    caption_embeds_attention_mask_list = []
    for caption in caption_batch:
        tokens = tokenizer(caption, max_length=300, padding="max_length", truncation=True, return_tensors="pt").to(device)
        caption_embeds = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
        caption_embeds_attention_mask = tokens.attention_mask

        caption_embeds_list.append(caption_embeds.squeeze(dim=0))
        caption_embeds_attention_mask_list.append(caption_embeds_attention_mask.squeeze(dim=0))
    caption_embeds_list_t = torch.stack(caption_embeds_list)
    caption_embeds_attention_mask_list_t = torch.stack(caption_embeds_attention_mask_list)

    return caption_embeds_list_t.to(dtype=dtype), caption_embeds_attention_mask_list_t.to(dtype=dtype)

num_frames = 120
pexles_val = PexelsDataset(data_dir, val_meta, [num_frames,3,128,256], [[384, 672]], [1.], [30], include_caption=True)
video_val_dataloader = DataLoader(
    dataset=pexles_val,
    batch_size=2,
    shuffle=False,
    num_workers=8,
    pin_memory=False,
)

for i, batch in enumerate(video_val_dataloader):
    print('---------------------')
    print(i)
    folder = batch['folder']
    v_id = batch['id']
    print(f'{folder}/{v_id}.mp4')
    print(batch['video'].size())
    print(batch['v_feat_path'])

    captions = batch['caption_text']
    print(type(captions))
    print(len(captions))
    print(captions[0])

    caption_text_emb, caption_text_mask = get_caption_embed(captions)
    print(caption_text_emb.shape)
    print(caption_text_mask.shape)

    frames = rearrange(batch['video'], 'b c f h w -> (b f) c h w')

    save_root = './video_debug'
    os.makedirs(save_root, exist_ok=True)

    print(torch.max(frames))
    print(torch.min(frames)) 
    # pexel datatset output pixel range [-1,1]
    # so torch.clamp(((frames+1)/2) to make back [0,1]

    debug_video_path = os.path.join(save_root, str(i)+'.png')
    save_debug_frames = torch.clamp(((frames+1)/2).detach().to(torch.float), min=0, max=1.0)
    save_image(save_debug_frames, debug_video_path, padding=0)

    debug_video_path_gif = os.path.join(save_root, str(i)+'.gif')
    save_debug_frames_forgif = rearrange(save_debug_frames, 'f c h w -> f h w c')

    export_to_gif(torch.unbind(save_debug_frames_forgif, dim=0), debug_video_path_gif, duration=16.0/30)
    if i == 2:
        break
            

    

    
